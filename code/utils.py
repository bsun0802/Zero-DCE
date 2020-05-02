import os
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# 1. helper functions for Zero-DCE
def light_enhancement(x, alpha):
    '''element-wise gamma correction production'''
    return x + alpha * x * (1 - x)


def refine_image(img, Astack):
    An = torch.split(Astack, 3, 1)
    for A in An:
        img = light_enhancement(img, A)
    return img


def alpha_total_variation(A):
    delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

    # TV used here: L-1 norm, sum R,G,B independently
    # Other variation of TV loss can be found by google search
    tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
    return loss


def exposure_control_loss(enhances, rsize=16, E=0.6):
    avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray algo here: (R+G+B)/3
    exp_loss = (avg_intensity - E).abs().mean()
    return exp_loss


def color_constency_loss(enhances):
    batch_J = enhances.mean((2, 3))
    col_loss = torch.mean((batch_J[:, 0] - batch_J[:, 1]) ** 2
                          + (batch_J[:, 1] - batch_J[:, 2]) ** 2
                          + (batch_J[:, 2] - batch_J[:, 0]) ** 2)
    return col_loss


def get_kernels(device):
    # weighted RGB to gray
    K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1).to(device)

    # kernel for neighbor diff
    K2 = torch.tensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                       [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                       [[0, 0, 0], [0, 1, -1], [0, 0, 0]]], dtype=torch.float32)
    K2 = K2.unsqueeze(1).to(device)
    return K1, K2


def spatial_consistency_loss(enhances, originals, to_gray, neigh_diff, rsize=4):
    # convert to gray
    enh_gray = F.conv2d(enhances, to_gray)
    ori_gray = F.conv2d(originals, to_gray)

    # average intensity of local regision
    enh_avg = F.avg_pool2d(enh_gray, rsize)
    ori_avg = F.avg_pool2d(ori_gray, rsize)

    # calculate spatial consistency loss via convolution
    enh_pad = F.pad(enh_avg, (1, 1, 1, 1), mode='replicate')
    ori_pad = F.pad(ori_avg, (1, 1, 1, 1), mode='replicate')
    enh_diff = F.conv2d(enh_pad, neigh_diff)
    ori_diff = F.conv2d(ori_pad, neigh_diff)

    spa_loss = torch.pow((enh_diff - ori_diff), 2).sum(1).mean()
    return spa_loss


# 2.model helpers
def group_params(model):
    decay, no_decay = [], []
    for name, par in model.named_parameters():
        if 'bias' in name:
            no_decay.append(par)
        else:
            decay.append(par)

    groups = [dict(params=decay), dict(params=no_decay, weight_decay=0.)]
    return groups


# 3. training helpers
class Logger:
    TRAIN_INFO = '[TRAIN] - EPOCH {:d}/{:d}, Iters {:d}/{:d}, {:.1f} s/iter, \
LOSS / LOSS(AVG): {:.4f}/{:.4f}, Components / Comp. Avg.: {} / {}'.strip()

    VAL_INFO = '[Validation] - EPOCH {:d}/{:d} - Validation Avg. LOSS: {:.4f}, in {:.2f} secs  '
    VAL_INFO += '- ' + datetime.now().strftime('%X') + ' -'

    def __init__(self, n):
        self.val = np.zeros(n)
        self.sum = np.zeros(n)
        self.count = 0
        self.avg = np.zeros(n)

        self.val_losses = []

    def update(self, losses):
        self.val = np.array(losses)  # log the loss of current batch
        self.sum += self.val
        self.count += 1
        self.avg = self.sum / self.count  # averaged loss of batches seen so far


def save_ckpt(state, is_best, experiment, epoch, ckpt_dir):
    filename = os.path.join(ckpt_dir, f'{experiment}_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        print(f'[BEST MODEL] Saving best model, obtained on epoch = {epoch + 1}')
        shutil.copy(filename, os.path.join(ckpt_dir, f'{experiment}_best_model.pth'))


# 4. tensor and numpy
def to_numpy(t, squeeze=False, to_HWC=True):
    x = t.detach().cpu().numpy()
    if squeeze:
        x = x.squeeze()
    if to_HWC:
        x = x.transpose((1, 2, 0))
    return x


def unnormalize(x):
    'revert [-1,1] to [0, 1]'
    return x / 2 + 0.5


def standardize(x):
    'standardize a tensor/array to [0, 1]'
    mi, mx = x.min(), x.max()
    return (x - mi) / (mx - mi + 1e-10)


# 5. Visulization
def plot_staff(img, enhanced, Astack, n_LE, scaler=None):
    Ar, Ag, Ab = Astack[..., 0::3].mean(2), Astack[..., 1::3].mean(2), Astack[..., 2::3].mean(2)
    if scaler and Ar.min() < 0:
        Ar = scaler(Ar)
    if scaler and Ag.min() < 0:
        Ag = scaler(Ag)
    if scaler and Ab.min() < 0:
        Ab = scaler(Ab)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    fig.subplots_adjust(wspace=0.1)
    axes[0].imshow(img)
    axes[1].imshow(Ar, cmap='jet')
    axes[2].imshow(Ag, cmap='jet')
    axes[3].imshow(Ab, cmap='jet')
    axes[4].imshow(enhanced)
    titles = ['Original',
              r'$\mathcal{A}^{R}' + f'_{n_LE}$',
              r'$\mathcal{A}^{G}' + f'_{n_LE}$',
              r'$\mathcal{A}^{B}' + f'_{n_LE}$',
              'Enhanced']
    for i in range(5):
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    fig.tight_layout()
    return fig
