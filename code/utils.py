import os
import shutil
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


# 1. helper functions for Zero-DCE
def light_enhancement(x, alpha):
    '''element-wise gamma correction production'''
    return x + alpha * x * (1 - x)


def refine_image(img, Astack, eval=False, num_enh=999):
    An = torch.split(Astack, 3, 1)
    num_enh = min(num_enh, len(An))
    if not eval:
        for A in An[:num_enh]:
            img = light_enhancement(img, A)
        return img
    else:
        cache = [to_numpy(img, squeeze=True)]
        for A in An[:num_enh]:
            img = light_enhancement(img, A)
            cache.append(to_numpy(img, squeeze=True))
        return img, cache


def alpha_total_variation(A):
    '''
    Links: https://remi.flamary.com/demos/proxtv.html
           https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
    '''
    delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

    # TV used here: L-1 norm, sum R,G,B independently
    # Other variation of TV loss can be found by google search
    tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
    return loss


def exposure_control_loss(enhances, rsize=16, E=0.6):
    avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray: (R+G+B)/3
    exp_loss = (avg_intensity - E).abs().mean()
    return exp_loss


def color_constency_loss(enhances):
    plane_avg = enhances.mean((2, 3))
    col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                          + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                          + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
    return col_loss


def color_constency_loss2(enhances, originals):
    enh_cols = enhances.mean((2, 3))
    ori_cols = originals.mean((2, 3))
    rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
    gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
    br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
    col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
    return col_loss


def get_kernels(device):
    # weighted RGB to gray
    # K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1).to(device)
    K1 = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32).view(1, 3, 1, 1).to(device)

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
def gamma_correction(img, gamma):
    return np.power(img, gamma)


def gamma_like(img, enhanced):
    x, y = img.mean(), enhanced.mean()
    gamma = np.log(y) / np.log(x)
    return gamma_correction(img, gamma)


def to_numpy(t, squeeze=False, to_HWC=True):
    x = t.detach().cpu().numpy()
    if squeeze:
        x = x.squeeze()
    if to_HWC:
        x = x.transpose((1, 2, 0))
    return x


# 5. Visulization
def plot_result(img, enhanced, Astack, n_LE, scaler=None):
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

    rmap = axes[1].imshow(Ar, cmap='jet')
    gmap = axes[2].imshow(Ag, cmap='jet')
    bmap = axes[3].imshow(Ab, cmap='jet')
    fig.colorbar(rmap, ax=axes[1])
    fig.colorbar(gmap, ax=axes[2])
    fig.colorbar(bmap, ax=axes[3])

    axes[4].imshow(enhanced)

    titles = ['Original',
              r'$\mathcal{A}^{R}' + f'_{n_LE}$', r'$\mathcal{A}^{G}' + f'_{n_LE}$',
              r'$\mathcal{A}^{B}' + f'_{n_LE}$', 'Enhanced']
    for i in range(5):
        axes[i].set_title(titles[i])
        axes[i].axis('off')

    fig.tight_layout()
    return fig


def plot_LE(cache):
    n = len(cache)
    fig, axes = plt.subplots(1, n, figsize=(10, 2))
    for i, img in enumerate(cache):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'n={i}')
    axes[0].set_title('Original')
    axes[-1].set_title(f'n={i+1}, Enhanced')

    fig.tight_layout()
    return fig


def plot_alpha_hist(Astack):
    n = Astack.shape[2] // 3
    figsize = (15, 1.5 * 3) if n > 4 else (10, 2 * 3)
    fig, axes = plt.subplots(3, n, figsize=figsize)

    for r in range(3):
        channels = Astack[..., r::3]
        for c in range(n):
            axes[r][c].hist(channels[..., c].ravel())

    for c in range(n):
        axes[0][c].set_title(c + 1)

    fig.tight_layout()
    return fig


def putText(im, *args):
    text, pos, font, size, color, scale = args
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = cv2.putText(im, text, pos, font, size, color, scale)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def row_arrange(inp, fixed, adaptive, algo):
    pos = (25, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (118 / 255., 117 / 255., 0.)

    inp = putText(inp, 'Input', pos, font, 1.5, color, 2)
    fixed = putText(fixed, 'Gamma(fixed=0.4)', pos, font, 1.5, color, 2)
    adaptive = putText(adaptive, 'Gamma(adaptive)', pos, font, 1.5, color, 2)
    algo = putText(algo, 'ZeroDCE', pos, font, 1.5, color, 2)

    return cv2.hconcat([inp, fixed, adaptive, algo])


def make_grid(dataset, vsep=8):
    n = len(dataset)
    img = to_numpy(dataset[0]['img'])
    h, w, _ = img.shape
    grid = np.ones((n * h + vsep * (n - 1), 4 * w, 3), dtype=np.float32)
    return grid, vsep


# system path
def create_dir(path):
    'create directory if not exist'
    if isinstance(path, str):
        path = Path(path).expanduser().resolve()

    if path.exists():
        if path.is_dir():
            print('Output dir already exists.')
        else:
            sys.exit('[ERROR] You specified a file, not a folder. Please revise --outputDir')
    else:
        path.mkdir(parents=True)
    return path


# DEPRECATED
# def unnormalize(x):
#     'revert [-1,1] to [0, 1]'
#     return x / 2 + 0.5


# def standardize(x):
#     'standardize a tensor/array to [0, 1]'
#     mi, mx = x.min(), x.max()
#     return (x - mi) / (mx - mi + 1e-10)
