import os
import sys
import time
import shutil
import argparse

from datetime import datetime
from subprocess import call

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from dataset import *
from utils import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device name, cuda:0')
    parser.add_argument('--experiment', required=True,
                        help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
    parser.add_argument('--baseDir', type=str, required=True,
                        help='baseDir/train, baseDir/val will be used')
    parser.add_argument('--testDir', type=str, required=True, help='path to test images')
    parser.add_argument('--weights', type=float, nargs='+', required=True,
                        help=('A list of weights for [w_spa, w_exp, w_col, w_tvA]. '
                              'That is, spatial loss, exposure loss, color constancy, '
                              'and total variation respectively'))

    parser.add_argument('--numEpoch', type=int, default=120)
    parser.add_argument('--loss', type=int, default=1, choices=[1, 2])
    args = parser.parse_args()
    return args


def train(loaders, model, optimizer, scheduler, epoch, num_epochs, **kwargs):
    w_spa, w_exp, w_col, w_tvA = kwargs['w_spa'], kwargs['w_exp'], kwargs['w_col'], kwargs['w_tvA']
    to_gray, neigh_diff = kwargs['to_gray'], kwargs['neigh_diff']
    spa_rsize, exp_rsize = kwargs['spa_rsize'], kwargs['exp_rsize']

    model = model.train()
    print(f'--- Epoch {epoch}, LR = {[group["lr"] for group in optimizer.param_groups]} ---')

    logger = Logger(n=5)

    total_i = len(loaders['train'])
    prev = time.time()
    for i, sample in enumerate(loaders['train']):
        optimizer.zero_grad()

        img_batch = sample['img'].to(device)
        results, Astack = model(img_batch)
        enhanced_batch = results[-1]

        L_spa = w_spa * spatial_consistency_loss(
            enhanced_batch, img_batch, to_gray, neigh_diff, spa_rsize)
        L_exp = w_exp * exposure_control_loss(enhanced_batch, exp_rsize, E=0.62)
        if args.loss == 1:
            L_col = w_col * color_constency_loss(enhanced_batch)
        elif args.loss == 2:
            L_col = w_col * color_constency_loss2(enhanced_batch, img_batch)
        L_tvA = w_tvA * alpha_total_variation(Astack)
        loss = L_spa + L_exp + L_col + L_tvA

        logger.update([l.item() for l in (L_spa, L_exp, L_col, L_tvA, loss)])

        loss.backward()

        optimizer.step()

        timeElapsed = (time.time() - prev)
        print(logger.TRAIN_INFO.format(
            epoch + 1, num_epochs, i + 1, total_i, timeElapsed,
            logger.val[-1], logger.avg[-1],
            ', '.join([f'{logger.val[0] * 100:.4f}(100x)']  # L_spa displayed in 100x
                      + ['{:.4f}'.format(l) for l in logger.val[1:-1]]),
            ', '.join([f'{logger.avg[0] * 100:.4f}(100x)']  # avg. L_spa displayed in 100x
                      + ['{:.4f}'.format(l) for l in logger.avg[1:-1]])), flush=True)

        prev = time.time()

    loss_history.append(logger.avg)  # store avg. train losses for this epoch

    model = model.eval()
    start = time.time()
    with torch.no_grad():
        for i, sample in enumerate(loaders['val']):
            img_batch = sample['img'].to(device)
            results, Astack = model(img_batch)
            enhanced_batch = results[-1]

            L_spa = w_spa * spatial_consistency_loss(
                enhanced_batch, img_batch, to_gray, neigh_diff, spa_rsize)
            L_exp = w_exp * exposure_control_loss(enhanced_batch, exp_rsize, E=0.62)
            if args.loss == 1:
                L_col = w_col * color_constency_loss(enhanced_batch)
            elif args.loss == 2:
                L_col = w_col * color_constency_loss2(enhanced_batch, img_batch)
            L_tvA = w_tvA * alpha_total_variation(Astack)
            loss = L_spa + L_exp + L_col + L_tvA

            logger.val_losses.append(loss.item())

    val_loss = np.mean(logger.val_losses)
    scheduler.step(val_loss)

    duration = (time.time() - start)  # how long does one round of validation takes
    print(logger.VAL_INFO.format(epoch + 1, num_epochs, val_loss, duration), flush=True)

    return val_loss


args = parse_args()


if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)


basedir = args.baseDir
train_dir = os.path.join(basedir, 'train')
val_dir = os.path.join(basedir, 'val')

train_dataset = SICEPart1(train_dir, transform=transforms.ToTensor())
trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
val_dataset = SICEPart1(val_dir, transform=transforms.ToTensor())
valloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

loaders = {'train': trainloader, 'val': valloader}


w_spa, w_exp, w_col, w_tvA = args.weights
hp = dict(lr=1e-4, wd=0, lr_decay_factor=0.97,
          w_spa=w_spa, w_exp=w_exp, w_col=w_col, w_tvA=w_tvA,
          spa_rsize=4, exp_rsize=16)

model = DCENet(n=8, return_results=[4, 6, 8])
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=hp['wd'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, mode='min', factor=hp['lr_decay_factor'], threshold=3e-4)

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

experiment = args.experiment

ckpt_dir = '../train-jobs/ckpt'

logfile = open(os.path.join('../train-jobs/log', args.experiment + '.log'), 'w')
sys.stdout = logfile

loss_history, best_loss = [], float('inf')
num_epochs = args.numEpoch
print(f'[START TRAINING JOB] -{experiment} on {datetime.now().strftime("%b %d %Y %H:%M:%S")}')
for epoch in range(num_epochs):
    val_loss = train(loaders, model, optimizer, scheduler, epoch, num_epochs,
                     to_gray=to_gray, neigh_diff=neigh_diff,
                     w_spa=hp['w_spa'], w_exp=hp['w_exp'], w_col=hp['w_col'], w_tvA=hp['w_tvA'],
                     spa_rsize=hp['spa_rsize'], exp_rsize=hp['exp_rsize'])
    # loss_history.append(val_loss)  # total validation loss isn't important
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)

    save_ckpt({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'HyperParam': hp,
        'val_loss': val_loss,
        'loss_history': loss_history,
        'model_src': open('./model.py', 'rt').read(),
        'train_src': open('./train.py', 'rt').read()
    }, is_best, experiment, epoch, ckpt_dir)

    # Evaluation per 30 epoch
    if (epoch + 1) % 40 == 0:
        CMD = ['python', 'eval.py', '--device=0', f'--testDir={args.testDir}',
               f'--ckpt=../train-jobs/ckpt/{args.experiment}_ckpt.pth']
        call(CMD)


logfile.close()

np.save(
    os.path.join('../train-jobs/evaluation', experiment, 'loss_hist.npy'),
    np.array(loss_history)
)

shutil.make_archive(f'../train-jobs/evaluation/{args.experiment}',
                    'zip', f'../train-jobs/evaluation/{args.experiment}')
