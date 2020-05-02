import os
import sys
import time
import argparse
from datetime import datetime

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from dataset import *
from utils import *
from model import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device name, cuda:0')
    parser.add_argument('--experiment', required=True, help='experiment_best_model.pth')
    parser.add_argument('--baseDir', type=str, required=True, 
                        help='baseDir/train, baseDir/val will be used')
    
    parser.add_argument('--numEpoch', type=int, default=120)
    args = parser.parse_args()
    return args


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
valloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

loaders = {'train': trainloader, 'val': valloader}


def train(loaders, model, optimizer, scheduler, epoch, num_epochs, **kwargs):
    w_exp, w_col, w_tvA = kwargs['w_exp'], kwargs['w_col'], kwargs['w_tvA']
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
        Astack = model(img_batch)  # [B, 3*n_LE, H, W], A ranges in [-1, 1] because of tanh
        # for DEBUG 
        # np.save(f'Astack_{epoch}_{i}', Astack.detach().cpu().numpy())
        # print(Astack.min(), Astack.max())

        enhanced_batch = refine_image(img_batch, Astack)

        L_spa = spatial_consistency_loss(enhanced_batch, img_batch, to_gray, neigh_diff, spa_rsize)
        L_exp = w_exp * exposure_control_loss(enhanced_batch, exp_rsize)
        L_col = w_col * color_constency_loss(enhanced_batch)
        L_tvA = w_tvA * alpha_total_variation(Astack)
        loss = L_spa + L_exp + L_col + L_tvA

        logger.update([l.item() for l in (L_spa, L_exp, L_col, L_tvA, loss)])

        loss.backward()

        optimizer.step()

        gstep = epoch * len(loaders['train']) + i + 1

        # for DEBUG: tracking changes of A
        # if gstep % 200 == 0:
        #     np.save(os.path.join(dump_dir, f'Astack_{epoch}_{i}'), Astack.detach().cpu().numpy())

        timeElapsed = (time.time() - prev)
        print(logger.TRAIN_INFO.format(
                epoch + 1, num_epochs, i + 1, total_i, timeElapsed,
                logger.val[-1], logger.avg[-1],
                ', '.join([f'{logger.val[0] * 100:.4f}(100x)'] +  # L_spa displayed in 100x
                          ['{:.4f}'.format(l) for l in logger.val[1:-1]]),
                ', '.join([f'{logger.avg[0] * 100:.4f}(100x)'] + # avg. L_spa displayed in 100x
                          ['{:.4f}'.format(l) for l in logger.avg[1:-1]])), flush=True)
        prev = time.time()

    model = model.eval()
    start = time.time()
    with torch.no_grad():
        for i, sample in enumerate(loaders['val']):
            img_batch = sample['img'].to(device)
            Astack = model(img_batch)

            enhanced_batch = refine_image(img_batch, Astack)

            L_spa = spatial_consistency_loss(
                enhanced_batch, img_batch, to_gray, neigh_diff, spa_rsize)
            L_exp = w_exp * exposure_control_loss(enhanced_batch, exp_rsize)
            L_col = w_col * color_constency_loss(enhanced_batch)
            L_tvA = w_tvA * alpha_total_variation(Astack)
            loss = L_spa + L_exp + L_col + L_tvA

            logger.val_losses.append(loss.item())

    val_loss = np.mean(logger.val_losses)
    scheduler.step(val_loss)

    duration = (time.time() - start)  # how long does one round of validation takes
    print(logger.VAL_INFO.format(epoch + 1, num_epochs, val_loss, duration), flush=True)

    return val_loss

hp = dict(lr=5e-4, wd=0.0, lr_decay_factor=0.9,
          n_LE=8, std=0.05, 
          w_exp=5, w_col=0.5, w_tvA=1,
          spa_rsize=4, exp_rsize=8)  # exp_rsize = 8 for 256x256, 16 for 512x512

model = DCENet(n_LE=hp['n_LE'], std=hp['std'])
model.to(device)

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

grouped_params = group_params(model)  # group params into decay(weight) and no_decay(bias)

optimizer = optim.Adam(grouped_params, lr=hp['lr'], weight_decay=hp['wd'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.numEpoch // 10, mode='min', factor=hp['lr_decay_factor'], threshold=3e-4)

experiment = args.experiment

ckpt_dir = '../train-jobs/ckpt'
dump_dir = os.path.join('../train-jobs/log', experiment)
os.makedirs(dump_dir, exist_ok=True)

logfile = open(os.path.join('../train-jobs/log', args.experiment + '.log'), 'w')
sys.stdout = logfile

loss_history, best_loss = [], float('inf')
num_epochs = args.numEpoch
print(f'[START TRAINING JOB] -{experiment} on {datetime.now().strftime("%b %d %Y %H:%M:%S")}')
for epoch in range(num_epochs):
    val_loss = train(loaders, model, optimizer, scheduler, epoch, num_epochs,
                     to_gray=to_gray, neigh_diff=neigh_diff,
                     w_exp=hp['w_exp'], w_col=hp['w_col'], w_tvA=hp['w_tvA'],
                     spa_rsize=hp['spa_rsize'], exp_rsize=hp['exp_rsize'])
    loss_history.append(val_loss)
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)

    save_ckpt({
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'HyperParam': hp,
        'val_loss': val_loss,
        'loss_history': loss_history
    }, is_best, experiment, epoch, ckpt_dir)

logfile.close()
