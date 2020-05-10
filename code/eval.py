import os
import sys
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from model import *
from utils import *
from dataset import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device num, cuda:0')
    parser.add_argument('--testDir', type=str, required=True, help='path to test images')
    parser.add_argument('--ckpt', type=str, required=True, help='path to *_best_model.pth')

    args = parser.parse_args()
    return args


args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

checkpoint = torch.load(args.ckpt, map_location=device)
hp = checkpoint['HyperParam']
print(hp)
model = DCENet(n_LE=hp['n_LE'], std=hp['std'])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

test_dataset = SICEPart1(args.testDir, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

experiment = os.path.basename(args.ckpt).split('_')[0]
output_dir = os.path.join('../train-jobs/evaluation', experiment)
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for i, sample in enumerate(test_loader):
        name = sample['name'][0][:-4]
        img_batch = sample['img'].to(device)

        Astack = model(img_batch)
        enhanced_batch, cache = refine_image(img_batch, Astack, eval=True)

        img = to_numpy(img_batch, squeeze=True)
        enhanced = to_numpy(enhanced_batch, squeeze=True)
        Astack = to_numpy(Astack, squeeze=True)

        np.savez_compressed(os.path.join(output_dir, name),
                            original=img, enhanced=enhanced, Astack=Astack)

        fig = plot_LE(cache)
        fig.savefig(os.path.join(output_dir, 'LE_' + name + '.jpg'), dpi=300)
        plt.close(fig)

        fig = plot_result(img, enhanced, Astack, hp['n_LE'], scaler=None)
        fig.savefig(os.path.join(output_dir, 'res_' + name + '.jpg'), dpi=300)
        plt.close(fig)

        fig = plot_alpha_hist(Astack)
        fig.savefig(os.path.join(output_dir, 'A_' + name + '.jpg'), dpi=150)
        plt.close(fig)
