import os
import argparse

# import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn
# import torch.nn.functional as F
import torchvision
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
epoch = checkpoint['epoch']
print(hp)
model = DCENet(n=8, return_results=[4, 6, 8])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

test_dataset = SICEPart1(args.testDir, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

experiment = os.path.basename(args.ckpt).split('_')[0]
output_dir = os.path.join('../train-jobs/evaluation', experiment, str(epoch))
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for i, sample in enumerate(test_loader):
        name = sample['name'][0][:-4]
        img_batch = sample['img'].to(device)

        results, alpha_stacked = model(img_batch)
        # !! NOTE, for this demo pre-trained model, we train with number of iteration n=8
        # !! but take the 4-th iterations as result
        enhanced_batch = results[1]

        img = to_numpy(img_batch, squeeze=True)
        enhanced = to_numpy(enhanced_batch, squeeze=True)
        alpha_stacked = to_numpy(alpha_stacked, squeeze=True)

        fp = os.path.join(output_dir, 'LE_' + name + '.jpg')
        torchvision.utils.save_image(torch.cat(results, 0), fp)

        fig = plot_result(img, enhanced, alpha_stacked, n_LE='4', scaler=None)
        fig.savefig(os.path.join(output_dir, 'res_4_outof_8' + name + '.jpg'), dpi=300)
        plt.close(fig)

        fig = plot_alpha_hist(alpha_stacked)
        fig.savefig(os.path.join(output_dir, 'A_' + name + '.jpg'), dpi=150)
        plt.close(fig)
