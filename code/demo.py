# import sys
import argparse
from pathlib import Path

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
# import torch.nn
# import torch.nn.functional as F
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
    parser.add_argument('--output-dir', type=str, required=True, help='path to save output')

    args = parser.parse_args()
    return args


args = parse_args()

if torch.cuda.is_available() and args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

outPath = create_dir(args.output_dir)
out_prefix = Path(args.ckpt).stem.split('_')[0]

checkpoint = torch.load(args.ckpt, map_location=device)
hp = checkpoint['HyperParam']
model = DCENet(n_LE=hp['n_LE'], std=hp['std'])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

test_dataset = SICEPart1(args.testDir, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

grid, vsep = make_grid(test_dataset)
curr_h = 0
with torch.no_grad():
    for i, sample in enumerate(test_loader):
        name = sample['name'][0][:-4]
        img_batch = sample['img'].to(device)
        Astack = model(img_batch)
        enhanced_batch = refine_image(img_batch, Astack)

        img = to_numpy(img_batch, squeeze=True)
        enhanced = to_numpy(enhanced_batch, squeeze=True)

        # < Snippet: Comparison>
        fixed_gamma = gamma_correction(img, 0.4)
        adaptive_gamma = gamma_like(img, enhanced)

        row = row_arrange(img, fixed_gamma, adaptive_gamma, enhanced)
        to_be_saved = Image.fromarray((row * 255).astype(np.uint8), mode='RGB')
        to_be_saved.save(outPath.joinpath(f'{out_prefix}_{name}.png'))
        h, _, _ = row.shape
        grid[curr_h:curr_h + h, :, :] = row

        curr_h += (h + vsep)

grid_image = Image.fromarray((grid * 255).astype(np.uint8), mode='RGB')
grid_image.save(outPath.joinpath(f'{out_prefix}_cmp.pdf'))

# <-- Old Snippet, ident 2 times to the right -->
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(img)
# axes[1].imshow(enhanced)
# axes[0].axis('off')
# axes[1].axis('off')

# fig.savefig(outPath.joinpath(name + '_img_enh.pdf'))

# img_jpg = Image.fromarray((img * 255.).astype(np.uint8), mode='RGB')
# enh_jpg = Image.fromarray((enhanced * 255.).astype(np.uint8), mode='RGB')
# img_jpg.save(outPath.joinpath(name + '_original.jpg'))
# enh_jpg.save(outPath.joinpath(name + '_enhanved.jpg'))
# <-- END Old Snippet -->
