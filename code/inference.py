import sys
import argparse
from pathlib import Path

from PIL import Image
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
    parser.add_argument('--ckpt', type=str, required=True, help='path to *_best_model.pth')
    parser.add_argument('--outputDir', type=str, required=True, help='path to save output')
    parser.add_argument('--inputDir', type=str, required=True, help='path to save output')
    parser.add_argument('--numEnh', type=int, required=True, help='number of LE to use')

    args = parser.parse_args()
    return args


args = parse_args()

if torch.cuda.is_available() and args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

dirname = Path(args.ckpt).expanduser().resolve().stem
outDir = Path(args.outputDir).joinpath(dirname)
create_dir(outDir)

checkpoint = torch.load(args.ckpt, map_location=device)
hp = checkpoint['HyperParam']
print(hp)
model = DCENet(n_LE=hp['n_LE'], std=hp['std'])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

test_dataset = SICEPart1(args.inputDir, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
with torch.no_grad():
    for i, sample in enumerate(test_loader):
        name = sample['name'][0][:-4]
        img_batch = sample['img'].to(device)
        Astack = model(img_batch)
        enhanced_batch = refine_image(img_batch, Astack, eval=False, num_enh=args.numEnh)  # !!!

        enhanced = to_numpy(enhanced_batch, squeeze=True)
        enh_jpg = Image.fromarray((enhanced * 255.).astype(np.uint8), mode='RGB')
        enh_jpg.save(outDir.joinpath(name + '_enh.png'))
