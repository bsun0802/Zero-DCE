import tarfile
import argparse
import subprocess
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from model import *
from utils import *
from dataset import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True, help='device num, cuda:0')
    parser.add_argument('--ckpt', type=str, required=True, help='path to *_best_model.pth')
    parser.add_argument('--inputDir', type=str, required=True, help='path to save output')

    args = parser.parse_args()
    return args


args = parse_args()

if torch.cuda.is_available() and args.device >= 0:
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)


test_dataset = SICEPart1(args.inputDir, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

inPath = Path(args.inputDir).expanduser().resolve()
outDir = inPath.parent.joinpath(inPath.stem + '_output')
create_dir(outDir)

checkpoint = torch.load(args.ckpt, map_location=device)

model = DCENet(n=8, return_results=[4, 6, 8])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

to_gray, neigh_diff = get_kernels(device)  # conv kernels for calculating spatial consistency loss

with torch.no_grad():
    for i, sample in enumerate(test_loader):
        names = sample['name']
        img_batch = sample['img'].to(device)
        results, alpha_stacked = model(img_batch)
        enhanced_batch = results[1]

        for name, enhanced in zip(names, enhanced_batch):
            torchvision.utils.save_image(enhanced, str(outDir.joinpath(name[:-4] + '_enh.jpg')))

with tarfile.open(inPath + '.tar.gz', 'w:gz') as tar:
    tar.add(outDir, arcname=outDir.stem)
    tar.add(args.ckpt, arcname=os.path.basename(args.ckpt))

# CMD = ['tar', 'czfP', inPath + '.tar.gz', args.ckpt, args.inputDir, str(outDir)]
# subprocess.call(CMD)
