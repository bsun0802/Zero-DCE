import glob
import os
import random

import numpy as np
import cv2
from torch.utils.data import Dataset


def train_val_split(part1_rootdir, dst_dir, splitAt=2421, resized=(512, 512)):
    '''resize the image to 512x512 and put it them in one folder'''
    JPGs = glob.iglob(part1_rootdir + '**/*.JPG', recursive=True)
    JPGs = [jpg for jpg in JPGs if 'Label' not in jpg]
    # assert len(JPGs) == 3021
    random.shuffle(JPGs)

    for jpg in JPGs[:splitAt]:
        img = cv2.imread(jpg)
        img = cv2.resize(img, resized)
        names = jpg.split('/')
        pref, imname = names[-2], names[-1]
        cv2.imwrite(os.path.join(dst_dir, 'train', pref + '_' + imname), img)

    for jpg in JPGs[splitAt:]:
        img = cv2.imread(jpg)
        img = cv2.resize(img, resized)
        names = jpg.split('/')
        pref, imname = names[-2], names[-1]
        cv2.imwrite(os.path.join(dst_dir, 'val', pref + '_' + imname), img)


class SICEPart1(Dataset):
    def __init__(self, img_dir, transform=None):
        self.root_dir = img_dir
        self.images = [im_name for im_name in os.listdir(img_dir)
                       if im_name.split('.')[-1].lower() in ('jpg', 'png')]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        img = cv2.imread(os.path.join(self.root_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        sample = {'name': name, 'img': img}
        return sample


if __name__ == '__main__':
    train_val_split(
        part1_rootdir='/Users/bsun/Downloads/Dataset_Part1/',
        dst_dir='/Users/bsun/repos/Zero-DCE/data/part1-512',
        resized=(512, 512)
    )
