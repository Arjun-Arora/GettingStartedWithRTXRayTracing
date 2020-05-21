import array
import numpy as np

import os
import OpenEXR, Imath

import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class TracerDataset(Dataset):
    def __init__(self, file_handle_csv: str, inp_dirs: List, out_dir: str):
        self.files_frame = pd.read_csv(file_handle_csv, names=['fh'])
        self.inp_dirs = inp_dirs
        self.out_dir = out_dir

    def __len__(self):
        return len(self.files_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        for dir in self.inp_dirs:
            img_name = os.path.join(dir, self.files_frame.values[idx][0])

            image = OpenEXR.InputFile(img_name)
            channels = image.header()['channels'].keys()

            img_arr = []
            for channel in channels:
                # TODO: will need to change the pixel type to HALF.
                img_channel = list(image.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)))
                img_arr.append(img_channel)

            handle = dir.split('/')[-1]
            sample[handle] = np.array(img_arr)

        return sample


data = TracerDataset('data/data.csv', ['data/super-resolution/input/g-buffer',
                                       'data/super-resolution/input/rt-half-image-1spp'],
                     'data/super-resolution/output/rt-full-image-1spp')

