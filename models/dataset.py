import os
import random

import argparse
import pyexr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import struct
import torch
import torchvision
from torch import Tensor
from torch.utils.data import Dataset

# small epsilon to avoid nan
SMALL_EPSILON = 1e-6

def random_crop_tensor(input, crop_size):
    assert(len(input.shape) == 3)
    random.seed()
    _, h, w = input.shape
    crop_h = max(0, h - crop_size)
    crop_w = max(0, w - crop_size)
    random_anchor = [int(random.random() * crop_h), int(random.random() * crop_w)]
    return input[:, random_anchor[0] : min(random_anchor[0] + crop_size, h), random_anchor[1] : min(random_anchor[1] + crop_size, w)]

class SupersampleDataset(Dataset):
    def __init__(self, src_folder: str, input_types: list, crop_size=256, log_trans=true):
        csv_path = os.path.join(src_folder, "data.csv")
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
            build_dataset_csv(src_folder)

        self.src_folder = src_folder
        self.fh_frame = pd.read_csv(csv_path)

        self.data_types_to_fetch = input_types
        self.crop_size = crop_size
        self.log_trans = log_trans

    def __len__(self):
        return len(self.fh_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        for i, fh in enumerate(self.fh_frame.values[idx]):
            data_type = self.fh_frame.columns[i]
            if data_type in self.data_types_to_fetch:
                img_path = os.path.join(self.src_folder, fh)
                if data_type in ["half", "full", "clean", "mat_diffuse"]:
                    if self.log_trans:
                        image = torch.log(torch.load(img_path)[:, :1060, :])
                    else:
                        image = torch.clamp(torch.load(img_path)[:, :1060, :], 0, 1)
                elif data_type in ["mat_ref", "mat_spec_rough"]:
                    image = torch.unsqueeze(torch.load(img_path)[0, :, :], 0)
                else:
                    image = torch.load(img_path)

                image = random_crop_tensor(image, self.crop_size)
                sample[data_type] = image.half()

        return sample


class DenoiseDataset(Dataset):
    def __init__(self, src_folder: str, crop_size=256, log_trans=true):
        csv_path = os.path.join(src_folder, "data.csv")
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
            build_dataset_csv(src_folder)

        self.src_folder = src_folder
        self.fh_frame = pd.read_csv(csv_path)

        self.data_types_to_fetch = ["full", "mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos", "clean"]
        self.crop_size = crop_size
        self.log_trans = log_trans

    def __len__(self):
        return len(self.fh_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        for i, fh in enumerate(self.fh_frame.values[idx]):
            data_type = self.fh_frame.columns[i]
            if data_type in self.data_types_to_fetch:
                img_path = os.path.join(self.src_folder, fh)
                if data_type in ["full", "clean", "mat_diffuse"]:
                    if self.log_trans:
                        image = torch.log(torch.load(img_path)[:, :1060, :])
                    else:
                        image = torch.clamp(torch.load(img_path)[:, :1060, :], 0, 1)
                elif data_type in ["mat_ref", "mat_spec_rough"]:
                    image = torch.unsqueeze(torch.load(img_path)[0, :1060, :], 0)
                else:
                    image = torch.load(img_path)[:, :1060, :]
                image = random_crop_tensor(image, self.crop_size)
                sample[data_type] = image.half()
        return sample


def build_dataset_csv(src_folder: str):
    src_file_denoise_format = "Clean"
    src_file_gbuff_format = ["MaterialDiffuse", "MaterialIoR",
                             "MaterialSpecRough", "WorldNormal", "WorldPosition"]
    src_file_rt_full_1spp = "Full"
    src_file_rt_half_0_5spp = "Half"
    src_file_rt_full_4spp = "4spp"

    data = {'clean': {}, 'full': {}, 'half': {}, 'mat_diffuse': {}, 'mat_ref': {},
            'mat_spec_rough': {}, "world_normal": {}, "world_pos": {}, "4spp": {}}
    files = os.listdir(src_folder)
    for file in files:
        file_type = file.split('-')[0]
        idx = int(file.split('-')[-1].split('.')[0])

        if file_type == src_file_denoise_format:
            data['clean'][idx] = file
        elif file_type == src_file_rt_full_1spp:
            data['full'][idx] = file
        elif file_type == src_file_rt_half_0_5spp:
            data['half'][idx] = file
        elif file_type == src_file_gbuff_format[0]:
            data['mat_diffuse'][idx] = file
        elif file_type == src_file_gbuff_format[1]:
            data['mat_ref'][idx] = file
        elif file_type == src_file_gbuff_format[2]:
            data['mat_spec_rough'][idx] = file
        elif file_type == src_file_gbuff_format[3]:
            data['world_normal'][idx] = file
        elif file_type == src_file_gbuff_format[4]:
            data['world_pos'][idx] = file
        elif file_type == src_file_rt_full_4spp:
            data['4spp'][idx] = file
        else:
            raise NotImplementedError

    for key, value in data.items():
        idx = list(value.keys())
        fh = list(value.values())

        zipped_lists = zip(idx, fh)
        sorted_zipped_lists = sorted(zipped_lists)

        sorted_list = [handle for _, handle in sorted_zipped_lists]
        data[key] = sorted_list

    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(src_folder, "data.csv"), index=False)


def exr_to_tensor(exr_filepath: str, half: bool) -> torch.Tensor:
    image = pyexr.open(exr_filepath)
    channels = image.channels

    img_array = image.get(precision=image.channel_precision[channels[0]])

    if half:
        img_array = img_array[:image.height//2, :image.width//2, :]

    return torch.tensor(np.transpose(img_array, (2, 0, 1)), dtype=torch.float32)


def convert_exrs_to_tensors(src: str, tgt: str):
    for i, exr_file_name in enumerate(os.listdir(src)):
        file_name, file_type = exr_file_name.split('.')

        if file_type == 'exr' and not os.path.exists(os.path.join(tgt, "{name}.pt".format(name=file_name))):
            if file_name.split('-')[0] == 'Half':
                img_tensor = exr_to_tensor(os.path.join(src, exr_file_name), True)
            else:
                img_tensor = exr_to_tensor(os.path.join(src, exr_file_name), False)

            if not os.path.exists(tgt):
                os.mkdir(tgt)

            torch.save(img_tensor, os.path.join(tgt, "{name}.pt".format(name=file_name)))
            print("Image {i}: Converted {name}.exr to {name}.pt".format(i=i, name=file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Preprocess exrs for model runs."))
    parser.add_argument('--input', default='data')
    parser.add_argument('--output', default='processed')
    args = parser.parse_args()

    convert_exrs_to_tensors(args.input, args.output)
