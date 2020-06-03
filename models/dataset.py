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
from tqdm import tqdm

# small epsilon to avoid nan
SMALL_EPSILON = 1e-6

GAMMA = 2.2
INV_GAMMA = 1 / GAMMA

def random_crop_tensor(input, crop_size):
    assert(len(input.shape) == 3)
    random.seed()
    _, h, w = input.shape
    crop_h = max(0, h - crop_size)
    crop_w = max(0, w - crop_size)
    random_anchor = [int(random.random() * crop_h), int(random.random() * crop_w)]
    return input[:, random_anchor[0] : min(random_anchor[0] + crop_size, h), random_anchor[1] : min(random_anchor[1] + crop_size, w)]

class SupersampleDataset(Dataset):
    def __init__(self, src_folder: str, data_types_to_fetch: list, crop_size=256):
        csv_path = os.path.join(src_folder, "data.csv")
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
            build_dataset_csv(src_folder)

        self.src_folder = src_folder
        self.fh_frame = pd.read_csv(csv_path)

        self.data_types_to_fetch = data_types_to_fetch
        self.crop_size = crop_size

    def __len__(self):
        return len(self.fh_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        
        # mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos
        for i, fh in enumerate(self.fh_frame.values[idx]):
            data_type = self.fh_frame.columns[i]
            if data_type in self.data_types_to_fetch:
                img_path = os.path.join(self.src_folder, fh)
                if data_type in ["half", "full", "clean"]:
                    # min_max_arr = np.load(os.path.join(self.src_folder, '{}_min_max.npy'.format(data_type)))
                    # img_np = torch.load(img_path)[:, :1016, :].numpy()

                    # img_np = np.transpose(img_np, axes=(2, 1, 0)) - min_max_arr[0]
                    # img_np = img_np / (min_max_arr[1] - min_max_arr[0])

                    # img_tensor = torch.tensor(np.transpose(img_np, axes=(2, 1, 0)))
                    
                    image = torch.load(img_path)[:, :1016, :]
                    image = torch.pow(image, INV_GAMMA)
                    image = torch.clamp(image, 0, 1)
                elif data_type in ["mat_ref", "mat_spec_rough"]:
                    image = torch.unsqueeze(torch.load(img_path)[0, :1016, :], 0)
                else:
                    image = torch.load(img_path)[:, :1016, :]

                # image = random_crop_tensor(image, self.crop_size)
                sample[data_type] = image

        return sample


class DenoiseDataset(Dataset):
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, src_folder: str, crop_size=256, log_trans=True):
=======
    def __init__(self, src_folder: str, crop_size=256, gamma_trans=true):
>>>>>>> e7fd05b... gamma trans by default
=======
    def __init__(self, src_folder: str, crop_size=256, gamma_trans=True):
>>>>>>> eebd4f7... fix: boolean case
        csv_path = os.path.join(src_folder, "data.csv")
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
            build_dataset_csv(src_folder)

        self.src_folder = src_folder
        self.fh_frame = pd.read_csv(csv_path)

        self.data_types_to_fetch = ["full", "mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos", "clean"]
        self.crop_size = crop_size
        self.gamma_trans = gamma_trans

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
                if data_type in ["full", "clean"]:
                    if self.gamma_trans:
                        image = torch.pow(torch.load(img_path)[:, :1016, :], INV_GAMMA)
                    else:
                        image = torch.clamp(torch.load(img_path)[:, :1016, :], 0, 1)
                elif data_type in ["mat_ref", "mat_spec_rough"]:
                    image = torch.unsqueeze(torch.load(img_path)[0, :1016, :], 0)
                else:
                    image = torch.load(img_path)[:, :1016, :]
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


def fetch_min_max_of_datatype(src: str, data_type: str):
    df = pd.read_csv(os.path.join(src, 'data.csv'))

    dataset_min = None
    dataset_max = None
    for idx, row in tqdm(df.iterrows()):
        data_tensor = torch.load(os.path.join(src, row[data_type]))
        curr_min = np.min(data_tensor.numpy(), axis=(1, 2))
        curr_max = np.max(data_tensor.numpy(), axis=(1, 2))

        if dataset_min is None:
            dataset_min = curr_min
        else:
            to_swap = dataset_min > curr_min
            dataset_min[to_swap] = curr_min[to_swap]

        if dataset_max is None:
            dataset_max = curr_max
        else:
            to_swap = dataset_max < curr_max
            dataset_max[to_swap] = curr_max[to_swap]
    np.save(os.path.join(src, '{}_min_max.npy'.format(data_type)), np.vstack((dataset_min, dataset_max)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Preprocess exrs for model runs."))
    parser.add_argument('--input', default='data')
    parser.add_argument('--output', default='processed')
    parser.add_argument('--compute_min_max', action='store_true')
    args = parser.parse_args()

    if args.compute_min_max:
        for dt in ['half', 'full', 'clean']:
            fetch_min_max_of_datatype(args.input, dt)
            print(np.load(os.path.join(args.input, '{}_min_max.npy'.format(dt))))
    else:
        convert_exrs_to_tensors(args.input, args.output)
