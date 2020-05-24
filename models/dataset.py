import os

import Imath
import OpenEXR
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import struct
import torch
from torch.utils.data import Dataset


class SupersampleDataset(Dataset):
    def __init__(self, src_folder: str):
        csv_path = os.path.join(src_folder, "data.csv")
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
            build_dataset_csv(src_folder)

        self.src_folder = src_folder
        self.fh_frame = pd.read_csv(csv_path)

        self.data_types_to_fetch = ["full", "half"]

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
                image = torch.load(img_path)

                sample[data_type] = image

        return sample


def build_dataset_csv(src_folder: str):
    src_file_denoise_format = "Clean"
    src_file_gbuff_format = ["MaterialDiffuse", "MaterialIoR",
                             "MaterialSpecRough", "WorldNormal", "WorldPosition"]
    src_file_rt_full_1spp = "Full"
    src_file_rt_half_0_5spp = "Half"

    data = {'clean': {}, 'full': {}, 'half': {}, 'mat_diffuse': {}, 'mat_ref': {},
            'mat_spec_rough': {}, "world_normal": {}, "world_pos": {}}
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
    image = OpenEXR.InputFile(exr_filepath)
    channels = image.header()['channels'].keys()
    dw = image.header()['dataWindow']
    img_dim = (len(channels), dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    if half:
        img_array = np.zeros((img_dim[0], img_dim[1] // 2, img_dim[2] // 2))
    else:
        img_array = np.zeros(img_dim)

    for i, channel in enumerate(channels):
        ch_bytes = image.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        ch_np = np.fromstring(ch_bytes, dtype=np.float32)
        ch_np = ch_np.reshape((img_dim[1], img_dim[2]))

        if channel == 'R':
            img_array[0] = ch_np
        elif channel == 'G':
            img_array[1] = ch_np
        else:
            img_array[2] = ch_np

    return torch.tensor(img_array, dtype=torch.float32)


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
