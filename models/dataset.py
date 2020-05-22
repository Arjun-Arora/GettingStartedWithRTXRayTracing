import numpy as np

import os
import OpenEXR, Imath

import pandas as pd

import torch
from torch.utils.data import Dataset


class TracerDataset(Dataset):
    def __init__(self, src_folder: str):
        csv_path = os.path.join(src_folder, "data.csv")
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
            TracerDataset.build_dataset_csv(src_folder)

        self.src_folder = src_folder
        self.fh_frame = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.fh_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}

        for i, fh in enumerate(self.fh_frame.values[idx]):
            img_name = os.path.join(self.src_folder, fh)
            image = OpenEXR.InputFile(img_name)
            channels = image.header()['channels'].keys()

            img_arr = []
            for channel in channels:
                img_channel = list(image.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)))
                img_arr.append(img_channel)

            data_type = self.fh_frame.columns[i]
            sample[data_type] = np.array(img_arr, dtype=np.float32)

        return sample

    @classmethod
    def build_dataset_csv(cls, src_folder: str):
        if not os.path.exists(os.path.join(src_folder, "data.csv")):
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
