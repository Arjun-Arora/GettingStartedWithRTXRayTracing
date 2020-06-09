
#torch
import dataset
import torch
import torchvision
import viz
import torch.nn.functional as F
import numpy as np

#vis
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

#std
from collections import OrderedDict
import argparse

# our imports 
from experiments import SingleImageSuperResolution
from experiments import Denoise
from experiments import experiment4b
from experiments import experiment4c
from experiments import experiment4sppPSNR
from experiments import experiment4d
import supersample_model

import os


def main(seed: int,
         run_name: str,
         writer_folder: str,
         dataset_folder: str,
         dataloader_params: dict,
         num_epochs: int,
         train_percentage: float,
         experiment_name: str,
         model_superres_dir,
         model_denoise_dir
         ):
    #setting universal experiment params
    torch.manual_seed(seed)
    np.random.seed(seed)
    writer = SummaryWriter(os.path.join(writer_folder, run_name))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device", device)
    
    #grabbing dataset from dataset folder
    if experiment_name == 'SingleImageSuperResolution':
        types_to_load = ['full', 'half']
        data = dataset.SupersampleDataset(dataset_folder, types_to_load)
    elif experiment_name == 'GbufferOnlySuperResolution':
        types_to_load = ['mat_diffuse', 'mat_ref', 'mat_spec_rough', 'world_normal', 'world_pos', 'full']
        data = dataset.SupersampleDataset(dataset_folder, types_to_load)
    elif experiment_name == 'HalfandGBufferSuperResolution':
        types_to_load = ['half','mat_diffuse', 'mat_ref', 'mat_spec_rough', 'world_normal', 'world_pos', 'full']
        data = dataset.SupersampleDataset(dataset_folder,types_to_load)
    elif experiment_name == 'Denoise':
        data = dataset.DenoiseDataset(dataset_folder)
    elif experiment_name == 'experiment4a':
        types_to_load = ['half','mat_diffuse', 'mat_ref', 'mat_spec_rough', 'world_normal', 'world_pos', 'full','clean']
        data = dataset.SupersampleDataset(dataset_folder,types_to_load)
    elif experiment_name == 'experiment4b':
        types_to_load = ['half','mat_diffuse', 'mat_ref', 'mat_spec_rough', 'world_normal', 'world_pos', 'full','clean']
        data = dataset.SupersampleDataset(dataset_folder,types_to_load)
    elif experiment_name == 'experiment4c':
        types_to_load = ['half','mat_diffuse', 'mat_ref', 'mat_spec_rough', 'world_normal', 'world_pos', 'full','clean']
        data = dataset.SupersampleDataset(dataset_folder,types_to_load)
    elif experiment_name == 'experiment4d':
        types_to_load = ['half','mat_diffuse', 'mat_ref', 'mat_spec_rough', 'world_normal', 'world_pos', 'full','clean']
        data = dataset.SupersampleDataset(dataset_folder,types_to_load)
    elif experiment_name == 'experiment4sppPSNR':
        types_to_load = ['half','full','clean', "4spp"]
        data = dataset.SupersampleDataset(dataset_folder,types_to_load)
    else:
        raise NotImplementedError
    #calculating train-val split
    train_size = int(train_percentage * len(data))
    val_size = len(data) - train_size
    train_set, val_set = random_split(data, [train_size, val_size])

    #instantiate train-val iterators
    train_gen = DataLoader(train_set, **dataloader_params)
    val_gen = DataLoader(val_set, **dataloader_params)

    # checkpoint folder
    chkpoint_folder = os.path.join('model_checkpoints', run_name)

    if experiment_name == 'SingleImageSuperResolution':
        model_params = {'input_types': ['half'], 'upscale_factor': 2, 
        'input_channel_size': 3, 'output_channel_size': 3}
        SingleImageSuperResolution(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)
    elif experiment_name == 'GbufferOnlySuperResolution':
        model_params = {'input_types': [ "mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],'upscale_factor': 1,
        'input_channel_size': 11, 'output_channel_size': 3}
        SingleImageSuperResolution(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)
    elif experiment_name == 'HalfandGBufferSuperResolution':
        model_params = {'input_types': [ "half","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],'upscale_factor': 1,
        'input_channel_size': 14, 'output_channel_size': 3}
        SingleImageSuperResolution(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)
    elif experiment_name == 'experiment4a':
        model_params = {'input_types': [ "half","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],'upscale_factor': 1,
        'input_channel_size': 14, 'output_channel_size': 3}
        assert model_superres_dir != None,"model_superres_dir needs to be a path, not None"
        assert model_denoise_dir != None,"model_denoise_dir needs to be a path, not None"
        experiment4b(writer, device, val_gen,model_superres_dir,model_denoise_dir, model_params)
    elif experiment_name == 'experiment4b':
        model_params = {'input_types': [ "half","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],'upscale_factor': 1,
        'input_channel_size': 14, 'output_channel_size': 3}
        experiment4b(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)
    elif experiment_name == 'experiment4c':
        model_params = {'input_types': [ "half","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],'upscale_factor': 1,
        'input_channel_size': 14, 'output_channel_size': 3}
        experiment4c(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)
    elif experiment_name == 'experiment4d':
        model_params = {'input_types': [ "half","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],'upscale_factor': 1,
        'input_channel_size': 14, 'output_channel_size': 3}
        experiment4d(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)   
    elif experiment_name == 'experiment4sppPSNR':
        experiment4sppPSNR(writer, device, val_gen)
    elif experiment_name == 'Denoise':
        model_params = {'input_types': ["full", "mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],
        'input_channel_size': 14}
        Denoise(writer, device, train_gen, val_gen, num_epochs, chkpoint_folder, model_params)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Run experiments for ray-tracing model experiments"))
    parser.add_argument('--seed',default =348)
    # seed = 348
    parser.add_argument('--run_name',default ='super_res',help=("name of this run for visualization purposes"))
    # run_name = 'runs/super_res'
    parser.add_argument('--batch_size',default = 2)
    parser.add_argument('--shuffle',default = True)
    parser.add_argument('--num_workers',default=2)
    parser.add_argument('--train_percentage',default = 0.9,help=("percentage of dataset to use in train vs val"))
    parser.add_argument('--num_epochs',default=100)
    parser.add_argument('--dataset_folder',default='processed')
    parser.add_argument('--writer_folder',default='runs')
    parser.add_argument('--experiment_name',default='SingleImageSuperResolution',help=("decides which experiment to run"))
    parser.add_argument('--model_superres_dir',default=None,help=("path to superres model"))
    parser.add_argument('--model_denoise_dir',default=None,help=("path to denoise model"))
    args = parser.parse_args()
    seed = int(args.seed)
    dataloader_params = {'batch_size': int(args.batch_size),
                         'shuffle': bool(args.shuffle),
                         'num_workers': int(args.num_workers)}
    train_percentage = float(args.train_percentage)
    num_epochs = int(args.num_epochs)
    dataset_folder= str(args.dataset_folder)
    writer_folder=str(args.writer_folder)
    experiment_name = str(args.experiment_name)
    run_name = str(args.run_name)
    model_superres_dir = str(args.model_superres_dir)
    model_denoise_dir = str(args.model_denoise_dir)
    main(seed,
        run_name,
        writer_folder,
        dataset_folder,
        dataloader_params,
        num_epochs,
        train_percentage,
        experiment_name,
        model_superres_dir,
        model_denoise_dir
        )
