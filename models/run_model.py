
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
import supersample_model



def main(seed: int,
         run_name: str,
         dataset_folder: str,
         dataloader_params: dict,
         num_epochs: int,
         train_percentage: float,
         experiment_name: str
         ):
    #setting universal experiment params
    torch.manual_seed(seed)
    np.random.seed(seed)
    writer = SummaryWriter(run_name)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device", device)
    
    #grabbing dataset from dataset folder
    if experiment_name == 'SingleImageSuperResolution':
        input_types = ['full', 'half']
        data = dataset.SupersampleDataset(dataset_folder, input_types)

    if experiment_name == 'Denoise':
        data = dataset.DenoiseDataset(dataset_folder)

    #calculating train-val split
    train_size = int(train_percentage * len(data))
    val_size = len(data) - train_size
    train_set, val_set = random_split(data, [train_size, val_size])

    #instantiate train-val iterators
    train_gen = DataLoader(train_set, **dataloader_params)
    val_gen = DataLoader(val_set, **dataloader_params)

    if experiment_name == 'SingleImageSuperResolution':
        model_params = {'input_types': ['half'], 'upscale_factor': 2, 
        'input_channel_size': 3, 'output_channel_size': 3}
        SingleImageSuperResolution(writer, device, train_gen, val_gen, num_epochs, model_params)

    if experiment_name == 'Denoise':
        model_params = {'input_types': ["full", "mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"],
        'input_channel_size': 14}
        Denoise(writer, device, train_gen, val_gen, num_epochs, model_params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Run experiments for ray-tracing model experiments"))
    parser.add_argument('--seed',default =348)
    # seed = 348
    parser.add_argument('--run_name',default ='runs/super_res',help=("name of this run for visualization purposes"))
    # run_name = 'runs/super_res'
    parser.add_argument('--batch_size',default = 2)
    parser.add_argument('--shuffle',default = True)
    parser.add_argument('--num_workers',default=2)
    parser.add_argument('--train_percentage',default = 0.9,help=("percentage of dataset to use in train vs val"))
    parser.add_argument('--num_epochs',default=100)
    parser.add_argument('--dataset_folder',default='processed')
    parser.add_argument('--experiment_name',default='SingleImageSuperResolution',help=("decides which experiment to run"))
    args = parser.parse_args()
    seed = int(args.seed)
    dataloader_params = {'batch_size': int(args.batch_size),
                         'shuffle': bool(args.shuffle),
                         'num_workers': int(args.num_workers)}
    train_percentage = float(args.train_percentage)
    num_epochs = int(args.num_epochs)
    dataset_folder= str(args.dataset_folder)
    experiment_name = str(args.experiment_name)
    run_name = str(args.run_name)
    main(seed,
        run_name,
        dataset_folder,
        dataloader_params,
        num_epochs,
        train_percentage,
        experiment_name)
