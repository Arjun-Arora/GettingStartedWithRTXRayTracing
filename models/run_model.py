import dataset
import torch
import torchvision
import viz
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

# our imports 
from experiments import SingleImageSuperResolution
import supersample_model


def main(seed: int,
         run_name: str,
         datasetFolder: str,
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
    Dataset = dataset.SupersampleDataset(datasetFolder)

    #calculating train-val split
    train_size = int(train_percentage * len(Dataset))
    val_size = len(Dataset) - train_size
    train_set, val_set = random_split(Dataset, [train_size, val_size])

    #instantiate train-val iterators
    train_gen = DataLoader(train_set, **dataloader_params)
    val_gen = DataLoader(val_set, **dataloader_params)

    if experiment_name == 'SingleImageSuperResolution':
        SingleImageSuperResolution(writer,device,train_gen,val_gen,num_epochs)

if __name__ == '__main__':
    seed = 348
    run_name = 'runs/super_res'
    dataloader_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 2}
    train_percentage = 0.9
    num_epochs = 100
    datasetFolder= 'processed'
    experiment_name = "SingleImageSuperResolution"
    main(seed,
        run_name,
        datasetFolder,
        dataloader_params,
        num_epochs,
        train_percentage,
        experiment_name)
