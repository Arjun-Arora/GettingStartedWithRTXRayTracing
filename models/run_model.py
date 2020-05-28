import dataset
import torch
import torchvision
import supersample_model
import unet
import viz
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


def main(seed: int,
         run_name: str,
         datasetFolder: str,
         dataloader_params: dict,
         num_epochs: int,
         train_percentage: float
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

 
    #grab model
    model = supersample_model.ESPCN(2).to(device)
    # model = unet.UNet().to(device)
    loss_criterion = torch.nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    running_loss = 0
    global_step = 0
    print("Running training.")
    for epoch in range(num_epochs):
        # training
        loss = 0
        for i, batch in enumerate(train_gen):
            y, x = batch['full'][:, :, :1060, :].to(device), batch['half'].to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            loss = loss_criterion(y_hat, y)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            global_step = epoch * len(train_gen) + i

            # visualization of input, output.
            if global_step % 10 == 0 and global_step > 0:
                with torch.no_grad():
                    writer.add_scalar('Training Loss', running_loss/10, global_step=global_step)
                    running_loss = 0
            if global_step % 1000 == 0:
                with torch.no_grad():
                    x_cpu = x.cpu()
                    y_hat_cpu = y_hat.cpu()
                    y_cpu = y.cpu()

                    img_grid = torchvision.utils.make_grid(x_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Training Input', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_hat_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Model Output', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Ground Truth', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_cpu - y_hat_cpu)
                    img_grid = viz.tensor_preprocess(img_grid, difference=True)
                    writer.add_image('Difference (GT and Model Output)', img_grid, global_step=global_step, dataformats='HW')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()},
            "model_checkpoints/super_res/exp_smooth_l1_loss_{epoch}.pt".format(epoch=epoch))

        with torch.set_grad_enabled(False):
            running_val_loss = 0
            x, y, y_hat = None, None, None
            for j, batch in enumerate(val_gen):
                x, y = batch['half'].to(device), batch['full'][:, :, :1060, :].to(device)

                y_hat = model(x)

                running_val_loss += loss_criterion(y_hat, y).item()

            x_cpu = x.cpu()
            y_hat_cpu = y_hat.cpu()
            y_cpu = y.cpu()

            writer.add_scalar('Validation Loss', running_val_loss / len(val_gen), global_step=global_step)
            running_val_loss = 0

            img_grid = torchvision.utils.make_grid(x_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Input', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_hat_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Model Output', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Ground Truth', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_cpu - y_hat_cpu)
            img_grid = viz.tensor_preprocess(img_grid, difference=True)
            writer.add_image('Val Difference (GT and Model Output)', img_grid, global_step=global_step, dataformats='HW')

if __name__ == '__main__':
    seed = 348
    run_name = 'runs/super_res'
    dataloader_params = {'batch_size': 2, 'shuffle': True, 'num_workers': 2}
    train_percentage = 0.9
    num_epochs = 100
    datasetFolder= 'processed'
    main(seed=348,
        run_name,
        datasetFolder,
        dataloader_params,
        num_epochs,
        train_percentage)
