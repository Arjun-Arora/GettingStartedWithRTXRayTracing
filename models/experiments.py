import dataset
import torch
import torchvision
import supersample_model
import denoise_model
import viz
import torch.nn.functional as F
import numpy as np
import os

from dataset import INV_GAMMA

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm

def get_PSNR(model_output, target):
    I_hat = model_output.cpu().detach().numpy()
    I = target.cpu().detach().numpy()
    mse = (np.square(I - I_hat)).mean(axis=None)
    PSNR = 10 * np.log10(1.0 / mse)
    return PSNR

def SingleImageSuperResolution(writer,
							 device,
							 train_gen,
							 val_gen,
							 num_epochs,
                             chkpoint_folder,
                             model_params: dict):
	#grab model

    model = supersample_model.ESPCN(upscale_factor=model_params['upscale_factor'],
                                    input_channel_size=model_params['input_channel_size'],
                                    output_channel_size=model_params['output_channel_size']).to(device)
    # model = unet.UNet().to(device)
    loss_criterion = torch.nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    running_loss = 0
    running_psnr = 0
    global_step = 0
    best_val_psnr = 0
    print("Running training...")
    for epoch in tqdm(range(num_epochs)):
        # training
        loss = 0
        for i, batch in enumerate(train_gen):
            #batch["half","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"]
            y = batch['full'][:, :, :1016, :].to(device)
            if len(model_params['input_types']) > 1:
                x = []
                for p in model_params['input_types']:
                    if p == 'half':
                        x.append(F.interpolate(batch[p], scale_factor=2).to(device))
                    x.append(batch[p].to(device))
                x = torch.cat(x,dim=1)
            else:
                x = batch[model_params['input_types'][0]]
            x = x.to(device)

            optimizer.zero_grad()

            y_hat = model(x)
            #print(y_hat.size())
            #print(y.size())
            loss = loss_criterion(y_hat, y)
            loss.backward()
            #print("loss:", loss.item())
            optimizer.step()
            running_loss += loss.item()
            with torch.no_grad():
                running_psnr += get_PSNR(y_hat, y)

            global_step = epoch * len(train_gen) + i

            # visualization of input, output.
            if global_step % 10 == 0 and global_step > 0:
                with torch.no_grad():
                    writer.add_scalar('Training Loss', running_loss/10, global_step=global_step)
                    writer.add_scalar('Training PSNR (dB)', running_psnr/10, global_step=global_step)
                    running_loss = 0
                    running_psnr = 0
            if global_step % 1000 == 0:
                with torch.no_grad():
                    x_cpu = x.cpu()[:, :3, :, :]
                    y_hat_cpu = y_hat.cpu()[:, :3, :, :]
                    y_cpu = y.cpu()[:, :3, :, :]

                    # x_cpu = torch.pow(x_cpu, INV_GAMMA)
                    # x_cpu = torch.clamp(x_cpu, 0, 1)

                    # y_hat_cpu = torch.pow(y_hat_cpu, INV_GAMMA)
                    # y_hat_cpu = torch.clamp(y_hat_cpu, 0, 1)

                    # y_cpu = torch.pow(y_cpu, INV_GAMMA)
                    # y_cpu = torch.clamp(y_cpu, 0, 1)

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

        

        with torch.set_grad_enabled(False):
            running_val_loss = 0
            running_val_psnr = 0
            x, y, y_hat = None, None, None
            for j, batch in enumerate(val_gen):
                y = batch['full'][:, :, :1016, :].to(device)
                if len(model_params['input_types']) > 1:
                    x = []
                    for p in model_params['input_types']:
                        if p == 'half':
                            x.append(F.interpolate(batch[p], scale_factor=2).to(device))
                        x.append(batch[p].to(device))
                    x = torch.cat(x,dim=1)
                else:
                    x = batch[model_params['input_types'][0]]
                
                x = x.to(device)

                y_hat = model(x)

                running_val_loss += loss_criterion(y_hat, y).item()
                running_val_psnr += get_PSNR(y_hat, y)

            x_cpu = x.cpu()[:, :3, :, :]
            y_hat_cpu = y_hat.cpu()[:, :3, :, :]
            y_cpu = y.cpu()[:, :3, :, :]

            # x_cpu = torch.pow(x_cpu, INV_GAMMA)
            # x_cpu = torch.clamp(x_cpu, 0, 1)

            # y_hat_cpu = torch.pow(y_hat_cpu, INV_GAMMA)
            # y_hat_cpu = torch.clamp(y_hat_cpu, 0, 1)

            # y_cpu = torch.pow(y_cpu, INV_GAMMA)
            # y_cpu = torch.clamp(y_cpu, 0, 1)

            scheduler.step(running_val_loss / len(val_gen))

            writer.add_scalar('Validation Loss', running_val_loss / len(val_gen), global_step=global_step)
            writer.add_scalar('Validation PSNR (dB)', running_val_psnr / len(val_gen), global_step=global_step)
            if running_val_psnr > best_val_psnr:
                running_val_psnr = best_val_psnr
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()},
                os.path.join(chkpoint_folder, "exp_smooth_l1_loss_{epoch}.pt".format(epoch=epoch)))

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


def Denoise(writer,
			 device,
			 train_gen,
			 val_gen,
			 num_epochs,
             chkpoint_folder,
             model_params: dict):
	#grab model

    model = denoise_model.KPCN_light(input_channels=model_params['input_channel_size']).half().to(device)
    apply_kernel = denoise_model.ApplyKernel(21).to(device)
    # model = unet.UNet().to(device)
    loss_criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    running_loss = 0
    running_psnr = 0
    global_step = 0
    best_val_psnr = 0
    print("Running training on denoiser...")
    for epoch in range(num_epochs):
        # training
        loss = 0
        for i, batch in enumerate(train_gen):
            #batch["full","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"]
            x = []
            for i in model_params['input_types']:
                x.append(batch[i])
            x = torch.cat(x,dim=1)

            y = batch['clean'][:, :, :1060, :].to(device)
            x = x.to(device)
            optimizer.zero_grad()

            kernel = model(x)
            y_hat = apply_kernel.forward(x, kernel, padding=True)
            loss = loss_criterion(y_hat, y)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            running_psnr += get_PSNR(y_hat,y)

            global_step = epoch * len(train_gen) + i

            # visualization of input, output.
            if global_step % 10 == 0 and global_step > 0:
                with torch.no_grad():
                    writer.add_scalar('Training Loss', running_loss/10, global_step=global_step)
                    writer.add_scalar('Training PSNR (dB)', running_psnr/10, global_step=global_step)
                    running_psnr = 0
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

        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss.item()},
        #     "model_checkpoints/denoise/exp_MSE_loss_{epoch}.pt".format(epoch=epoch))

        with torch.set_grad_enabled(False):
            running_val_loss = 0
            running_val_psnr = 0
            x, y, y_hat = None, None, None
            for j, batch in enumerate(val_gen):
                x, y = batch['half'].to(device), batch['full'][:, :, :1060, :].to(device)

                y_hat = model(x)

                running_val_loss += loss_criterion(y_hat, y).item()
                running_val_psnr += get_PSNR(y_hat,y)

            x_cpu = x.cpu()
            y_hat_cpu = y_hat.cpu()
            y_cpu = y.cpu()

            scheduler.step(running_val_loss / len(val_gen))

            writer.add_scalar('Validation Loss', running_val_loss / len(val_gen), global_step=global_step)
            writer.add_scalar('Validation PSNR (dB)', running_val_psnr / len(val_gen), global_step=global_step)
            if running_val_psnr > best_val_psnr:
                running_val_psnr = best_val_psnr
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()},
                os.path.join(chkpoint_folder, "exp_smooth_l1_loss_{epoch}.pt".format(epoch=epoch)))

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

