import dataset
import torch
import torchvision
import supersample_model
import denoise_model
import end2end_model
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


def experiment4b(writer,
                 device,
                 train_gen,
                 val_gen,
                 num_epochs,
                 chkpoint_folder,
                 model_params: dict):

    model_1 = supersample_model.ESPCN(upscale_factor=model_params['upscale_factor'],
                                    input_channel_size=model_params['input_channel_size'],
                                    output_channel_size=model_params['output_channel_size']).to(device)

    loss_criterion_1 = torch.nn.SmoothL1Loss().to(device)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=1e-4)
    scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1)

    model_2 = denoise_model.KPCN_light(input_channels=model_params['input_channel_size'], kernel_size=3).to(device)
    apply_kernel = denoise_model.ApplyKernel(kernel_size=3).to(device)

    loss_criterion_2 = torch.nn.MSELoss().to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=1e-4)
    scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2)
    global_step = 0

    running_loss_1 = 0
    running_psnr_1 = 0
    best_val_psnr_1 = 0

    running_loss_2 = 0
    running_psnr_2 = 0
    best_val_psnr_2 = 0

    print("Running training...")
    for epoch in tqdm(range(num_epochs)):
        loss_1 = 0
        loss_2 = 0
        for i,batch in enumerate(tqdm(train_gen)):
            y1 = batch['full'][:, :, :1016, :].to(device)
            N,C,H,W = y1.size()
            x1 = []
            for p in model_params['input_types']:
                if p == 'half':
                    x1.append(F.interpolate(batch[p], size=(H,W)))
                    # print(F.interpolate(batch[p], size=(H,W)).to(device).size())
                else:
                    x1.append(batch[p])
            x1 = torch.cat(x1,dim=1)
            x1 = x1.to(device)

            optimizer_1.zero_grad()
            y_hat_1 = model_1(x1)
            loss_1  = loss_criterion_1(y_hat_1, y1)
            loss_1.backward()

            optimizer_1.step()
            running_loss_1 += loss_1.item()
            running_psnr_1 += get_PSNR(y_hat_1, y1)

            #set y_hat_1 separate from previous model
            y_hat_1 = y_hat_1.detach()

            x2 = y_hat_1
            y2 = batch['clean'][:, :, :1016, :].to(device)

            for p in model_params['input_types']:
                if p != 'half':
                    x2 = torch.cat((x2,batch[p].to(device)),dim=1)

            optimizer_2.zero_grad()

            kernel = model_2(x2)
            y_hat_2 = apply_kernel.forward(x2[:, :3], kernel, padding=True)
            loss_2 = loss_criterion_2(y_hat_2, y2)
            loss_2.backward()

            optimizer_2.step()
            running_loss_2 += loss_2.item()
            running_psnr_2 += get_PSNR(y_hat_2,y2)

            global_step = epoch * len(train_gen) + i

            if global_step % 10 == 0 and global_step > 0:
                with torch.no_grad():
                    writer.add_scalar('Training Loss_1', running_loss_1/10, global_step=global_step)
                    writer.add_scalar('Training PSNR_1 (dB)', running_psnr_1/10, global_step=global_step)
                    writer.add_scalar('Training Loss_2', running_loss_2/10, global_step=global_step)
                    writer.add_scalar('Training PSNR_2 (dB)', running_psnr_2/10, global_step=global_step)
                    running_loss_1 = 0
                    running_psnr_1 = 0
                    running_loss_2 = 0
                    running_psnr_2 = 0 
            if global_step % 1000 == 0:
                with torch.no_grad():
                    x1_cpu = x1.cpu()[:, :3, :, :]
                    y_hat_1cpu = y_hat_1.cpu()[:, :3, :, :]
                    y1_cpu = y1.cpu()[:, :3, :, :]

                    img_grid = torchvision.utils.make_grid(x1_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Training Input 1', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_hat_1cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Model Output 1', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y1_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Ground Truth 1', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y1_cpu - y_hat_1cpu)
                    img_grid = viz.tensor_preprocess(img_grid, difference=True)
                    writer.add_image('Difference (GT and Model Output) 1', img_grid, global_step=global_step, dataformats='HW')


                    #second set
                    x2_cpu = x2.cpu()[:, :3, :, :]
                    y_hat_2cpu = y_hat_2.cpu()[:, :3, :, :]
                    y2_cpu = y2.cpu()[:, :3, :, :]
                    img_grid = torchvision.utils.make_grid(x2_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)

                    writer.add_image('Training Input 2', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_hat_2cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Model Output 2', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y2_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Ground Truth 2', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y2_cpu - y_hat_2cpu)
                    img_grid = viz.tensor_preprocess(img_grid, difference=True)
                    writer.add_image('Difference (GT and Model Output) 2', img_grid, global_step=global_step, dataformats='HW')

        with torch.set_grad_enabled(False):
            running_val_loss_1 = 0
            running_val_psnr_1 = 0

            running_val_loss_2 = 0
            running_val_psnr_2 = 0
            x1, y1, y1_hat = None, None, None
            for j, batch in enumerate(val_gen):
                y1 = batch['full'][:, :, :1016, :].to(device)
                N,C,H,W = y1.size()
                x1 = []
                for p in model_params['input_types']:
                    if p == 'half':
                        x1.append(F.interpolate(batch[p], size=(H,W)))
                        # print(F.interpolate(batch[p], size=(H,W)).to(device).size())
                    else:
                        x1.append(batch[p])
                x1 = torch.cat(x1,dim=1)
                x1 = x1.to(device)
                
                y_hat_1 = model_1(x1)

                running_val_loss_1 += loss_criterion_1(y_hat_1, y1).item()
                running_val_psnr_1 += get_PSNR(y_hat_1, y1)

                #set y_hat_1 separate from previous model
                y_hat_1 = y_hat_1.detach()

                x2 = y_hat_1
                y2 = batch['clean'][:, :, :1016, :].to(device)

                for p in model_params['input_types']:
                    if p != 'half':
                        x2 = torch.cat((x2,batch[p].to(device)),dim=1)

                kernel = model_2(x2)
                y_hat_2 = apply_kernel.forward(x2[:, :3], kernel, padding=True)

                running_val_loss_2 += loss_criterion_2(y_hat_2, y2).item()
                running_val_psnr_2 += get_PSNR(y_hat_2,y2)

            x1_cpu = x1.cpu()[:, :3, :, :]
            y_hat_1cpu = y_hat_1.cpu()[:, :3, :, :]
            y1_cpu = y1.cpu()[:, :3, :, :]

            x2_cpu = x2.cpu()[:, :3, :, :]
            y_hat_2cpu = y_hat_2.cpu()[:, :3, :, :]
            y2_cpu = y2.cpu()[:, :3, :, :]

            scheduler_1.step(running_val_loss_1 / len(val_gen))
            scheduler_2.step(running_val_loss_2 / len(val_gen))

            writer.add_scalar('Validation Loss 1', running_val_loss_1 / len(val_gen), global_step=global_step)
            writer.add_scalar('Validation PSNR (dB) 1', running_val_psnr_1 / len(val_gen), global_step=global_step)

            writer.add_scalar('Validation Loss 2', running_val_loss_2 / len(val_gen), global_step=global_step)
            writer.add_scalar('Validation PSNR (dB) 2', running_val_psnr_2 / len(val_gen), global_step=global_step)

            if running_val_psnr_1 > best_val_psnr_1:
                running_val_psnr_1 = best_val_psnr_1
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_1.state_dict(),
                'optimizer_state_dict': optimizer_1.state_dict(),
                'loss': loss_1.item()},
                os.path.join(chkpoint_folder, "super_res_{epoch}.pt".format(epoch=epoch)))
            if running_val_psnr_2 > best_val_psnr_2:
                running_val_psnr_2 = best_val_psnr_2
                torch.save({
                'epoch': epoch,
                'model_state_dict': model_2.state_dict(),
                'optimizer_state_dict': optimizer_2.state_dict(),
                'loss': loss_2.item()},
                os.path.join(chkpoint_folder, "denoiser_{epoch}.pt".format(epoch=epoch)))


            img_grid = torchvision.utils.make_grid(x1_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            
            writer.add_image('Val Input 1', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_hat_1cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Model Output 1', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y1_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Ground Truth 1', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y1_cpu - y_hat_1cpu)
            img_grid = viz.tensor_preprocess(img_grid, difference=True)
            writer.add_image('Val Difference (GT and Model Output) 1', img_grid, global_step=global_step, dataformats='HW')

            #second
            img_grid = torchvision.utils.make_grid(x2_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            
            writer.add_image('Val Input 2', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_hat_2cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Model Output 2', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y2_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Ground Truth 2', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y2_cpu - y_hat_2cpu)
            img_grid = viz.tensor_preprocess(img_grid, difference=True)
            writer.add_image('Val Difference (GT and Model Output) 2', img_grid, global_step=global_step, dataformats='HW')


def experiment4c(writer,
                 device,
                 train_gen,
                 val_gen,
                 num_epochs,
                 chkpoint_folder,
                 model_params: dict):

    model = end2end_model.ESPCN_KPCN(upscale_factor=model_params['upscale_factor'],
                                    input_channel_size=model_params['input_channel_size'],
                                    kernel_size=3).to(device)

    loss_criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    global_step = 0

    running_loss = 0
    running_psnr = 0
    running_psnr_sr = 0
    best_val_psnr = 0

    print("Running training...")
    for epoch in tqdm(range(num_epochs)):
        loss = 0
        for i,batch in enumerate(train_gen):
            y = batch['clean'][:, :, :1016, :].to(device)
            y_full = batch['full'][:, :, :1016, :].to(device)
            N,C,H,W = y.size()
            x = F.interpolate(batch['half'], size=(H,W)).to(device)
            g = []
            for p in model_params['input_types']:
                if p != 'half':
                    g.append(batch[p])
            g = torch.cat(g,dim=1)
            g = g.to(device)

            optimizer.zero_grad()
            y_denoise, y_supersample = model(x, g)
            loss  = loss_criterion(y_denoise, y)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            running_psnr += get_PSNR(y_denoise, y)
            running_psnr_sr += get_PSNR(y_supersample, y_full)

            global_step = epoch * len(train_gen) + i

            if global_step % 10 == 0 and global_step > 0:
                with torch.no_grad():
                    writer.add_scalar('Training Loss', running_loss/10, global_step=global_step)
                    writer.add_scalar('Training PSNR (dB)', running_psnr/10, global_step=global_step)
                    writer.add_scalar('Training PSNR SR(dB)', running_psnr_sr/10, global_step=global_step)
                    running_loss = 0
                    running_psnr = 0
                    running_psnr_sr = 0
            if global_step % 1000 == 0:
                with torch.no_grad():
                    x_cpu = x.cpu()[:, :3, :, :]
                    y_denoise_cpu = y_denoise.cpu()[:, :3, :, :]
                    y_supersample_cpu = y_supersample.cpu()[:, :3, :, :]
                    y_cpu = y.cpu()[:, :3, :, :]
                    y_full_cpu = y_full.cpu()[:, :3, :, :]

                    img_grid = torchvision.utils.make_grid(x_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Training Input', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_denoise_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Model Output Denoise', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_supersample_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Model Output Supersample', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Ground Truth', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_full_cpu)
                    img_grid = viz.tensor_preprocess(img_grid)
                    writer.add_image('Ground Truth Supersample', img_grid, global_step=global_step)

                    img_grid = torchvision.utils.make_grid(y_cpu - y_denoise_cpu)
                    img_grid = viz.tensor_preprocess(img_grid, difference=True)
                    writer.add_image('Difference (GT and Model Output)', img_grid, global_step=global_step, dataformats='HW')

                    img_grid = torchvision.utils.make_grid(y_full_cpu - y_supersample_cpu)
                    img_grid = viz.tensor_preprocess(img_grid, difference=True)
                    writer.add_image('Difference (SR GT and Model SR Output)', img_grid, global_step=global_step, dataformats='HW')

        with torch.set_grad_enabled(False):
            running_val_loss = 0
            running_val_psnr = 0
            running_val_psnr_sr = 0

            for j, batch in enumerate(val_gen):
                y = batch['clean'][:, :, :1016, :].to(device)
                y_full = batch['full'][:, :, :1016, :].to(device)
                N,C,H,W = y.size()
                x = F.interpolate(batch['half'], size=(H,W)).to(device)
                g = []
                for p in model_params['input_types']:
                    if p != 'half':
                        g.append(batch[p])
                g = torch.cat(g,dim=1)
                g = g.to(device)

                y_denoise, y_supersample = model(x, g)
                loss  = loss_criterion(y_denoise, y)

                running_val_loss += loss.item()
                running_val_psnr += get_PSNR(y_denoise, y)
                running_val_psnr_sr += get_PSNR(y_supersample, y_full)

            x_cpu = x.cpu()[:, :3, :, :]
            y_denoise_cpu = y_denoise.cpu()[:, :3, :, :]
            y_supersample_cpu = y_supersample.cpu()[:, :3, :, :]
            y_cpu = y.cpu()[:, :3, :, :]
            y_full_cpu = y_full.cpu()[:, :3, :, :]

            scheduler.step(running_val_loss / len(val_gen))

            writer.add_scalar('Validation Loss', running_val_loss / len(val_gen), global_step=global_step)
            writer.add_scalar('Validation PSNR (dB)', running_val_psnr / len(val_gen), global_step=global_step)
            writer.add_scalar('Validation PSNR SR(dB)', running_val_psnr_sr / len(val_gen), global_step=global_step)

            if running_val_psnr > best_val_psnr:
                running_val_psnr = best_val_psnr
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()},
                os.path.join(chkpoint_folder, "end_to_end_{epoch}.pt".format(epoch=epoch)))

            img_grid = torchvision.utils.make_grid(x_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            
            writer.add_image('Val Input', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_denoise_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Model Output Denoise', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_supersample_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Model Output Supersample', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Ground Truth', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_full_cpu)
            img_grid = viz.tensor_preprocess(img_grid)
            writer.add_image('Val Ground Truth Supersample', img_grid, global_step=global_step)

            img_grid = torchvision.utils.make_grid(y_cpu - y_denoise_cpu)
            img_grid = viz.tensor_preprocess(img_grid, difference=True)
            writer.add_image('Val Difference (GT and Model Output)', img_grid, global_step=global_step, dataformats='HW')

            img_grid = torchvision.utils.make_grid(y_full_cpu - y_supersample_cpu)
            img_grid = viz.tensor_preprocess(img_grid, difference=True)
            writer.add_image('Val Difference (SR GT and Model SR Output)', img_grid, global_step=global_step, dataformats='HW')

def experiment4sppPSNR(writer,
                 device,
                 val_gen):

    psnr_1spp = 0
    psnr_4spp = 0

    print("Running PSNR testing...")
    with torch.set_grad_enabled(False):
        for j, batch in enumerate(val_gen):
            y = batch['clean'][:, :, :1016, :].to(device)
            N,C,H,W = y.size()
            x_1spp = batch['full'][:, :, :1016, :].to(device)
            x_4spp = batch['4spp'][:, :, :1016, :].to(device)

            psnr_1spp += get_PSNR(x_1spp, y)
            psnr_4spp += get_PSNR(x_4spp, y)
        
        psnr_1spp = psnr_1spp / len(val_gen)
        psnr_4spp = psnr_4spp / len(val_gen)

        print("PSNR 1spp/clean Valset: {}".format(psnr_1spp))
        print("PSNR 4spp/clean Valset: {}".format(psnr_4spp))
        
        x_1spp = x_1spp.cpu()[:, :3, :, :]
        x_4spp = x_4spp.cpu()[:, :3, :, :]
        y = y.cpu()[:, :3, :, :]

        writer.add_scalar('Validation PSNR 1spp(dB)', psnr_1spp, global_step=0)
        writer.add_scalar('Validation PSNR 4spp(dB)', psnr_4spp, global_step=0)

        img_grid = torchvision.utils.make_grid(x_1spp)
        img_grid = viz.tensor_preprocess(img_grid)
        writer.add_image('Val 1spp', img_grid, global_step=0)

        img_grid = torchvision.utils.make_grid(x_4spp)
        img_grid = viz.tensor_preprocess(img_grid)
        writer.add_image('Val 4spp', img_grid, global_step=0)

        img_grid = torchvision.utils.make_grid(y)
        img_grid = viz.tensor_preprocess(img_grid)
        writer.add_image('Val Clean', img_grid, global_step=0)

        img_grid = torchvision.utils.make_grid(x_1spp - y)
        img_grid = viz.tensor_preprocess(img_grid, difference=True)
        writer.add_image('Val Difference (1spp - Clean)', img_grid, global_step=0, dataformats='HW')

        img_grid = torchvision.utils.make_grid(x_4spp - y)
        img_grid = viz.tensor_preprocess(img_grid, difference=True)
        writer.add_image('Val Difference (4spp - Clean)', img_grid, global_step=0, dataformats='HW')



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
            N,C,H,W = y.size()
            if len(model_params['input_types']) > 1:
                x = []
                for p in model_params['input_types']:
                    if p == 'half':
                        x.append(F.interpolate(batch[p], size=(H,W)).to(device))
                        # print(F.interpolate(batch[p], size=(H,W)).to(device).size())
                    else:
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
                        else:
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

    model = denoise_model.KPCN_light(input_channels=model_params['input_channel_size'], kernel_size=3).to(device)
    apply_kernel = denoise_model.ApplyKernel(kernel_size=3).to(device)

    loss_criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    running_loss = 0
    running_psnr = 0
    global_step = 0
    best_val_psnr = 0
    print("Running training on denoiser...")
    for epoch in tqdm(range(num_epochs)):
        # training
        loss = 0
        for i, batch in enumerate(tqdm(train_gen)):
            #batch["full","mat_diffuse", "mat_ref", "mat_spec_rough", "world_normal", "world_pos"]
            x, y, y_hat = None, None, None
            y = batch['clean'][:, :, :1016, :].to(device)
            x = []
            for p in model_params['input_types']:
                x.append(batch[p].to(device))
            x = torch.cat(x, dim=1)

            x = x.to(device)

            optimizer.zero_grad()

            kernel = model(x)
            y_hat = apply_kernel.forward(x[:, :3], kernel, padding=True)
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
            if global_step % 500 == 0:
                with torch.no_grad():
                    x_cpu = x.cpu()[:, :3, :, :]
                    y_hat_cpu = y_hat.cpu()[:, :3, :, :]
                    y_cpu = y.cpu()[:, :3, :, :]

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
                y = batch['clean'][:, :, :1016, :].to(device)
                x = []
                for p in model_params['input_types']:
                    x.append(batch[p].to(device))
                x = torch.cat(x, dim=1)

                x = x.to(device)

                kernel = model(x)
                y_hat = apply_kernel.forward(x[:, :3], kernel, padding=True)

                running_val_loss += loss_criterion(y_hat, y).item()
                running_val_psnr += get_PSNR(y_hat, y)

            x_cpu = x.cpu()[:, :3, :, :]
            y_hat_cpu = y_hat.cpu()[:, :3, :, :]
            y_cpu = y.cpu()[:, :3, :, :]

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
                os.path.join(chkpoint_folder, "exp_mse_loss_{epoch}.pt".format(epoch=epoch)))

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

