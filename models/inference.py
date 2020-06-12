import dataset
import end2end_model
import torch
import numpy as np
import os

from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image


def run4cd(save_dir, device, test_gen, checkpoint_path, model_params):
    model = end2end_model.ESPCN_KPCN(upscale_factor=model_params['upscale_factor'],
                                    input_channel_size=model_params['input_channel_size'],
                                    kernel_size=3)
    
    chkpoint = torch.load(checkpoint_path)

    model.load_state_dict(chkpoint['model_state_dict'])

    model.eval().to(device)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(os.path.join(save_dir, 'denoise')):
        os.mkdir(os.path.join(save_dir, 'denoise'))
    
    if not os.path.exists(os.path.join(save_dir, 'supersample')):
        os.mkdir(os.path.join(save_dir, 'supersample'))


    with torch.set_grad_enabled(False):
        for j, batch in enumerate(tqdm(test_gen)):
            y = batch['clean'][:, :, :1016, :].to(device)
            # y_full = batch['full'][:, :, :1016, :].to(device)

            N, C, H, W = y.size()

            x = F.interpolate(batch['half'], size = (H, W)).to(device)

            g = []
            for p in model_params['input_types']:
                if p != 'half':
                    g.append(batch[p])
            g = torch.cat(g, dim=1)
            g = g.to(device)

            y_denoise, y_supersample = model(x, g)

            save_image(y_supersample.cpu()[:3, :, :], os.path.join(save_dir, 'supersample', '{}.png'.format(j)))
            save_image(y_denoise.cpu()[:3, :, :], os.path.join(save_dir, 'denoise', '{}.png'.format(j)))


# def run4d(save_dir, device, test_gen, checkpoint_path, model_params):
#     model = end2end_model.ESPCN_KPCN(upscale_factor=model_params['upscale_factor'],
#                                     input_channel_size=model_params['input_channel_size'],
#                                     kernel_size=3)

#     chkpoint = torch.load(checkpoint_path)

#     model.load_state_dict(chkpoint['model_state_dict'])

#     model.eval().to(device)

#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
    
#     if not os.path.exists(os.path.join(save_dir, 'denoise')):
#         os.mkdir(os.path.je, test_gen, checkpoint_path, model_params):
#     model = end2end_model.oin(save_dir, 'denoise'))
    
#     if not os.path.exists(os.path.join(save_dir, 'supersample')):
#         os.mkdir(os.path.join(save_dir, 'supersample'))


#     with torch.set_grad_enabled(False):
#         for j, batch in enumerate(tqdm(test_gen)):
#             y = batch['clean'][:, :, :1016, :].to(device)
#             # y_full = batch['full'][:, :, :1016, :].to(device)

#             N, C, H, W = y.size()

#             x = F.interpolate(batch['half'], size = (H, W)).to(device)

#             g = []
#             for p in model_params['input_types']:
#                 if p != 'half':
#                     g.append(batch[p])
#             g = torch.cat(g, dim=1)
#             g = g.to(device)

#             y_denoise, y_supersample = model(x, g)

#             save_image(y_supersample.cpu()[:3, :, :], os.path.join(save_dir, 'supersample', '{}.png'.format(j)))
#             save_image(y_denoise.cpu()[:3, :, :], os.path.join(save_dir, 'denoise', '{}.png'.format(j)))
