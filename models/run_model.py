import dataset
import torch
import supersample_model
import viz

import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print("Using device", device)

params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
max_epochs = 100

supersample_dataset = dataset.SupersampleDataset('processed')

train_percentage = 0.9
train_size = int(train_percentage * len(supersample_dataset))
val_size = len(supersample_dataset) - train_size

train_set, val_set = random_split(supersample_dataset, [train_size, val_size])

train_gen = DataLoader(train_set, **params)
val_gen = DataLoader(val_set, **params)

model = supersample_model.UNet().to(device)
loss_criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

iteration = 0
print("Running training.")
for epoch in range(max_epochs):

    if epoch % 10 == 0:
        save = True
    else:
        save = False

    # training
    for i, batch in enumerate(train_gen):
        x, y = batch['half'][:, :, :64, :64].to(device), batch['full'][:, :, :128, :128].to(device)
        y_hat = model(torch.nn.Upsample(scale_factor=2, mode='bilinear')(x))

        loss = loss_criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            with torch.no_grad():
                to_viz = OrderedDict()
                to_viz['x'] = x.cpu()
                to_viz['y_hat'] = y_hat.cpu()
                to_viz['y'] = y.cpu()

                viz.visualize_outputs(to_viz, epoch, i, save)

        with torch.no_grad():
            if iteration % 5 == 0:
                viz.plot_loss(loss.cpu().item(), iteration, save)
                print("Loss @ {} iteration:".format(iteration), loss.cpu().item())

        # forward pass through model
        # y_hat = model(x)

        # print(x.size())
        # print(y_hat.size())
        iteration += 1
    with torch.set_grad_enabled(False):
        for batch in val_gen:
            x, y = batch['half'].to(device), batch['full'].to(device)

            # model computations.
