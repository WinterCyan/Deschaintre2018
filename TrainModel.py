import torch
import torch.nn as nn
from torch import optim
import numpy as np
import math
import time
from torch.utils.data import DataLoader, random_split
from Model import *
from Dataset import MaterialDataset
from SaveLoad import *

# Core Training 5 Steps
# output, input, loss, label are all PyTorch Variable.
# if using GPU, send model/Tensor/Variable to GPU with cuda()

# 1. output = net(input), predict the output value.
# 2. loss = loss_function(output, label), calculate loss with loss function.
# 3. optimizer.zero_grad(), clear previous gradients.
# 4. loss.backward(), compute all gradients.
# 5. optimizer.step(), perform weight update.

# train details:
# train mode -> dataloader -> train_batch, label_batch -> move to GPU -> convert to Variable
# -> compute output and loss -> zero grad -> backward -> step -> compute summary

# summary with metrics, which is a dictionary of functions that compute a metric (every several steps):
# move output_batch, labels_batch to CPU, convert to numpy -> compute metrics -> append
# or, in every batch training, update average loss
# or, compute mean of all metrics

# validation
# evaluate mode -> dataloader -> data_batch, label_batch -> move to GPU -> fetch next batch
# -> compute output and loss -> move to CPU -> compute all metrics

if __name__ == '__main__':
    dataset_path = '/media/winter/M2/UbuntuDownloads/Deschaintre2018_Dataset/Data_Deschaintre18/trainBlended'
    device = 'cuda:0'
    model = MaterialNet().to(device)
    data = MaterialDataset(data_dir=dataset_path)
    split_ratio = 0.01
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    training_data_loader = DataLoader(training_data, batch_size=8, pin_memory=True, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=8, pin_memory=True, shuffle=False)
    batch_num = int(math.ceil(len(training_data)/training_data_loader.batch_size))  # 24635
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    in_net_renderer = InNetworkRenderer()
    loss_func = MixLoss(renderer=in_net_renderer)
    # loss_func = L1Loss()
    # loss_func = RenderingLoss(renderer=in_net_renderer)


    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # ckp_path = 'modelsavings/checkpoint.pt'
    # model, optimizer, load_epoch = load_ckp(ckp_path, model, optimizer)

    model.train()
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_start_time = time.time()
        for batch_idx, sample in enumerate(training_data_loader):
            img_batch = sample["img"].to(device)
            target_svbrdf_batch = sample["svbrdf"].to(device)
            estimated_svbrdf_batch = model(img_batch)
            loss = loss_func(estimated_svbrdf_batch, target_svbrdf_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 1000 == 0:
                print("Epoch {:d}, Batch {:d}, loss: {:f}".format(epoch, batch_idx+1, loss.item()))
            if batch_idx == batch_num-1:
                print("Epoch {:d}, Batch {:d}, loss: {:f}".format(epoch+1, batch_idx+1, loss.item()))
        epoch_end_time = time.time()
        epoch_time = epoch_end_time-epoch_start_time
        print('----------------------EPOCH{}---------------------'.format(epoch))
        print("epoch loss: "+str(epoch_loss))
        print("epoch time: "+str(epoch_time))
        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            is_best = True if epoch == epochs-1 else False
            save_ckp(checkpoint, is_best, 'modelsavings', 'modelsavings')

