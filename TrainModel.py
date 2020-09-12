import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

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
