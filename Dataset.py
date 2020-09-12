import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

transformer = transforms.Compose([transforms.ToTensor()])


class MaterialDataset(Dataset):
    def __init__(self, data_dir):
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.png')]
        self.transformer = transformer

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        img = self.transformer(img)
        return img, self.filenames[idx]


def batch_to_tensor(img_batch):
    # img_batch: [8, 3, 288, 288*5]
    N, C, H, W_total = img_batch.shape
    W = int(W_total/5)
    C_total = int(C*5)
    tensor_batch = torch.stack([
        img_batch[:, :, :, 0:W],
        img_batch[:, :, :, W:2*W],
        img_batch[:, :, :, 2*W:3*W],
        img_batch[:, :, :, 3*W:4*W],
        img_batch[:, :, :, 4*W:5*W]], dim=1)
    # tensor_batch: [8, 15, 288, 288]
    tensor_batch = torch.reshape(tensor_batch, [N, C_total, H, W])
    return tensor_batch
