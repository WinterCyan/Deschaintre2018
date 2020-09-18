import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms

# ImageFile.LOAD_TRUNCATED_IMAGES = True
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
        img = self.transformer(img)  # [3,288,1440]
        rendered_img, svbrdf = img_to_tensor(img)  # [3,288,288], [12,288,288]
        return {'img': rendered_img, 'svbrdf': svbrdf}


def img_to_tensor(input_img):
    # input_img: [3,288,1440]
    C, H, W_total = input_img.shape
    W = int(W_total/5)
    output_tensor = torch.stack([
        input_img[:, :, 0:W],
        input_img[:, :, W:2*W],
        input_img[:, :, 2*W:3*W],
        input_img[:, :, 3*W:4*W],
        input_img[:, :, 4*W:5*W]], dim=0)  # [5,3,288,288]
    rendered_img = output_tensor[0]  # [3,288,288]
    svbrdf = output_tensor[1:5]  # [4,3,288,288]
    svbrdf = svbrdf.reshape([svbrdf.shape[0]*svbrdf.shape[1], svbrdf.shape[2], svbrdf.shape[3]])  # [12,288,288]
    return crop(rendered_img, svbrdf)


def crop(input_img, input_svbrdf):
    output_size = 256
    input_size = input_img.shape[-1]  # 288
    left_upper = torch.IntTensor(
        [np.random.randint(low=0, high=input_size-output_size),
         np.random.randint(low=0, high=input_size-output_size)])
    cropped_img = input_img[:, left_upper[0]:left_upper[0]+output_size, left_upper[1]:left_upper[1]+output_size]
    cropped_svbrdf = input_svbrdf[:, left_upper[0]:left_upper[0]+output_size, left_upper[1]:left_upper[1]+output_size]
    return cropped_img, cropped_svbrdf  # [3,256,256], [12,256,256]



