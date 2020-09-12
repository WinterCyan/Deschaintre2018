from Model import *
import time
import PIL
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
from Dataset import *
from matplotlib import pyplot as plt
from InNetworkRenderer import *

if __name__ == '__main__':
    # x = torch.randn(5, 64, 128, 128)
    # x_gb = torch.randn(5, 128)
    # encoder = Encoder(c_in=64, c_out=128, c_gb_in=128, c_gb_out=256)
    # unet_output, global_output = encoder.forward(unet_input=x, global_input=x_gb)
    # print('running encoder')
    # print(unet_output.shape)
    # print(global_output.shape)
    # print()

    to_img = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    dataset_path = '/media/winter/M2/UbuntuDownloads/Deschaintre2018_Dataset/Data_Deschaintre18/testBlended'
    print('fetching data...')
    dataset = MaterialDataset(data_dir=dataset_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    demo_batch = []
    for i, img_batch in enumerate(data_loader):
        if i == 0:
            demo_batch = batch_to_tensor(img_batch[0])  # [8,15,288,288]
    N, C_total, H, W = demo_batch.shape
    svbrdf = demo_batch[:, 3:15, :, :]

    lightpos = generate_direction(N)
    viewpos = generate_direction(N)
    result = tf_render(svbrdf, lightpos, viewpos)
    print(result.shape)
    img1 = to_img(result[1, :, :, :])
    plt.imshow(img1)
    plt.show()

