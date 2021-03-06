from Model import *
import time
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from Dataset import *
from matplotlib import pyplot as plt
from InNetworkRenderer import *
from Utils import *

if __name__ == '__main__':
    # x = torch.randn(5, 64, 128, 128)
    # x_gb = torch.randn(5, 128)
    # encoder = Encoder(c_in=64, c_out=128, c_gb_in=128, c_gb_out=256)
    # unet_output, global_output = encoder.forward(unet_input=x, global_input=x_gb)
    # print('running encoder')
    # print(unet_output.shape)
    # print(global_output.shape)
    # print()

    # to_img = transforms.ToPILImage()
    # to_tensor = transforms.ToTensor()
    #
    # img = Image.open('5.png')
    # img = np.float32(img)/255.0
    # img = to_tensor(img)
    # img = img.unsqueeze(dim=0)
    # img_tensor = torch.squeeze(batch_to_tensor(img), dim=0).cuda()
    # svbrdf_single = img_tensor[3:, :, :]
    # lightpos = torch.tensor([0.0, -1.0, 2.0])
    # viewpos = torch.tensor([0.0, 0.0, 2.0])
    # # lightpos = torch.squeeze(lightpos, dim=0)
    # lightpos = lightpos.squeeze(dim=0)
    # viewpos = viewpos.squeeze(dim=0)
    # renderer = InNetworkRenderer()
    # scene = generate_specular_scenes(1)[0]
    # result = renderer.render(scene, svbrdf_single)
    # result = result.cpu()
    # print(result.shape)
    # rendered = gamma(result)
    # rendered = to_img(rendered)
    # plt.imshow(rendered)
    # plt.show()

    # dataset_path = '/media/winter/M2/UbuntuDownloads/Deschaintre2018_Dataset/Data_Deschaintre18/copied'
    dataset_path = 'E:\\UbuntuDownloads\\Deschaintre2018_Dataset\\Data_Deschaintre18\\testBlended'
    print('fetching data...')
    dataset = MaterialDataset(data_dir=dataset_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    demo_batch = []
    order = 4
    in_net_renderer = InNetworkRenderer()
    # for i, (img_batch, svbrdf_batch) in enumerate(data_loader):
    #     displayimg(img_batch[order])
    #     displaybrdf(svbrdf_batch[order])
    #     scenes = generate_random_scenes(count=3) + generate_specular_scenes(count=3)
    #     for scene in scenes:
    #         displayimg(renderer.render(scene, svbrdf_batch[order]))

    device = 'cuda:0'
    img_batch = []
    svbrdf_batch = []
    model = MaterialNet().to(device)
    l1_loss_func = L1Loss()
    rendering_loss_func = RenderingLoss(renderer=in_net_renderer)
    mix_loss_func = MixLoss(renderer=in_net_renderer)
    for i, sample in enumerate(data_loader):
        if i == 0:
            img_batch = sample["img"].to(device)  # [8,3,256,256]
            svbrdf_batch = sample["svbrdf"].to(device)  # [8,12,256,256]
            break
        break

    print(img_batch.device, svbrdf_batch.device)
    print(img_batch.shape, svbrdf_batch.shape)
    estimated_svbrdf_batch = model(img_batch)
    print(estimated_svbrdf_batch.device, estimated_svbrdf_batch.shape)
    loss = mix_loss_func(estimated_svbrdf_batch, svbrdf_batch)
    print(loss.item(), loss.device)
