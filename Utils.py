import torch
import math
import torch.nn as nn
import torch.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms


def expand_split_svbrdf(input_svbrdf):
    # input_svbrdf: [N, 9, H, W], 2, 3, 1, 3; between [-1,1]
    splited = input_svbrdf.split(1, dim=-3)

    diffuse = map_to_img(torch.cat(splited[2:5], dim=-3))  # [8,3,H,W], [0,1]
    roughness_scalar = map_to_img(torch.cat(splited[5:6], dim=-3))  # [8,1,H,W], [0,1]
    specular = map_to_img(torch.cat(splited[6:9], dim=-3))  # [8,3,H,W], [0,1]

    roughness_shape = [1, 3, 1, 1]
    roughness = roughness_scalar.repeat(roughness_shape)

    normals_xy = torch.cat(splited[0:2], dim=-3)  # [8,2,H,W], [-1,1]
    # normals_x, normals_y = torch.split(normals_xy.mul(3.0), 1, dim=-3)
    normals_x, normals_y = torch.split(normals_xy, 1, dim=-3)  # [-1,1]
    normals_z = torch.ones_like(normals_x)
    normals = torch.cat([normals_x, normals_y, normals_z], dim=-3)
    norm = torch.sqrt(torch.sum(torch.pow(normals, 2.0), dim=-3, keepdim=True))
    normals = torch.div(normals, norm)  # norm, [N, 3, H, W], [-1,1]
    normals = map_to_img(normals)

    # 4*[N,3,H,W], between [0,1]
    return normals, diffuse, roughness, specular


def split_svbrdf(input_svbrdf):
    # input_svbrdf: [N, 12, H, W], between [0,1]
    splited = input_svbrdf.split(1, dim=-3)

    normals = torch.cat(splited[0:3], dim=-3)  # [8,3,H,W], [0,1]
    diffuse = torch.cat(splited[3:6], dim=-3)  # [8,3,H,W], [0,1]
    roughness = torch.cat(splited[6:9], dim=-3)  # [8,3,H,W], [0,1]
    specular = torch.cat(splited[9:12], dim=-3)  # [8,3,H,W], [0,1]

    return normals, diffuse, roughness, specular


def expand_svbrdf(input_svbrdf):
    # input_svbrdf: [N, 9, H, W], 2, 3, 1, 3; between [-1,1]
    normals, diffuse, roughness, specular = expand_split_svbrdf(input_svbrdf)
    # [N,12,H,W], between [0,1]
    expanded = torch.cat([normals, diffuse, roughness, specular], dim=-3)

    return expanded


def generate_normalized_random_direction(count):
    # def generate_normalized_random_direction(count=5, min_eps=0.001, max_eps=0.05):
    theta = torch.Tensor(count, 1).uniform_(1.0/6.0, 1.0/4.0)*math.pi
    phi = torch.Tensor(count, 1).uniform_(0.0, 1.0)*math.pi
    z = torch.cos(theta)
    xy = torch.sin(theta)
    x = xy * torch.cos(phi)
    y = xy * torch.sin(phi)
    return torch.cat([x, y, z], dim=-1)


def generate_distance():
    distance = torch.Tensor(1).uniform_(6, 8)
    # distance = torch.Tensor(1).uniform_(8, 10)
    return torch.sqrt(distance)


to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
scale_trans = transforms.Resize([288, 288])


def map_to_img(x):
    return x/2.0 + 0.5


def de_map(x):
    return 2.0*x - 1.0


def displayimg(t):
    img = to_img(t)
    plt.imshow(img)
    plt.show()


def displaybrdf(t):
    # t: [12,H,W]
    C_total, H, W = t.shape
    t = t.reshape(4, 3, H, W)  # [3,4,H,W]
    normals = t[0]
    albedo = t[1]
    roughness = t[2]
    specular = t[3]
    img = torch.cat([normals, albedo, roughness, specular], dim=-1)
    img = to_img(img.reshape(3, H, W*4))
    plt.imshow(img)
    plt.show()
