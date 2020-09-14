import torch
import math
import torch.nn as nn
import torch.functional as F


def expand_split_svbrdf(input_svbrdf):
    # input_svbrdf: [N, 9, H, W], 2, 3, 1, 3; between [-1,1]
    # output_svbrdf: [N, 12, H, W]
    splited = input_svbrdf.split(1, dim=-3)

    normals_xy = torch.cat(splited[0:2], dim=-3)  # [8,2,H,W]
    diffuse = torch.cat(splited[2:5], dim=-3)  # [8,3,H,W]
    roughness_scalar = torch.cat(splited[5:6], dim=-3)  # [8,1,H,W]
    specular = torch.cat(splited[6:9], dim=-3)  # [8,3,H,W]

    roughness_shape = [1, 3, 1, 1]
    roughness = roughness_scalar.repeat(roughness_shape)

    normals_x, normals_y = torch.split(normals_xy.mul(3.0), 1, dim=-3)
    normals_z = torch.ones_like(normals_x)
    normals = torch.cat([normals_x, normals_y, normals_z], dim=-3)
    norm = torch.sqrt(torch.sum(torch.pow(normals, 2.0), dim=-3, keepdim=True))
    normals = torch.div(normals, norm)

    return normals, diffuse, roughness, specular



def expand_svbrdf(input_svbrdf):
    normals, diffuse, roughness, specular = expand_split_svbrdf(input_svbrdf)
    expanded = torch.cat([normals, diffuse, roughness, specular], dim=-3)  # [8,12,H,W]

    return expanded


def generate_normalized_random_direction(count, min_eps=0.001, max_eps=0.05):
    r1 = torch.Tensor(count, 1).uniform_(0.0 + min_eps, 1.0 - max_eps)
    r2 = torch.Tensor(count, 1).uniform_(0.0, 1.0)

    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - r ** 2)

    return torch.cat([x, y, z], axis=-1)


def generate_distance(batch_size):
    norm_distribution = torch.distributions.Normal(loc=0.5, scale=0.75)
    expo = norm_distribution.sample([batch_size, 1])
    return torch.exp(expo)
