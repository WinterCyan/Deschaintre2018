import torch
from Environment import *
from Utils import *
import math
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
import torch.nn.functional as F


def dot_vec(x, y):
    # x,y: [N,C,H,W]
    return torch.sum(torch.mul(x, y), dim=-3, keepdim=True)


def norm_vec(x):
    # vec: [N,3]
    # len = torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
    return torch.div(x, torch.sqrt(dot_vec(x, x)))


def xi(x):
    return (x > 0.0)*torch.ones_like(x)


def gamma(x):
    return torch.pow(x, 1.0/2.2)


def de_gamma(x):
    return torch.pow(x, 2.2)


class InNetworkRenderer:
    def diffuse_term(self, diffuse, ks):
        kd = 1.0-ks
        return kd*diffuse/math.pi

    def ndf(self, roughness, NH):
        alpha = roughness**2
        alpha_squared = alpha**2
        NH_squared = NH**2
        denom = torch.clamp(NH_squared*(alpha_squared+(1.0-NH_squared)/NH_squared), min=0.001)
        return (alpha_squared * xi(NH))/(math.pi * denom**2)


    def fresnel(self, specular, VH):
        return specular+(1.0-specular)*(1.0-VH)**5


    def g1(self, roughness, XH, XN):
        alpha = roughness**2
        alpha_squared = alpha**2
        XN_squared = XN**2
        return 2 * xi(XH/XN)/(1.0+torch.sqrt(1.0+alpha_squared*(1.0-XN_squared)/XN_squared))


    def geometry(self, roughness, VH, LH, VN, LN):
        return self.g1(roughness, VH, VN) * self.g1(roughness, LH, LN)


    def specular_term(self, wi, wo, normals, diffuse, roughness, specular):
        H = norm_vec((wi+wo)/2.0)
        NH = torch.clamp(dot_vec(normals, H), min=0.001)
        VH = torch.clamp(dot_vec(wo, H), min=0.001)
        LH = torch.clamp(dot_vec(wi, H), min=0.001)
        VN = torch.clamp(dot_vec(wo, normals), min=0.001)
        LN = torch.clamp(dot_vec(wi, normals), min=0.001)
        F = self.fresnel(specular, VH)
        G = self.geometry(roughness, VH, LH, VN, LN)
        D = self.ndf(roughness, NH)
        return F*G*D/(4.0*VN*LN), F


    def brdf(self, wi, wo, normals, diffuse, roughness, specular):
        spec, ks = self.specular_term(wi, wo, normals, diffuse, roughness, specular)
        diff = self.diffuse_term(diffuse, ks)
        return spec+diff


    def render(self, scene, svbrdf):
        device = svbrdf.device
        # svbrdf: [12, H, W], [0,1]
        normal = svbrdf[0:3, :, :]
        normal = de_map(normal)
        diffuse = svbrdf[3:6, :, :]
        roughness = svbrdf[6:9, :, :]
        roughness = torch.clamp(roughness, min=0.001)
        specular = svbrdf[9:12, :, :]

        coords_row = torch.linspace(-1.0, 1.0, svbrdf.shape[-1], device=device)
        coordsx = coords_row.unsqueeze(0).expand(svbrdf.shape[-2], svbrdf.shape[-1]).unsqueeze(0)
        coordsy = -1.0*torch.transpose(coordsx, dim0=1, dim1=2)
        coords = torch.cat((coordsx, coordsy, torch.zeros_like(coordsx)), dim=0)

        camera_pos = torch.Tensor(scene.camera.pos).unsqueeze(-1).unsqueeze(-1).to(device)
        light_pos = torch.Tensor(scene.light.pos).unsqueeze(-1).unsqueeze(-1).to(device)
        relative_camera_pos = camera_pos - coords
        relative_light_pos = light_pos - coords
        wi = norm_vec(relative_camera_pos)
        wo = norm_vec(relative_light_pos)

        f = self.brdf(wi, wo, normal, diffuse, roughness, specular)
        LN = torch.clamp(dot_vec(wi, normal), min=0.0)
        falloff = 1.0/torch.sqrt(dot_vec(relative_light_pos, relative_light_pos))**2
        lightcolor = torch.Tensor([50.0, 50.0, 50.0]).unsqueeze(-1).unsqueeze(-1).to(device)
        # lightcolor = torch.Tensor([10.0, 10.0, 10.0]).unsqueeze(-1).unsqueeze(-1).to(device)
        # lightcolor = torch.Tensor([30.0, 30.0, 30.0]).unsqueeze(-1).unsqueeze(-1).to(device)
        f = torch.clamp(f, min=0.0, max=1.0)
        radiance = torch.mul(torch.mul(f, lightcolor*falloff), LN)
        radiance = torch.clamp(radiance, min=0.01, max=1.0)

        return radiance

