import torch
import math
import torch.nn.functional as F


def map_to_img(x):
    return x/2.0 + 0.5


def de_map(x):
    return 2.0*x - 1.0


def generate_direction(batch_size, low_eps=0.001, high_eps=0.05):
    r1 = low_eps + (1.0 - high_eps - low_eps) * torch.rand([batch_size, 1], dtype=torch.float32)  # [0.001, 0.05]
    r2 = torch.rand([batch_size, 1], dtype=torch.float32)  # [0, 1]
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - torch.square(r))
    final_vec = torch.cat([x,y,z], dim=-1)  # [N, 3], every vec for a single sample in the batch
    return final_vec


def generate_distance(batch_size):
    norm_distribution = torch.distributions.Normal(loc=0.5, scale=0.75)
    expo = norm_distribution.sample([batch_size, 1])
    return torch.exp(expo)


def norm_vec(x):
    # vec: [N,3]
    len = torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
    return torch.div(x, len)


def dot_vec(x,y):
    # x,y: [N,C,H,W]
    return torch.mul(x,y)


# svbrdf: [N,12,H,W], normal, diffuse, roughness, specular
# wi, wo: [N, 3]
def tf_render(svbrdf, wi, wo):
    normal = svbrdf[:, 0:3, :, :]
    diffuse = torch.clamp(map_to_img(svbrdf[:, 3:6, :, :]), min=0.0, max=1.0)
    roughness = torch.clamp(map_to_img(svbrdf[:, 6:9, :, :]), min=0.001, max=1.0)
    specular = torch.clamp(map_to_img(svbrdf[:, 9:12, :, :]), min=0.0, max=1.0)
    wi_norm = norm_vec(wi)
    wo_norm = norm_vec(wo)
    wi_norm = torch.unsqueeze(torch.unsqueeze(wi_norm, -1), -1)  # [N,3,1,1]
    wo_norm = torch.unsqueeze(torch.unsqueeze(wo_norm, -1), -1)  # [N,3,1,1]
    h = norm_vec(torch.add(wi_norm, wo_norm)/2.0)  # [N,3,1,1]
    NdotH = torch.mul(normal, h)
    NdotL = torch.mul(normal, wi_norm)
    NdotV = torch.mul(normal, wo_norm)
    VdotH = torch.mul(wo_norm, h)
    NdotH_pos = torch.clamp(NdotH, min=0.0)
    NdotL_pos = torch.clamp(NdotL, min=0.0)
    NdotV_pos = torch.clamp(NdotV, min=0.0)
    VdotH_pos = torch.clamp(VdotH, min=0.0)

    diffuse_rendered = diffuse * (1.0 - specular)/math.pi

    # D
    alpha = torch.square(roughness)
    denominator = 1.0/torch.clamp((torch.square(NdotH_pos)*(torch.square(alpha)-1.0)+1.0), min=0.001)
    D = torch.square(alpha*denominator)/math.pi

    # F
    sphg = torch.pow(((-5.55473*VdotH_pos)-6.98316)*VdotH_pos, exponent=2.0)
    F = specular+(1.0-specular)*sphg

    # G
    k = torch.square(roughness)/2.0
    g1 = 1.0/torch.clamp((NdotL_pos*(1.0-k)+k), min=0.001)
    g2 = 1.0/torch.clamp((NdotV_pos*(1.0-k)+k), min=0.001)
    G = g1 + g2

    specular_rendered = F*(G*D*0.25)
    result = specular_rendered
    result = torch.add(result, diffuse_rendered)

    light_intensity = 1.0
    light_factor = light_intensity * math.pi

    result = result * light_factor

    # This division is to compensate for the cosinus distribution of the intensity in the rendering
    # result = result * NdotL_pos / tf.expand_dims(tf.maximum(wiNorm[:,:,:,2], 0.001), axis=-1)
    # result = result * NdotL_pos / torch.unsqueeze(torch.clamp(wi_norm[:, :, :, 2], min=0.001), dim=-1)

    # return [result, D, G, F, diffuse_rendered, diffuse]
    return result
