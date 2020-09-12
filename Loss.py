import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# prediction: img of [288,1440] (rendered, normal, albedo, roughness, specular)

to_img = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
img = Image.open("1.png")
img_tensor = to_tensor(img)


# sub images: [3,288,288]
# rendered = img_tensor[:,:,0:288]
# normal = img_tensor[:,:,288:2*288]
# albedo = img_tensor[:,:,2*288:3*288]
# roughness = img_tensor[:,:,3*288:4*288]
# specular = img_tensor[:,:,4*288:5*288]


# train pipeline:
# input pair: [N,3*5,H,W], light pos, view pos
# [N,0:3,H,W] -> PredictionNet -> [N,9,H,W](+lightpos, viewpos) -> Renderer -> [N,3,H,W]
#                   |<------- loss --<---|---------<---------------------------------|

# input_img: [N, 3*5, H, W] + lightpos + viewpos
# output_params: [N, 9, H, W], normal + albedo +
def render_loss(input_params, lightpos, viewpos, output_params):
    N, C_total, H, W = input_params.shape
    C = C_total / 5.0
    GT_rendered = input_params[:, 0:C, :, :]
    GT_normal = input_params[:, C:2 * C, :, :]
    GT_albedo = input_params[:, 2 * C:3 * C, :, :]
    GT_roughness = input_params[:, 3 * C:4 * C, :, :]
    GT_specular = input_params[:, 4 * C:5 * C, :, :]
    extended_params = extend_output(output_params)  # [N,3*4,H,W], normal + albedo + roughness + specular
    output_normal = extended_params[:, 0:C, :, :]
    output_albedo = extended_params[:, C:2 * C, :, :]
    output_roughness = extended_params[:, 2 * C:3 * C, :, :]
    output_specular = extended_params[:, 3 * C:4 * C, :, :]
    output_rendered = torch_renderer(
        normal=output_normal,
        albedo=output_albedo,
        roughness=output_roughness,
        specular=output_specular,
        lightpos=lightpos,
        viewpos=viewpos
    )
    loss = img_l2_loss(GT_rendered, output_rendered)  # confirm that img are > 0, < 1 ???
    return loss


def extend_output(params):
    extended = params
    return extended


def torch_renderer(normal, albedo, roughness, specular, lightpos, viewpos):
    rendering = normal
    return rendering


def img_l2_loss(img1, img2):
    diff = torch.log(img1+0.01) - torch.log(img2+0.01)
    loss = torch.sum(diff**2)
    return loss