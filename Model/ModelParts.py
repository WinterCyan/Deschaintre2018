import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F


class ENCODER(nn.Module):
    def __init__(self, c_in, c_out, c_gb_in, c_gb_out):
        super().__init__()
        self.CONV = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)  # [N,Cin,Hin,Win] --> [N,Cout,Hout,Wout]
        self.NORM = nn.InstanceNorm2d(c_out)  # operate on each channel
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)  # operate element-wise
        self.Global2UnetFC = nn.Linear(c_gb_in, c_out)  # [N,CGBin] --> [N,Cout]
        self.GlobalFC = nn.Sequential(
            nn.Linear(c_gb_in+c_out, c_gb_out),  # [N, CGBin+Cout] --> [N, CGBout]
            nn.SELU()  # operate element-wise
        )

    # input: [N,C,H,W]
    def forward(self,unet_input,global_input):
        convolved = self.CONV(unet_input)
        normed = self.NORM(convolved)
        mean = torch.mean(convolved, [2, 3], True)  # calculate mean on H, W, get [N, Cout]
        global_feature = self.Global2UnetFC(global_input)
        torch.unsqueeze(global_feature,-1)
        expand_global_feature = torch.unsqueeze(torch.unsqueeze(global_feature, -1), -1).expand(-1, -1, 64, 64)
        unet_output = self.ReLU(torch.add(normed, expand_global_feature))
        squeezed_mean = torch.squeeze(torch.squeeze(mean, -1), -1)
        concatenated = torch.cat((global_input, squeezed_mean), dim=-1)  # get [N, CGBin+Cout]
        global_output = self.GlobalFC(concatenated)

        return unet_output, global_output


class DECODER(nn.Module):
    def __init__(self):
        super.__init__()
        self.DECONV = nn.ConvTranspose2d()
        self.NORM = nn.InstanceNorm2d()
        self.ReLU = nn.LeakyReLU()
        self.Global2UnetFC = nn.Linear()
        self.GlobalFC = nn.Sequential(
            nn.Linear(),
            nn.SELU()
        )

    def forward(self, encoder_link_input, unet_input, global_input):
        concatenated_input = torch.cat((encoder_link_input,unet_input), dim=-1)
        convolved = self.DECONV(concatenated_input)
        normed = self.NORM(convolved)
        mean = torch.mean(convolved, [1, 2], True)
        global_feature = self.Global2UnetFC(global_input)
        unet_output = self.ReLU(normed + global_feature)
        concatenated_global = torch.cat((global_input, mean), dim=-1)
        global_output = self.GlobalFC(concatenated_global)

        return unet_output, global_output
