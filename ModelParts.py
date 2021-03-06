import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, c_in, c_out, c_gb_in, c_gb_out):
        super().__init__()
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)  # operate element-wise
        self.CONV = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)  # [N,Cin,Hin,Win] --> [N,Cout,Hout,Wout]
        self.NORM = nn.InstanceNorm2d(c_out)  # operate on each channel
        self.Global2UnetFC = nn.Linear(c_gb_in, c_out)  # [N,CGBin] --> [N,Cout]
        self.GlobalFC = nn.Sequential(
            nn.Linear(c_gb_in+c_out, c_gb_out),  # [N, CGBin+Cout] --> [N, CGBout]
            nn.SELU()  # operate element-wise
        )

    # input: [N,C,H,W]
    def forward(self, unet_input, global_input):
        # relu -> conv -> norm
        relu_result = self.ReLU(unet_input)
        conv_result = self.CONV(relu_result)
        h_out = conv_result.shape[2]
        w_out = conv_result.shape[3]
        mean = torch.mean(conv_result, [2, 3])
        norm_result = self.NORM(conv_result)
        global_feature = self.Global2UnetFC(global_input)
        expand_global_feature = global_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_out, w_out)
        unet_output = torch.add(norm_result, expand_global_feature)
        concat_global = torch.cat((global_input, mean), dim=-1)
        global_output = self.GlobalFC(concat_global)
        return unet_output, global_output

        # convolved = self.CONV(unet_input)
        # normed = self.NORM(convolved)
        # mean = torch.mean(convolved, [2, 3])  # calculate mean on H, W, get [N, Cout]
        # global_feature = self.Global2UnetFC(global_input)
        # expand_global_feature = torch.unsqueeze(torch.unsqueeze(global_feature, -1), -1).expand(-1, -1, h_out, w_out)
        # unet_output = self.ReLU(torch.add(normed, expand_global_feature))
        #
        # concatenated_global = torch.cat((global_input, mean), dim=-1)  # get [N, CGBin+Cout]
        # global_output = self.GlobalFC(concatenated_global)
        #
        # return unet_output, global_output


class Decoder(nn.Module):
    def __init__(self, c_in, c_link, c_out, c_gb_in, c_gb_out):
        super().__init__()
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)
        self.DeCONV = nn.ConvTranspose2d(c_in+c_link, c_out, kernel_size=4, stride=2, padding=1)
        self.NORM = nn.InstanceNorm2d(c_out)
        self.Global2UnetFC = nn.Linear(c_gb_in, c_out)
        self.GlobalFC = nn.Sequential(
            nn.Linear(c_gb_in+c_out, c_gb_out),
            nn.SELU()
        )

    def forward(self, unet_input, link_input, global_input, dropout=0.0):
        # concat -> relu -> deconv -> norm
        concat_result = torch.cat((link_input, unet_input), dim=1)
        relu_result = self.ReLU(concat_result)
        conv_result = self.DeCONV(relu_result)
        norm_result = self.NORM(conv_result)
        h_out = norm_result.shape[2]
        w_out = norm_result.shape[3]
        mean = torch.mean(conv_result, [2, 3])
        global_feature = self.Global2UnetFC(global_input)
        expand_global_feature = global_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_out, w_out)
        unet_output = torch.add(norm_result, expand_global_feature)
        concat_global = torch.cat((global_input, mean), dim=-1)
        global_output = self.GlobalFC(concat_global)
        if dropout > 0.0:
            unet_output = F.dropout(unet_output, p=dropout)
        return unet_output, global_output

        # concatenated_input = torch.cat((link_input, unet_input), dim=1)
        # convolved = self.DeCONV(concatenated_input)
        # normed = self.NORM(convolved)
        # h_out = normed.shape[2]
        # w_out = normed.shape[3]
        # mean = torch.mean(convolved, [2, 3])
        # global_feature = self.Global2UnetFC(global_input)
        # expand_global_feature = torch.unsqueeze(torch.unsqueeze(global_feature, -1), -1).expand(-1, -1, h_out, w_out)
        # unet_output = self.ReLU(torch.add(normed, expand_global_feature))
        # concatenated_global = torch.cat((global_input, mean), dim=-1)
        # global_output = self.GlobalFC(concatenated_global)
        # if dropout > 0.0:
        #     unet_output = F.dropout(unet_output, p=dropout)
        #
        # return unet_output, global_output


class InitEncoder(nn.Module):
    def __init__(self, c_in, c_out, c_gb_out):
        super().__init__()
        self.CONV = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.GlobalFC = nn.Sequential(
            nn.Linear(c_in, c_gb_out),
            nn.SELU()
        )

    def forward(self, unet_input):
        unet_output = self.CONV(unet_input)
        mean = torch.mean(unet_input, [2, 3])
        global_output = self.GlobalFC(mean)

        return unet_output, global_output


class LastEncoder(nn.Module):
    def __init__(self, c_in, c_out, c_gb_in, c_gb_out):
        super().__init__()
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)
        self.CONV = nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.Global2UnetFC = nn.Linear(c_gb_in, c_out)
        self.GlobalFC = nn.Sequential(
            nn.Linear(c_gb_in+c_out, c_gb_out),
            nn.SELU()
        )

    def forward(self,unet_input,global_input):
        # relu -> conv, no norm
        relu_result = self.ReLU(unet_input)
        conv_result = self.CONV(relu_result)
        mean = torch.mean(conv_result, [2, 3])
        global_feature = self.Global2UnetFC(global_input)
        expand_global_feature = global_feature.unsqueeze(-1).unsqueeze(-1)
        unet_output = torch.add(conv_result, expand_global_feature)
        concat_global = torch.cat((global_input, mean), dim=-1)
        global_output = self.GlobalFC(concat_global)
        return unet_output, global_output

        # convolved = self.CONV(unet_input)
        # global_feature = self.Global2UnetFC(global_input)
        # expand_global_feature = torch.unsqueeze(torch.unsqueeze(global_feature, -1), -1)
        # convolved = torch.add(convolved, expand_global_feature)
        # mean = torch.mean(convolved, [2, 3])
        # unet_output = self.ReLU(convolved)
        # concatenated_global = torch.cat((global_input, mean), dim=-1)
        # global_output = self.GlobalFC(concatenated_global)
        # return unet_output, global_output


class InitDecoder(nn.Module):
    def __init__(self, c_in, c_out, c_gb_in, c_gb_out):
        super().__init__()
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)
        self.DeCONV = nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1)
        self.NORM = nn.InstanceNorm2d(c_out)
        self.Global2UnetFC = nn.Linear(c_gb_in, c_out)
        self.GlobalFC = nn.Sequential(
            nn.Linear(c_gb_in+c_out, c_gb_out),
            nn.SELU()
        )

    def forward(self, unet_input, global_input, dropout=0.0):
        relu_result = self.ReLU(unet_input)
        conv_result = self.DeCONV(relu_result)
        h_out = conv_result.shape[2]
        w_out = conv_result.shape[3]
        mean = torch.mean(conv_result, [2, 3])
        norm_result = self.NORM(conv_result)
        global_feature = self.Global2UnetFC(global_input)
        expand_global_feature = global_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_out, w_out)
        unet_output = torch.add(norm_result, expand_global_feature)
        concat_global = torch.cat((global_input, mean), dim=-1)
        global_output = self.GlobalFC(concat_global)
        if dropout > 0.0:
            unet_output = F.dropout(unet_output, p=dropout)
        return unet_output, global_output

        # convolved = self.DeCONV(unet_input)
        # normed = self.NORM(convolved)
        # h_out = normed.shape[2]
        # w_out = normed.shape[3]
        # mean = torch.mean(convolved, [2, 3])
        # global_feature = self.Global2UnetFC(global_input)
        # expand_global_feature = torch.unsqueeze(torch.unsqueeze(global_feature, -1), -1).expand(-1, -1, h_out, w_out)
        # unet_output = self.ReLU(torch.add(normed, expand_global_feature))
        # concatenated_global = torch.cat((global_input, mean), dim=-1)
        # global_output = self.GlobalFC(concatenated_global)
        # if dropout > 0.0:
        #     unet_output = F.dropout(unet_output, p=dropout)
        #
        # return unet_output, global_output


class LastDecoder(nn.Module):
    def __init__(self, c_in, c_link, c_out, c_gb_in):
        super().__init__()
        self.ReLU = nn.LeakyReLU(negative_slope=0.2)
        self.DeCONV = nn.ConvTranspose2d(c_in+c_link, c_out, kernel_size=4, stride=2, padding=1)
        self.Global2UnetFC = nn.Linear(c_gb_in, c_out)

    def forward(self, unet_input, link_input, global_input):
        concat_result = torch.cat((link_input, unet_input), dim=1)
        relu_result = self.ReLU(concat_result)
        conv_result = self.DeCONV(relu_result)
        global_feature = self.Global2UnetFC(global_input)
        h_out = conv_result.shape[2]
        w_out = conv_result.shape[3]
        expand_global_feature = global_feature.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_out, w_out)
        unet_output = torch.tanh(torch.add(conv_result, expand_global_feature))
        return unet_output

        # concatenated_input = torch.cat((link_input, unet_input), dim=1)
        # convolved = self.DeCONV(concatenated_input)
        # h_out = convolved.shape[2]
        # w_out = convolved.shape[3]
        # global_feature = self.Global2UnetFC(global_input)
        # expand_global_feature = torch.unsqueeze(torch.unsqueeze(global_feature, -1), -1).expand(-1, -1, h_out, w_out)
        # unet_output = torch.add(convolved, expand_global_feature)
        # unet_output = torch.tanh(unet_output)
        #
        # return unet_output
