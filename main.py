# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from Model.ModelParts import *


if __name__ == '__main__':
    x = torch.randn(5, 64, 128, 128)
    x_gb = torch.randn(5, 128)
    encoder = ENCODER(64,128,128,256)
    unet_output, global_output = ENCODER.forward(encoder,x,x_gb)
    print('running')
    print(unet_output.shape)
    print(global_output.shape)
