import torch.nn.functional as F
import torch.nn as nn

def get_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

def get_conv_elu(in_channels, out_channels, alpha=1, kernel_size=3, stride=1, padding=0): 
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ELU(alpha= alpha, inplace=True)
    )

def get_conv_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

def get_conv_lrelu(in_channels, out_channels, slope=1e-2, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.LeakyReLU(negative_slope=slope, inplace=True)
    )

def get_elu(alpha=1):
    return nn.ELU(alpha=alpha, inplace=True)

def get_conv_preactivation(in_channels, out_channels, kernel_size=1, stride=1, padding=1):
    return nn.Sequential(
        nn.ELU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )

def get_conv_preactivation_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=1):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )

def get_max_pool(kernel_size, stride):
    return nn.Sequential(
        nn.MaxPool2d(kernel_size, stride)
    )

class ResConv(nn.Module):
    def __init__(self, ndf):
        super(ResConv, self).__init__()
        """
        Args:
            ndf: constant number from channels
            dil: dilation value - parameter for convolutional layers
            norma_type: normalization type (elu | batch norm)
        """
        self.ndf = ndf
        self.conv1 = get_conv_preactivation_relu(self.ndf, self.ndf, kernel_size=3, stride=1, padding=1)
        self.conv2 = get_conv_preactivation_relu(self.ndf, self.ndf, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        return out

#old one
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = get_conv_elu(in_channels, out_channels, stride, padding)
        self.conv2 = get_conv(out_channels, out_channels)      
        self.elu = get_elu()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        out = self.elu(out)
        return out

def get_residual(in_channels, out_channels, stride=1, padding=0):
    return ResidualBlock(in_channels, out_channels, stride, padding)