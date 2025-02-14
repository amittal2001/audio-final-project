import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCondenser(nn.Module):
    def __init__(self, name, in_channels, mid_channels, out_channels):
        super(AttentionCondenser, self).__init__()
        self.condense = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, groups=1)
        self.pointwise_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.scale = nn.Parameter(torch.Tensor(1))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.expand_conv = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.name = name

    def forward(self, x):
        residual = x
        Q = self.condense(x)
        K = F.relu(self.group_conv(Q))
        K = F.relu(self.pointwise_conv(K))
        A = self.upsample(K)
        A = self.expand_conv(A)
        S = torch.sigmoid(A)
        V_prime = residual * S * self.scale
        V_prime += residual
        return V_prime


class Attn_BN_Block(nn.Module):
    def __init__(self, name, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1):
        super(Attn_BN_Block, self).__init__()
        self.name = name
        self.layer1 = AttentionCondenser(f"{self.name}_AttnCond1", in_channels, mid_channels_0, out_channels_0)
        self.layer2 = nn.BatchNorm2d(num_features=in_channels)
        self.layer3 = AttentionCondenser(f"{self.name}_AttnCond2", in_channels, mid_channels_1, out_channels_1)
        self.layer4 = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        x_ = self.layer1(x)
        x_ = self.layer2(x_)
        x_ = self.layer3(x_)
        x_ = self.layer4(x_)
        x_ += x
        return x_
