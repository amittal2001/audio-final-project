import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCondenser(nn.Module):
    def __init__(self, name, in_channels, mid_channels, out_channels):
        super(AttentionCondenser, self).__init__()
        self.condense = nn.MaxPool2d(kernel_size=2, stride=2)
        self.embedding1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same")
        self.embedding2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same")
        self.embedding3 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding="same")
        self.expand = nn.Upsample(scale_factor=2, mode='nearest')
        self.S = nn.Parameter(torch.Tensor(1))
        self.name = name

    def forward(self, x):
        V = x
        Q = self.condense(V)
        K_ = F.relu(self.embedding1(Q))
        K_ = F.relu(self.embedding2(K_))
        K = F.relu(self.embedding3(K_))
        A = self.expand(K)
        V_prime = V * self.S + A
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


class Attn_Block(nn.Module):
    def __init__(self, name, in_channels, mid_channels_0, out_channels_0, mid_channels_1, out_channels_1):
        super(Attn_Block, self).__init__()
        self.name = name
        self.layer1 = AttentionCondenser(f"{self.name}_AttnCond1", in_channels, mid_channels_0, out_channels_0)
        self.layer2 = AttentionCondenser(f"{self.name}_AttnCond2", in_channels, mid_channels_1, out_channels_1)

    def forward(self, x):
        x_ = self.layer1(x)
        x_ = self.layer2(x_)
        x_ += x
        return x_


