import torch
import torch.nn as nn
import torch.nn.functional as F


def lcd(a, b):
    lcd = 2
    while lcd < a and lcd < b:
        if a % lcd == 0 and b % lcd == 0:
            return lcd
        lcd += 1
    return lcd

class Attn_Condenser_Block(nn.Module):
    def __init__(self, name, in_channels, mid_channels_0, out_channels_0,
                 mid_channels_1, out_channels_1, batch_norm=True):
        super(Attn_Condenser_Block, self).__init__()
        self.name = name
        self.batch_norm = batch_norm

        self.pool_0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv_0 = nn.Conv2d(in_channels, mid_channels_0, kernel_size=3,
                                      padding="same", groups=lcd(in_channels, mid_channels_0))
        self.pointwise_conv_0 = nn.Conv2d(mid_channels_0, out_channels_0, kernel_size=1, padding="same")
        self.unpool_0 = nn.Upsample(scale_factor=2, mode='nearest')
        self.expand_conv_0 = nn.Conv2d(out_channels_0, in_channels, kernel_size=1, padding="same")
        #self.batch_nrom_0 = nn.BatchNorm2d(num_features=out_channels_0)
        self.batch_nrom_0 = nn.BatchNorm2d(num_features=in_channels)
        self.S_0 = nn.Parameter(torch.Tensor(1))

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_conv_1 = nn.Conv2d(out_channels_0, mid_channels_1, kernel_size=3,
                                      padding="same", groups=lcd(out_channels_0, mid_channels_1))
        self.pointwise_conv_1 = nn.Conv2d(mid_channels_1, out_channels_1, kernel_size=1, padding="same")
        self.unpool_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.batch_nrom_1 = nn.BatchNorm2d(num_features=out_channels_1)
        self.S_1 = nn.Parameter(torch.Tensor(1))

    def forward(self, x):
        V_0 = x
        Q_0 = self.pool_0(V_0)
        K_0 = F.leaky_relu(self.group_conv_0(Q_0))
        K_0 = F.leaky_relu(self.pointwise_conv_0(K_0))
        A_0 = self.unpool_0(K_0)
        A_0_expend = F.leaky_relu(self.expand_conv_0(A_0))
        V_1 = V_0 * self.S_0 + A_0_expend
        if self.batch_norm:
            V_1 = self.batch_nrom_0(V_1)
        Q_1 = self.pool_1(A_0)
        K_1 = F.leaky_relu(self.group_conv_1(Q_1))
        K_1 = F.leaky_relu(self.pointwise_conv_1(K_1))
        A_1 = self.unpool_1(K_1)
        V_prime = V_1 * self.S_1 + A_1
        if self.batch_norm:
            V_prime = self.batch_nrom_1(V_prime)
        return V_prime
