import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnCondenserBlock(nn.Module):
    """
    A simplified attention condenser block.

    This block performs:
      - A condensation branch:
           • Downsample the input using AvgPool2d,
           • Apply a depthwise convolution (with groups=in_channels) followed by a pointwise convolution,
           • Upsample the result to the original resolution,
           • Apply an expansion convolution to bring the channels back to in_channels,
           • Fuse with the original input via a learnable scale.

      - An attention branch:
           • Use the condensed feature (before upsampling) from the condensation branch,
           • Apply a depthwise convolution (with groups equal to out_channels0) and a pointwise convolution,
           • Upsample the result so that its spatial dimensions match the input,
           • Fuse with the output of the condensation branch via a learnable scale.

      - Optionally applies BatchNorm after each fusion.

    :param name: Identifier for the block.
    :param in_channels: Number of input channels.
    :param mid_channels0: Intermediate channels for the condensation branch.
    :param out_channels0: Output channels for the condensation branch.
    :param mid_channels1: Intermediate channels for the attention branch.
    :param out_channels1: Output channels for the attention branch (should equal in_channels for addition).
    :param batch_norm: Whether to apply BatchNorm (default: True).
    """

    def __init__(self, name, in_channels, mid_channels0, out_channels0, mid_channels1, out_channels1, batch_norm=True):
        super(AttnCondenserBlock, self).__init__()
        self.name = name
        self.batch_norm = batch_norm

        # --- Condensation branch ---
        # Downsample input
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Depthwise conv: groups=in_channels minimizes parameters.
        self.conv1 = nn.Conv2d(in_channels, mid_channels0, kernel_size=3, padding=1, groups=in_channels)
        # Pointwise conv
        self.conv2 = nn.Conv2d(mid_channels0, out_channels0, kernel_size=1, padding=0)
        # Depthwise conv: groups=in_channels minimizes parameters.
        self.conv3 = nn.Conv2d(out_channels0, mid_channels1, kernel_size=3, padding=1, groups=out_channels0)
        # Pointwise conv
        self.conv4 = nn.Conv2d(mid_channels1, out_channels1, kernel_size=1, padding=0)
        if self.batch_norm:
            self.normalize = nn.BatchNorm2d(in_channels)
        # Upsample to recover original spatial dimensions
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.S = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Condensation branch
        x_pool = self.pool(x)  # (B, in_channels, H/2, W/2)
        y = self.conv1(x_pool)  # (B, mid_channels0, H/2, W/2)
        y = F.leaky_relu(self.conv2(y))  # (B, out_channels0, H/2, W/2)
        z = self.conv3(y)  # (B, mid_channels1, H/2, W/2)
        z = F.leaky_relu(self.conv4(z))  # (B, out_channels1, H/2, W/2)
        if self.batch_norm:
            z = self.normalize(z)
        z_up = self.up(z)  # (B, out_channels1, H, W)
        out = x * self.S + z_up   # (B, in_channels, H, W)
        return out
