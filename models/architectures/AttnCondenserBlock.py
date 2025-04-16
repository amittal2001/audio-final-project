import torch.nn.functional as F
import torch.nn as nn
import torch


class AttnCondenserBlock(nn.Module):
    """
    A simplified attention condenser block.

    This block performs:
      - A condensation branch:
           • Downsample the input using MaxPool2d,
           • Apply a depthwise convolution (with groups=in_channels) followed by a pointwise convolution,
           • Upsample the result to the original resolution using interpolation,
           • Apply an expansion convolution to bring the channels back to in_channels,
           • Fuse with the original input via a learnable scale.

      - An attention branch:
           • Use the condensed feature (from conv2) directly,
           • Apply a depthwise convolution (with groups equal to out_channels0) and a pointwise convolution,
           • Upsample the result so that its spatial dimensions match the input using interpolation,
           • Fuse with the output of the condensation branch via a learnable scale.

      - Optionally applies BatchNorm after each fusion.
    """

    def __init__(self, name, in_channels, mid_channels0, out_channels0, mid_channels1, out_channels1, batch_norm=True):
        super(AttnCondenserBlock, self).__init__()
        self.name = name
        self.batch_norm = batch_norm

        # --- Condensation branch ---
        # Downsample input using max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Note: self.up is removed since we will use F.interpolate for dynamic resizing.
        # Depthwise conv: groups=in_channels minimizes parameters.
        self.conv1 = nn.Conv2d(in_channels, mid_channels0, kernel_size=3, padding=1, groups=in_channels)
        # Pointwise conv
        self.conv2 = nn.Conv2d(mid_channels0, out_channels0, kernel_size=1, padding=0)
        # Expansion conv to match original channel count
        self.expand_conv = nn.Conv2d(out_channels0, in_channels, kernel_size=1, padding=0)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
        # Learnable scaling for branch fusion
        self.S0 = nn.Parameter(torch.ones(1))

        # --- Attention branch ---
        # Use the condensed feature (output of conv2) directly
        self.conv3 = nn.Conv2d(out_channels0, mid_channels1, kernel_size=3, padding=1, groups=out_channels0)
        self.conv4 = nn.Conv2d(mid_channels1, out_channels1, kernel_size=1, padding=0)
        if self.batch_norm:
            self.bn2 = nn.BatchNorm2d(out_channels1)
        self.S1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Store original spatial dimensions
        H, W = x.shape[2], x.shape[3]

        # Condensation branch
        x_pool = self.pool(x)  # (B, in_channels, H/2, W/2)
        y = F.leaky_relu(self.conv1(x_pool))  # (B, mid_channels0, H/2, W/2)
        y = F.leaky_relu(self.conv2(y))  # (B, out_channels0, H/2, W/2)
        # Interpolate to exactly match original spatial dims
        y_up = F.interpolate(y, size=(H, W), mode='nearest')
        branch1 = x * self.S0 + self.expand_conv(y_up)  # (B, in_channels, H, W)
        if self.batch_norm:
            branch1 = self.bn1(branch1)

        # Attention branch
        # Use the same condensed feature y directly
        z = F.leaky_relu(self.conv3(y))  # (B, mid_channels1, H/2, W/2)
        z = F.leaky_relu(self.conv4(z))  # (B, out_channels1, H/2, W/2)
        # Interpolate to match the original spatial dims
        z_up = F.interpolate(z, size=(H, W), mode='nearest')
        out = branch1 * self.S1 + z_up  # (B, in_channels, H, W) assuming out_channels1 == in_channels
        if self.batch_norm:
            out = self.bn2(out)
        return out
