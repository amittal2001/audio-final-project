import torch.nn as nn
import torch.nn.functional as F
from models.architectures.AttnCondenserBlock import AttnCondenserBlock


class TinySpeechX(nn.Module):
    """
    TinySpeechX architecture.

    Architecture:
      - Input conv: 1 → 15 channels.
      - 6 AttnCondenserBlocks with varying configurations.
      - Final conv: 15 → 38 channels.
      - Global average pooling and fully connected layer.
    """

    def __init__(self, num_classes):
        super(TinySpeechX, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1, padding=1)
        self.block1 = AttnCondenserBlock("AttnCondBlock1", 15, 30, 6, 12, 15)
        self.block2 = AttnCondenserBlock("AttnCondBlock2", 15, 30, 6, 12, 15)
        self.block3 = AttnCondenserBlock("AttnCondBlock3", 15, 30, 6, 12, 15)
        self.block4 = AttnCondenserBlock("AttnCondBlock4", 15, 30, 19, 38, 15)
        self.block5 = AttnCondenserBlock("AttnCondBlock5", 15, 30, 26, 52, 15)
        self.block6 = AttnCondenserBlock("AttnCondBlock6", 15, 30, 18, 36, 15)
        self.conv2 = nn.Conv2d(15, 38, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(38, num_classes)

    def __str__(self):
        return "TinySpeechX"

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinySpeechY(nn.Module):
    """
    TinySpeechY architecture.

    Architecture:
      - Input conv: 1 → 10 channels.
      - 6 AttnCondenserBlocks with lower channel counts.
      - Final conv: 10 → 6 channels.
      - Global average pooling and fully connected layer.
    """

    def __init__(self, num_classes):
        super(TinySpeechY, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1, padding=1)
        self.block1 = AttnCondenserBlock("AttnCondBlock1", 15, 30, 6, 12, 15)
        self.block2 = AttnCondenserBlock("AttnCondBlock2", 15, 30, 6, 12, 15)
        self.block3 = AttnCondenserBlock("AttnCondBlock3", 15, 30, 6, 12, 15)
        self.block4 = AttnCondenserBlock("AttnCondBlock4", 15, 30, 19, 38, 15)
        self.block5 = AttnCondenserBlock("AttnCondBlock5", 15, 30, 26, 52, 15)
        self.block6 = AttnCondenserBlock("AttnCondBlock6", 15, 30, 18, 36, 15)
        self.conv2 = nn.Conv2d(15, 6, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, num_classes)

    def __str__(self):
        return "TinySpeechY"

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinySpeechZ(nn.Module):
    """
    TinySpeechZ architecture.

    Architecture:
      - Input conv: 1 → 7 channels.
      - 6 AttnCondenserBlocks with lower channel numbers.
      - Final conv: 7 → 17 channels.
      - Global average pooling and fully connected layer.
    """

    def __init__(self, num_classes):
        super(TinySpeechZ, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)
        self.block1 = AttnCondenserBlock("AttnCondBlock1", 7, 14, 3, 6, 7)
        self.block2 = AttnCondenserBlock("AttnCondBlock2", 7, 14, 3, 6, 7)
        self.block3 = AttnCondenserBlock("AttnCondBlock3", 7, 14, 2, 4, 7)
        self.block4 = AttnCondenserBlock("AttnCondBlock4", 7, 14, 11, 22, 7)
        self.block5 = AttnCondenserBlock("AttnCondBlock5", 7, 14, 14, 28, 7)
        self.block6 = AttnCondenserBlock("AttnCondBlock6", 7, 14, 10, 20, 7)
        self.conv2 = nn.Conv2d(7, 17, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(17, num_classes)

    def __str__(self):
        return "TinySpeechZ"

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TinySpeechM(nn.Module):
    """
    TinySpeechM architecture.

    For TinySpeechM (designed with microcontroller constraints), batch normalization is omitted.

    Architecture:
      - Input conv: 1 → 9 channels.
      - 3 AttnCondenserBlocks (batch_norm=False).
      - Final conv: 9 → 40 channels.
      - Global average pooling and fully connected layer.
    """

    def __init__(self, num_classes):
        super(TinySpeechM, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, kernel_size=3, stride=1, padding=1)
        self.block1 = AttnCondenserBlock("AttnCondBlock1", 9, 18, 6, 12, 9, batch_norm=False)
        self.block2 = AttnCondenserBlock("AttnCondBlock2", 9, 18, 9, 18, 9, batch_norm=False)
        self.block3 = AttnCondenserBlock("AttnCondBlock3", 9, 18, 8, 16, 9, batch_norm=False)
        self.conv2 = nn.Conv2d(9, 40, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(40, num_classes)

    def __str__(self):
        return "TinySpeechM"

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Example: Instantiate TinySpeechX with 35 classes and print parameter count.
    model_x = TinySpeechX(num_classes=35)
    print("Total parameters in TinySpeechX:", count_parameters(model_x))
