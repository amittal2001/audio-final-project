from models.architectures.helpers import *


class TinySpeechZ(nn.Module):
    def __init__(self, num_classes):
        super(TinySpeechZ, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_BN_Block("AttnCondBlock1", in_channels=7, mid_channels_0=14, out_channels_0=3,
                                    mid_channels_1=6, out_channels_1=7)
        self.block2 = Attn_BN_Block("AttnCondBlock2", in_channels=7, mid_channels_0=14, out_channels_0=3,
                                    mid_channels_1=6, out_channels_1=7)
        self.block3 = Attn_BN_Block("AttnCondBlock3", in_channels=7, mid_channels_0=14, out_channels_0=2,
                                    mid_channels_1=4, out_channels_1=7)
        self.block4 = Attn_BN_Block("AttnCondBlock4", in_channels=7, mid_channels_0=14, out_channels_0=11,
                                    mid_channels_1=22, out_channels_1=7)
        self.block5 = Attn_BN_Block("AttnCondBlock5", in_channels=7, mid_channels_0=14, out_channels_0=14,
                                    mid_channels_1=28, out_channels_1=7)
        self.block6 = Attn_BN_Block("AttnCondBlock6", in_channels=7, mid_channels_0=14, out_channels_0=10,
                                    mid_channels_1=20, out_channels_1=7)

        self.conv2 = nn.Conv2d(in_channels=7, out_channels=17, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(17, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def __str__(self):
        return "TinySpeechZ"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
