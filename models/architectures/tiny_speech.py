from models.architectures.attn_condenser import *


class TinySpeechX(nn.Module):
    def __init__(self, num_classes):
        super(TinySpeechX, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_Condenser_Block("AttnCondBlock1", in_channels=15, mid_channels_0=30, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=15)
        self.block2 = Attn_Condenser_Block("AttnCondBlock2", in_channels=15, mid_channels_0=30, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=15)
        self.block3 = Attn_Condenser_Block("AttnCondBlock3", in_channels=15, mid_channels_0=30, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=15)
        self.block4 = Attn_Condenser_Block("AttnCondBlock4", in_channels=15, mid_channels_0=30, out_channels_0=18,
                                    mid_channels_1=36, out_channels_1=15)
        self.block5 = Attn_Condenser_Block("AttnCondBlock5", in_channels=15, mid_channels_0=30, out_channels_0=27,
                                    mid_channels_1=54, out_channels_1=15)
        self.block6 = Attn_Condenser_Block("AttnCondBlock6", in_channels=15, mid_channels_0=30, out_channels_0=18,
                                    mid_channels_1=36, out_channels_1=15)

        self.conv2 = nn.Conv2d(in_channels=15, out_channels=38, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(38, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def __str__(self):
        return "TinySpeechX"

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


class TinySpeechY(nn.Module):
    def __init__(self, num_classes):
        super(TinySpeechY, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_Condenser_Block("AttnCondBlock1", in_channels=10, mid_channels_0=20, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=10)
        self.block2 = Attn_Condenser_Block("AttnCondBlock2", in_channels=10, mid_channels_0=20, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=10)
        self.block3 = Attn_Condenser_Block("AttnCondBlock3", in_channels=10, mid_channels_0=20, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=10)
        self.block4 = Attn_Condenser_Block("AttnCondBlock4", in_channels=10, mid_channels_0=20, out_channels_0=18,
                                    mid_channels_1=36, out_channels_1=10)
        self.block5 = Attn_Condenser_Block("AttnCondBlock5", in_channels=10, mid_channels_0=20, out_channels_0=26,
                                    mid_channels_1=52, out_channels_1=10)
        self.block6 = Attn_Condenser_Block("AttnCondBlock6", in_channels=10, mid_channels_0=20, out_channels_0=18,
                                    mid_channels_1=36, out_channels_1=10)

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def __str__(self):
        return "TinySpeechY"

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


class TinySpeechZ(nn.Module):
    def __init__(self, num_classes):
        super(TinySpeechZ, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_Condenser_Block("AttnCondBlock1", in_channels=7, mid_channels_0=14, out_channels_0=3,
                                    mid_channels_1=6, out_channels_1=7)
        self.block2 = Attn_Condenser_Block("AttnCondBlock2", in_channels=7, mid_channels_0=14, out_channels_0=3,
                                    mid_channels_1=6, out_channels_1=7)
        self.block3 = Attn_Condenser_Block("AttnCondBlock3", in_channels=7, mid_channels_0=14, out_channels_0=2,
                                    mid_channels_1=4, out_channels_1=7)
        self.block4 = Attn_Condenser_Block("AttnCondBlock4", in_channels=7, mid_channels_0=14, out_channels_0=11,
                                    mid_channels_1=22, out_channels_1=7)
        self.block5 = Attn_Condenser_Block("AttnCondBlock5", in_channels=7, mid_channels_0=14, out_channels_0=14,
                                    mid_channels_1=28, out_channels_1=7)
        self.block6 = Attn_Condenser_Block("AttnCondBlock6", in_channels=7, mid_channels_0=14, out_channels_0=10,
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


class TinySpeechM(nn.Module):
    def __init__(self, num_classes):
        super(TinySpeechM, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, kernel_size=3, stride=1, padding=1)

        self.block1 = Attn_Condenser_Block("AttnCondBlock1", in_channels=9, mid_channels_0=18, out_channels_0=6,
                                    mid_channels_1=12, out_channels_1=9, batch_norm=False)
        self.block2 = Attn_Condenser_Block("AttnCondBlock2", in_channels=9, mid_channels_0=18, out_channels_0=9,
                                    mid_channels_1=18, out_channels_1=9, batch_norm=False)
        self.block3 = Attn_Condenser_Block("AttnCondBlock3", in_channels=9, mid_channels_0=18, out_channels_0=8,
                                    mid_channels_1=16, out_channels_1=9, batch_norm=False)

        self.conv2 = nn.Conv2d(in_channels=9, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(40, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def __str__(self):
        return "TinySpeechM"

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x