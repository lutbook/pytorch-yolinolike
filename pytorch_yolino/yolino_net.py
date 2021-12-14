import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x, maxpool=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if maxpool:
            x = self.pool(x)

        return x

class net(nn.Module):
    def __init__(self, p):
        super(net, self).__init__()
        
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        self.conv5 = ConvBlock(256, 512)
        self.conv6 = ConvBlock(512, 1024)

        self.up = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2, padding=0)
        self.conv_3_1 = nn.Conv2d(1536, 1024, kernel_size=3, padding=1)
        self.conv_1_2 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.conv_3_3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv_1_4 = nn.Conv2d(1024, p * 6, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x, maxpool=False)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        skip = x
        x = self.conv6(x)

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_3_1(x)
        x = self.conv_1_2(x)
        x = self.conv_3_3(x)
        x = self.conv_1_4(x)

        return x