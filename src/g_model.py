import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DModel(nn.Module):
    def __init__(
        self, block=ResidualBlock, all_connections=[2, 3, 3, 2]
    ):  # Reduced layers
        super().__init__()
        self.inputs = 16  # Reduced from 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer0 = self.makeLayer(block, 32, all_connections[0], stride=1)
        self.layer1 = self.makeLayer(block, 64, all_connections[1], stride=2)
        self.layer2 = self.makeLayer(block, 128, all_connections[2], stride=2)
        self.layer3 = self.makeLayer(block, 256, all_connections[3], stride=2)

        self.avgpool = nn.AvgPool2d(16, stride=1)
        self.fc = nn.Linear(256, 1)

    def makeLayer(self, block, outputs, connections, stride=1):
        downsample = None
        if stride != 1 or self.inputs != outputs:
            downsample = nn.Sequential(
                nn.Conv2d(self.inputs, outputs, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outputs),
            )
        layers = [block(self.inputs, outputs, stride, downsample)]
        self.inputs = outputs
        for _ in range(1, connections):
            layers.append(block(self.inputs, outputs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(-1, 256)
        x = self.fc(x).flatten()
        return torch.sigmoid(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class GModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = DoubleConv(3, 16)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = DoubleConv(16, 32)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = DoubleConv(32, 64)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = DoubleConv(64, 128)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_5 = DoubleConv(128, 256)
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Removed last two encoder layers

        self.upconv_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_6 = DoubleConv(256, 128)

        self.upconv_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_7 = DoubleConv(128, 64)

        self.upconv_3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_8 = DoubleConv(64, 32)

        self.upconv_4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_9 = DoubleConv(32, 16)

        self.output = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, batch):
        conv_1_out = self.conv_1(batch)
        pool_1_out = self.pool_1(conv_1_out)

        conv_2_out = self.conv_2(pool_1_out)
        pool_2_out = self.pool_2(conv_2_out)

        conv_3_out = self.conv_3(pool_2_out)
        pool_3_out = self.pool_3(conv_3_out)

        conv_4_out = self.conv_4(pool_3_out)
        pool_4_out = self.pool_4(conv_4_out)

        conv_5_out = self.conv_5(pool_4_out)

        upconv_1_out = self.upconv_1(conv_5_out)
        conv_6_out = self.conv_6(torch.cat([upconv_1_out, conv_4_out], dim=1))

        upconv_2_out = self.upconv_2(conv_6_out)
        conv_7_out = self.conv_7(torch.cat([upconv_2_out, conv_3_out], dim=1))

        upconv_3_out = self.upconv_3(conv_7_out)
        conv_8_out = self.conv_8(torch.cat([upconv_3_out, conv_2_out], dim=1))

        upconv_4_out = self.upconv_4(conv_8_out)
        conv_9_out = self.conv_9(torch.cat([upconv_4_out, conv_1_out], dim=1))

        return torch.sigmoid(self.output(conv_9_out))
