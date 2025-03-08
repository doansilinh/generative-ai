import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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
    def __init__(self, block=ResidualBlock, all_connections=[2, 3, 4, 2, 2]):
        super().__init__()
        self.inputs = 16  # Reduced from 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Reduced from 32
            nn.BatchNorm2d(16),  # Reduced from 32
            nn.ReLU(),
        )  # 512x512
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x256

        self.layer0 = self.makeLayer(
            block, 32, all_connections[0], stride=1
        )  # Reduced from 64, connections = 2, shape: 256x256
        self.layer1 = self.makeLayer(
            block, 64, all_connections[1], stride=2
        )  # Reduced from 128, connections = 3, shape: 128x128
        self.layer2 = self.makeLayer(
            block, 128, all_connections[2], stride=2
        )  # Reduced from 256, connections = 4, shape: 64x64
        self.layer3 = self.makeLayer(
            block, 256, all_connections[3], stride=2
        )  # Reduced from 512, connections = 2, shape: 32x32
        self.layer4 = self.makeLayer(
            block, 512, all_connections[4], stride=2
        )  # Reduced from 1024, connections = 2, shape: 16x16

        # Removed layer5 completely (was 2048 features)

        self.avgpool = nn.AvgPool2d(
            16, stride=1
        )  # Changed from 8 to 16 since we have one less layer
        self.fc = nn.Linear(512, 1)  # Changed from 2048 to 512

        # Enable memory efficient forward pass
        self.use_checkpointing = True

    def makeLayer(self, block, outputs, connections, stride=1):
        downsample = None
        if stride != 1 or self.inputs != outputs:
            downsample = nn.Sequential(
                nn.Conv2d(self.inputs, outputs, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outputs),
            )
        layers = []
        layers.append(block(self.inputs, outputs, stride, downsample))
        self.inputs = outputs
        for i in range(1, connections):
            layers.append(block(self.inputs, outputs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        # Use gradient checkpointing to save memory during backward pass
        if self.use_checkpointing and self.training:
            x = checkpoint(self.layer0, x, use_reentrant=False)
            x = checkpoint(self.layer1, x, use_reentrant=False)
            x = checkpoint(self.layer2, x, use_reentrant=False)
            x = checkpoint(self.layer3, x, use_reentrant=False)
            x = checkpoint(self.layer4, x, use_reentrant=False)
        else:
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(-1, 512)  # Changed from 2048 to 512
        x = self.fc(x).flatten()
        return torch.sigmoid(x)
