import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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
        self.conv_1 = DoubleConv(3, 16)  # Reduced from 32, 512x512
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x256

        self.conv_2 = DoubleConv(16, 32)  # Reduced from 64, 256x256
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x128

        self.conv_3 = DoubleConv(32, 64)  # Reduced from 128, 128x128
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64

        self.conv_4 = DoubleConv(64, 128)  # Reduced from 256, 64x64
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32

        self.conv_5 = DoubleConv(128, 256)  # Reduced from 512, 32x32
        self.pool_5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16

        self.conv_6 = DoubleConv(256, 512)  # Reduced from 1024, 16x16
        self.pool_6 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8

        self.conv_7 = DoubleConv(512, 1024)  # Reduced from 2048, 8x8

        # DECODER
        self.upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16x16
        self.conv_8 = DoubleConv(
            1024, 512
        )  # Concatenated features: 512+512=1024, 16x16

        self.upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 32x32
        self.conv_9 = DoubleConv(512, 256)  # Concatenated features: 256+256=512, 32x32

        self.upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 64x64
        self.conv_10 = DoubleConv(256, 128)  # Concatenated features: 128+128=256, 64x64

        self.upconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128x128
        self.conv_11 = DoubleConv(128, 64)  # Concatenated features: 64+64=128, 128x128

        self.upconv_5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 256x256
        self.conv_12 = DoubleConv(64, 32)  # Concatenated features: 32+32=64, 256x256

        self.upconv_6 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 512x512
        self.conv_13 = DoubleConv(32, 16)  # Concatenated features: 16+16=32, 512x512

        self.output = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)  # 3x512x512

        # Enable memory efficient forward pass
        self.use_checkpointing = True

    def forward(self, batch):
        # Encoder path
        conv_1_out = self.conv_1(batch)
        pool_1_out = self.pool_1(conv_1_out)

        conv_2_out = self.conv_2(pool_1_out)
        pool_2_out = self.pool_2(conv_2_out)

        conv_3_out = self.conv_3(pool_2_out)
        pool_3_out = self.pool_3(conv_3_out)

        conv_4_out = self.conv_4(pool_3_out)
        pool_4_out = self.pool_4(conv_4_out)

        conv_5_out = self.conv_5(pool_4_out)
        pool_5_out = self.pool_5(conv_5_out)

        conv_6_out = self.conv_6(pool_5_out)
        pool_6_out = self.pool_6(conv_6_out)

        # Bottleneck
        conv_7_out = self.conv_7(pool_6_out)

        # Decoder path with gradient checkpointing for memory efficiency
        if self.use_checkpointing and self.training:
            # Using gradient checkpointing for decoder path
            upconv_1_out = self.upconv_1(conv_7_out)
            concat_1 = torch.cat([upconv_1_out, conv_6_out], dim=1)
            conv_8_out = checkpoint(self.conv_8, concat_1, use_reentrant=False)

            upconv_2_out = self.upconv_2(conv_8_out)
            concat_2 = torch.cat([upconv_2_out, conv_5_out], dim=1)
            conv_9_out = checkpoint(self.conv_9, concat_2, use_reentrant=False)

            upconv_3_out = self.upconv_3(conv_9_out)
            concat_3 = torch.cat([upconv_3_out, conv_4_out], dim=1)
            conv_10_out = checkpoint(self.conv_10, concat_3, use_reentrant=False)

            upconv_4_out = self.upconv_4(conv_10_out)
            concat_4 = torch.cat([upconv_4_out, conv_3_out], dim=1)
            conv_11_out = checkpoint(self.conv_11, concat_4, use_reentrant=False)

            upconv_5_out = self.upconv_5(conv_11_out)
            concat_5 = torch.cat([upconv_5_out, conv_2_out], dim=1)
            conv_12_out = checkpoint(self.conv_12, concat_5, use_reentrant=False)

            upconv_6_out = self.upconv_6(conv_12_out)
            concat_6 = torch.cat([upconv_6_out, conv_1_out], dim=1)
            conv_13_out = checkpoint(self.conv_13, concat_6, use_reentrant=False)
        else:
            # Standard forward pass without checkpointing
            conv_8_out = self.conv_8(
                torch.cat([self.upconv_1(conv_7_out), conv_6_out], dim=1)
            )
            conv_9_out = self.conv_9(
                torch.cat([self.upconv_2(conv_8_out), conv_5_out], dim=1)
            )
            conv_10_out = self.conv_10(
                torch.cat([self.upconv_3(conv_9_out), conv_4_out], dim=1)
            )
            conv_11_out = self.conv_11(
                torch.cat([self.upconv_4(conv_10_out), conv_3_out], dim=1)
            )
            conv_12_out = self.conv_12(
                torch.cat([self.upconv_5(conv_11_out), conv_2_out], dim=1)
            )
            conv_13_out = self.conv_13(
                torch.cat([self.upconv_6(conv_12_out), conv_1_out], dim=1)
            )

        output = self.output(conv_13_out)
        return torch.sigmoid(output)
