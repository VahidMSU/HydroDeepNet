import torch
import torch.nn as nn

# Convolutional Block with BatchNorm and ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# Downsampling block with MaxPool
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # return conv features and pooled output for skip connections

# Upsampling block with ConvTranspose2d for upsampling and ConvBlock for processing
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)  # concatenate doubled channels after skip connection

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concatenate along the channel axis
        x = self.conv(x)
        return x

# Advanced CNN for Continuous Segmentation (based on U-Net style architecture)
class AdvancedCNN(nn.Module):
    def __init__(self, num_channels, num_classes=1):
        super(AdvancedCNN, self).__init__()

        # Encoder (Downsampling)
        self.down1 = DownBlock(num_channels, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck (lowest resolution part of the U-Net)
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (Upsampling)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # Final convolution to get the desired output channels (e.g., 1 for regression)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Downsampling path
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling path with skip connections
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Final output
        x = self.final_conv(x)

        x = self.relu(x)  # apply ReLU to the output
        return x
