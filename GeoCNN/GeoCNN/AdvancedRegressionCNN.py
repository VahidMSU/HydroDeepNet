import torch
import torch.nn as nn

# Convolutional Block with BatchNorm, SE block, and ReLU
class ConvBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockWithSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.se = SqueezeExcitation(out_channels, out_channels // 16)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        return x

# Squeeze-and-Excitation Block for Channel Attention
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SqueezeExcitation, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale

# Downsampling block with MaxPool and SE block
class DownBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlockWithSE, self).__init__()
        self.conv = ConvBlockWithSE(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # return conv features and pooled output for skip connections

# Upsampling block with ConvTranspose2d for upsampling and ConvBlockWithSE for processing
class UpBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlockWithSE, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlockWithSE(in_channels, out_channels)  # concatenate doubled channels after skip connection

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concatenate along the channel axis
        x = self.conv(x)
        return x

# Improved CNN for Continuous Regression (based on U-Net style architecture)
class AdvancedRegressorCNN(nn.Module):
    def __init__(self, num_channels, num_classes=1):
        super(AdvancedRegressorCNN, self).__init__()

        # Encoder (Downsampling)
        self.down1 = DownBlockWithSE(num_channels, 64)
        self.down2 = DownBlockWithSE(64, 128)
        self.down3 = DownBlockWithSE(128, 256)
        self.down4 = DownBlockWithSE(256, 512)

        # Bottleneck (lowest resolution part of the U-Net)
        self.bottleneck = ConvBlockWithSE(512, 1024)

        # Decoder (Upsampling)
        self.up1 = UpBlockWithSE(1024, 512)
        self.up2 = UpBlockWithSE(512, 256)
        self.up3 = UpBlockWithSE(256, 128)
        self.up4 = UpBlockWithSE(128, 64)

        # Final convolution to get the desired output channels (1 for regression)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

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

        # Final output (without ReLU, as this is a regression task)
        x = self.final_conv(x)

        return x
