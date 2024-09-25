import torch
import torch.nn as nn

# Simple Convolutional Block (Without Residuals)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        # Convolution layers with BatchNorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First conv block
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Second conv block
        out = self.relu(self.bn2(self.conv2(out)))

        return out

# Simple CNN Model for Regression (No Residual Connections)
class SimpleCNN(nn.Module):
    def __init__(self, num_channel, num_classes=1):
        super(SimpleCNN, self).__init__()

        # Stack of convolutional blocks
        self.conv_blocks = nn.Sequential(
            ConvBlock(num_channel, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        )

        # Final convolution to down-sample to the desired number of output classes (1 for regression)
        self.conv_final = nn.Conv2d(256, num_classes, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the convolutional blocks
        x = self.conv_blocks(x)

        # Apply the final convolution to get the output
        x = self.conv_final(x)

        x = self.relu(x)

        return x


