import torch.nn.functional as F
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class SparseFCN(nn.Module):
    """ Sparse Fully Convolutional Network for 2D data with Residual Blocks
        Description:
        - 5 convolutional layers with residual blocks
        - Batch normalization after each convolutional layer
        - ReLU activation function
        - Dropout layers added for regularization
        - 1x1 convolutional layer to output the final prediction
    """
    def __init__(self, dropout_prob=0.5):
        super(SparseFCN, self).__init__()
        self.resblock1 = ResidualConvBlock(12, 32)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.resblock2 = ResidualConvBlock(32, 64)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.resblock3 = ResidualConvBlock(64, 128)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.resblock4 = ResidualConvBlock(128, 256)
        self.dropout4 = nn.Dropout(dropout_prob)
        self.resblock5 = ResidualConvBlock(256, 512)
        self.dropout5 = nn.Dropout(dropout_prob)
        self.conv_final = nn.Conv2d(512, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.resblock2(x)
        x = self.dropout2(x)
        x = self.resblock3(x)
        x = self.dropout3(x)
        x = self.resblock4(x)
        x = self.dropout4(x)
        x = self.resblock5(x)
        x = self.dropout5(x)
        x = self.conv_final(x)
        return x
    
class SimpleSparseFCN(nn.Module):
    """ Simple Sparse Fully Convolutional Network for 2D data with Residual Blocks
        Description:
        - 3 convolutional layers with residual blocks
        - Batch normalization after each convolutional layer
        - ReLU activation function
        - Dropout layers added for regularization
        - 1x1 convolutional layer to output the final prediction
    """
    def __init__(self, dropout_prob=0.5):
        super(SimpleSparseFCN, self).__init__()
        self.resblock1 = ResidualConvBlock(12, 32)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.resblock2 = ResidualConvBlock(32, 64)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.resblock3 = ResidualConvBlock(64, 128)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.conv_final = nn.Conv2d(128, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.resblock1(x)
        x = self.dropout1(x)
        x = self.resblock2(x)
        x = self.dropout2(x)
        x = self.resblock3(x)
        x = self.dropout3(x)
        x = self.conv_final(x)
        return x
############################################

class SimpleFCN(nn.Module):
    """ Simplified Fully Convolutional Network for 2D data
        Description:
        - 3 convolutional layers with 3x3 kernels
        - Batch normalization after each convolutional layer
        - ReLU activation function
        - Final 1x1 convolutional layer adapted for multi-class classification
    """
    def __init__(self, num_classes=10):
        super(SimpleFCN, self).__init__()
        self.conv1 = nn.Conv2d(20, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv_final = nn.Conv2d(64, num_classes, kernel_size=1)  # Output num_classes channels
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv_final(x)
        return x  # Output raw logits for use with CrossEntropyLoss

class ClassSparseFCN(nn.Module):
    """ Sparse Fully Convolutional Network for 2D data
        Description:
        - 5 convolutional layers with 3x3 kernels
        - Batch normalization after each convolutional layer
        - ReLU activation function
        - Final 1x1 convolutional layer adapted for multi-class classification

    """
    def __init__(self, num_classes):
        super(ClassSparseFCN, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_final = nn.Conv2d(512, num_classes, kernel_size=1)  # Output num_classes channels
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv_final(x)
        return x  # Output raw logits for use with CrossEntropyLoss

