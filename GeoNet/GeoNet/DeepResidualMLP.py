import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualFullyConnectedBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualFullyConnectedBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.1)

        if input_dim != output_dim:
            self.downsample = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out

class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super(DeepResidualMLP, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.resblock1 = ResidualFullyConnectedBlock(512, 512)
        self.resblock2 = ResidualFullyConnectedBlock(512, 256)
        self.resblock3 = ResidualFullyConnectedBlock(256, 128)
        self.resblock4 = ResidualFullyConnectedBlock(128, 64)
        self.resblock5 = ResidualFullyConnectedBlock(64, 32)
        self.resblock6 = ResidualFullyConnectedBlock(32, 16)

        self.final = nn.Sequential(
            nn.Linear(16, 8),
            nn.GELU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.final(x)
        return x
