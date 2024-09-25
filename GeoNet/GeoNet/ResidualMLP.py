import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLinearBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out

class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super(ResidualMLP, self).__init__()
        self.initial = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.resblock1 = ResidualLinearBlock(128, 64)
        self.resblock2 = ResidualLinearBlock(64, 32)

        self.final = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.final(x)
        return x
