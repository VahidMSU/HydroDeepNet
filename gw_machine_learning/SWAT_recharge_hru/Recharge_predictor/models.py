import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Add an extra dimension for batch size compatibility
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_weights = self.softmax(torch.bmm(queries, keys.transpose(1, 2)) / (x.size(-1) ** 0.5))
        out = torch.bmm(attention_weights, values)
        
        # Remove the extra dimension
        out = out.squeeze(1)
        
        return out + x.squeeze(1)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.swish = nn.SiLU()
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(0.3)

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
        out = self.swish(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.swish(out)

        return out

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc_initial = nn.Linear(input_dim, 512)
        self.bn_initial = nn.BatchNorm1d(512)
        self.swish = nn.SiLU()
        self.resblock1 = ResidualBlock(512, 512)
        self.resblock2 = ResidualBlock(512, 256)
        self.resblock3 = ResidualBlock(256, 256)
        self.resblock4 = ResidualBlock(256, 128)
        self.attention1 = AttentionBlock(128)
        self.resblock5 = ResidualBlock(128, 128)
        self.resblock6 = ResidualBlock(128, 64)
        self.attention2 = AttentionBlock(64)
        self.resblock7 = ResidualBlock(64, 64)
        self.resblock8 = ResidualBlock(64, 32)
        self.fc_final = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc_initial(x)
        x = self.bn_initial(x)
        x = self.swish(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.attention1(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.attention2(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.fc_final(x)
        return x

