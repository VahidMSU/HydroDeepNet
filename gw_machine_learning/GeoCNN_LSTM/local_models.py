import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out

class FullyConvModel(nn.Module):
    def __init__(self, number_of_channels=15, height=553, width=367):
        super(FullyConvModel, self).__init__()
        self.conv1 = SimpleConvBlock(number_of_channels, 64)
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 64)
        self.res_block3 = ResidualBlock(64, 64)
        self.res_block4 = ResidualBlock(64, 64)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output a single channel
        self.height = height
        self.width = width

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.conv_final(x)
        
        x = x.view(batch_size, seq_len, 1, self.height, self.width)
        
        return x




class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(SimpleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        output = self.fc(attn_output)
        return output

class CNN_LSTM(nn.Module):
    def __init__(self, dropout_prob=0.5, hidden_dim=128, num_layers=7, number_of_channels=12, height=553, width=367):
        super(CNN_LSTM, self).__init__()
        self.conv1 = SimpleConvBlock(number_of_channels, 64)
        self.res_block1 = ResidualConvBlock(64, 64)
        self.res_block2 = ResidualConvBlock(64, 64)
        self.res_block3 = ResidualConvBlock(64, 64)
        self.res_block4 = ResidualConvBlock(64, 64)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)
        self.relu_final = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.attention = AttentionBlock(hidden_dim, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, height * width)
        self.dropout = nn.Dropout(dropout_prob)
        self.height = height
        self.width = width

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.conv1(c_in)
        c_out = self.res_block1(c_out)
        c_out = self.res_block2(c_out)
        c_out = self.res_block3(c_out)
        c_out = self.res_block4(c_out)
        c_out = self.conv_final(c_out)
        c_out = self.relu_final(c_out)
        
        c_out = self.global_avg_pool(c_out)
        c_out = c_out.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(c_out)
        lstm_out = self.attention(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        output = F.relu(output)
        
        output = output.view(batch_size * seq_len, 1, self.height, self.width)
        output = F.interpolate(output, size=(self.height, self.width), mode='bilinear', align_corners=False)
        output = output.view(batch_size, seq_len, 1, self.height, self.width)
        
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4, dropout_prob=0.5, number_of_channels=12, height=553, width=367):
        super(TransformerModel, self).__init__()
        self.input_dim = number_of_channels * height * width
        self.height = height
        self.width = width
        self.flatten_dim = number_of_channels * height * width

        self.embedding = nn.Linear(self.flatten_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead=8, dropout=dropout_prob, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, height * width)  # Output one channel
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._initialize_weights()

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, -1)  # Flatten the spatial dimensions

        x = self.embedding(x)
        x = self.dropout(x)

        x = x.view(batch_size, seq_len, -1)  # Reshape for transformer

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)

        output = self.fc(x)
        output = F.relu(output)

        output = output.view(batch_size, seq_len, 1, self.height, self.width)  # Correctly reshape back to original dimensions

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

# Example usage
if __name__ == '__main__':
    number_of_channels = 12
    height = 553
    width = 367
    input_dim = number_of_channels * height * width
    model = TransformerModel(hidden_dim=128, num_layers=4, dropout_prob=0.5, number_of_channels=number_of_channels, height=height, width=width)
    x = torch.randn(1, 15, number_of_channels, height, width)
    output = model(x)
    print(output.size())  # Should print (1, 15, 1, 553, 367)
    print(f"Output contains negative values: {(output < 0).any()}")
