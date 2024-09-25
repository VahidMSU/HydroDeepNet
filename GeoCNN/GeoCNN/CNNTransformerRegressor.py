import torch
import torch.nn as nn
import math

# Positional Encoding to add spatial information for Transformers
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Convolutional Block with BatchNorm, SE block, and ReLU
class ConvBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockWithSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()
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
        self.relu = nn.PReLU()
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

# Transformer Encoder Module with Positional Encoding
class TransformerEncoderWithPositionalEncoding(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout, num_layers):
        super(TransformerEncoderWithPositionalEncoding, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=forward_expansion * embed_size, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_size)  # Add LayerNorm for stability
    
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = self.norm(x)  # Apply normalization
        return x
        
class CNNTransformerRegressor(nn.Module):
    def __init__(self, num_channels=4, embed_size=1024, num_heads=8, forward_expansion=4, num_layers=2, dropout=0.1, num_classes=1):
        super(CNNTransformerRegressor, self).__init__()

        # Encoder (Downsampling)
        self.down1 = DownBlockWithSE(num_channels, 64)
        self.down2 = DownBlockWithSE(64, 128)
        self.down3 = DownBlockWithSE(128, 256)
        self.down4 = DownBlockWithSE(256, 512)

        # Bottleneck (lowest resolution part of the U-Net)
        self.bottleneck = ConvBlockWithSE(512, embed_size)

        # Transformer Encoder with positional encoding
        self.transformer = TransformerEncoderWithPositionalEncoding(embed_size=embed_size, num_heads=num_heads, forward_expansion=forward_expansion, dropout=dropout, num_layers=num_layers)

        # Decoder (Upsampling)
        self.up1 = UpBlockWithSE(embed_size, 512)
        self.up2 = UpBlockWithSE(512, 256)
        self.up3 = UpBlockWithSE(256, 128)
        self.up4 = UpBlockWithSE(128, 64)

        # Final convolution to get the desired output channels (1 for regression)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape  # Expect 5D input: [batch_size, time_steps, channels, height, width]
        #print(f"Forward pass=================================")
        #print(f"Input shape: {x.shape}")
        #print(f"Batch size: {batch_size}")
        #print(f"Time steps: {time_steps}")
        #print(f"Channels: {channels}")
        #print(f"Height: {height}")
        ##print(f"Width: {width}")
        #print(f"=============================================")
        
        cnn_features = []
        skip_connections = []

        # Apply CNNs to each time step independently
        for t in range(time_steps):
         #   print(f"Time step: {t}")
            x_t = x[:, t, :, :, :]  # Extract input for time step t
            skip1, x_t = self.down1(x_t)
            skip2, x_t = self.down2(x_t)
            skip3, x_t = self.down3(x_t)
            skip4, x_t = self.down4(x_t)

            # Bottleneck
            x_t = self.bottleneck(x_t)
            cnn_features.append(x_t)
            skip_connections.append((skip1, skip2, skip3, skip4))

        # Stack features across time dimension: [batch_size, time_steps, embed_size, height, width]
        x = torch.stack(cnn_features, dim=1)
        #print(f"Stacked CNN features shape: {x.shape}")

        # Flatten spatial dimensions to use with Transformer
        batch_size, time_steps, embed_size, height, width = x.shape
        x = x.view(batch_size, time_steps, embed_size, height * width)
        #print(f"Reshaped CNN features shape: {x.shape}")

        # Apply global pooling over spatial dimensions
        x = x.mean(dim=-1)  # Reduce to [batch_size, time_steps, embed_size]
        #print(f"Global pooling shape: {x.shape}")

        # Apply Transformer for temporal modeling
        x = self.transformer(x)  # Shape: [batch_size, time_steps, embed_size]
        #print(f"Transformer output shape: {x.shape}")

        # Upsampling path (time dimension treated independently)
        up_features = []
        for t in range(time_steps):
            x_t = x[:, t, :].view(batch_size, embed_size, 1, 1).expand(batch_size, embed_size, height, width)
            skip1, skip2, skip3, skip4 = skip_connections[t]

            # Apply upsampling with skip connections
            x_t = self.up1(x_t, skip4)
            x_t = self.up2(x_t, skip3)
            x_t = self.up3(x_t, skip2)
            x_t = self.up4(x_t, skip1)
            up_features.append(x_t)

        # Stack the upsampled features back into the time dimension
        x = torch.stack(up_features, dim=1)
       # print(f"Stacked upsampled features shape: {x.shape}")

        # Merge the time dimension with the batch dimension
        x = x.view(batch_size * time_steps, x.size(2), x.size(3), x.size(4))
       # print(f"Reshaped for final conv: {x.shape}")

        # Final convolution to produce the output
        x = self.final_conv(x)  # Shape: [batch_size * time_steps, num_classes, height, width]
       # print(f"Final output shape: {x.shape}")

        # Reshape back to [batch_size, time_steps, num_classes, height, width] [1,3,64,64]
        x = x.view(batch_size, time_steps, x.size(1), x.size(2), x.size(3))
    
        return x
