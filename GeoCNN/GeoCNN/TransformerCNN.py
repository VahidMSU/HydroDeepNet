import torch
import torch.nn as nn
import math
from GeoCNN.ResBlocks import ResidualBlock
import torch
import torch.nn as nn
from GeoCNN.Positional_encoding import FourierFeaturePositionalEncoding2D, LearnablePositionalEncoding2D
from GeoCNN.TransformerBlock import TransformerEncoder
import torch
import torch.nn as nn
import math



class TransformerCNN(nn.Module):
    def __init__(self, num_channel, d_model=256, num_heads=4, forward_expansion=4, num_layers=2, patch_size=16, height=320, width=352, verbose=False):
        super(TransformerCNN, self).__init__()

        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.verbose = verbose
        
        # Residual Convolutional Blocks
        self.residual_blocks = nn.Sequential(
            ResidualBlock(num_channel, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256)
        )
        
        # Convolution to adjust channels to match transformer input (d_model)
        self.flatten_conv = nn.Conv2d(256, d_model, kernel_size=1)

        # Compute total number of patches
        self.num_patches_height = height // patch_size
        self.num_patches_width = width // patch_size
        self.num_patches = self.num_patches_height * self.num_patches_width

        # Positional Encoding
        self.pos_encoding = FourierFeaturePositionalEncoding2D(d_model, self.num_patches, self.height, self.width, verbose=self.verbose)
        
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(num_layers, d_model, num_heads, forward_expansion)
        
        # Final regression layer
        self.conv_final = nn.Conv2d(d_model, 1, kernel_size=1)

    def forward(self, x):
        # Apply CNN blocks
        x = self.residual_blocks(x)

        # Get dimensions
        batch_size, channels, height, width = x.shape
        assert channels == 256, f"Expected 256 channels after residual blocks, got {channels}"

        # Ensure height and width are divisible by patch size
        assert height % self.patch_size == 0, "Height not divisible by patch size"
        assert width % self.patch_size == 0, "Width not divisible by patch size"

        # Convert the number of channels to d_model before applying transformer
        x = self.flatten_conv(x)
        assert x.shape[1] == 256, f"Expected 256 channels after flatten_conv, got {x.shape[1]}"

        # Step 1: Split the input into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        if self.verbose:
            print(f"patches shape before permute: {patches.shape}")

        # Reshape the patches for transformer input [batch_size * num_patches, d_model, patch_size * patch_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [batch_size, num_patches_height, num_patches_width, channels, patch_size, patch_size]
        patches = patches.view(batch_size * self.num_patches, 256, self.patch_size * self.patch_size)
        if self.verbose:
            print(f"patches shape after permute and reshape: {patches.shape}")

        # Step 2: Apply the positional encoding
        patches_encoded = self.pos_encoding(patches)

        # Step 3: Apply the Transformer Encoder
        transformer_output = self.transformer_encoder(patches_encoded)
        if self.verbose:
            print(f"transformer_output shape: {transformer_output.shape}")  # [batch_size * num_patches, 256, patch_size * patch_size]

        # Step 4: Reshape back to the spatial dimensions
        # Reshape transformer output back to [batch_size * num_patches, 256, patch_size, patch_size]
        transformer_output = transformer_output.view(batch_size * self.num_patches, 256, self.patch_size, self.patch_size)
        if self.verbose:
            print(f"transformer_output reshaped to patches: {transformer_output.shape}")

        # Step 5: Reshape to [batch_size, 256, num_patches_height * patch_size, num_patches_width * patch_size]
        transformer_output = transformer_output.view(batch_size, self.num_patches_height, self.num_patches_width, 256, self.patch_size, self.patch_size)
        transformer_output = transformer_output.permute(0, 3, 1, 4, 2, 5).contiguous()  # [batch_size, 256, num_patches_height, patch_size, num_patches_width, patch_size]
        transformer_output = transformer_output.view(batch_size, 256, height, width)  # [batch_size, 256, height, width]
        if self.verbose:
            print(f"transformer_output final reshaped: {transformer_output.shape}")  # Expecting [batch_size, 256, height, width]

        # Step 6: Final convolution to generate the output
        output = self.conv_final(transformer_output)
        ## relu to make it possitive
        output = torch.relu(output)
        return output


