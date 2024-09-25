import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import math
class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, num_patches):
        super(LearnablePositionalEncoding2D, self).__init__()
        # Create learnable positional encoding for each patch
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

    def forward(self, x, verbose=False):
        if verbose:
            print(f"x shape: {x.shape}")
            print(f"pos_embedding shape: {self.pos_embedding.shape}")
        batch_size, num_patches, d_model = x.shape[:3]
        if verbose:
            print(f"batch_size: {batch_size}, num_patches: {num_patches}, d_model: {d_model}")
        # Apply positional encoding
        return x + self.pos_embedding[:, :num_patches, :].to(x.device)
    

class FourierFeaturePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width, scale=10.0, verbose=False):
        """
        Positional encoding using Fourier features.
        Args:
            d_model: The dimensionality of the model (embedding dimension).
            height: Number of patches along the height.
            width: Number of patches along the width.
            scale: Controls the range of frequencies for the Fourier features.
        """
        super(FourierFeaturePositionalEncoding2D, self).__init__()
        
        # The number of features should be half of d_model to account for sin and cos
        assert d_model % 2 == 0, "d_model must be divisible by 2."
        
        self.height = height
        self.width = width
        self.d_model = d_model
        self.verbose = verbose

        # Create a grid of patch positions and apply scaling for Fourier encoding
        self.pos_embedding = self.generate_fourier_features(height, width, d_model, scale)

    def generate_fourier_features(self, height, width, d_model, scale):
        """
        Generate Fourier features based on spatial positions.
        Args:
            height: Number of patches in height.
            width: Number of patches in width.
            d_model: Dimensionality of the positional encoding.
            scale: Scale factor for frequencies.
        Returns:
            Fourier encoded positional embeddings (height * width, d_model).
        """
        # Create a meshgrid of patch positions (normalized between 0 and 1)
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width))
        if self.verbose:
            print(f"grid_y shape: {grid_y.shape}, grid_x shape: {grid_x.shape}")
            
        # Stack the grids along the last dimension to form a tensor of shape (height, width, 2)
        grid = torch.stack([grid_y, grid_x], dim=-1).view(height * width, 2)  # Shape: (height * width, 2)
        if self.verbose:
            print(f"grid shape: {grid.shape}")

        # Generate frequency bands (logarithmic scaling)
        freqs = torch.linspace(1.0, scale, d_model // 4)  # Shape: (d_model // 4,)
        if self.verbose:
            print(f"freqs shape: {freqs.shape}")

        # Multiply the frequencies with both x and y coordinates
        pos_x = grid[:, 0:1] * freqs  # Shape: (num_patches, d_model // 4)
        pos_y = grid[:, 1:2] * freqs  # Shape: (num_patches, d_model // 4)

        # Apply sin and cos to the frequencies
        pos_emb = torch.cat([torch.sin(pos_x), torch.cos(pos_x), torch.sin(pos_y), torch.cos(pos_y)], dim=-1)
        pos_emb = pos_emb.view(height * width, d_model)  # Shape: (num_patches, d_model)

        if self.verbose:
            print(f"pos_emb shape: {pos_emb.shape}")

        return nn.Parameter(pos_emb)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model).
        Returns:
            Positional encoded tensor with the same shape as input.
        """
        batch_size, num_patches, d_model = x.shape
        if self.verbose:
            print(f"x shape: {x.shape}")
            print(f"pos_embedding shape: {self.pos_embedding.shape}")

        # Ensure that the positional embedding matches the number of patches
        pos_emb = self.pos_embedding[:num_patches, :].unsqueeze(0).repeat(batch_size, 1, 1)
        
        return x + pos_emb.to(x.device)
