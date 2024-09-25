import torch
import torch.nn as nn
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.2, use_gelu=True, weight_init_gain=1.0):
        super(ResidualBlock, self).__init__()
        
        # Convolutional layers with BatchNorm and Dropout for regularization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)
        
        # Use GELU (Gaussian Error Linear Unit) or PReLU based on configuration
        self.activation = nn.GELU() if use_gelu else nn.PReLU()
        
        # Residual connection, adjusted if input and output channels differ
        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Initialize weights for stability
        self._init_weights(weight_init_gain)

    def forward(self, x, verbose=False):
        if verbose:
            print(f"Input shape: {x.shape}")
        
        # Residual connection
        residual = self.residual_connection(x)

        # First convolution + batchnorm + activation + dropout
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout1(out)

        # Second convolution + batchnorm + dropout
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)

        # Add the residual (skip connection)
        out += residual

        # Final activation
        out = self.activation(out)

        if verbose:
            print(f"Output shape: {out.shape}")

        return out

    def _init_weights(self, gain):
        """ Weight initialization for stability and control over learning dynamics """
        nn.init.xavier_uniform_(self.conv1.weight, gain=gain)
        nn.init.xavier_uniform_(self.conv2.weight, gain=gain)
        if isinstance(self.residual_connection, nn.Conv2d):
            nn.init.xavier_uniform_(self.residual_connection.weight, gain=gain)


