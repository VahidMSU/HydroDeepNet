from torch import nn
from torchvision.models import resnet152


class ModifiedResNet(nn.Module):
    def __init__(self, output_size, num_input_channels): 
        super(ModifiedResNet, self).__init__()
        self.output_size = output_size  # Desired output size (height, width)
        
        # Load the ResNet152 model
        self.model = resnet152(weights=None)

        # Modify the first convolutional layer to accept num_input_channels
        self.model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=2, stride=1, padding=3, bias=False)

        # Remove the global average pooling layer
        self.model.avgpool = nn.Identity()

        # Remove the fully connected layer
        self.model.fc = nn.Identity()

        # Add a transposed convolutional block to recover spatial dimensions
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)  # Upsample by 2x
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)   # Upsample by 2x
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1)    # Upsample by 1x (finer resolution)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1)    # Upsample by 1x (finer resolution)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)     # Upsample by 2x

        # Use bilinear interpolation to refine spatial resolution further
        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final convolution to get the desired output size (1 channel for regression)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Pass through ResNet layers (no avgpool or fc)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # Apply transposed convolutions to recover spatial dimensions
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        # Apply bilinear interpolation to increase resolution
        x = self.interpolate(x)

        # Final output layer to produce 1-channel output
        x = self.final_conv(x)

        return x[:, :, :self.output_size[0], :self.output_size[1]]  # Crop to desired size
