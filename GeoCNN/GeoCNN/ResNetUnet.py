import torch
import torch.nn as nn
from torchvision.models import resnet152
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

class ModifiedResNetUNet(nn.Module):
    def __init__(self, num_input_channels=9, output_size=(320, 352)):
        super(ModifiedResNetUNet, self).__init__()

        # Load the ResNet152 model
        self.encoder = resnet18(weights=None)

        # Modify the first convolutional layer to accept num_input_channels
        self.encoder.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=2, stride=1, padding=3, bias=False)

        # Remove the global average pooling and fully connected layers
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        # Decoder (mirroring the encoder)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final convolution to get the desired output size (1 channel for regression)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x1)  # Output from the first ResNet block
        x3 = self.encoder.layer2(x2)  # Output from the second ResNet block
        x4 = self.encoder.layer3(x3)  # Output from the third ResNet block
        x5 = self.encoder.layer4(x4)  # Output from the fourth ResNet block

        # Decoder with cropping for skip connections to handle dimension mismatch
        d4 = self.crop_and_add(self.deconv1(x5), x4)  # Skip connection from encoder's third block
        d3 = self.crop_and_add(self.deconv2(d4), x3)  # Skip connection from encoder's second block
        d2 = self.crop_and_add(self.deconv3(d3), x2)  # Skip connection from encoder's first block
        d1 = self.crop_and_add(self.deconv4(d2), x1)  # Skip connection from encoder's input

        d0 = self.deconv5(d1)  # Final upsampling

        # Final output layer to produce 1-channel output
        output = self.final_conv(d0)

        return output[:, :, :320, :352]  # Crop to the original size if necessary

    def crop_and_add(self, upsampled, skip):
        """
        Crop the upsampled tensor to match the skip connection tensor size.
        """
        _, _, h_skip, w_skip = skip.size()
        _, _, h_up, w_up = upsampled.size()

        # Calculate the cropping needed
        h_crop = (h_up - h_skip) // 2
        w_crop = (w_up - w_skip) // 2

        # Crop the upsampled tensor
        upsampled_cropped = upsampled[:, :, h_crop:h_crop + h_skip, w_crop:w_crop + w_skip]

        # Add the cropped upsampled tensor with the skip connection
        return upsampled_cropped + skip


