import torch
from torch import nn
import warnings


"""
All parameter names and keyword arguments will be explicitly specified for better understanding
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Kernel size will be 3x3, stride of 1, and use padding=same to prevent dimensionality reduction

        # Adding BN to the output of the Conv blocks, as used by others
        # BN isn't used in the original UNET paper, which is most likely due to the Batch Normalization and UNET paper
        # are published within months of each other, so original authors probably didn't have time to thoroughly test it

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),

            # For second convolution, the number of filters doesn't change
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
        )

    def forward(self, x):
        return self.seq(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Padding of 0 is fine where images are all divisible by kernel size

    def forward(self, x):
        skip = self.conv_block(x)  # This will be the saved skip connected used in Decoder portion
        output = self.max_pool(skip)  # Downscaled image after max pooling
        return skip, output


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Up-sampling by 2, so kernel_size of 2 and stride of 2, padding wouldn't be needed as our image is multiple of 2s
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, skip):
        # First, up-sample the given image
        x = self.up_sample(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate the skip connection via channels dimension (1)
        x = self.conv_block(x)

        return x


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # First the encoder portion (contractive path)
        self.enc1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.enc2 = EncoderBlock(in_channels=64, out_channels=128)
        self.enc3 = EncoderBlock(in_channels=128, out_channels=256)
        self.enc4 = EncoderBlock(in_channels=256, out_channels=512)

        # Now the bottleneck portion
        self.bottle_neck = ConvBlock(in_channels=512, out_channels=1024)

        # Finally, the decoder portion (expansive path)
        self.dec1 = DecoderBlock(in_channels=1024, out_channels=512)
        self.dec2 = DecoderBlock(in_channels=512, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, out_channels=128)
        self.dec4 = DecoderBlock(in_channels=128, out_channels=64)

        # The final layer that gives the output of original input dimensions
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
            warnings.warn("\nInput data is not divisible by 16 along the width/height dimension"
                          "\nWould likely lead to errors. Highly recommended to resize image to fit this criteria"
                          f"\nCurrent shape: {x.shape}")

        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottle_neck(x)

        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        return self.output_layer(x)


if __name__ == "__main__":
    # Testing if the network is working as expected

    from PIL import Image


    image = Image.open("./Dataset/ScaledImages/001.jpg").convert("RGB")
    image.show()

    width, height = image.size  # Both should be the same for square images, 448
    data = torch.tensor(image.getdata(), dtype=torch.float32)

    print(f"Data shape: {data.shape}")

    # Since it's of shape (num_pixels, RGB), it would need to be reshaped based on it's dimensions
    data = data.reshape(width, height, 3)
    # Note that channels is first dimension, in front of width and height, as pytorch expects it as such for conv layers
    data = data.permute(2, 0, 1)
    # BatchNorm2d expects a batch dimension. Current shape is (Batch, Channels, Width, Height)
    data = data.unsqueeze(0)

    print(f"Data shape after reshaping: {data.shape}")

    # Now pass it through the UNET to test
    unet = UNET(in_channels=3, out_channels=3)
    result = unet(data)

    print(f"Final shape after network: {result.shape}")


    # Now converting the result back into an image
    # First, clip the upper and lower bound of values and element-wise round to nearest integer
    result = torch.clamp(result, min=0, max=255).type(torch.uint8).round()

    # Remove batch dimension and permute back dimension where channel is last dim, as PIL.Image expects
    result = result.squeeze(0).permute(1, 2, 0)

    # Should look like a black image, since this is just a single pass over an initialized network, where the init
    # of parameters would be centered around 0. After ReLU and rounding, most should be extremely low values
    image = Image.fromarray(result.detach().numpy()).convert("RGB")
    image.show()

