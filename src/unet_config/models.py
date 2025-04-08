from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = self.conv_block(128, 256)

        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv4 = self.conv_block(256, 128)

        self.up5 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv5 = self.conv_block(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        bottleneck = self.bottleneck(p2)

        u4 = self.up4(bottleneck)
        u4 = torch.cat([u4, c2], dim=1)
        c4 = self.up_conv4(u4)

        u5 = self.up5(c4)
        u5 = torch.cat([u5, c1], dim=1)
        c5 = self.up_conv5(u5)

        outputs = self.out_conv(c5)

        return outputs


def configure_unet():
    unet = UNet()
    weights = torch.load("src/unet_config/unet_model.pth", map_location=device)
    unet.load_state_dict(weights)
    unet.to(device)
    return unet
