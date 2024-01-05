import torch
import torch.nn as nn
import torchvision.transforms as transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.upconv(x)
        _, _, h, w = x.shape
        x_skip = transforms.CenterCrop((h, w))(x_skip)
        x = torch.cat([x, x_skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, size):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.downs = nn.ModuleList(
            [Down(n_channels[i], n_channels[i + 1]) for i in range(len(n_channels) - 2)]
        )
        self.conv1 = DoubleConv(n_channels[-2], n_channels[-1])
        self.ups = nn.ModuleList(
            [
                Up(n_channels[-(i + 1)], n_channels[-(i + 2)])
                for i in range(len(n_channels) - 2)
            ]
        )
        self.conv2 = nn.Conv2d(n_channels[1], n_classes, 1)
        self.sigmoid = nn.Sigmoid()
        self.resize = transforms.Resize(size, antialias=True)

    def forward(self, x):
        x_skips = []
        for down in self.downs:
            x_skip, x = down(x)
            x_skips.append(x_skip)
        x = self.conv1(x)
        for i, up in enumerate(self.ups):
            x = up(x, x_skips[-(i + 1)])
        x = self.conv2(x)
        x = self.sigmoid(x)
        return self.resize(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = (256, 256)
    unet = UNet((1, 32, 64, 128, 256, 512), 1, size)
    unet.to(device)

    x = torch.randn(1, 1, *size).to(device)
    y = unet(x)
    print(y.shape)


if __name__ == "__main__":
    main()
