import torch
import torch.nn as nn


class UNet(nn.Module):
    """https://github.com/milesial/Pytorch-UNet"""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv1 = DoubleConv(n_channels[0], n_channels[1])
        self.downs = nn.ModuleList(
            [Down(n_channels[i - 1], n_channels[i]) for i in range(2, len(n_channels))]
        )
        self.ups = nn.ModuleList(
            [Up(n_channels[-i], n_channels[-(i + 1)]) for i in range(1, len(n_channels) - 1)]
        )
        self.conv2 = nn.Conv2d(n_channels[1], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        xn = []
        for down in self.downs:
            xn.append(x)
            x = down(x)
        for i, up in enumerate(self.ups):
            x = up(x, xn[-(i + 1)])
        return self.conv2(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bnorm(self.conv1(x)))
        x = self.relu(self.bnorm(self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
