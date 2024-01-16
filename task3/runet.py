import torch
import torch.nn as nn


class RUNet(nn.Module):
    """https://github.com/LeeJunHyun/Image_Segmentation"""

    def __init__(self, n_channels, n_classes, n_rblocks=1, n_recursions=2):
        super().__init__()
        self.conv1 = RConv(n_channels[0], n_channels[1], n_rblocks, n_recursions)
        self.downs = nn.ModuleList(
            [
                RDown(n_channels[i - 1], n_channels[i], n_rblocks, n_recursions)
                for i in range(2, len(n_channels))
            ]
        )
        self.ups = nn.ModuleList(
            [
                RUp(n_channels[-i], n_channels[-(i + 1)], n_rblocks, n_recursions)
                for i in range(1, len(n_channels) - 1)
            ]
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


class RConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_rblocks, n_recursions):
        super().__init__()
        self.n_rblocks = n_rblocks
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = RBlock(out_channels, n_recursions)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        for _ in range(self.n_rblocks - 1):
            x1 = self.conv2(x1)
        return x + x1


class RBlock(nn.Module):
    def __init__(self, out_channels, n_recursions):
        self.n_recursions = n_recursions
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv(x)
        for _ in range(self.n_recursions - 1):
            x1 = self.conv(x + x1)
        return x1


class RDown(nn.Module):
    def __init__(self, in_channels, out_channels, n_rblocks, n_recursions):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            RConv(in_channels, out_channels, n_rblocks, n_recursions),
        )

    def forward(self, x):
        return self.down(x)


class RUp(nn.Module):
    def __init__(self, in_channels, out_channels, n_rblocks, n_recursions):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = RConv(in_channels, out_channels, n_rblocks, n_recursions)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
