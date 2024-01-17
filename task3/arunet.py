import torch
import torch.nn as nn
from aunet import AttBlock
from runet import RConv, RDown


class AttRUNet(nn.Module):
    """https://github.com/LeeJunHyun/Image_Segmentation"""

    def __init__(self, n_channels, n_classes, n_rblocks=2, n_recursions=2):
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
                AttRUp(n_channels[-i], n_channels[-(i + 1)], n_rblocks, n_recursions)
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


class AttRUp(nn.Module):
    def __init__(self, in_channels, out_channels, n_rblocks, n_recursions):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.att = AttBlock(Fg=out_channels, Fl=out_channels, Fint=out_channels // 2)
        self.conv = RConv(in_channels, out_channels, n_rblocks, n_recursions)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x3 = self.att(x1, x2)
        x = torch.cat([x3, x1], dim=1)
        return self.conv(x)
