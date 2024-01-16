import torch
import torch.nn as nn

from unet import DoubleConv, Down


class AttUNet(nn.Module):
    """https://github.com/LeeJunHyun/Image_Segmentation"""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.conv1 = DoubleConv(n_channels[0], n_channels[1])
        self.downs = nn.ModuleList(
            [Down(n_channels[i - 1], n_channels[i]) for i in range(2, len(n_channels))]
        )
        self.ups = nn.ModuleList(
            [AttUp(n_channels[-i], n_channels[-(i + 1)]) for i in range(1, len(n_channels) - 1)]
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


class AttUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.att = AttBlock(Fg=out_channels, Fl=out_channels, Fint=out_channels // 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x3 = self.att(x1, x2)
        x = torch.cat([x3, x1], dim=1)
        return self.conv(x)


class AttBlock(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super().__init__()
        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, kernel_size=1),
            nn.BatchNorm2d(Fint),
        )
        self.Wx = nn.Sequential(
            nn.Conv2d(Fl, Fint, kernel_size=1),
            nn.BatchNorm2d(Fint),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(Fint, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.Wg(g)
        x1 = self.Wx(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
