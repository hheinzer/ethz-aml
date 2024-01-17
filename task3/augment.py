import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T


class RandomErasing(nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
        super().__init__()
        self.erase = T.RandomErasing(p, scale, ratio, value, inplace)

    def forward(self, img, label):
        return self.erase(img), label


class RandomEraseFromLabel(nn.Module):
    def __init__(self, radius, channels=None, p=0.5):
        super().__init__()
        self.radius = radius
        self.channels = channels
        self.p = p

    def forward(self, img, label):
        if torch.rand(1) > self.p:
            return img, label

        _, H, W = label.shape

        _, y, x = torch.argwhere(label > 0.5).t()
        i = torch.randint(len(y), (1,))
        y, x = y[i], x[i]

        r = torch.rand(1) * (self.radius[1] - self.radius[0]) + self.radius[0]
        r = int(min(H, W) * r)

        yy, xx = map(torch.from_numpy, np.ogrid[:H, :W])
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r**2

        new_img = img.clone()
        if self.channels is None:
            new_img[:, mask] = 0
        else:
            for c in self.channels:
                new_img[c, mask] = 0

        return new_img, label
