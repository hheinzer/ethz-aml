import torch
from torch.utils.data import Dataset

try:
    from torchvision.tv_tensors import Image, Mask  # 0.16.1 # pyright: ignore
except:
    from torchvision.datapoints import Image, Mask  # 0.15.2 # pyright: ignore


class EchoDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, Y = self.X[idx], self.Y[idx]
        if self.transform is not None:
            return self.transform(Image(X), Mask(Y))
        return X, Y
