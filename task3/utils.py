import gzip
import pickle

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.tv_tensors import Image, Mask


class EchoDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(Image(self.X[idx]), Mask(self.Y[idx]))
        return self.X[idx], self.Y[idx]


def load_pkl(fname):
    with gzip.open(fname, "rb") as f:
        return pickle.load(f)


def save_pkl(fname, data):
    with gzip.open(fname, "wb") as f:
        pickle.dump(data, f, 2)


def resize_frame(frame, size):
    dtype = frame.dtype
    frame = frame.astype(np.float32)
    h, w = frame.shape
    top, right = max(0, w - h), max(0, h - w)
    frame = cv2.copyMakeBorder(frame, 0, top, 0, right, cv2.BORDER_CONSTANT, 0)
    if size < max(h, w):
        frame = cv2.resize(frame, (size, size), cv2.INTER_AREA)
    else:
        frame = cv2.resize(frame, (size, size), cv2.INTER_CUBIC)
    frame = frame.astype(dtype)
    return frame
