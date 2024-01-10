import gzip
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from sklearn.model_selection import train_test_split
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# os.environ["OMP_NUM_THREADS"] = "6"


try:
    from torchvision.tv_tensors import Image, Mask  # 0.16.1 # pyright: ignore
except:
    from torchvision.datapoints import Image, Mask  # 0.15.2 # pyright: ignore

from checkpt import checkpoint
from explore import merge_pdfs, plot_frames
from rnmf import rnmf
from unet import UNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():
    print("device:", device)

    train, test = checkpoint("load_data", load_data)
    test = checkpoint("detect_box", detect_box, train, test)
    train = checkpoint("detect_train_movement", detect_movement, train)
    test = checkpoint("detect_test_movement", detect_movement, test)

    plot_frames("frames", train, test)


def load_pkl(fname):
    with gzip.open(fname, "rb") as file:
        return pickle.load(file)


def save_pkl(fname, data):
    with gzip.open(fname, "wb") as file:
        pickle.dump(data, file, 2)


def load_data():
    train = load_pkl("data/train.pkl")
    for data in train:
        data["shape"] = data["video"].shape[:2]
        data["video"] = np.moveaxis(data["video"], 2, 0).astype(np.float32) / 255.0
        data["label"] = np.moveaxis(data["label"], 2, 0)
    test = load_pkl("data/test.pkl")
    for data in test:
        data["shape"] = data["video"].shape[:2]
        data["video"] = np.moveaxis(data["video"], 2, 0).astype(np.float32) / 255.0
    return train, test


def detect_box(train, test):
    size = 128
    model = UNet((1, 32, 64, 128, 256, 512), 1).to(device)
    opti = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    trans = T.RandomPerspective(distortion_scale=0.5)

    try:
        model.load_state_dict(torch.load("models/box_model.pt"))
    except:
        Xa, Ya, Xe, Ye = [], [], [], []
        for data in train:
            if data["dataset"] == "amateur":
                Xa.append(data["video"][::5])
                Ya.append(np.repeat(data["box"][np.newaxis, :, :], len(Xa[-1]), axis=0))
            else:
                Xe.append(data["video"])
                Ye.append(np.repeat(data["box"][np.newaxis, :, :], len(Xe[-1]), axis=0))

        os.makedirs("train", exist_ok=True)
        train_model(Xa, Ya, model, opti, loss_fn, trans, size, 32, 5, "train/box_model_amateur")
        train_model(Xe, Ye, model, opti, loss_fn, trans, size, 32, 5, "train/box_model_expert")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/box_model.pt")

    test = eval_model(test, model, size, 32, "box")

    for data in test:
        box = data["box"].mean(axis=0)
        y, x = np.argwhere(box > 0.5).T.round().astype(int)
        box = np.zeros(box.shape)
        box[y.min() : y.max(), x.min() : x.max()] = 1.0
        data["box"] = box.astype(np.bool_)

    return test


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


def preprocess(X, size):
    H, W = X.shape[-2:]
    X = torch.from_numpy(X).to(device).float().unsqueeze(1)
    X = F.pad(X, [0, 0, max(0, H - W), max(0, W - H)])
    X = F.resize(X, size, antialias=True)
    return X.cpu()


def postprocess(X: torch.Tensor, H, W):
    X = F.resize(X.to(device), max(H, W), antialias=True)
    X = F.crop(X, 0, 0, H, W)
    X = X.squeeze().cpu().numpy()
    return X


def train_model(X, Y, model, opti, loss_fn, trans, size, batch_size, n_epochs, prefix=None):
    X, Y = map(lambda xs: [preprocess(x, size) for x in xs], (X, Y))

    X0, X1, Y0, Y1 = train_test_split(X, Y)  # split over videos
    X0, X1, Y0, Y1 = map(torch.cat, (X0, X1, Y0, Y1))

    loader0 = DataLoader(EchoDataset(X0, Y0, trans), batch_size, shuffle=True)
    loader1 = DataLoader(EchoDataset(X1, Y1), batch_size, shuffle=True)

    for epoch in range(n_epochs):
        train_loss = 0
        for x, y in tqdm(loader0, f"epoch {epoch + 1}/{n_epochs}"):
            pred = model(x.to(device))
            loss = loss_fn(pred, y.to(device))
            opti.zero_grad()
            loss.backward()
            opti.step()
            train_loss += loss.item()
        train_loss /= len(loader0)

        valid_loss = 0
        with torch.no_grad():
            for x, y in loader1:
                valid_loss += loss_fn(model(x.to(device)), y.to(device)).item()
        valid_loss /= len(loader1)
        print(f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

        if prefix is None:
            continue

        with torch.no_grad():
            x, y = next(iter(loader1))
            pred = sigmoid(model(x.to(device)))
            fig, axs = plt.subplots(4, 4, num=1, clear=True, figsize=(10, 10))
            for ax, xi, yi, pi in zip(axs.flatten(), x, y, pred):
                xi, yi, pi = map(lambda x: x.squeeze().cpu().numpy(), (xi, yi, pi))
                ax.imshow(xi, cmap="gray")
                ax.contour(yi, levels=[0.5], colors="tab:blue")
                ax.contour(pi, levels=[0.5], colors="tab:orange")
                ax.set_axis_off()
            fig.suptitle(
                f"epoch {epoch + 1}/{n_epochs}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}"
            )
            fig.tight_layout()
            fig.savefig(f"{prefix}_{epoch + 1:04d}.pdf", bbox_inches="tight")
    merge_pdfs(prefix)


def eval_model(test, model, size, batch_size, key):
    with torch.no_grad():
        for data in tqdm(test, "eval"):
            X = preprocess(data["video"], size)
            X = torch.split(X, batch_size)
            Y = [sigmoid(model(x.to(device))).cpu() for x in X]
            data[key] = postprocess(torch.cat(Y), *data["shape"])
    return test


def detect_movement(dataset):
    size = 128
    for data in tqdm(dataset, "movement"):
        video = preprocess(data["video"], size)
        _, _, movement, _, _ = rnmf(video.to(device), 2, 1.0)
        movement /= movement.max()
        movement = postprocess(movement, *data["shape"])
        data["movement"] = movement >= movement.mean()
    return dataset


if __name__ == "__main__":
    main()
