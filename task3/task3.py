import os

os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from checkpt import checkpoint
from explore import merge_pdfs, plot_frames
from rnmf import rnmf
from sklearn.model_selection import train_test_split
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm
from unet import UNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():
    print("device:", device)

    train, test = checkpoint("load_data", load_data, ())
    test = checkpoint("detect_box", detect_box, (train, test))
    train = checkpoint("detect_train_movement", detect_movement, (train,))
    test = checkpoint("detect_test_movement", detect_movement, (test,))

    plot_frames("frames", train, test)


def load_pkl(fname):
    with gzip.open(fname, "rb") as f:
        return pickle.load(f)


def save_pkl(fname, data):
    with gzip.open(fname, "wb") as f:
        pickle.dump(data, f, 2)


def load_data():
    train = load_pkl("data/train.pkl")
    for data in tqdm(train, "load train"):
        data["shape"] = data["video"].shape[:2]
        data["video"] = np.moveaxis(data["video"], 2, 0).astype(np.float32) / 255.0
        data["label"] = np.moveaxis(data["label"], 2, 0)
    test = load_pkl("data/test.pkl")
    for data in tqdm(test, "load test"):
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
        train_model(Xe, Ye, model, opti, loss_fn, trans, size, 16, 5, "train/box_model_expert")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/box_model.pt")

    test = eval_model(test, model, size, "box")

    for data in test:
        box = data["box"].mean(axis=0)
        _, y, x = np.argwhere(box > 0.5).T.round().astype(int)
        box = torch.zeros(box.shape)
        box[0, y.min() : y.max(), x.min() : x.max()] = 1.0
        data["box"] = postprocess(box, *data["shape"])

    return test


class EchoDataset(Dataset):
    def __init__(self, X, Y, transform=None):
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


def preprocess(x, size):
    x = torch.from_numpy(x).to(device).float().unsqueeze(1)
    H, W = x.shape[-2:]
    x = F.pad(x, (0, 0, max(0, H - W), max(0, W - H)))
    x = F.resize(x, size, antialias=True)
    return x


def postprocess(x, H, W):
    x = F.resize(x, max(H, W), antialias=True)
    x = F.crop(x, 0, 0, H, W)
    x = x.squeeze().cpu().numpy()
    return x


def train_model(X, Y, model, opti, loss_fn, trans, size, batch_size, n_epochs, prefix=None):
    X, Y = map(lambda xs: [preprocess(x, size) for x in xs], (X, Y))
    X0, X1, Y0, Y1 = train_test_split(X, Y)
    X0, X1, Y0, Y1 = map(torch.cat, (X0, X1, Y0, Y1))

    loader0 = DataLoader(EchoDataset(X0, Y0, trans), batch_size, shuffle=True)
    loader1 = DataLoader(EchoDataset(X1, Y1), batch_size, shuffle=True)

    for epoch in range(n_epochs):
        train_loss = 0
        for x, y in tqdm(loader0, f"epoch {epoch + 1}/{n_epochs}"):
            pred = model(x)
            loss = loss_fn(pred, y)
            opti.zero_grad()
            loss.backward()
            opti.step()
            train_loss += loss.item()
        train_loss /= len(loader0)

        valid_loss = 0
        with torch.no_grad():
            for x, y in loader1:
                valid_loss += loss_fn(model(x), y).item()
        valid_loss /= len(loader1)
        print(f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

        if prefix is None:
            continue

        with torch.no_grad():
            x, y = next(iter(loader1))
            pred = sigmoid(model(x))
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


def eval_model(test, model, size, key):
    with torch.no_grad():
        for data in tqdm(test, "eval"):
            X = preprocess(data["video"], size)
            data[key] = sigmoid(model(X)).cpu().numpy()
    return test


def detect_movement(dataset):
    size = 128
    for data in tqdm(dataset, "movement"):
        video = preprocess(data["video"], size)
        _, _, movement, _, _ = rnmf(video, 2, 1.0)
        movement = postprocess(movement, *data["shape"])
        movement /= movement.max()
        data["movement"] = movement
    return dataset


if __name__ == "__main__":
    main()
