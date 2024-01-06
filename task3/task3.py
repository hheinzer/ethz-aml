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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.tv_tensors import Image, Mask
from tqdm import tqdm

from explore import plot_frames
from fcache import fcache
from unet import UNet

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    print("Using", device)
    torch.manual_seed(42)
    np.random.seed(42)

    train, test = load_data()
    # plot_frames("frames/initial", train, test)

    test = detect_box(train, test)
    # plot_frames("frames/box", None, test)


@fcache
def load_data():
    convert_single = lambda x: torch.from_numpy(x).unsqueeze(0).float()
    convert_mutliple = lambda x: torch.from_numpy(x).unsqueeze(0).moveaxis(3, 0).float()

    train = load_pkl("data/train.pkl")
    for data in train:
        data["shape"] = data["video"].shape[:2]
        data["video"] = convert_mutliple(data["video"]) / 255.0
        data["box"] = convert_single(data["box"])
        data["label"] = convert_mutliple(data["label"])

    test = load_pkl("data/test.pkl")
    for data in test:
        data["shape"] = data["video"].shape[:2]
        data["video"] = convert_mutliple(data["video"]) / 255.0

    return train, test


def load_pkl(fname):
    with gzip.open(fname, "rb") as f:
        return pickle.load(f)


def detect_box(train, test):
    Xa, Ya, Xe, Ye = [], [], [], []
    for data in train:
        if data["dataset"] == "amateur":
            Xa.append(data["video"][::5])
            Ya.append(data["box"].unsqueeze(0).expand(Xa[-1].shape))
        else:
            Xe.append(data["video"])
            Ye.append(data["box"].unsqueeze(0).expand(Xe[-1].shape))

    size = 128
    model = UNet((1, 16, 32, 64, 128), 1).to(device)
    opti = optim.Adam(model.parameters(), weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    trans = T.RandomPerspective(distortion_scale=0.5)

    try:
        model.load_state_dict(torch.load("models/box_model.pt"))
    except:
        os.makedirs("train", exist_ok=True)
        train_model(
            Xa, Ya, model, opti, loss_fn, trans, size, 16, 5, "train/box_model_amateur"
        )
        train_model(
            Xe, Ye, model, opti, loss_fn, trans, size, 16, 5, "train/box_model_expert"
        )
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/box_model.pt")

    test = eval_model(test, model, size, "box")

    box_shape = []
    for data in train:
        if data["dataset"] == "amateur":
            continue
        box = preprocess(data["box"], size).squeeze()
        bounds = torch.argwhere(box > 0.5)[[0, -1], :]
        box_shape.append(bounds[1] - bounds[0])
    H, W = torch.mean(torch.stack(box_shape).float(), dim=0).round().int()

    for data in test:
        box = data["box"].mean(dim=0)
        _, y, x = torch.argwhere(box > 0.5).float().mean(dim=0).round().int()
        box = torch.zeros_like(box)
        box[0, y - H // 2 : y + H // 2, x - W // 2 : x + W // 2] = 1.0
        data["box"] = postprocess(box, *data["shape"])

    return test


def train_model(
    X, Y, model, opti, loss_fn, trans, size, batch_size, n_epochs, prefix=None
):
    X, Y = map(lambda XY: [preprocess(xy.to(device), size).cpu() for xy in XY], (X, Y))

    X0, X1, Y0, Y1 = train_test_split(X, Y)
    X0, X1, Y0, Y1 = map(lambda x: torch.cat(x), (X0, X1, Y0, Y1))

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
                pred = model(x.to(device))
                valid_loss += loss_fn(pred, y.to(device)).item()
        valid_loss /= len(loader1)
        print(f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

        if prefix is None:
            continue

        with torch.no_grad():
            x, y = next(iter(loader1))
            pred = model(x.to(device))
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


def preprocess(x, size):
    H, W = x.shape[-2:]
    x = F.pad(x, (0, 0, max(0, H - W), max(0, W - H)))
    x = F.resize(x, size, antialias=True)
    return x


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


def eval_model(test, model, size, key):
    with torch.no_grad():
        for data in test:
            X = preprocess(data["video"].to(device), size)
            data[key] = model(X).cpu()
    return test


def postprocess(x, H, W):
    x = F.resize(x, max(H, W), antialias=True)
    x = F.crop(x, 0, 0, H, W)
    return x


def save_pkl(fname, data):
    with gzip.open(fname, "wb") as f:
        pickle.dump(data, f, 2)


if __name__ == "__main__":
    main()
