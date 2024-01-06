import os

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from explore import plot_frames
from fcache import fcache
from unet import UNet
from utils import EchoDataset, load_pkl, resize_frame

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

    size = 128
    train = preprocess(train, size)
    test = preprocess(test, size)
    # plot_frames("frames/preprocess", train, test)

    test = detect_box(train, test)
    plot_frames("frames/box", None, test)


@fcache
def load_data():
    train = load_pkl("data/train.pkl")
    test = load_pkl("data/test.pkl")
    return train, test


@fcache
def preprocess(dataset, size):
    for data in tqdm(dataset, "preprocess"):
        nframes = data["video"].shape[2]

        video = data["video"]
        video = video.astype(np.float32)
        video = video / 255.0
        video = [resize_frame(video[:, :, j], size) for j in range(nframes)]
        video = np.moveaxis(video, 0, 2)
        data["video"] = video

        if "box" in data:
            box = data["box"]
            box = box.astype(np.float32)
            box = resize_frame(box, size)
            data["box"] = box

        if "label" in data:
            label = data["label"]
            label = label.astype(np.float32)
            label = [resize_frame(label[:, :, j], size) for j in range(nframes)]
            label = np.moveaxis(label, 0, 2)
            data["label"] = label

    return dataset


def detect_box(train, test):
    Xa, Ya, Xe, Ye = [], [], [], []
    for data in train:
        if data["dataset"] == "amateur":
            Xa.append(np.moveaxis(data["video"], 2, 0)[::5])
            Ya.append(np.repeat(data["box"][np.newaxis, :, :], len(Xa[-1]), axis=0))
        else:
            Xe.append(np.moveaxis(data["video"], 2, 0))
            Ye.append(np.repeat(data["box"][np.newaxis, :, :], len(Xe[-1]), axis=0))

    model = UNet(n_channels=(1, 16, 32, 64, 128), n_classes=1).to(device)
    opti = optim.Adam(model.parameters(), weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    trans = T.RandomPerspective(distortion_scale=0.5)

    if os.path.exists("models/box_model.pt"):
        model.load_state_dict(torch.load("models/box_model.pt"))
    else:
        os.makedirs("train", exist_ok=True)
        train_model(
            Xa, Ya, model, opti, loss_fn, trans, 16, 5, "train/box_model_amateur"
        )
        train_model(
            Xe, Ye, model, opti, loss_fn, trans, 16, 5, "train/box_model_expert"
        )
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/box_model.pt")

    test = eval_model(test, model, "box")

    box_shape = []
    for data in train:
        if data["dataset"] == "amateur":
            continue
        bounds = np.argwhere(data["box"] > 0.5)[[0, -1], :]
        box_shape.append(bounds[1] - bounds[0])
    H, W = np.round(np.mean(box_shape, axis=0)).astype(int)

    for data in test:
        box = np.mean(data["box"], axis=2)
        y, x = np.round(np.mean(np.argwhere(box > 0.5), axis=0)).astype(int)
        box = np.zeros_like(box, dtype=np.float32)
        box[y - H // 2 : y + H // 2, x - W // 2 : x + W // 2] = 1.0
        data["box"] = box

    return test


def train_model(X, Y, model, opti, loss_fn, trans, batch_size, n_epochs, prefix=None):
    X0, X1, Y0, Y1 = train_test_split(X, Y)
    X0, X1, Y0, Y1 = map(lambda x: np.concatenate(x), (X0, X1, Y0, Y1))
    X0, X1, Y0, Y1 = map(lambda x: torch.tensor(x).unsqueeze(1), (X0, X1, Y0, Y1))

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


def eval_model(test, model, key):
    with torch.no_grad():
        for data in test:
            X = np.moveaxis(data["video"], 2, 0)
            X = torch.tensor(X).unsqueeze(1)
            Y = model(X.to(device))
            Y = Y.squeeze().cpu().numpy()
            data[key] = np.moveaxis(Y, 0, 2)
    return test


if __name__ == "__main__":
    main()
