import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T
from sklearn.model_selection import train_test_split
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm

from arunet import AttRUNet
from augment import RandomEraseFromLabel, RandomErasing
from checkpt import checkpoint
from dataset import EchoDataset
from explore import merge_pdfs, plot_frames, plot_intermediate
from process import postprocess, preprocess
from rnmf import rnmf
from utils import load_pkl, save_pkl

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():
    print("device:", device)

    train, test = checkpoint("load_data", load_data)

    boxes = checkpoint("predict_boxes", predict_boxes, train, test)
    for data, box in zip(test, boxes):
        data["box"] = box

    movements = checkpoint("compute_train_movement", compute_movement, train)
    for data, movement in zip(train, movements):
        data["movement"] = movement

    movements = checkpoint("compute_test_movement", compute_movement, test)
    for data, movement in zip(test, movements):
        data["movement"] = movement

    valves = checkpoint("predict_valves", predict_valves, train, test)
    for data, valve in zip(test, valves):
        data["label"] = valve

    plot_frames("frames", None, test)


def load_data():
    train = load_pkl("data/train.pkl")
    for data in train:
        data["shape"] = data["video"].shape[:2]
        data["video"] = np.moveaxis(data["video"], 2, 0)
        data["label"] = np.moveaxis(data["label"], 2, 0)
    test = load_pkl("data/test.pkl")
    for data in test:
        data["shape"] = data["video"].shape[:2]
        data["video"] = np.moveaxis(data["video"], 2, 0)
    return train, test


def predict_boxes(train, test):
    size = 128
    model = AttRUNet((1, 32, 64, 128, 256, 512), 1).to(device)
    opti = optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    trans = T.Compose(
        [
            RandomErasing(scale=(0.1, 0.2)),
            T.RandomAffine(
                degrees=(-15, 15),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-15, 15),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomPerspective(),
        ]
    )

    try:
        model.load_state_dict(torch.load("models/boxes.pth"))
    except:
        Xa, Ya, Xe, Ye = [], [], [], []
        for data in train:
            if data["dataset"] == "amateur":
                Xa.append(data["video"][::5])
                Ya.append(np.repeat(data["box"][np.newaxis], len(Xa[-1]), axis=0))
            else:
                Xe.append(data["video"])
                Ye.append(np.repeat(data["box"][np.newaxis], len(Xe[-1]), axis=0))

        os.makedirs("boxes", exist_ok=True)
        train_model(Xa, Ya, model, opti, loss_fn, trans, size, 16, 100, 3, "boxes/amateur_")
        train_model(Xe, Ye, model, opti, loss_fn, trans, size, 16, 100, 3, "boxes/expert_")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/boxes.pth")

    raw_boxes = []
    with torch.no_grad():
        for data in tqdm(test, "eval"):
            X = preprocess(data["video"], size, device)
            X = torch.split(X, 64)
            Y = [sigmoid(model(x.to(device))) for x in X]
            raw_boxes.append(postprocess(torch.cat(Y), *data["shape"]))

    boxes = []
    for raw_box in raw_boxes:
        raw_box = raw_box.mean(axis=0)
        y, x = np.argwhere(raw_box > 0.5).T.round().astype(int)
        box = np.zeros(raw_box.shape, dtype=bool)
        box[y.min() : y.max(), x.min() : x.max()] = True
        boxes.append(box)

    return boxes


def compute_movement(dataset):
    size = 128
    movements = []
    for data in tqdm(dataset, "movement"):
        video = preprocess(data["video"], size, device).to(device)
        video /= video.max()
        _, _, movement, _, _ = rnmf(video, 2, 0.1)
        movement = postprocess(movement, *data["shape"])
        movement /= movement.max()
        movements.append(movement)
    return movements


def predict_valves(train, test):
    size = 128
    n_features = 3
    model = AttRUNet((n_features, 64, 128, 256, 512, 1024), 1).to(device)
    opti = optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    trans = T.Compose(
        [
            RandomEraseFromLabel(radius=(0.02, 0.04), channels=(0, 2), p=0.9),
            T.RandomAffine(
                degrees=(-15, 15),
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-15, 15),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.RandomPerspective(),
        ]
    )

    try:
        model.load_state_dict(torch.load("models/valves.pth"))
    except:
        Xa, Ya, Xe, Ye = [], [], [], []
        for data in train:
            frames = data["frames"]
            X = np.stack(
                (
                    data["video"][frames],
                    np.repeat(data["box"][np.newaxis], len(frames), axis=0),
                    data["movement"][frames],
                ),
                axis=1,
            )
            if data["dataset"] == "amateur":
                Xa.append(X)
                Ya.append(data["label"][frames])
            else:
                Xe.append(X)
                Ye.append(data["label"][frames])

        os.makedirs("valves", exist_ok=True)
        train_model(Xa, Ya, model, opti, loss_fn, trans, size, 8, 100, 5, "valves/amateur_")
        train_model(Xe, Ye, model, opti, loss_fn, trans, size, 8, 100, 5, "valves/expert_")
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/valves.pth")

    raw_valves = []
    with torch.no_grad():
        for data in tqdm(test, "eval"):
            X = np.stack(
                (
                    data["video"],
                    np.repeat(data["box"][np.newaxis], len(data["video"]), axis=0),
                    data["movement"],
                ),
                axis=1,
            )
            X = preprocess(X, size, device)
            X = torch.split(X, 64)
            Y = [sigmoid(model(x.to(device))) for x in X]
            raw_valves.append(postprocess(torch.cat(Y), *data["shape"]))

    valves = [raw_valve > 0.5 for raw_valve in raw_valves]

    return valves


def train_model(X, Y, model, opti, loss_fn, trans, size, batch_size, n_epochs, patience, prefix):
    X, Y = map(lambda xs: [preprocess(x, size, device) for x in xs], (X, Y))

    X0, X1, Y0, Y1 = train_test_split(X, Y)  # split over videos
    X0, X1, Y0, Y1 = map(torch.cat, (X0, X1, Y0, Y1))

    loader0 = DataLoader(EchoDataset(X0, Y0, trans), batch_size, shuffle=True)
    loader1 = DataLoader(EchoDataset(X1, Y1), batch_size, shuffle=True)

    best_loss, counter = float("inf"), 0
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

        with torch.no_grad():
            plot_intermediate(loader0, model, epoch, n_epochs, train_loss, prefix + "train")
            plot_intermediate(loader1, model, epoch, n_epochs, valid_loss, prefix + "valid")

        if valid_loss < best_loss:
            best_loss = valid_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if prefix is not None:
        merge_pdfs(prefix + "train")
        merge_pdfs(prefix + "valid")


def create_submission(test):
    pred = [
        {
            "name": data["name"],
            "prediction": np.moveaxis(data["label"], 0, 2),
        }
        for data in test
    ]
    save_pkl(pred, "submission.pkl")


if __name__ == "__main__":
    main()
