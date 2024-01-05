import os

os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from explore import plot_frames
from fcache import fcache


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    train, test = load_data()
    # plot_frames(train, test)
    print(len(train), len(test))

    train, test = map(preprocess_size, [train, test])
    # plot_frames(train, test)


@fcache
def load_data():
    train = load_pkl("data/train.pkl")
    test = load_pkl("data/test.pkl")
    return train, test


def load_pkl(fname):
    with gzip.open(fname, "rb") as f:
        return pickle.load(f)


@fcache
def preprocess_size(data, size=256):
    for i in range(len(data)):
        frames = data[i]["video"]
        nframes = np.zeros((size, size, frames.shape[2]), dtype=frames.dtype)
        for j in range(frames.shape[2]):
            nframes[:, :, j] = resize_frame(frames[:, :, j], size)
        data[i]["video"] = nframes

        if "box" in data[i]:
            data[i]["box"] = resize_frame(data[i]["box"], size)

        if "label" in data[i]:
            labels = data[i]["label"]
            nlabels = np.zeros((size, size, frames.shape[2]), dtype=labels.dtype)
            for j in range(labels.shape[2]):
                nlabels[:, :, j] = resize_frame(labels[:, :, j], size)
            data[i]["label"] = nlabels

        data[i]["size"] = tuple(frames.shape[:2])
    return data


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


def save_pkl(fname, data):
    with gzip.open(fname, "wb") as f:
        pickle.dump(data, f, 2)


if __name__ == "__main__":
    main()
