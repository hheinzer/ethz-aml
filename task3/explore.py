import os

os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfMerger
from tqdm import tqdm


def plot_frames(train, test):
    with mp.Pool(6) as pool:
        train = [(i, data) for i, data in enumerate(train)]
        list(tqdm(pool.imap(plot_train, train), total=len(train)))
        test = [(i, data) for i, data in enumerate(test)]
        list(tqdm(pool.imap(plot_test, test), total=len(test)))


def plot_train(args):
    i, data = args
    frames, box, labels = data["video"], data["box"], data["label"]
    for j in range(frames.shape[2]):
        frame = frames[:, :, j]
        label = labels[:, :, j]
        fig, ax = plt.subplots(num=1, clear=True)
        ax.imshow(frame, cmap="gray")
        ax.contour(box, levels=[0.5], colors="tab:blue")
        ax.imshow(np.where(label, 2, np.nan), cmap="tab10", vmin=1, vmax=10, alpha=0.6)
        ax.axis("off")
        ax.set_title(f"({data['dataset']}) frame: {j:4d}/{frames.shape[2]}")
        fig.savefig(f"frames/train_{i:04d}_{j:04d}.pdf", bbox_inches="tight")
    merge_pdfs(f"frames/train_{i:04d}")


def plot_test(args):
    i, data = args
    frames = data["video"]
    for j in range(frames.shape[2]):
        frame = frames[:, :, j]
        fig, ax = plt.subplots(num=1, clear=True)
        ax.imshow(frame, cmap="gray")
        ax.axis("off")
        ax.set_title(f"frame: {j:4d}/{frames.shape[2]}")
        fig.savefig(f"frames/test_{i:04d}_{j:04d}.pdf", bbox_inches="tight")
    merge_pdfs(f"frames/test_{i:04d}")


def merge_pdfs(prefix):
    pdfs = sorted(glob(prefix + "?*.pdf"))
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(prefix + ".pdf")
    merger.close()
    for pdf in pdfs:
        os.remove(pdf)
