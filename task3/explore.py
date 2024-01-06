import os

os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing as mp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfMerger
from tqdm import tqdm


def plot_frames(prefix, train, test):
    os.makedirs(prefix, exist_ok=True)
    with mp.Pool(6) as pool:
        if train is not None:
            train = [(prefix + "/train", i, data) for i, data in enumerate(train)]
            list(tqdm(pool.imap(_plot_frames, train), total=len(train)))
        if test is not None:
            test = [(prefix + "/test", i, data) for i, data in enumerate(test)]
            list(tqdm(pool.imap(_plot_frames, test), total=len(test)))


def _plot_frames(args):
    prefix, i, data = args
    frames = data["video"]
    vmin, vmax = frames.min(), frames.max()
    box = data["box"] if "box" in data else np.zeros_like(frames[:, :, 0])
    labels = data["label"] if "label" in data else np.zeros_like(frames)
    dataset = f"({data['dataset']})" if "dataset" in data else ""
    for j in range(frames.shape[2]):
        fig, ax = plt.subplots(num=1, clear=True)
        im = ax.imshow(frames[:, :, j], cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(box, levels=[0.5], colors="tab:blue")
        ax.imshow(np.where(labels[:, :, j], 2, np.nan), cmap="tab10", vmax=9, alpha=0.6)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"{dataset} frame: {j + 1:4d}/{frames.shape[2]}")
        fig.savefig(f"{prefix}_{i:04d}_{j:04d}.pdf", bbox_inches="tight")
    merge_pdfs(f"{prefix}_{i:04d}")


def merge_pdfs(prefix):
    pdfs = sorted(glob(prefix + "?*.pdf"))
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(prefix + ".pdf")
    merger.close()
    for pdf in pdfs:
        os.remove(pdf)
