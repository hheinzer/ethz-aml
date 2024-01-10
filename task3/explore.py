import multiprocessing as mp
import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pypdf import PdfMerger
from tqdm import tqdm


def plot_frames(prefix, train, test):
    os.makedirs(prefix, exist_ok=True)
    if train is not None:
        with mp.Pool(6) as pool:
            train = [(prefix + "/train", i, data) for i, data in enumerate(train)]
            list(tqdm(pool.imap(_plot_frames, train), "plot train", total=len(train)))
    if test is not None:
        with mp.Pool(6) as pool:
            test = [(prefix + "/test", i, data) for i, data in enumerate(test)]
            list(tqdm(pool.imap(_plot_frames, test), "plot test", total=len(test)))


def _plot_frames(args):
    prefix, i, data = args

    video = data["video"]
    box = data["box"] if "box" in data else None
    movement = data["movement"] if "movement" in data else None
    label = data["label"] if "label" in data else None
    dataset = f"({data['dataset']})" if "dataset" in data else ""

    vmin, vmax = video.min(), video.max()
    mpl.rcParams.update(mpl.rcParamsDefault)
    for j in range(len(video)):
        fig, ax = plt.subplots(num=1, clear=True)
        ax.imshow(video[j], cmap="gray", vmin=vmin, vmax=vmax, interpolation=None)
        if box is not None:
            ax.contour(box, levels=[0.5], colors="tab:blue")
        if movement is not None:
            ax.imshow(movement[j], cmap=alpha_cmap("Oranges"), alpha=0.6, interpolation=None)
        if label is not None:
            ax.imshow(label[j], cmap=alpha_cmap("Greens"), alpha=0.6, interpolation=None)
        ax.set_axis_off()
        ax.set_title(f"{dataset} frame: {j + 1:4d}/{len(video)}")
        fig.savefig(f"{prefix}_{i:04d}_{j:04d}.pdf", bbox_inches="tight")

    merge_pdfs(f"{prefix}_{i:04d}")


def alpha_cmap(cmap):
    colors = plt.get_cmap(cmap)([0, 255])
    colors[:, -1] = np.linspace(0.0, 1.0, 2)
    cmap = LinearSegmentedColormap.from_list("alpha_" + cmap, colors)
    return cmap


def merge_pdfs(prefix):
    pdfs = sorted(glob(prefix + "?*.pdf"))
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(prefix + ".pdf")
    merger.close()
    for pdf in pdfs:
        os.remove(pdf)
