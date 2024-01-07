import multiprocessing as mp
import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

    get = lambda x: x.squeeze().cpu().numpy()
    video = get(data["video"])
    box = get(data["box"]) if "box" in data else np.zeros_like(video[0])
    labels = get(data["label"]) if "label" in data else np.zeros_like(video)
    dataset = f"({data['dataset']})" if "dataset" in data else ""

    mpl.rcParams.update(mpl.rcParamsDefault)
    vmin, vmax = video.min(), video.max()
    for j, (frame, label) in enumerate(zip(video, labels)):
        fig, ax = plt.subplots(num=1, clear=True)
        ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(box, levels=[0.5], colors="tab:blue")
        ax.imshow(np.where(label, 2, np.nan), cmap="tab10", vmin=0.5, vmax=10.5, alpha=0.6)
        ax.set_axis_off()
        ax.set_title(f"{dataset} frame: {j + 1:4d}/{len(video)}")
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
