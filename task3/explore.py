import multiprocessing as mp
import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pypdf import PdfMerger
from torch.nn.functional import sigmoid
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

    mpl.rcParams.update(mpl.rcParamsDefault)
    for j in range(len(video)):
        fig, ax = plt.subplots(num=1, clear=True)
        ax.imshow(video[j], cmap="gray", vmin=0, vmax=255)
        if box is not None:
            ax.contour(box, levels=[0.5], colors="tab:blue")
        if movement is not None:
            ax.imshow(movement[j], cmap=alpha_cmap("Oranges"), vmin=0.0, vmax=1.0, alpha=0.8)
        if label is not None:
            ax.contour(label[j], levels=[0.5], colors="tab:green")
        ax.set_axis_off()
        ax.set_title(f"{dataset} frame: {j + 1:4d}/{len(video)}")
        fig.savefig(f"{prefix}_{i:04d}_{j:04d}.pdf", bbox_inches="tight")

    merge_pdfs(f"{prefix}_{i:04d}")


def alpha_cmap(cmap):
    colors = plt.get_cmap(cmap)(range(256))
    colors[:, -1] = np.linspace(0.0, 1.0, 256)
    cmap = LinearSegmentedColormap.from_list("alpha_" + cmap, colors)
    return cmap


def plot_intermediate(loader, model, epoch, n_epochs, loss, prefix):
    device = next(model.parameters()).device
    x, y = next(iter(loader))
    pred = sigmoid(model(x.to(device)))
    fig, axs = plt.subplots(2, 4, num=1, clear=True, figsize=(10, 5))
    x, y, pred = map(lambda x: x.squeeze().cpu().numpy(), (x, y, pred))
    if x.ndim == 4:
        x = x[:, 0, :, :]
    for ax, xi, yi, pi in zip(axs.flatten(), x, y, pred):
        ax.imshow(xi, cmap="gray")
        ax.contour(yi, levels=[0.5], colors="tab:blue")
        ax.contour(pi, levels=[0.5], colors="tab:orange")
        ax.set_axis_off()
    fig.suptitle(f"epoch {epoch + 1}/{n_epochs}, loss: {loss:.4f}")
    fig.tight_layout()
    fig.savefig(f"{prefix}_{epoch + 1:04d}.pdf", bbox_inches="tight")


def merge_pdfs(prefix):
    pdfs = sorted(glob(prefix + "?*.pdf"))
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(prefix + ".pdf")
    merger.close()
    for pdf in pdfs:
        os.remove(pdf)
