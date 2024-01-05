import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfMerger


def plot_ecgs(X, y, n):
    mpl.rcParams.update(mpl.rcParamsDefault)
    for j in np.unique(y):
        for i, (idx, x) in enumerate(
            [(i, x) for i, x in enumerate(X) if y[i] == j][:n]
        ):
            clean, rpeaks, epochs, _, _, _ = x

            _, ax = plt.subplots(2, 2, num=1, clear=True, figsize=(10, 10))
            ax = ax.flatten()

            clip = clean[rpeaks[0] : rpeaks[5]]
            time = np.linspace(0, len(clip) / 300, len(clip))
            ax[0].plot(time, clip)
            ax[0].set_xlabel("time")
            ax[0].set_ylabel("signal")
            ax[0].set_xlim(time[0], time[-1])

            clip = clean[rpeaks[0] : rpeaks[-1]]
            freq = np.fft.rfftfreq(len(clip), 1 / 300)
            spec = np.abs(np.fft.rfft(clip)) / len(clip)
            ax[1].plot(freq, spec)
            ax[1].set_xlabel("frequency")
            ax[1].set_ylabel("spectrum")
            ax[1].set_xlim(0, 50)

            time = np.linspace(0, len(clip) / 300, len(clip))
            autocorr = np.correlate(clip, clip, mode="full") / len(clip)
            autocorr = autocorr[autocorr.size // 2 :] / autocorr.max()
            ax[2].plot(time, autocorr)
            ax[2].set_xlabel("time")
            ax[2].set_ylabel("autocorrelation")
            ax[2].set_xlim(0, 2.0)

            time = np.linspace(0, epochs.shape[1] / 300, epochs.shape[1]) - 0.2
            mean = np.nanmean(epochs, axis=0)
            std = np.nanstd(epochs, axis=0)
            ax[3].plot(time, mean)
            ax[3].fill_between(time, mean - std, mean + std, alpha=0.3)
            ax[3].set_xlabel("time")
            ax[3].set_ylabel("heart beat")
            ax[3].set_xlim(-0.2, 0.4)

            ax[0].set_title(f"{idx}: class = {j}")
            plt.savefig(f"ecg_{j}_{i:05}.pdf", bbox_inches="tight")

    merge_pdfs("ecg")


def merge_pdfs(prefix):
    pdfs = sorted(glob(prefix + "?*.pdf"))
    merger = PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(prefix + ".pdf")
    merger.close()
    for pdf in pdfs:
        os.remove(pdf)
