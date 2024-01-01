import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


def plot_outliers(X_train, X_test, pred):
    _, ax = plt.subplots(num=1, clear=True)

    ax.scatter(X_train[:, 0], X_train[:, 1], c="tab:blue", marker="o", alpha=0.4)
    ax.scatter(X_train[pred < 0, 0], X_train[pred < 0, 1], c="tab:red", marker="x")
    ax.scatter(X_test[:, 0], X_test[:, 1], c="tab:orange", marker="o", alpha=0.4)

    gx, gy = np.mgrid[-40:40:100j, -20:20:200j]
    Z = sp.interpolate.griddata(X_train, pred, (gx, gy))
    ax.contour(gx, gy, Z, levels=[0], colors="black")

    ax.legend(["train", "outlier", "test"])
    ax.set_title(f"{np.count_nonzero(pred < 0)} outliers removed")
    plt.savefig("outliers.pdf", bbox_inches="tight")
