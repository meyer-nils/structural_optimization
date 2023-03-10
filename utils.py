import matplotlib.pyplot as plt
import torch


def plot_contours(x, y, z, figsize=(4, 3), levels=25, title=None):
    with torch.no_grad():
        plt.figure(figsize=figsize)
        plt.contour(x, y, z, levels=levels, colors="k", linewidths=1)
        plt.contourf(x, y, z, levels=levels, cmap="plasma")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
