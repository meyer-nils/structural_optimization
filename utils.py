import matplotlib.pyplot as plt
import torch


def plot_contours(x, y, z, figsize=(4, 3), levels=25, title=None, paths=[]):
    with torch.no_grad():
        plt.figure(figsize=figsize)
        plt.contour(x, y, z, levels=levels, colors="k", linewidths=1)
        plt.contourf(x, y, z, levels=levels, cmap="plasma")
        for path in paths:
            xp = [p[0] for p in path]
            yp = [p[1] for p in path]
            plt.plot(xp, yp, "o-", linewidth=3)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
