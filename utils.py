import matplotlib.pyplot as plt
import torch


def plot_contours(
    x, y, z, opti=[], figsize=(8, 6), levels=25, title=None, patches=[], paths={}
):
    with torch.no_grad():
        plt.figure(figsize=figsize)
        plt.contour(x, y, z, levels=levels, colors="k", linewidths=1)
        plt.contourf(x, y, z, levels=levels, cmap="plasma")
        for patch in patches:
            plt.gca().add_patch(patch)
        for label, path in paths.items():
            xp = [p[0] for p in path]
            yp = [p[1] for p in path]
            plt.plot(xp, yp, "o-", linewidth=3, label=label)
            plt.legend()
        if opti:
            plt.plot(opti[0], opti[1], "ow")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
