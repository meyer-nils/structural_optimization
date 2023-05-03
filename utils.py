import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.double)


def plot_contours(
    x, y, z, opti=[], figsize=(8, 6), levels=25, title=None, box=None, paths={}
):
    with torch.no_grad():
        plt.figure(figsize=figsize)
        plt.contour(x, y, z, levels=levels, colors="k", linewidths=0.5)
        # opts = {}
        if box is not None:
            cond = (x > box[0][0]) & (x < box[1][0]) & (y > box[0][1]) & (y < box[1][1])
            rect = patches.Rectangle(
                (box[0][0], box[0][1]),
                box[1][0] - box[0][0],
                box[1][1] - box[0][1],
                edgecolor="k",
                facecolor="none",
                zorder=2,
            )
            plt.gca().add_patch(rect)
            plt.contourf(
                x,
                y,
                z,
                levels=levels,
                cmap="plasma",
                alpha=0.5,
                vmin=z.min(),
                vmax=z.max(),
            )
            plt.contourf(
                torch.where(cond, x, torch.nan),
                torch.where(cond, y, torch.nan),
                torch.where(cond, z, torch.nan),
                levels=levels,
                cmap="plasma",
                vmin=z.min(),
                vmax=z.max(),
            )
        else:
            plt.contourf(
                x, y, z, levels=levels, cmap="plasma", vmin=z.min(), vmax=z.max()
            )
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
        plt.tight_layout()
