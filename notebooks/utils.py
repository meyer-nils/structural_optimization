import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

torch.set_default_dtype(torch.double)


def plot_contours(
    x,
    f,
    opti=[],
    figsize=(8, 6),
    levels=25,
    title=None,
    box=None,
    paths={},
    colorbar=False,
):
    with torch.no_grad():
        plt.figure(figsize=figsize)
        plt.contour(x[..., 0], x[..., 1], f, levels=levels, colors="k", linewidths=0.5)
        if box is not None:
            cond = (
                (x[..., 0] > box[0][0])
                & (x[..., 0] < box[1][0])
                & (x[..., 1] > box[0][1])
                & (x[..., 1] < box[1][1])
            )
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
                x[..., 0],
                x[..., 1],
                f,
                levels=levels,
                cmap="plasma",
                alpha=0.5,
                vmin=f.min(),
                vmax=f.max(),
            )
            plt.contourf(
                torch.where(cond, x[..., 0], torch.nan),
                torch.where(cond, x[..., 1], torch.nan),
                torch.where(cond, f, torch.nan),
                levels=levels,
                cmap="plasma",
                vmin=f.min(),
                vmax=f.max(),
            )
        else:
            plt.contourf(
                x[..., 0],
                x[..., 1],
                f,
                levels=levels,
                cmap="plasma",
                vmin=f.min(),
                vmax=f.max(),
            )
        if colorbar:
            plt.colorbar()
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
