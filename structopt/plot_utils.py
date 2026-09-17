import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def plot_contours(
    x: Tensor,
    f: Tensor,
    opti: Tensor | list | None = None,
    figsize: tuple[float, float] = (8, 6),
    levels: int = 25,
    title: str = "",
    box: list[Tensor] | None = None,
    paths: dict[str, list] | None = None,
    colorbar: bool = False,
):
    """Function to plot contours of a function f(x) in 2D.
    Only for educational purposes.

    Args:
        x: 2D tensor of shape (N, 2) representing the coordinates.
        f: 2D tensor of shape (N,) representing the function values.
        opti: 2D tensor of shape (2,) representing the optimal point.
        figsize: Tuple representing the figure size.
        levels: Number of contour levels.
        title: Title of the plot.
        box: Tuple representing the box coordinates for the contour plot.
        paths: Dictionary of paths to plot.
        colorbar: Boolean indicating whether to show the colorbar."""
    opti = [] if opti is None else opti
    paths = {} if paths is None else paths

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
                (box[0][0].item(), box[0][1].item()),
                box[1][0].item() - box[0][0].item(),
                box[1][1].item() - box[0][1].item(),
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
        plt.axis("equal")
        plt.title(title)
        plt.tight_layout()
