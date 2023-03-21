import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from matplotlib import cm


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


class Truss:
    def __init__(self, nodes, elements, forces, constraints):
        self.nodes = nodes
        self.elements = elements
        self.forces = forces
        self.constraints = constraints

    def _element_stiffness(self, element, E, A):
        n1 = int(element[0])
        n2 = int(element[1])
        dx = self.nodes[n1][0] - self.nodes[n2][0]
        dy = self.nodes[n1][1] - self.nodes[n2][1]
        L = torch.sqrt(dx**2 + dy**2)
        c = dx / L
        s = dy / L
        m = torch.tensor(
            [
                [c**2, s * c, -(c**2), -s * c],
                [s * c, s**2, -s * c, -(s**2)],
                [-(c**2), -s * c, (c**2), s * c],
                [-s * c, -(s**2), s * c, s**2],
            ]
        )
        return E * A / L * m

    def stiffness(self, E, A):
        n_dofs = torch.numel(self.nodes)
        K = torch.zeros((n_dofs, n_dofs))
        for i, element in enumerate(self.elements):
            n1 = int(element[0])
            n2 = int(element[1])
            k = self._element_stiffness(element, E[i], A[i])
            K[n1 * 2 : n1 * 2 + 2, n1 * 2 : n1 * 2 + 2] += k[0:2, 0:2]
            K[n1 * 2 : n1 * 2 + 2, n2 * 2 : n2 * 2 + 2] += k[0:2, 2:4]
            K[n2 * 2 : n2 * 2 + 2, n1 * 2 : n1 * 2 + 2] += k[2:4, 0:2]
            K[n2 * 2 : n2 * 2 + 2, n2 * 2 : n2 * 2 + 2] += k[2:4, 2:4]
        return K

    def solve(self, E, A):
        # Compute global stiffness matrix
        K = self.stiffness(E, A)
        # Get reuced stiffness matrix
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        K_red = torch.index_select(K, 0, uncon)
        K_red = torch.index_select(K_red, 1, uncon)
        f_red = self.forces.ravel()[uncon]
        # Solve for displacement
        u_red = torch.linalg.solve(K_red, f_red)
        u = torch.zeros_like(self.nodes).ravel()
        u[uncon] = u_red

        # Evaluate stress
        f = (K @ u).reshape((-1, 2))
        sigma = torch.zeros((self.elements.shape[0]))
        for i, element in enumerate(self.elements):
            n1 = int(element[0])
            n2 = int(element[1])
            df = f[n2] - f[n1]
            dx = self.nodes[n2] - self.nodes[n1]
            sigma[i] = torch.dot(df, dx / torch.norm(dx)) / A[i]

        return [u.reshape((-1, 2)), sigma]

    def plot(self, u=0.0, sigma=None, A=None, node_labels=True):
        # Line widths from diameters (if present)
        if A is not None:
            A_max = torch.max(A)
            linewidth = 3.0 * A / A_max
        else:
            linewidth = 2.0 * torch.ones(self.elements.shape[0])

        # Line color from stress (if present)
        if sigma is not None:
            cmap = cm.viridis
            vmin = sigma.min()
            vmax = sigma.max()
            color = cmap((sigma - vmin) / (vmax - vmin))
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            plt.colorbar(sm, label="Stress")
        else:
            color = self.elements.shape[0] * ["black"]

        # Nodes
        pos = self.nodes + u
        plt.scatter(pos[:, 0], pos[:, 1], color="black", marker="o")
        if node_labels:
            for i, node in enumerate(pos):
                plt.annotate(i, (node[0] + 0.01, node[1] + 0.1), color="black")

        # Trusses
        for i, element in enumerate(self.elements):
            n1 = int(element[0])
            n2 = int(element[1])
            x = [pos[n1][0], pos[n2][0]]
            y = [pos[n1][1], pos[n2][1]]
            plt.plot(x, y, linewidth=linewidth[i], c=color[i])

        # Forces
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                x = pos[i][0]
                y = pos[i][1]
                plt.arrow(x, y, force[0], force[1], width=0.05, facecolor="gray")
        for i, constraint in enumerate(self.constraints):
            x = pos[i][0]
            y = pos[i][1]
            if constraint[0]:
                plt.plot(x - 0.1, y, ">", color="gray")
            if constraint[1]:
                plt.plot(x, y - 0.1, "^", color="gray")

        # Adjustments
        nmin = pos.min(dim=0).values
        nmax = pos.max(dim=0).values
        plt.axis([nmin[0] - 0.5, nmax[0] + 0.5, nmin[1] - 0.5, nmax[1] + 0.5])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")
