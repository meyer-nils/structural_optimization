import matplotlib.pyplot as plt
import torch
from matplotlib import cm

torch.set_default_dtype(torch.double)


class Truss:
    def __init__(self, nodes, elements, forces, constraints, areas, E):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = len(self.elements)
        self.forces = forces
        self.constraints = constraints
        self.areas = areas
        self.E = E

        # Precompute mapping from local to global indices
        gidx_1 = []
        gidx_2 = []
        for element in self.elements:
            indices = torch.tensor([2 * n + i for n in element for i in range(2)])
            idx_1, idx_2 = torch.meshgrid(indices, indices, indexing="xy")
            gidx_1.append(idx_1)
            gidx_2.append(idx_2)
        self.gidx_1 = torch.stack(gidx_1)
        self.gidx_2 = torch.stack(gidx_2)

    def k(self):
        nodes = self.nodes[self.elements, :]
        dx = nodes[:, 0, 0] - nodes[:, 1, 0]
        dy = nodes[:, 0, 1] - nodes[:, 1, 1]
        l0 = self.element_lengths()
        c = dx / l0
        s = dy / l0
        c2 = c * c
        s2 = s * s
        cs = c * s
        m = torch.stack(
            [
                torch.stack([c2, cs, -c2, -cs], dim=-1),
                torch.stack([cs, s2, -cs, -s2], dim=-1),
                torch.stack([-c2, -cs, c2, cs], dim=-1),
                torch.stack([-cs, -s2, cs, s2], dim=-1),
            ],
            dim=-1,
        )
        return self.areas[:, None, None] * self.E / l0[:, None, None] * m

    def element_lengths(self):
        nodes = self.nodes[self.elements, :]
        dx = nodes[:, 0, :] - nodes[:, 1, :]
        return torch.linalg.norm(dx, dim=-1)

    def element_strain_energies(self, u):
        w = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            u_j = torch.stack([u[n1, 0], u[n1, 1], u[n2, 0], u[n2, 1]])
            k0 = self.k()[j] / self.areas[j]
            w[j] = 0.5 * u_j @ k0 @ u_j
        return w

    def stiffness(self):
        K = torch.zeros((self.n_dofs, self.n_dofs))
        K.index_put_((self.gidx_1, self.gidx_2), self.k(), accumulate=True)
        return K

    def solve(self):
        # Compute global stiffness matrix
        K = self.stiffness()

        # Get reduced stiffness matrix
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        K_red = K[uncon][:, uncon]
        f_red = self.forces.ravel()[uncon]

        # Solve for displacement
        u_red = torch.linalg.solve(K_red, f_red)
        u = torch.zeros_like(self.nodes).ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        # Reshape
        u = u.reshape((-1, 2))
        f = f.reshape((-1, 2))

        # Evaluate stress
        sigma = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            dx = self.nodes[n1][0] - self.nodes[n2][0]
            dy = self.nodes[n1][1] - self.nodes[n2][1]
            l0 = torch.sqrt(dx**2 + dy**2)
            c = dx / l0
            s = dy / l0
            m = torch.tensor([c, s, -c, -s])
            u_j = torch.tensor([u[n1, 0], u[n1, 1], u[n2, 0], u[n2, 1]])
            sigma[j] = self.E / l0 * torch.inner(m, u_j)

        return [u, f, sigma]

    @torch.no_grad()
    def plot(
        self,
        u=0.0,
        sigma=None,
        node_labels=True,
        show_thickness=False,
        thickness_threshold=0.0,
        default_color="black",
    ):
        # Line widths from areas
        if show_thickness:
            a_max = torch.max(self.areas)
            linewidth = 8.0 * self.areas / a_max
        else:
            linewidth = 2.0 * torch.ones(self.n_elem)
            linewidth[self.areas < thickness_threshold] = 0.0

        # Line color from stress (if present)
        if sigma is not None:
            cmap = cm.viridis
            vmin = min(sigma.min(), 0.0)
            vmax = max(sigma.max(), 0.0)
            color = cmap((sigma - vmin) / (vmax - vmin))
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            plt.colorbar(sm, label="Stress", shrink=0.5)
        else:
            color = self.n_elem * [default_color]

        # Nodes
        pos = self.nodes + u
        plt.scatter(pos[:, 0], pos[:, 1], color=default_color, marker="o")
        if node_labels:
            for i, node in enumerate(pos):
                plt.annotate(i, (node[0] + 0.01, node[1] + 0.1), color=default_color)

        # Trusses
        for j, element in enumerate(self.elements):
            n1 = element[0]
            n2 = element[1]
            x = [pos[n1][0], pos[n2][0]]
            y = [pos[n1][1], pos[n2][1]]
            plt.plot(x, y, linewidth=linewidth[j], c=color[j])

        # Forces
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                x = pos[i][0]
                y = pos[i][1]
                plt.arrow(x, y, force[0], force[1], width=0.05, facecolor="gray")

        # Constraints
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
