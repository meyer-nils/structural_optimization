from math import sqrt

import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection

torch.set_default_dtype(torch.double)


class Tria:
    def __init__(self):
        self.nodes = 3

    def N(self, xi):
        N_1 = 1.0 - xi[..., 0] - xi[..., 1]
        N_2 = xi[..., 0]
        N_3 = xi[..., 1]
        return torch.stack([N_1, N_2, N_3], dim=2)

    def B(self, _):
        return torch.tensor([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

    def ipoints(self):
        return [[1.0 / 3.0, 1.0 / 3.0]]

    def iweights(self):
        return [0.5]


class Quad:
    def __init__(self):
        self.nodes = 4

    def N(self, xi):
        N_1 = (1.0 - xi[..., 0]) * (1.0 - xi[..., 1])
        N_2 = (1.0 + xi[..., 0]) * (1.0 - xi[..., 1])
        N_3 = (1.0 + xi[..., 0]) * (1.0 + xi[..., 1])
        N_4 = (1.0 - xi[..., 0]) * (1.0 + xi[..., 1])
        return 0.25 * torch.stack([N_1, N_2, N_3, N_4], dim=2)

    def B(self, xi):
        return 0.25 * torch.tensor(
            [
                [-(1.0 - xi[1]), (1.0 - xi[1]), (1.0 + xi[1]), -(1.0 + xi[1])],
                [-(1.0 - xi[0]), -(1.0 + xi[0]), (1.0 + xi[0]), (1.0 - xi[0])],
            ]
        )

    def ipoints(self):
        return [
            [xi_1 / sqrt(3.0), xi_2 / sqrt(3.0)]
            for xi_2 in [-1.0, 1.0]
            for xi_1 in [-1.0, 1.0]
        ]

    def iweights(self):
        return [1.0, 1.0, 1.0, 1.0]


class FEM:
    def __init__(self, nodes, elements, forces, constraints, E, nu, etype=Quad()):
        self.nodes = nodes
        self.n_dofs = torch.numel(self.nodes)
        self.elements = elements
        self.n_elem = self.elements.shape[0]
        self.forces = forces
        self.constraints = constraints
        self.etype = etype
        # Plain strain state
        self.C = (E / ((1.0 + nu) * (1.0 - 2.0 * nu))) * torch.tensor(
            [[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 - nu]]
        )
        # Save distances between element centers
        ecenters = [torch.mean(self.nodes[e], dim=0) for e in self.elements]
        self.dist = torch.zeros((self.n_elem, self.n_elem))
        for i, ec_i in enumerate(ecenters):
            for j, ec_j in enumerate(ecenters):
                self.dist[i, j] = torch.linalg.norm(ec_j - ec_i)

    def areas(self):
        areas = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            nodes = self.nodes[element, :]
            area = 0.0
            for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
                J = (self.etype.B(q) @ nodes).T
                area += w * torch.linalg.det(J)
            areas[j] = area
        return areas

    def element_strain_energies(self, u):
        w = torch.zeros((self.n_elem))
        for j, element in enumerate(self.elements):
            u_j = torch.tensor([u[int(n), i] for n in element for i in [0, 1]])
            w[j] = 0.5 * u_j @ self.k0(element) @ u_j
        return w

    def k0(self, element):
        # Element stiffness matrix
        k = torch.zeros((2 * self.etype.nodes, 2 * self.etype.nodes))
        B = torch.zeros((3, 2 * self.etype.nodes))
        nodes = self.nodes[element, :]
        for w, q in zip(self.etype.iweights(), self.etype.ipoints()):
            J = (self.etype.B(q) @ nodes).T
            JB = torch.linalg.inv(J) @ self.etype.B(q)
            B[0, 0::2] = JB[0, :]
            B[1, 1::2] = JB[1, :]
            B[2, 0::2] = JB[1, :]
            B[2, 1::2] = JB[0, :]
            k += w * B.T @ self.C @ B * torch.linalg.det(J)
        return k

    def stiffness(self, d):
        # Assemble global stiffness matrix
        K = torch.zeros((self.n_dofs, self.n_dofs))
        for j, element in enumerate(self.elements):
            k = d[j] * self.k0(element)
            for i, I in enumerate(element):
                for j, J in enumerate(element):
                    K[2 * I, 2 * J] += k[2 * i, 2 * j]
                    K[2 * I + 1, 2 * J] += k[2 * i + 1, 2 * j]
                    K[2 * I + 1, 2 * J + 1] += k[2 * i + 1, 2 * j + 1]
                    K[2 * I, 2 * J + 1] += k[2 * i, 2 * j + 1]
        return K

    def solve(self, d):
        # Compute global stiffness matrix
        K = self.stiffness(d)

        # Get reduced stiffness matrix
        uncon = torch.nonzero(~self.constraints.ravel(), as_tuple=False).ravel()
        K_red = torch.index_select(K, 0, uncon)
        K_red = torch.index_select(K_red, 1, uncon)
        f_red = self.forces.ravel()[uncon]

        # Solve for displacement
        u_red = torch.linalg.solve(K_red, f_red)
        u = torch.zeros_like(self.nodes).ravel()
        u[uncon] = u_red

        # Evaluate force
        f = K @ u

        u = u.reshape((-1, 2))
        f = f.reshape((-1, 2))
        return u, f

    @torch.no_grad()
    def plot(self, u=0.0, node_property=None, element_property=None):
        # Compute deformed positions
        pos = self.nodes + u

        # Color surface with interpolated nodal properties (if provided)
        if node_property is not None:
            plt.tricontourf(pos[:, 0], pos[:, 1], node_property)

        # Color surface with element properties (if provided)
        if element_property is not None:
            ax = plt.gca()
            verts = pos[self.elements]
            pc = PolyCollection(verts, cmap="gray_r")
            pc.set_array(element_property)
            ax.add_collection(pc)

        # Nodes
        if len(pos) < 200:
            plt.scatter(pos[:, 0], pos[:, 1], color="black", marker="o")

        # Elements
        for element in self.elements:
            x1 = [pos[node, 0] for node in element] + [pos[element[0], 0]]
            x2 = [pos[node, 1] for node in element] + [pos[element[0], 1]]
            plt.plot(x1, x2, color="black")

        # Forces
        for i, force in enumerate(self.forces):
            if torch.norm(force) > 0.0:
                x = pos[i][0]
                y = pos[i][1]
                plt.arrow(
                    x, y, force[0], force[1], width=0.3, facecolor="gray", zorder=10
                )

        # Contraints
        for i, constraint in enumerate(self.constraints):
            x = pos[i][0]
            y = pos[i][1]
            if constraint[0]:
                plt.plot(x - 0.1, y, ">", color="gray")
            if constraint[1]:
                plt.plot(x, y - 0.1, "^", color="gray")

        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")


def get_cantilever(size, Lx, Ly, E=100, nu=0.3, etype=Quad()):
    # Dimensions
    Nx = int(Lx / size)
    Ny = int(Ly / size)

    # Create nodes
    n1 = torch.linspace(0.0, Lx, Nx)
    n2 = torch.linspace(0.0, Ly, Ny)
    n1, n2 = torch.stack(torch.meshgrid(n1, n2, indexing="xy"))
    nodes = torch.stack([n1.ravel(), n2.ravel()], dim=1)

    # Create elements connecting nodes
    elem = []
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            if type(etype) == Quad:
                # Quad elements
                n0 = i + j * Nx
                elem.append([n0, n0 + 1, n0 + Nx + 1, n0 + Nx])
            else:
                # Tria elements
                n0 = i + j * Nx
                elem.append([n0, n0 + 1, n0 + Nx + 1])
                elem.append([n0 + Nx + 1, n0 + Nx, n0])
    elements = torch.tensor(elem)

    # Load at tip
    forces = torch.zeros_like(nodes)
    # forces[len(nodes) - 1, 1] = -1.0
    forces[(int(Ny / 2) + 1) * Nx - 1, 1] = -1.0

    # Constrained displacement at left end
    constraints = torch.zeros_like(nodes, dtype=bool)
    for i in range(Ny):
        constraints[i * Nx, :] = True

    return FEM(nodes, elements, forces, constraints, E, nu, etype=etype)
