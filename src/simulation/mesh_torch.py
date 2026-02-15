import torch
import numpy as np


class SimpleMesh:
    """Simple PyTorch replacement for discretize.TensorMesh"""

    def __init__(self, cell_sizes, origin="00N"):
        """
        cell_sizes: list of [(cell_size_x, nx), (cell_size_y, ny), (cell_size_z, nz)]
        origin: string like "00N" meaning origin at x=0, y=0, z=negative
        """
        self.cell_sizes = cell_sizes
        self.n_cells = [int(n) for (_, n) in cell_sizes]
        self.hx = cell_sizes[0][0]
        self.hy = cell_sizes[1][0]
        self.hz = cell_sizes[2][0]

        # Calculate total dimensions
        self.extent_x = self.hx * self.n_cells[0]
        self.extent_y = self.hy * self.n_cells[1]
        self.extent_z = self.hz * self.n_cells[2]

        # Set origin based on string
        if origin == "00N":
            self.origin = torch.tensor([0.0, 0.0, -self.extent_z])
        else:
            self.origin = torch.tensor([0.0, 0.0, 0.0])

        # Generate cell centers
        self._generate_cell_centers()

    def _generate_cell_centers(self):
        """Generate cell center coordinates"""
        # Cell center positions in each dimension
        x_centers = torch.linspace(
            self.hx / 2, self.extent_x - self.hx / 2, self.n_cells[0]
        )
        y_centers = torch.linspace(
            self.hy / 2, self.extent_y - self.hy / 2, self.n_cells[1]
        )
        z_centers = torch.linspace(
            self.hz / 2, self.extent_z - self.hz / 2, self.n_cells[2]
        )

        # 3D grid
        grid_x, grid_y, grid_z = torch.meshgrid(
            x_centers, y_centers, z_centers, indexing="ij"
        )

        # Apply origin offset
        self.gridCC = torch.stack(
            [grid_x.flatten(), grid_y.flatten(), grid_z.flatten() + self.origin[2]],
            dim=1,
        )

    @property
    def nC(self):
        """Total number of cells"""
        return self.n_cells[0] * self.n_cells[1] * self.n_cells[2]

    def __repr__(self):
        return f"SimpleMesh({self.n_cells}, origin={self.origin})"


def active_from_xyz(mesh, topo_xyz):
    """
    Determine active cells (below topography) using PyTorch.

    Parameters:
    -----------
    mesh : SimpleMesh
        The mesh object
    topo_xyz : torch.Tensor or np.ndarray
        Topography points (N, 3)

    Returns:
    --------
    torch.Tensor
        Boolean mask of active cells
    """
    if isinstance(topo_xyz, np.ndarray):
        topo_xyz = torch.from_numpy(topo_xyz).float()

    # Get cell centers
    cell_centers = mesh.gridCC

    # For each cell, check if it's below the topography
    # Interpolate topography height at cell's (x, y) position
    ind_active = torch.zeros(mesh.nC, dtype=torch.bool)

    # Reshape cell centers back to 3D for easier indexing
    centers_3d = cell_centers.reshape(*mesh.n_cells, 3)
    topo_3d = topo_xyz.reshape(mesh.n_cells[0], mesh.n_cells[1], 3)

    # For each (x, y) column, check if cell z is below topo z
    for i in range(mesh.n_cells[0]):
        for j in range(mesh.n_cells[1]):
            topo_z = topo_3d[i, j, 2]
            for k in range(mesh.n_cells[2]):
                cell_idx = (
                    i * mesh.n_cells[1] * mesh.n_cells[2] + j * mesh.n_cells[2] + k
                )
                cell_z = centers_3d[i, j, k, 2]
                ind_active[cell_idx] = cell_z <= topo_z

    return ind_active


class IdentityMap:
    """Simple identity mapping"""

    def __init__(self, nP):
        self.nP = nP

    def __call__(self, x):
        return x


# Compatibility aliases for imports
TensorMesh = SimpleMesh
