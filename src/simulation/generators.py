import numpy as np
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps


def create_topo(
    x_dom: float = 1.6e3,
    y_dom: float = 1.6e3,
    dx: float = 50,
    dy: float = 50,
    fbm_amp: float = 0.0,
    seed: int = 0,
    noise_sigma: float = 0.0,
    cycles1: tuple = (1.0, 1.0),
    cycles2: tuple = (3.0, 2.0),
    phase: float = 0.0,
):
    """Return (N,3) synthetic topography points."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, x_dom, int(round(x_dom / dx)))
    ys = np.linspace(0, y_dom, int(round(y_dom / dy)))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    U = (X - x_dom / 2) / x_dom
    V = (Y - y_dom / 2) / y_dom
    w1x, w1y = 2 * np.pi * cycles1[0], 2 * np.pi * cycles1[1]
    w2x, w2y = 2 * np.pi * cycles2[0], 2 * np.pi * cycles2[1]
    Z = fbm_amp * (
        np.sin(w1x * U + phase) * np.cos(w1y * V + 0.5 * phase)
        + 0.5 * np.sin(w2x * U) * np.cos(w2y * V)
    )
    if noise_sigma:
        Z += rng.normal(0.0, noise_sigma, size=Z.shape)
    return np.c_[X.ravel(), Y.ravel(), Z.ravel()]


def create_mesh(
    topo_xyz: np.ndarray, n_xy: int = 32, n_z: int = 16, z_dom: float = 800.0
):
    """
    Returns tensor with origin (0, 0, 0) where x and y is +ve and z is -ve.
    Inferred from topo_xyz.
    """
    hx = [(topo_xyz[-1, 0] / n_xy, n_xy)]
    hy = [(topo_xyz[-1, 1] / n_xy, n_xy)]
    hz = [(z_dom / n_z, n_z)]
    return TensorMesh([hx, hy, hz], "00N")


def create_mesh_from_bounds(bounds, resolution):
    """
    Create mesh from explicit bounds and resolution.
    bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    resolution: (nx, ny, nz)
    """
    hx = [(bounds[0][1] / resolution[0], resolution[0])]
    hy = [(bounds[1][1] / resolution[1], resolution[1])]
    hz = [(bounds[2][1] / resolution[2], resolution[2])]
    return TensorMesh([hx, hy, hz], "00N")


def init_model(mesh, topo_xyz, background_density=0.0):
    """
    Initialize the density model and active cell indices.
    """
    ind_active = active_from_xyz(mesh, topo_xyz)
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)
    true_model = background_density * np.ones(nC)
    return ind_active, nC, model_map, true_model


def add_random_blocks(
    mesh,
    ind_active,
    model,
    n_blocks=1,
    size_frac_range=(0.05, 0.30),
    density_range=(0.0, 1.0),
    seed=0,
    max_tries=50,
    enforce_nonoverlap=False,
):
    """
    Density contrast generator. Returns density contrast model and tracks inidces of blocks in 'occupied'.
    """
    rng = np.random.default_rng(seed)
    fmin, fmax = size_frac_range
    assert fmax > 0 and fmin > 0 and fmax >= fmin, "size_frac_range must be > 0"
    CCa = mesh.gridCC[ind_active]  # only generate for active cells
    mins, maxs = CCa.min(0), CCa.max(0)
    span = maxs - mins
    occupied = np.zeros_like(model, dtype=bool)  # track occupied cells
    for k in range(n_blocks):
        for _ in range(max_tries):
            size = rng.uniform(fmin, fmax, 3) * span  # size in x,y,z
            center = rng.uniform(mins + size / 2, maxs - size / 2)  # center in x,y,z
            lo, hi = center - size / 2, center + size / 2  # bounds in x,y,z
            mask = np.all((CCa >= lo) & (CCa <= hi), axis=1)  # mask on active cells
            if not mask.any():  # check if blocks are generated within tensor bounds
                continue
            if enforce_nonoverlap and np.any(occupied & mask):  # check for overlap
                continue
            model[mask] = rng.uniform(*density_range)
            occupied |= mask
            break
        else:
            raise RuntimeError(
                f"Could not place block {k + 1}/{n_blocks} after {max_tries} tries."
            )
    return model, occupied
