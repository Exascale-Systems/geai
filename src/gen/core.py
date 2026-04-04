import numpy as np
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from scipy.interpolate import LinearNDInterpolator
from simpeg import maps
from simpeg.potential_fields import gravity


def create_topo(
        x_dom: float = 1.6e3, y_dom: float = 1.6e3,   # domain size in x/y (m)
        dx: float = 50, dy: float = 50,               # grid spacing in x/y (m)
        fbm_amp: float = 0.0,                         # amplitude of topography (set to 0 for flat)
        seed: int = 0,                                # random seed
        noise_sigma: float = 0.0,                     # stddev of Gaussian noise added to topography (m)
        cycles1: tuple = (1.0, 1.0),                  # number of long-wavelength cycles in x/y
        cycles2: tuple = (3.0, 2.0),                  # number of finer cycles in x/y
        phase: float = 0.0,                           # phase offset
    ):
    """Return (N,3) synthetic topography points."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, x_dom, int(round(x_dom/dx)))
    ys = np.linspace(0, y_dom, int(round(y_dom/dy)))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    U = (X - x_dom/2) / x_dom
    V = (Y - y_dom/2) / y_dom
    w1x, w1y = 2*np.pi*cycles1[0], 2*np.pi*cycles1[1]
    w2x, w2y = 2*np.pi*cycles2[0], 2*np.pi*cycles2[1]
    Z = fbm_amp * (
        np.sin(w1x*U + phase) * np.cos(w1y*V + 0.5*phase)
        + 0.5*np.sin(w2x*U) * np.cos(w2y*V)
    )
    if noise_sigma:
        Z += rng.normal(0.0, noise_sigma, size=Z.shape)
    return np.c_[X.ravel(), Y.ravel(), Z.ravel()]


def create_mesh(
        topo_xyz: np.ndarray,   # (N,3) array of topography points
        n_xy: int = 32,         # number of cells in x and y
        n_z: int = 16,          # number of cells in z
        z_dom: float = 800.0    # domain size in z (m)
    ):
    """Returns tensor mesh inferred from topo_xyz, origin (0,0,0), x/y +ve, z -ve."""
    hx = [(topo_xyz[-1,0] / n_xy, n_xy)]
    hy = [(topo_xyz[-1,1] / n_xy, n_xy)]
    hz = [(z_dom / n_z, n_z)]
    return TensorMesh([hx, hy, hz], "00N")


def mesh_from_bounds(bounds, resolution):
    """
    Returns tensor mesh from explicit bounds and resolution.
    bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    resolution: (nx, ny, nz)
    """
    hx = [(bounds[0][1] / resolution[0], resolution[0])]
    hy = [(bounds[1][1] / resolution[1], resolution[1])]
    hz = [(bounds[2][1] / resolution[2], resolution[2])]
    return TensorMesh([hx, hy, hz], "00N")


def init_model(mesh, topo_xyz, background_density=0.0):
    """Initialize the density model and active cell indices."""
    ind_active = active_from_xyz(mesh, topo_xyz)
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)
    true_model = background_density * np.ones(nC)
    return ind_active, nC, model_map, true_model


def add_random_blocks(              # should this be a random walk? does that better reflect geophyics? is it worth it at this stage?
    mesh,
    ind_active,
    model,
    n_blocks=1,                     # number of blocks
    size_frac_range=(0.05, 0.30),   # size as fraction of domain size
    density_range=(0.0, 1.0),       # density contrast range (g/cc)
    seed=0,
    max_tries=50,
    enforce_nonoverlap=False,       # enforce non-overlapping blocks
    ):
    """Density contrast generator. Returns density contrast model and tracks indices of blocks in 'occupied'."""
    rng = np.random.default_rng(seed)
    fmin, fmax = size_frac_range
    assert fmax > 0 and fmin > 0 and fmax >= fmin, "size_frac_range must be > 0"
    CCa = mesh.gridCC[ind_active]   # only generate for active cells
    mins, maxs = CCa.min(0), CCa.max(0)
    span = maxs - mins
    occupied = np.zeros_like(model, dtype=bool) # track occupied cells
    for k in range(n_blocks):
        for _ in range(max_tries):
            size = rng.uniform(fmin, fmax, 3) * span # size in x,y,z
            center = rng.uniform(mins + size / 2, maxs - size / 2) # center in x,y,z
            lo, hi = center - size / 2, center + size / 2 # bounds in x,y,z
            mask = np.all((CCa >= lo) & (CCa <= hi), axis=1) # mask on active cells
            if not mask.any(): # check if blocks are generated within tensor bounds
                continue
            if enforce_nonoverlap and np.any(occupied & mask): # check for overlap
                continue
            model[mask] = rng.uniform(*density_range)
            occupied |= mask
            break
        else:
            raise RuntimeError(
                f"Could not place block {k+1}/{n_blocks} after {max_tries} tries."
            )
    return model, occupied


def gravity_survey(
    topo_xyz,
    n_per_axis=32,             # number of receivers per axis (x/y), should match tensor discretization
    components=("gz",),        # gravity components to measure
    ):
    """Build receiver grid at topography surface; returns (receiver_locations, survey)."""
    rx_xy = np.array(np.meshgrid(
        np.linspace(topo_xyz[-1,0], topo_xyz[0,0], n_per_axis),
        np.linspace(topo_xyz[-1,1], topo_xyz[0,1], n_per_axis),
        indexing="xy"
    )).reshape(2, -1).T
    fun = LinearNDInterpolator(topo_xyz[:, :2], topo_xyz[:, 2], fill_value=np.nan)
    z = fun(rx_xy)
    if np.isnan(z).any():
        raise ValueError(f"{np.isnan(z).sum()} receivers outside convex hull; values set to NaN.")
    receiver_locations = np.c_[rx_xy, z]
    rx = gravity.receivers.Point(receiver_locations, components=list(components))
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    return receiver_locations, survey


def sim_from_sample(sample_data, shape_cells, h, components=("gz",)):
    """
    Reconstruct a SIMPEG simulation from a saved dataset sample (used during eval).

    Args:
        sample_data: dict with 'receiver_locations' and 'ind_active' tensors.
        shape_cells: (nx, ny, nz)
        h: (hx, hy, hz) cell spacings
        components: gravity components

    Returns:
        (sim, mesh, survey, model_map, ind_active)
    """
    rx = sample_data["receiver_locations"].cpu().numpy()
    ind = sample_data["ind_active"].cpu().numpy().astype(bool)

    nx, ny, nz = shape_cells
    hx, hy, hz = h

    mesh = mesh_from_bounds(
        bounds=((0, nx * hx[0]), (0, ny * hy[0]), (0, nz * hz[0])),
        resolution=(nx, ny, nz),
    )

    # Build survey directly from stored receiver locations to preserve ordering.
    # gravity_survey() would reverse the grid (uses rx[-1]/rx[0] as bounds).
    rx_obj = gravity.receivers.Point(rx, components=list(components))
    src = gravity.sources.SourceField(receiver_list=[rx_obj])
    survey = gravity.survey.Survey(src)
    _, _, model_map, _ = init_model(mesh, rx, 0)

    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )

    return sim, mesh, survey, model_map, ind


def main():
    topo_xyz = create_topo()
    mesh = create_mesh(topo_xyz, n_xy=32, n_z=16, z_dom=800.0)
    ind_active, nC, model_map, true_model = init_model(mesh, topo_xyz)
    true_model, blocks_mask = add_random_blocks(
        mesh=mesh,
        ind_active=ind_active,
        model=true_model,
        n_blocks=1,
        size_frac_range=(0.08, 0.30),
        density_range=(0.0, 2.0),
        enforce_nonoverlap=True,
    )
    receiver_locations, survey = gravity_survey(topo_xyz, n_per_axis=32, components=("gz",))
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        engine="choclo",
    )
    y = sim.dpred(true_model)
    import matplotlib.pyplot as plt
    plt.hist(y, bins=50, density=False, alpha=0.7)
    plt.show()
    try:
        from src.evaluation import (
            plot_density_contrast_3D,
            plot_gravity_measurements,
            plot_topography,
        )
        plot_topography(topo_xyz)
        plot_density_contrast_3D(mesh, ind_active, blocks_mask)
        plot_gravity_measurements(receiver_locations, y)
    except Exception as e:
        print("[plot skipped]", e)
    return dict(
        mesh=mesh, ind_active=ind_active, model=true_model,
        blocks_mask=blocks_mask, survey=survey, receivers=receiver_locations, sim=sim
    )


if __name__ == "__main__":
    main()
