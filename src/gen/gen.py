import numpy as np
from scipy.interpolate import LinearNDInterpolator
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.potential_fields import gravity


def create_topo(
    x_dom=2000, y_dom=2000,
    dx=10, dy=10,
    base_level=0.0,
    fbm_amp=0.0,         # amplitude of topography (set to 0 for flat)
    seed=0,
    noise_sigma=0.0, 
    cycles1=(1.0, 1.0),   # number of long-wavelength cycles in x/y
    cycles2=(3.0, 2.0),   # number of finer cycles in x/y
    phase=0.0,            # phase offset
):
    """Return (N,3) synthetic topography points."""
    rng = np.random.default_rng(seed)
    x_min, x_max=-x_dom/2, x_dom/2
    y_min, y_max=-y_dom/2, y_dom/2
    xs = np.arange(x_min, x_max + 1e-9, dx)
    ys = np.arange(y_min, y_max + 1e-9, dy)
    Lx, Ly = float(x_max - x_min), float(y_max - y_min)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    U = (X - x_min) / Lx
    V = (Y - y_min) / Ly

    w1x, w1y = 2*np.pi*cycles1[0], 2*np.pi*cycles1[1]
    w2x, w2y = 2*np.pi*cycles2[0], 2*np.pi*cycles2[1]

    Z = base_level + fbm_amp * (
        np.sin(w1x*U + phase) * np.cos(w1y*V + 0.5*phase)
        + 0.5*np.sin(w2x*U) * np.cos(w2y*V)
    )

    if noise_sigma:
        Z += rng.normal(0.0, noise_sigma, size=Z.shape)

    return np.c_[X.ravel(), Y.ravel(), Z.ravel()]


def create_mesh(topo_xyz, n_xy=40, n_z=20, z_dom=500.0):
    """
    Square XY extent covering topo; 'CCN' => centered x/y, z goes negative.
    """
    x_min, x_max = float(topo_xyz[:,0].min()), float(topo_xyz[:,0].max())
    y_min, y_max = float(topo_xyz[:,1].min()), float(topo_xyz[:,1].max())
    L = max(x_max - x_min, y_max - y_min)
    hx = [(L / n_xy, n_xy)]
    hy = [(L / n_xy, n_xy)]
    hz = [(z_dom / n_z, n_z)]
    return TensorMesh([hx, hy, hz], "CCN")


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
    base_model=None,
    n_blocks=1,
    size_frac_range=(0.05, 0.30),
    density_range=(0.0, 2.0), 
    seed=90,
    max_tries=200,
    enforce_nonoverlap=True,
):
    """
    Paint axis-aligned rectangular prisms on ACTIVE cells.
    Returns (model, blocks_mask_on_active).
    """
    fmin, fmax = size_frac_range
    assert fmax > 0 and fmin > 0 and fmax >= fmin, "size_frac_range must be > 0"
    rng = np.random.default_rng(seed)

    CCa = mesh.gridCC[ind_active]
    x, y, z = CCa[:, 0], CCa[:, 1], CCa[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z.min()), float(z.max())
    Lx, Ly, Lz = x_max - x_min, y_max - y_min, z_max - z_min

    model = np.zeros(ind_active.sum(), dtype=float) if base_model is None else base_model.copy()
    occupied = np.zeros_like(model, dtype=bool)

    for k in range(n_blocks):
        placed = False
        for _ in range(max_tries):
            sx, sy, sz = (rng.uniform(fmin, fmax)*Lx, rng.uniform(fmin, fmax)*Ly, rng.uniform(fmin, fmax)*Lz)
            cx = rng.uniform(x_min + sx/2, x_max - sx/2)
            cy = rng.uniform(y_min + sy/2, y_max - sy/2)
            cz = rng.uniform(z_min + sz/2, z_max - sz/2)
            x0, x1 = cx - sx/2, cx + sx/2
            y0, y1 = cy - sy/2, cy + sy/2
            z0, z1 = cz - sz/2, cz + sz/2

            mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
            if mask.sum() == 0:
                continue
            if enforce_nonoverlap and np.any(occupied & mask):
                continue

            model[mask] = rng.uniform(*density_range)
            occupied |= mask
            placed = True
            break

        if not placed:
            raise RuntimeError(f"Could not place block {k+1}/{n_blocks} after {max_tries} tries."
                               " Adjust size_frac_range or disable enforce_nonoverlap.")

    return model, occupied


def gravity_survey(
    topo_xyz,
    n_per_axis=20,
    components=("gz",),
):
    """
    Build receiver grid at topography surface; returns (receiver_locations, survey).
    Components e.g. ('gz',), ('gx','gy','gz'), or ('gxx','gxy',...).
    """
    x_min, x_max = float(topo_xyz[:, 0].min()), float(topo_xyz[:, 0].max())
    y_min, y_max = float(topo_xyz[:, 1].min()), float(topo_xyz[:, 1].max())
    xs = np.linspace(x_min, x_max, n_per_axis)
    ys = np.linspace(y_min, y_max, n_per_axis)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    rx_xy = np.c_[X.ravel(), Y.ravel()]

    fun = LinearNDInterpolator(topo_xyz[:, :2], topo_xyz[:, 2], fill_value=np.nan)
    z = fun(rx_xy)
    if np.isnan(z).any():
        n_bad = int(np.isnan(z).sum())
        bad_ix = np.flatnonzero(np.isnan(z))[:min(5, n_bad)]
        bad_pts = rx_xy[bad_ix]
        msg = (
            f"[gravity_survey] ERROR: {n_bad} receiver(s) fall outside the convex hull "
            f"of topo_xyz. Example out-of-hull XY points:\n{bad_pts}\n"
            "Fix: shrink the XY grid, densify/expand topo_xyz, or pre-interpolate onto a full grid."
        )
        raise ValueError(msg)

    receiver_locations = np.c_[rx_xy, z]
    rx = gravity.receivers.Point(receiver_locations, components=list(components))
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    return receiver_locations, survey


def add_noise(data, noise_level=0.05, seed=100):
    """
    Add Gaussian noise to data.
    `noise_level` is relative to max(|data|).
    """
    rng = np.random.default_rng(seed)
    sigma = noise_level * np.abs(data).max()
    return data + rng.normal(0.0, sigma, size=data.shape)


def main():
    topo_xyz = create_topo()

    mesh = create_mesh(topo_xyz, n_xy=40, n_z=20, z_dom=500.0)
    
    ind_active, nC, model_map, true_model = init_model(mesh, topo_xyz)

    true_model, blocks_mask = add_random_blocks(
        mesh=mesh,
        ind_active=ind_active,
        base_model=true_model,
        n_blocks=4,
        size_frac_range=(0.08, 0.30),
        density_range=(0.0, 2.0),
        enforce_nonoverlap=True,
    )

    receiver_locations, survey = gravity_survey(topo_xyz, n_per_axis=20, components=("gz",))

    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        engine="choclo",
    )

    data = add_noise(sim.dpred(true_model))

    try:
        from src.viz.samples import (
            plot_topography,
            plot_density_contrast_3D_voxels,
            plot_gravity_measurements,
        )
        plot_topography(topo_xyz)
        plot_density_contrast_3D_voxels(mesh, ind_active, blocks_mask)
        plot_gravity_measurements(receiver_locations, data)
    except Exception as e:
        print("[plot skipped]", e)

    return dict(mesh=mesh, ind_active=ind_active, model=true_model,
                blocks_mask=blocks_mask, survey=survey, receivers=receiver_locations, sim=sim)


if __name__ == "__main__":
    main()
