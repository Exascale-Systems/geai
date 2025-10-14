import numpy as np
from scipy.interpolate import LinearNDInterpolator
from simpeg import maps
from simpeg.potential_fields import gravity
from simpeg.utils import model_builder
from discretize import TensorMesh
from discretize.utils import mkvc, active_from_xyz
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)

from plots import plot_topography, plot_density_contrast_2D, plot_density_contrast_3D, plot_density_contrast_3D_voxels, plot_gravity_measurements


def synthetic_topography(
    x_min=-1000, x_max=1000,
    y_min=-1000, y_max=1000,
    dx=10, dy=10,
    base_level=0.0,
    fbm_amp=20.0,
    seed=4,
    noise_sigma=0.5,
):
    """
    Return (N,3) array of x,y,z points representing synthetic topography.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float
        Domain bounds.
    dx, dy : float
        Sampling grid spacing.
    base_level : float
        Base elevation (meters).
    fbm_amp : float
        Relief amplitude (meters).
    seed : int
        RNG seed.
    noise_sigma : float
        Std deviation of Gaussian noise on z.

    Returns
    -------
    xyz : ndarray, shape (N,3)
        Point cloud of synthetic topography.
    """
    rng = np.random.default_rng(seed)

    xs = np.arange(x_min, x_max + 1e-9, dx)
    ys = np.arange(y_min, y_max + 1e-9, dy)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # synthetic hills from sinusoids
    Z = base_level + fbm_amp * (
        np.sin(0.01*X + seed) * np.cos(0.01*Y + seed/2)
        + 0.5*np.sin(0.03*X) * np.cos(0.02*Y)
    )

    # add noise
    Z += rng.normal(0.0, noise_sigma, size=Z.shape)

    # flatten
    XX, YY, ZZ = mkvc(X), mkvc(Y), mkvc(Z)

    return np.c_[XX, YY, ZZ]


def create_mesh(topo_xyz, n_xy=40, n_z=20, z_down=500.0):
    """
    Create a TensorMesh based on topography points.
    Parameters
    ----------
    topo_xyz : np.ndarray
        Topography XYZ locations.
    n_xy : int
        Number of cells in x and y directions.
    n_z : int
        Number of cells in z direction.
    z_down : float
        Depth extent (downwards).
    """
    x_min, x_max = float(topo_xyz[:,0].min()), float(topo_xyz[:,0].max())
    y_min, y_max = float(topo_xyz[:,1].min()), float(topo_xyz[:,1].max())
    Lx = x_max - x_min
    Ly = y_max - y_min
    # choose square lateral extent = max(Lx, Ly)
    L = max(Lx, Ly)
    hx = [(L / n_xy, n_xy)]
    hy = [(L / n_xy, n_xy)]
    hz = [(z_down / n_z, n_z)]   # depth extent (downwards)
    mesh = TensorMesh([hx, hy, hz], "CCN")  # centered in x,y; z goes negative
    return mesh


def initialize_model_and_active_cells(mesh, topo_xyz, background_density=0.0):
    """
    Initialize the density model and active cell indices.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        The mesh object.
    topo_xyz : np.ndarray
        Topography XYZ locations.
    background_density : float, optional
        Background density value (default is 0.0).

    Returns
    -------
    ind_active : np.ndarray
        Boolean array of active cells.
    nC : int
        Number of active cells.
    model_map : simpeg.maps.IdentityMap
        Identity map for the model.
    true_model : np.ndarray
        Initialized model array.
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
    n_blocks=4,
    size_frac_range=(0.05, 0.30),     # each block spans 5–30% of domain per axis
    density_range=(0, 2),     # g/cc; choose negatives for density deficit
    seed=90,
    max_tries=200,
    enforce_nonoverlap=True,
):
    """
    Returns (model, blocks_mask) where:
      - model is a copy of base_model (or zeros) with random blocks painted in
      - blocks_mask is a boolean mask (on active cells) of all block cells
    """
    rng = np.random.default_rng(seed)
    # Active cell centers (x, y, z)
    CCa = mesh.gridCC[ind_active]  # shape (n_active, 3)
    x, y, z = CCa[:, 0], CCa[:, 1], CCa[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    z_min, z_max = float(z.min()), float(z.max())

    Lx, Ly, Lz = x_max - x_min, y_max - y_min, z_max - z_min
    # Start from provided base_model or zeros
    if base_model is None:
        model = np.zeros(ind_active.sum(), dtype=float)
    else:
        model = base_model.copy()

    # Track “occupied” cells if we want non-overlap
    occupied = np.zeros_like(model, dtype=bool)

    for k in range(n_blocks):
        placed = False
        for _ in range(max_tries):
            # Random physical sizes (axis-aligned prisms)
            fx = rng.uniform(*size_frac_range)
            fy = rng.uniform(*size_frac_range)
            fz = rng.uniform(*size_frac_range)
            sx, sy, sz = fx * Lx, fy * Ly, fz * Lz

            # Guard tiny or degenerate sizes
            if sx <= 0 or sy <= 0 or sz <= 0:
                continue

            # Random centers (keep the full block inside domain extents)
            cx = rng.uniform(x_min + sx / 2, x_max - sx / 2)
            cy = rng.uniform(y_min + sy / 2, y_max - sy / 2)
            cz = rng.uniform(z_min + sz / 2, z_max - sz / 2)

            # Axis-aligned bounds
            x0, x1 = cx - sx / 2, cx + sx / 2
            y0, y1 = cy - sy / 2, cy + sy / 2
            z0, z1 = cz - sz / 2, cz + sz / 2

            # Boolean mask on *active* cells only
            mask_k = (
                (x >= x0) & (x <= x1) &
                (y >= y0) & (y <= y1) &
                (z >= z0) & (z <= z1)
            )

            # Ensure we actually captured some cells below the surface
            if mask_k.sum() == 0:
                continue

            # Optionally avoid overlap with previous blocks
            if enforce_nonoverlap and np.any(occupied & mask_k):
                continue

            dens_k = rng.uniform(*density_range)  # g/cc
            model[mask_k] = dens_k
            occupied |= mask_k
            placed = True
            break

        if not placed:
            print(f"[warn] Block {k+1}/{n_blocks} could not be placed after {max_tries} tries.")

    return model, occupied


def gravity_survey(
    x_bounds,
    y_bounds,
    topo_xyz,
    dh=5.0,
    n_points=20,
    components=["gz"]
):
    """
    Create gravity survey locations and SimPEG survey objects from topography.

    Parameters
    ----------
    x_bounds : tuple
        (min, max) bounds for x locations (meters).
    y_bounds : tuple
        (min, max) bounds for y locations (meters).
    topo_xyz : ndarray, shape (N, 3)
        Topography points.
    dh : float
        Height above topography for receivers (meters).
    n_points : int
        Number of points per axis for grid.
    components : list
        Gravity components (default ["gz"]).

    Returns
    -------
    receiver_locations : ndarray
        (N, 3) array of receiver locations.
    receiver_list : list
        List of SimPEG gravity receivers.
    source_field : SimPEG gravity SourceField
    survey : SimPEG gravity Survey

    Notes
    -----
    All units are in meters.
    """
    x = np.linspace(x_bounds[0], x_bounds[1], n_points)
    y = np.linspace(y_bounds[0], y_bounds[1], n_points)
    x, y = np.meshgrid(x, y)
    x, y = mkvc(x.T), mkvc(y.T)
    x_topo, y_topo, z_topo = topo_xyz[:, 0], topo_xyz[:, 1], topo_xyz[:, 2]
    fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
    z = fun_interp(np.c_[x, y]) + dh
    receiver_locations = np.c_[x, y, z]
    components = ["gz"]

    receiver_list = gravity.receivers.Point(receiver_locations, components=components)
    receiver_list = [receiver_list]
    source_field = gravity.sources.SourceField(receiver_list=receiver_list)
    survey = gravity.survey.Survey(source_field)
    return receiver_locations, receiver_list, source_field, survey


if __name__ == "__main__":
    topo_xyz = synthetic_topography()

    mesh = create_mesh(topo_xyz)

    ind_active, nC, model_map, true_model = initialize_model_and_active_cells(mesh, topo_xyz)

    true_model, blocks_mask = add_random_blocks(
        mesh=mesh,
        ind_active=ind_active,
        base_model=true_model,     # start from your background model
        n_blocks=4,
        size_frac_range=(0, 0.3),   # tweak as you like
        density_range=(0, 2),   # around your -0.2 g/cc
        seed=2,                         # remove or change for different realizations
        enforce_nonoverlap=True
    )

    xb = (topo_xyz[:,0].min(), topo_xyz[:,0].max())
    yb = (topo_xyz[:,1].min(), topo_xyz[:,1].max())
    receiver_locations, receiver_list, source_field, survey = gravity_survey(xb, yb, topo_xyz)

    simulation = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        engine="choclo",
    )

    plot_topography(topo_xyz)
    # plot_density_contrast_2D(mesh, ind_active, true_model)
    # plot_density_contrast_3D(mesh, ind_active, blocks_mask)
    plot_density_contrast_3D_voxels(mesh, ind_active, blocks_mask)
    plot_gravity_measurements(receiver_list[0].locations, simulation.dpred(true_model))