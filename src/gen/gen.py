import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import norm
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.potential_fields import gravity

def create_topo(
        x_dom=1.6e3, y_dom=1.6e3,   # domain size in x/y (m)
        dx=50, dy=50,           # grid spacing in x/y (m)   
        fbm_amp=0.0,            # amplitude of topography (set to 0 for flat)
        seed=0,                 # random seed
        noise_sigma=0.0,        # stddev of Gaussian noise added to topography (m)
        cycles1=(1.0, 1.0),     # number of long-wavelength cycles in x/y
        cycles2=(3.0, 2.0),     # number of finer cycles in x/y
        phase=0.0,              # phase offset
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
        topo_xyz, 
        n_xy=32,    # number of cells in x and y
        n_z=16,     # number of cells in z
        z_dom=500.0 # domain size in z (m)
    ):
    """
    Returns tensor with origin (0, 0, 0) where x and y is +ve and z is -ve.
    """
    hx = [(topo_xyz[-1,0] / n_xy, n_xy)]
    hy = [(topo_xyz[-1,1] / n_xy, n_xy)]
    hz = [(z_dom / n_z, n_z)]
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
    """
    Density contrast generator. Returns density contrast model and tracks inidces of blocks in 'occupied'.
    """
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
    """
    Build receiver grid at topography surface; returns (receiver_locations, survey).
    """
    rx_xy = np.array(np.meshgrid(
    np.linspace(topo_xyz[-1,0], topo_xyz[0,0], n_per_axis),
    np.linspace(topo_xyz[-1,1], topo_xyz[0,1], n_per_axis),
    indexing="xy"
    )).reshape(2, -1).T
    fun = LinearNDInterpolator(topo_xyz[:, :2], topo_xyz[:, 2], fill_value=np.nan)
    z = fun(rx_xy)
    if np.isnan(z).any():
        msg = (f"{np.isnan(z).sum()} receivers outside convex hull; values set to NaN.")
        raise ValueError(msg)
    receiver_locations = np.c_[rx_xy, z]
    rx = gravity.receivers.Point(receiver_locations, components=list(components))
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    return receiver_locations, survey


def add_noise(shape, accuracy, confidence=0.95, seed=0):
    """
    Simulate measurement uncertainty by adding Gaussian noise to data. 

    Eg. gravimeter accuracy is 0.1 mGal with 95% confidence.
    """
    rng = np.random.default_rng(seed)
    z = (1.0 + confidence) / 2.0
    sigma = accuracy / norm.ppf(z)
    return rng.normal(0.0, sigma, size=shape)

def main():
    topo_xyz = create_topo()
    mesh = create_mesh(topo_xyz, n_xy=32, n_z=16, z_dom=500.0)
    ind_active, nC, model_map, true_model = init_model(mesh, topo_xyz)
    true_model, blocks_mask = add_random_blocks(
        mesh=mesh,
        ind_active=ind_active,
        model=true_model,
        n_blocks=3,
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
    y += add_noise(y.shape, accuracy=0.05, confidence=0.95, seed=0)
    try:
        from src.viz.samples import (
            plot_topography,
            plot_density_contrast_3D_voxels,
            plot_gravity_measurements,
        )
        plot_topography(topo_xyz)
        plot_density_contrast_3D_voxels(mesh, ind_active, blocks_mask)
        plot_gravity_measurements(receiver_locations, y)
    except Exception as e:
        print("[plot skipped]", e)
    return dict(
        mesh=mesh, ind_active=ind_active, model=true_model,
        blocks_mask=blocks_mask, survey=survey, receivers=receiver_locations, sim=sim
        )

if __name__ == "__main__":
    main()
