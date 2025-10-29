import numpy as np
import matplotlib as mpl
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import norm
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.potential_fields import gravity
from StructuralGeo.src.geogen.dataset import GeoData3DStreamingDataset  

def create_topo(
        dataset=None, k=0,
        x_dom=1.6e3, y_dom=1.6e3, z_dom=0.8e3,   # domain size in x/y (m)
        dx=50, dy=50, dz=50,           # grid spacing in x/y (m)   
    ):
    """Return (N,3) synthetic topography points."""
    xs = np.linspace(0, x_dom, int(round(x_dom/dx)))
    ys = np.linspace(0, y_dom, int(round(y_dom/dy)))
    zs = np.linspace(0, z_dom, int(round(z_dom/dz)))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    density = dataset.__getitem__(k)[0, ...].squeeze(0).numpy()
    signs = np.sign(density)
    signs = np.where(density >= 0, 1, -1)
    switch_mask = np.diff(signs, axis=2) != 0  # (Nx, Ny, Nz-1)
    z_centers = 0.5 * (zs[:-1] + zs[1:])
    Z_switch = np.full((density.shape[0], density.shape[1]), np.nan)
    for i in range(density.shape[0]):
        for j in range(density.shape[1]):
            idx = np.where(switch_mask[i, j, :])[0]
            if len(idx) > 0:
                Z_switch[i, j] = z_centers[idx[0]]
    mask = ~np.isnan(Z_switch)
    topo_xyz = np.c_[X[mask], Y[mask], Z_switch[mask]]
    return topo_xyz

def create_mesh(
        topo_xyz, 
        n_xy=32,    # number of cells in x and y
        n_z=16,     # number of cells in z
        z_dom=800.0 # domain size in z (m)
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

def get_sample(dataset, k):
    return dataset.__getitem__(k)[0, ...].squeeze(0).numpy().ravel(order="F")

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
    bounds = ((0, 1.6e3), (0, 1.6e3), (0, 0.8e3))
    resolution = (32, 32, 16)
    dataset = GeoData3DStreamingDataset(model_bounds=bounds, model_resolution=resolution)
    topo_xyz = create_topo(dataset, k=0)
    mesh = create_mesh(topo_xyz, n_xy=32, n_z=16, z_dom=800.0)
    ind_active, nC, model_map, true_model = init_model(mesh, topo_xyz)
    true_model = get_sample(dataset, k=0)
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
            plot_density_contrast_3D,
            plot_gravity_measurements,
        )
        plot_topography(topo_xyz)
        plot_density_contrast_3D(mesh, ind_active, true_model)
        plot_gravity_measurements(receiver_locations, y)
    except Exception as e:
        print("[plot skipped]", e)
    return dict(
        mesh=mesh, ind_active=ind_active, model=true_model, survey=survey, receivers=receiver_locations, sim=sim
        )

if __name__ == "__main__":
    main()
