import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import norm
from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg import maps
from simpeg.potential_fields import gravity
from StructuralGeo.src.geogen.dataset import GeoData3DStreamingDataset  

def get_sample(dataset, k):
    """
    Extract [C, X, Y, Z] out of StructuralGeo GeoData3DStreamingDataset class. Return (N,3) array from StructuralGeo synthetic model.
    """
    return dataset.__getitem__(k)[1, ...].squeeze(0).numpy()

def create_topo(
        density,                                 # density array (N,3)   
        x_dom=1.6e3, y_dom=1.6e3, z_dom=0.8e3,   # domain size in x/y (m)
        dx=50, dy=50, dz=50,                     # grid spacing in x/y (m)   
    ):
    """Return (N,3) synthetic topography points from StructuralGeo synthetic model."""
    xs = np.linspace(0, x_dom, int(round(x_dom/dx)))
    ys = np.linspace(0, y_dom, int(round(y_dom/dy)))
    zs = np.linspace(0, z_dom, int(round(z_dom/dz)+1))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    switch_mask = (density[:, :, :-1] > 0) & (density[:, :, 1:] <= 0)
    z_centers = 0.5 * (zs[:-1] + zs[1:])
    Z_switch = np.full((density.shape[0], density.shape[1]), np.nan)
    for i in range(density.shape[0]):
        for j in range(density.shape[1]):
            idx = np.where(switch_mask[i, j, :])[0]
            if len(idx) > 0:
                Z_switch[i, j] = z_centers[idx[0]]
    mask = ~np.isnan(Z_switch)
    topo_xyz = np.c_[X[mask], Y[mask], Z_switch[mask]]
    if topo_xyz.size == 0:
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        topo_xyz = np.c_[X.ravel(), Y.ravel(), np.full_like(X.ravel(), z_dom)]   
    return topo_xyz

def create_mesh(
        bounds = ((0, 3.2e4), (0, 3.2e4), (0, 1.6e4)),
        resolution = (32, 32, 16)
    ):
    """
    Returns tensor with origin (0, 0, 0) where x and y is +ve and z is -ve.
    """
    hx = [(bounds[0][1] / resolution[0], resolution[0])]
    hy = [(bounds[1][1]  / resolution[1], resolution[1])]
    hz = [(bounds[2][1]  / resolution[2], resolution[2])]
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

def gravity_survey(
    topo_xyz,
    components=("gz",),        # gravity components to measure
    ):
    """
    Build receiver grid at topography surface; returns (receiver_locations, survey).
    """
    receiver_locations = topo_xyz
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
    bounds = ((0, 3.2e4), (0, 3.2e4), (0, 1.6e4))
    resolution = (32, 32, 16)
    dataset = GeoData3DStreamingDataset(model_bounds=bounds, model_resolution=resolution)
    model = get_sample(dataset, 0)
    mesh = create_mesh(bounds, resolution)
    topo_xyz = create_topo(model)
    ind_active, nC, model_map, _ = init_model(mesh, topo_xyz)
    model = model.ravel(order="F")
    receiver_locations, survey = gravity_survey(topo_xyz, components=("gz",))
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        engine="choclo",
    )
    y = sim.dpred(model)
    y += add_noise(y.shape, accuracy=0.05, confidence=0.95, seed=0)
    try:
        from src.plot import (
            plot_topography,
            plot_density_contrast_3D,
            plot_gravity_measurements,
        )
        plot_topography(topo_xyz)
        plot_density_contrast_3D(mesh, ind_active, model)
        plot_gravity_measurements(receiver_locations, y)
    except Exception as e:
        print("[plot skipped]", e)
    return dict(
        mesh=mesh, ind_active=ind_active, model=model, survey=survey, receivers=receiver_locations, sim=sim
        )

if __name__ == "__main__":
    main()
