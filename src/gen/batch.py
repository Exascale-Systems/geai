from src.gen.gen import *
from src.gen.hdf5_writer import MasterWriter
from tqdm import tqdm  

def generate_batch(
    # out_path="datasets/test.h5",                                    # add path before generating!
    ds_size=10,                                                     # number of samples to generate 
    x_dom=1.6e3, y_dom=1.6e3, z_dom=0.8e3,                          # domain size (m)
    n_xy=32, n_z=16,                                                # mesh resolution
    n_blocks=1, size_frac=(0.08, 0.30), density_range=(0.0, 1.0),   # random blocks generator
    base_seed=0,
    accuracy = 0.5e-3, confidence = 0.95                            # gravimetry noise parameters
    ):
    # invariant across samples
    topo_xyz = create_topo(x_dom=x_dom, y_dom=y_dom)
    mesh = create_mesh(topo_xyz=topo_xyz, n_xy=n_xy, n_z=n_z, z_dom=z_dom)
    ind_active, nC, model_map, _ = init_model(mesh=mesh, topo_xyz=topo_xyz)
    receiver_locations, survey = gravity_survey(
        topo_xyz=topo_xyz, n_per_axis=n_xy, components=("gz",)
    )
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        store_sensitivities='ram',
        engine="choclo",
    )
    true_model = np.zeros(nC, dtype=float)
    with MasterWriter(out_path, mesh.shape_cells, mesh.h[0], mesh.h[1], mesh.h[2]) as master:
        for k in tqdm(range(ds_size), desc="Generating samples", unit="sample", ncols=100):
            seed = base_seed + k
            true_model.fill(0.0)
            true_model, _ = add_random_blocks(
                mesh=mesh,
                ind_active=ind_active,
                model=true_model,
                n_blocks=n_blocks,
                size_frac_range=size_frac,
                density_range=density_range,
                seed=seed,
                enforce_nonoverlap=True,
            )
            y = sim.dpred(true_model)
            y += add_noise(y.shape, accuracy=accuracy, confidence=confidence, seed=seed)
            master.add(gz=y, receiver_locations=receiver_locations, true_model=true_model, ind_active=ind_active, seed=seed)
    return out_path

if __name__ == "__main__":
    generate_batch()

