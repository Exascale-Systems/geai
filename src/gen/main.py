from generator import *
from hdf5_io import MasterWriter
from tqdm import tqdm  # progress bar


def generate_batch(
    out_path="data/master.h5",
    batch_size=10,
    x_dom=2000, y_dom=2000, z_dom=500.0, # domain size (m)
    n_xy=40, n_z=20, # mesh resolution
    n_per_axis=20, # survey receivers per axis
    n_blocks=1, size_frac=(0.08, 0.30), density_range=(0.0, 2.0), # random blocks generator
    base_seed=0,
    noise_level=0.05, # noise level (%)
):
    # invariant across samples
    topo_xyz = create_topo(x_dom, y_dom)
    mesh = create_mesh(topo_xyz, n_xy=n_xy, n_z=n_z, z_dom=z_dom)
    ind_active, nC, model_map, _ = init_model(mesh, topo_xyz)

    # survey/receivers once
    receiver_locations, survey = gravity_survey(
        topo_xyz, n_per_axis=n_per_axis, components=("gz",)
    )

    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind_active,
        engine="choclo",
    )

    with MasterWriter(out_path, mesh.shape_cells, mesh.h[0], mesh.h[1], mesh.h[2]) as master:
        for k in tqdm(range(batch_size), desc="Generating samples", unit="sample"):
            seed = base_seed + k

            true_model = np.zeros(nC, dtype=float)

            true_model, _ = add_random_blocks(
                mesh=mesh,
                ind_active=ind_active,
                base_model=true_model,
                n_blocks=n_blocks,
                size_frac_range=size_frac,
                density_range=density_range,
                seed=seed,
                enforce_nonoverlap=True,
            )

            y = sim.dpred(true_model)

            y = add_noise(y, noise_level=noise_level, seed=seed)

            print(type(seed))

            master.add(gz=y, receiver_locations=receiver_locations, true_model=true_model, ind_active=ind_active, seed=seed)

    return out_path


if __name__ == "__main__":
    generate_batch()

