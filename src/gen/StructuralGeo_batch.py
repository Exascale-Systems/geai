from src.gen.StructuralGeo_gen import *
from src.gen.hdf5_writer import MasterWriter
from tqdm import tqdm
from StructuralGeo.src.geogen.dataset import GeoData3DStreamingDataset


def generate_batch(
    out_path="datasets/sg.h5",
    ds_size=20000,  # number of samples to generate
    bounds=((0, 3.2e4), (0, 3.2e4), (0, 1.6e4)),  # domain bounds (m)
    resolution=(32, 32, 16),  # domain discretization (m)
):
    # invariant across samples
    dataset = GeoData3DStreamingDataset(
        model_bounds=bounds, model_resolution=resolution
    )
    mesh = create_mesh(bounds=bounds, resolution=resolution)
    with MasterWriter(
        out_path, mesh.shape_cells, mesh.h[0], mesh.h[1], mesh.h[2]
    ) as master:
        for k in tqdm(
            range(ds_size), desc="Generating samples", unit="sample", ncols=100
        ):
            model = get_sample(dataset, k)
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
            master.add(
                gz=y,
                receiver_locations=receiver_locations,
                true_model=model,
                ind_active=ind_active,
                seed=k,
            )
            model.fill(0.0)
    return out_path


if __name__ == "__main__":
    generate_batch()
