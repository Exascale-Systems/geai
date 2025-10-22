# scripts/visualize_sample.py
import numpy as np
from pathlib import Path
from src.viz.samples import (
    plot_topography,
    plot_density_contrast_3D_voxels,
    plot_gravity_measurements,
)
from src.gen.gen import create_mesh
from src.io.hdf5_i import MasterReader

def main():
    path = Path("data/master.h5")
    with MasterReader(path) as mr:
        seeds = mr.list_seeds()
        print(f"Available seeds: {seeds[:5]}")
        sample = mr.read(seeds[4])
        gz = sample["gz"]
        true_model = sample["true_model"]
        ind_active = sample["ind_active"].astype(bool)
        receiver_locations = sample["receiver_locations"]
        plot_topography(receiver_locations)
        plot_gravity_measurements(receiver_locations, gz)
        full_model = np.zeros_like(ind_active, dtype=float)
        full_model[ind_active] = true_model
        blocks_mask = full_model > 0.0
        receiver_locations = np.flip(receiver_locations, axis=0)
        mesh = create_mesh(receiver_locations, n_xy=64, n_z=32, z_dom=1.6e3)
        plot_density_contrast_3D_voxels(mesh, ind_active, blocks_mask)

if __name__ == "__main__":
    main()
