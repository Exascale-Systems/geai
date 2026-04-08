import argparse

import h5py
from discretize import TensorMesh

from src.evaluation.plotter import plot_density_contrast_3D, plot_gravity_measurements


def view(h5_path: str, idx: int = 0):
    with h5py.File(h5_path, "r") as f:
        g = f["globals"]
        hx, hy, hz = g["hx"][:], g["hy"][:], g["hz"][:]
        key = sorted(f["samples"].keys(), key=lambda k: int(k))[idx]
        s = f["samples"][key]
        true_model = s["true_model"][:]
        ind_active = s["ind_active"][:].astype(bool)
        receiver_locations = s["receiver_locations"][:]
        gravity_data = s["gravity_data"][:]

    mesh = TensorMesh([hx, hy, hz])
    n_pts = receiver_locations.shape[0]
    n_components = gravity_data.size // n_pts
    gravity_data = gravity_data.reshape(n_pts, n_components)

    print(f"Sample {key} | active cells: {ind_active.sum()} | gravity components: {n_components}")
    plot_density_contrast_3D(mesh, ind_active, true_model)
    labels = ["gx", "gy", "gz"] if n_components == 3 else [f"g{i}" for i in range(n_components)]
    for i, label in enumerate(labels):
        plot_gravity_measurements(receiver_locations, gravity_data[:, i], title=f"Gravity Anomaly ({label})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize raw samples from an HDF5 dataset")
    parser.add_argument("h5_path", help="Path to HDF5 file")
    parser.add_argument("--idx", type=int, default=0, help="Sample index (default: 0)")
    args = parser.parse_args()

    view(args.h5_path, idx=args.idx)
