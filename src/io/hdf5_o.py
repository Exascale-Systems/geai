import h5py
import numpy as np
from pathlib import Path


def save_sample_h5(path, gz, receiver_locations, true_model, ind_active, seed):
    """
    Save one gravity inversion sample to an individual HDF5 file. Uses `seed` as the unique identifier.

    Parameters
    ----------
    path : str or Path
        Output filename (.h5).
    gz : array (N_recv,)
        Gravity values at receiver locations.
    receiver_locations : array (N_recv, 3)
        Coordinates of each gravity measurement.
    true_model : array (n_active,)
        Density contrast per active cell.
    ind_active : array (mesh.nC,)
        Boolean or uint8 mask for active cells in the mesh.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["seed"] = int(seed)
        f.create_dataset("gz", data=np.asarray(gz, np.float32), compression="gzip")
        f.create_dataset("receiver_locations", data=np.asarray(receiver_locations, np.float32), compression="gzip")
        f.create_dataset("true_model", data=np.asarray(true_model, np.float32), compression="gzip")
        f.create_dataset("ind_active", data=np.asarray(ind_active, np.uint8), compression="gzip")


class MasterWriter:
    """
    Master HDF5 container for all samples + global mesh info.
      /globals/{shape_cells, hx, hy, hz}
      /samples/<seed>/{gz, receiver_locations, true_model, ind_active}
    """

    def __init__(self, master_path, shape_cells, hx, hy, hz, overwrite=True):
        self.path = Path(master_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.path.exists():
            self.path.unlink()

        self.f = h5py.File(self.path, "w")
        g = self.f.create_group("globals")
        g.create_dataset("shape_cells", data=np.asarray(shape_cells, np.int32))
        g.create_dataset("hx", data=np.asarray(hx, np.float32))
        g.create_dataset("hy", data=np.asarray(hy, np.float32))
        g.create_dataset("hz", data=np.asarray(hz, np.float32))
        self.samples = self.f.create_group("samples")

    def add(self, seed, gz, receiver_locations, true_model, ind_active):
        """Add one sample using seed as group name."""
        sg = self.samples.create_group(str(seed))
        print(type(seed))
        sg.attrs["seed"] = int(seed)
        sg.create_dataset("gz", data=np.asarray(gz, np.float32), compression="gzip")
        sg.create_dataset("receiver_locations", data=np.asarray(receiver_locations, np.float32), compression="gzip")
        sg.create_dataset("true_model", data=np.asarray(true_model, np.float32), compression="gzip")
        sg.create_dataset("ind_active", data=np.asarray(ind_active, np.uint8), compression="gzip")

    def close(self):
        """Close the master file."""
        if getattr(self, "f", None):
            self.f.flush()
            self.f.close()
            self.f = None

    def __enter__(self): return self

    def __exit__(self, *args): self.close()

    
