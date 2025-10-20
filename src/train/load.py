import h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, get_worker_info


class MasterDataset(Dataset):
    def __init__(self, master_path, transform=None):
        self.master_path = str(master_path)
        # Read just the index and globals once (fast)
        with h5py.File(self.master_path, "r") as f:
            self.seeds = sorted(map(int, f["samples"].keys()))
            g = f["globals"]
            self.shape_cells = tuple(np.array(g["shape_cells"], dtype=np.int32))
            self.hx = np.array(g["hx"], dtype=np.float32)
            self.hy = np.array(g["hy"], dtype=np.float32)
            self.hz = np.array(g["hz"], dtype=np.float32)
        self.transform = transform
        self._f = None  # per-process handle

    def _require_file(self):
        if self._f is None:
            self._f = h5py.File(self.master_path, "r", swmr=True, libver="latest")
        return self._f

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, idx):
        f = self._require_file()
        k = str(self.seeds[idx])
        s = f["samples"][k]
        sample = {
            "seed": int(s.attrs.get("seed", int(k))),
            "gz": np.array(s["gz"], dtype=np.float32),
            "receiver_locations": np.array(s["receiver_locations"], dtype=np.float32),
            "true_model": np.array(s["true_model"], dtype=np.float32),
            "ind_active": np.array(s["ind_active"], dtype=np.uint8),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def _worker_init_fn(_):
    # Ensure each worker opens its own HDF5 handle
    info = get_worker_info()
    if info is not None:
        info.dataset.close()
