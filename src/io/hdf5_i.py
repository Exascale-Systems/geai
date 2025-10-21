import h5py, numpy as np
from pathlib import Path

class MasterReader:
    def __init__(self, master_path):
        self.f = h5py.File(Path(master_path), "r")
        g = self.f["globals"]
        self.shape_cells = tuple(np.array(g["shape_cells"], dtype=np.int32))
        self.hx = np.array(g["hx"], dtype=np.float32)
        self.hy = np.array(g["hy"], dtype=np.float32)
        self.hz = np.array(g["hz"], dtype=np.float32)
        self._samples = self.f["samples"]

    def list_seeds(self):
        return sorted(int(k) for k in self._samples.keys())

    def read(self, seed):
        s = self._samples[str(seed)]
        return {
            "seed": int(s.attrs.get("seed", seed)),
            "gz": np.array(s["gz"], dtype=np.float32),
            "receiver_locations": np.array(s["receiver_locations"], dtype=np.float32),
            "true_model": np.array(s["true_model"], dtype=np.float32),
            "ind_active": np.array(s["ind_active"], dtype=np.uint8),
        }

    def close(self):
        self.f.close()

    def __enter__(self): return self
    def __exit__(self, *a): self.close()
