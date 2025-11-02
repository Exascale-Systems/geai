import h5py, numpy as np, torch
from torch.utils.data import Dataset, get_worker_info, Subset

class MasterDataset(Dataset):
    def __init__(self, master_path, transform=None, device="cpu"):
        self.master_path = str(master_path)
        with h5py.File(self.master_path, "r") as f:
            self.seeds = sorted(map(int, f["samples"].keys()))
            g = f["globals"]
            self.shape_cells = tuple(torch.as_tensor(g["shape_cells"][:], dtype=torch.int32))
            self.hx = torch.as_tensor(g["hx"][:], dtype=torch.float32)
            self.hy = torch.as_tensor(g["hy"][:], dtype=torch.float32)
            self.hz = torch.as_tensor(g["hz"][:], dtype=torch.float32)

        self.transform = transform
        self.device = torch.device(device)
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
            "seed": torch.tensor(int(s.attrs.get("seed", int(k))), dtype=torch.int32),
            "gz": torch.as_tensor(s["gz"][:], dtype=torch.float32, device=self.device),
            "receiver_locations": torch.as_tensor(s["receiver_locations"][:], dtype=torch.float32, device=self.device),
            "true_model": torch.as_tensor(s["true_model"][:], dtype=torch.float32, device=self.device),
            "ind_active": torch.as_tensor(s["ind_active"][:], dtype=torch.uint8, device=self.device),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

def _worker_init_fn(_):
    info = get_worker_info()
    if info is None:
        return
    ds = info.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    if hasattr(ds, "close"):
        ds.close()

def collate(b):
    xs, ys = zip(*[(x,y) for x,y,_,_ in b])
    return torch.stack(xs), torch.stack(ys)
