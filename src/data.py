import h5py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, get_worker_info
from src.utils import compute_stats, norm, denorm
from src.gen.StructuralGeo_gen import add_noise

class MasterDataset(Dataset):
    def __init__(self, master_path: str, transform=None, device:str="cpu"):
        self.master_path = master_path
        self.transform = transform
        self.device = torch.device(device)
        self._f = None
        with h5py.File(self.master_path, "r") as f:
            self.seeds = sorted(map(int, f["samples"].keys()))
            g = f["globals"]
            self.shape_cells = tuple(torch.as_tensor(g["shape_cells"][:], dtype=torch.int32))
            self.hx = torch.as_tensor(g["hx"][:], dtype=torch.float32)
            self.hy = torch.as_tensor(g["hy"][:], dtype=torch.float32)
            self.hz = torch.as_tensor(g["hz"][:], dtype=torch.float32)

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
            "gz": torch.as_tensor(s["gravity_data"][2::9], dtype=torch.float32, device=self.device),
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

def data_prep(ds_name: str, split_name: str, bs: int, load_splits: bool = True, transform: bool = True, accuracy: float = 0.01, confidence: float = 0.95):
    """
    return training dataset, validation dataset, and stats required for normalization.
    
    Args:
        ds_name: Dataset name
        split_name: Split name for saving/loading
        bs: Batch size
        load_splits: If True, load existing splits from file. If False, create new splits.
    """
    stats = compute_stats(f"data/{ds_name}.h5")                     # calculate distribution of data
    ds = MasterDataset(f"data/{ds_name}.h5")                        # load dataset
    if transform:
        ds.transform = make_transform(ds.shape_cells, stats, noise=(accuracy, confidence))   
    else:
        ds.transform = None
    if load_splits and Path(f"splits/{split_name}.npz").exists():
        splits = np.load(f"splits/{split_name}.npz")
        tr_indices, va_indices = splits['tr'], splits['va']
        tr_ds = Subset(ds, tr_indices)
        va_ds = Subset(ds, va_indices)
    else: # new splits
        n = len(ds)
        tr_ds, va_ds = random_split(ds, [int(0.8*n),n-int(0.8*n)], torch.Generator().manual_seed(0)) # 80% train, 20% val
        Path("splits").mkdir(parents=True, exist_ok=True); 
        np.savez(f"splits/{split_name}.npz", tr=np.array(tr_ds.indices), va=np.array(va_ds.indices))
    tr_ld = DataLoader(tr_ds,batch_size=bs,shuffle=True,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
    va_ld = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,worker_init_fn=_worker_init_fn,collate_fn=collate, pin_memory=True)
    return tr_ld, va_ld, stats

def make_transform(shape_cells, stats, noise=(0, 0)):
    """
    Transform generator output (already torch tensors) into model-ready tensors.
    Produces:
      x: (2, ny, nx)  -> [gz_norm, h_norm]
      y: (nz, ny, nx) -> normalized true_model
      m: (nz, ny, nx) -> bool mask
    """
    nx, ny, nz = map(int, shape_cells)

    def to_tensors(sample):
        g = sample["gz"]
        t = sample["true_model"]
        n = add_noise(g.shape, accuracy=noise[0], confidence=noise[1]) #data augementation of noise
        g+=n
        # gz = norm(g, stats["gz_min"], stats["gz_max"])
        # h = norm(sample["receiver_locations"][:,2], 0, 800) # survey height as a channel
        dev = g.device
        # tm = norm(sample["true_model"], stats["rho_min"], stats["rho_max"])
        a = g.to(dtype=torch.float32, device=dev).view(1, ny, nx)                   
        # b = h.to(dtype=torch.float32, device=dev).view(1, ny, nx)                     
        # x = torch.cat([a, b], dim=0)
        x = a  # single channel
        y = t.to(dtype=torch.float32, device=dev).reshape(nx, ny, nz).permute(2, 1, 0).contiguous()
        m = torch.as_tensor(sample["ind_active"]).to(device=dev).reshape(nx, ny, nz).permute(2, 1, 0).contiguous().to(torch.bool)
        return x, y, m, sample["seed"]
    
    return to_tensors

