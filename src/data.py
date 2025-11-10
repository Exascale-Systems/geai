import h5py
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split, get_worker_info
from src.transform import *
from src.io.hdf5_reader import MasterReader
from src.gen.StructuralGeo_gen import create_mesh

class MasterDataset(Dataset):
    def __init__(self, master_path: str, transform=None, device="cpu"):
        self.master_path = master_path
        with h5py.File(self.master_path, "r") as f:
            self.seeds = sorted(map(int, f["samples"].keys()))
            g = f["globals"]
            self.shape_cells = tuple(torch.as_tensor(g["shape_cells"][:], dtype=torch.int32))
            self.hx = torch.as_tensor(g["hx"][:], dtype=torch.float32)
            self.hy = torch.as_tensor(g["hy"][:], dtype=torch.float32)
            self.hz = torch.as_tensor(g["hz"][:], dtype=torch.float32)
        self.transform = transform
        self.device = torch.device(device)
        self._f = None

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

def data_prep(ds_name: dict, split_name: dict, bs: int, load_splits: bool = False):
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
    ds.transform = make_transform(ds.shape_cells, stats, noise=(0,1))   # convert to model ready tensors
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

def load_raw_sample(h5_path: Path, seed_index: int = 0):
    """
    Load sample defined by seed from dataset and return dict with:
        - seed
        - receiver locations (N,3) tuple
        - gravity values
        - shape
        - boolean array of active cells
        - true density
        - empty mesh 
     """
    ds = MasterDataset(h5_path) 
    nx, ny, nz = map(int, ds.shape_cells)
    with MasterReader(h5_path) as mr:
        seed = mr.list_seeds()[seed_index]
        s = mr.read(seed)
    ind = s["ind_active"].astype(bool)                 
    true = np.zeros_like(ind, float)
    true[ind] = s["true_model"]
    mesh = create_mesh(bounds=((0, nx*ds.hx[0]), (0, ny*ds.hy[0]), (0, nz*ds.hz[0])), resolution=(nx, ny, nz))
    return {
        "seed": seed,
        "rx": s["receiver_locations"],
        "gz": s["gz"],
        "shape": (nx, ny, nz),
        "ind_active": ind,
        "true_model": true,
        "mesh": mesh,
    }

def load_split_sample(dataset_path: str, splits_path: str, split: str = "val", index: int = 0):
    splits = np.load(splits_path)
    indices = splits["tr" if split == "train" else "va"]
    sample_data = load_raw_sample(Path(dataset_path), indices[index])
    stats = compute_stats(str(dataset_path))
    return sample_data, stats