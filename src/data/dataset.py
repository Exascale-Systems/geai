from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, get_worker_info, random_split

from src.data.transforms import add_noise, compute_stats

COMPONENT_MAP = {
    "gx": 0,
    "gy": 1,
    "gz": 2,
    "gxx": 3,
    "gxy": 4,
    "gxz": 5,
    "gyy": 6,
    "gyz": 7,
    "gzz": 8,
}


class MasterDataset(Dataset):
    def __init__(
        self, master_path: str, transform=None, device: str = "cpu", components=("gz",)
    ):
        self.master_path = master_path
        self.transform = transform
        self.device = torch.device(device)
        self.components = components
        self.component_indices = [COMPONENT_MAP[c] for c in components]
        self._f = None
        with h5py.File(self.master_path, "r") as f:
            # Cast to Any to avoid h5py typing issues
            samples_group = cast(Any, f["samples"])
            self.seeds = sorted(map(int, samples_group.keys()))
            g = cast(Any, f["globals"])
            self.shape_cells = tuple(
                torch.as_tensor(g["shape_cells"][:], dtype=torch.int32)
            )
            self.hx = torch.as_tensor(g["hx"][:], dtype=torch.float32)
            self.hy = torch.as_tensor(g["hy"][:], dtype=torch.float32)
            self.hz = torch.as_tensor(g["hz"][:], dtype=torch.float32)

    def _require_file(self):
        if self._f is None:
            self._f = h5py.File(self.master_path, "r", swmr=True, libver="latest")
        return self._f

    def __len__(self):
        return len(self.seeds)

    def __subgetitem__(self, idx, f):
        # Helper to avoid repetitive casting
        k = str(self.seeds[idx])
        s = cast(Any, f["samples"][k])
        return k, s

    def __getitem__(self, idx):
        f = self._require_file()
        k, s = self.__subgetitem__(idx, f)

        gravity_list = []
        raw_gravity = s["gravity_data"]
        for c_idx in self.component_indices:
            gravity_list.append(
                torch.as_tensor(
                    raw_gravity[c_idx::len(COMPONENT_MAP)],
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        sample = {
            "seed": torch.tensor(int(s.attrs.get("seed", int(k))), dtype=torch.int32),
            "gravity": torch.stack(gravity_list, dim=0),  # (C, N_receivers)
            "receiver_locations": torch.as_tensor(
                s["receiver_locations"][:], dtype=torch.float32, device=self.device
            ),
            "true_model": torch.as_tensor(
                s["true_model"][:], dtype=torch.float32, device=self.device
            ),
            "ind_active": torch.as_tensor(
                s["ind_active"][:], dtype=torch.uint8, device=self.device
            ),
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

    close_method = getattr(ds, "close", None)
    if callable(close_method):
        close_method()


def collate(b):
    xs, ys = zip(*[(x, y) for x, y, _, _ in b])
    return torch.stack(xs), torch.stack(ys)


def make_transform(shape_cells, stats, noise=(0, 0)):
    """
    Transform generator output (already torch tensors) into model-ready tensors.
    Produces:
      x: (C, ny, nx)  -> [gravity_norm]
      y: (nz, ny, nx) -> normalized true_model
      m: (nz, ny, nx) -> bool mask
    """
    nx, ny, nz = map(int, shape_cells)

    def to_tensors(sample):
        g = sample["gravity"]
        t = sample["true_model"]
        n = add_noise(
            g.shape, accuracy=noise[0], confidence=noise[1]
        )  # data augementation of noise
        g += n
        dev = g.device
        C = g.shape[0]
        a = g.to(dtype=torch.float32, device=dev).view(C, ny, nx)
        x = a
        y = (
            t.to(dtype=torch.float32, device=dev)
            .reshape(nx, ny, nz)
            .permute(2, 1, 0)
            .contiguous()
        )
        m = (
            torch.as_tensor(sample["ind_active"])
            .to(device=dev)
            .reshape(nx, ny, nz)
            .permute(2, 1, 0)
            .contiguous()
            .to(torch.bool)
        )
        return x, y, m, sample["seed"]

    return to_tensors


def data_prep(
    ds_name: str,
    split_name: str,
    bs: int,
    load_splits: bool = True,
    transform: bool = True,
    accuracy: float = 0.01,
    confidence: float = 0.95,
    components: tuple = ("gz",),
):
    """
    return training dataset, validation dataset, and stats required for normalization.
    """
    stats = compute_stats(f"data/{ds_name}.h5")  # calculate distribution of data
    ds = MasterDataset(f"data/{ds_name}.h5", components=components)  # load dataset
    if transform:
        ds.transform = make_transform(
            ds.shape_cells, stats, noise=(accuracy, confidence)
        )
    else:
        ds.transform = None

    if load_splits and Path(f"splits/{split_name}.npz").exists():
        splits = np.load(f"splits/{split_name}.npz")
        tr_indices, va_indices = splits["tr"], splits["va"]
        tr_ds = Subset(ds, tr_indices)
        va_ds = Subset(ds, va_indices)
    else:  # new splits
        n = len(ds)
        tr_ds, va_ds = random_split(
            ds, [int(0.8 * n), n - int(0.8 * n)], torch.Generator().manual_seed(0)
        )  # 80% train, 20% val
        Path("splits").mkdir(parents=True, exist_ok=True)
        np.savez(
            f"splits/{split_name}.npz",
            tr=np.array(tr_ds.indices),
            va=np.array(va_ds.indices),
        )
    tr_ld = DataLoader(
        tr_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        worker_init_fn=_worker_init_fn,
        collate_fn=collate,
        pin_memory=True,
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        worker_init_fn=_worker_init_fn,
        collate_fn=collate,
        pin_memory=True,
    )
    return tr_ld, va_ld, stats
