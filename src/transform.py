import torch
import h5py
import numpy as np
import math
from src.gen.StructuralGeo_gen import add_noise

def compute_stats(h5_path):
    gz_min, gz_max, rho_min, rho_max = math.inf, -math.inf, math.inf, -math.inf
    with h5py.File(h5_path, "r") as f:
        for s in f["samples"].values():
            gz = s["gz"][()]
            gz_min, gz_max = min(gz_min, gz.min()), max(gz_max, gz.max())
            tm = s["true_model"][()]
            vals = tm
            if vals.size:
                rho_min, rho_max = min(rho_min, vals.min()), max(rho_max, vals.max())
    return dict(gz_min=gz_min, gz_max=gz_max, rho_min=rho_min, rho_max=rho_max)

def norm(a, a0, a1): 
    a = torch.as_tensor(a)
    a0 = torch.as_tensor(a0, dtype=a.dtype, device=a.device)
    a1 = torch.as_tensor(a1, dtype=a.dtype, device=a.device)
    return 2*(a - a0)/(a1 - a0) - 1

def denorm(y_norm, stats, data_type="rho"):
    y = torch.clamp(torch.as_tensor(y_norm), -1.0, 1.0)
    if data_type == "rho":
        a = torch.as_tensor(stats["rho_min"], dtype=y.dtype, device=y.device)
        b = torch.as_tensor(stats["rho_max"], dtype=y.dtype, device=y.device)
    elif data_type == "gz":
        a = torch.as_tensor(stats["gz_min"], dtype=y.dtype, device=y.device)
        b = torch.as_tensor(stats["gz_max"], dtype=y.dtype, device=y.device)
    return ((y + 1.0) * 0.5) * (b - a) + a

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
        n = add_noise(g.shape, accuracy=noise[0], confidence=noise[1]) #data augementation of noise
        g+=n
        gz = norm(g, stats["gz_min"], stats["gz_max"])
        # h = norm(sample["receiver_locations"][:,2], 0, 800) # survey height as a channel
        dev = gz.device
        tm = norm(sample["true_model"], stats["rho_min"], stats["rho_max"])
        a = gz.to(dtype=torch.float32, device=dev).view(1, ny, nx)                   
        # b = h.to(dtype=torch.float32, device=dev).view(1, ny, nx)                     
        # x = torch.cat([a, b], dim=0)
        x = a  # single channel
        y = tm.to(dtype=torch.float32, device=dev).reshape(nx, ny, nz).permute(2, 1, 0).contiguous()
        m = torch.as_tensor(sample["ind_active"]).to(device=dev).reshape(nx, ny, nz).permute(2, 1, 0).contiguous().to(torch.bool)
        return x, y, m, sample["seed"]

    return to_tensors   