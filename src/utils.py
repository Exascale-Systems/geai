import torch
from pathlib import Path
import numpy as np
import h5py
import math
from src.model import GravInvNet

def load_model(model_path, device="cpu"):
    device = torch.device(device)
    model = GravInvNet().to(device)
    if Path(model_path).exists():
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state.get("model", state))
    model.eval()
    return model, device

def add_noise(shape, accuracy, confidence=0.95, seed=0):
    """
    Simulate measurement uncertainty by adding Gaussian noise to data. 
    Eg. gravimeter accuracy is 0.1 mGal with 95% confidence.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    z = (1.0 + confidence) / 2.0
    # Clamp z to avoid numerical issues at extremes
    z = np.clip(z, 1e-10, 1 - 1e-10)
    ppf_z = norm.ppf(z)
    sigma = accuracy / ppf_z
    return rng.normal(0.0, sigma, size=shape)

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

