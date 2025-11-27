import torch
from pathlib import Path
import numpy as np
import h5py
import math
from src.model import GravInvNet
from scipy.stats import norm

def load_model(model_name, device="cpu"):
    device = torch.device(device)
    model = GravInvNet().to(device)
    if Path(f"models/{model_name}.pt").exists():
        state = torch.load(f"models/{model_name}.pt", map_location=device)
        model.load_state_dict(state.get("model", state))
    model.eval()
    return model, device

def add_noise(shape, accuracy, confidence=0.95, seed=0):
    """
    Simulate measurement uncertainty by adding Gaussian noise to data. 
    Eg. gravimeter accuracy is 0.1 mGal with 95% confidence.
    """
    rng = np.random.default_rng(seed)
    z = (1.0 + confidence) / 2.0
    z = np.clip(z, 1e-10, 1 - 1e-10)
    from scipy.stats import norm as scipy_norm
    ppf_z = scipy_norm.ppf(z)
    sigma = accuracy / ppf_z
    return rng.normal(0.0, sigma, size=shape)

def compute_stats(h5_path):
    gx_min, gx_max = math.inf, -math.inf
    gy_min, gy_max = math.inf, -math.inf
    gz_min, gz_max = math.inf, -math.inf
    rho_min, rho_max = math.inf, -math.inf
    with h5py.File(h5_path, "r") as f:
        for s in f["samples"].values():
            gravity_data = s["gravity_data"][()]
            gx = gravity_data[0::9]  # Extract gx component
            gy = gravity_data[1::9]  # Extract gy component
            gz = gravity_data[2::9]  # Extract gz component
            gx_min, gx_max = min(gx_min, gx.min()), max(gx_max, gx.max())
            gy_min, gy_max = min(gy_min, gy.min()), max(gy_max, gy.max())
            gz_min, gz_max = min(gz_min, gz.min()), max(gz_max, gz.max())
            tm = s["true_model"][()]
            vals = tm
            if vals.size:
                rho_min, rho_max = min(rho_min, vals.min()), max(rho_max, vals.max())
    return dict(gx_min=gx_min, gx_max=gx_max, gy_min=gy_min, gy_max=gy_max, 
                gz_min=gz_min, gz_max=gz_max, rho_min=rho_min, rho_max=rho_max)

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
    elif data_type == "gx":
        a = torch.as_tensor(stats["gx_min"], dtype=y.dtype, device=y.device)
        b = torch.as_tensor(stats["gx_max"], dtype=y.dtype, device=y.device)
    elif data_type == "gy":
        a = torch.as_tensor(stats["gy_min"], dtype=y.dtype, device=y.device)
        b = torch.as_tensor(stats["gy_max"], dtype=y.dtype, device=y.device)
    elif data_type == "gz":
        a = torch.as_tensor(stats["gz_min"], dtype=y.dtype, device=y.device)
        b = torch.as_tensor(stats["gz_max"], dtype=y.dtype, device=y.device)
    return ((y + 1.0) * 0.5) * (b - a) + a

class TorchMetrics:
    def __init__(self, stats, threshold=0.1):
        self.stats = stats
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.sum_se = 0.0
        self.sum_ae = 0.0
        self.intersection = 0.0
        self.union = 0.0
        self.true_sum = 0.0
        self.pred_sum = 0.0
        self.n_samples = 0

    @torch.no_grad()
    def update(self, net, gravity_data, tgt, denorm_fn):
        """Accumulate metrics for a single batch."""
        pred = net(gravity_data)
        # Skip denormalization since data is not normalized
        pred_denorm = pred  # denorm_fn(pred, self.stats)
        tgt_denorm = tgt    # denorm_fn(tgt, self.stats)
        diff = tgt_denorm - pred_denorm
        self.sum_se += torch.sum(diff ** 2).item()
        self.sum_ae += torch.sum(torch.abs(diff)).item()
        self.n_samples += diff.numel()
        true_binary = tgt_denorm > self.threshold
        pred_binary = pred_denorm > self.threshold
        self.intersection += torch.sum(true_binary & pred_binary).item()
        self.union += torch.sum(true_binary | pred_binary).item()
        self.true_sum += torch.sum(true_binary).item()
        self.pred_sum += torch.sum(pred_binary).item()

    def compute(self):
        """Return final metrics: RMSE, L1, IoU, Dice."""
        rmse = (self.sum_se / self.n_samples) ** 0.5 if self.n_samples > 0 else 0
        l1 = self.sum_ae / self.n_samples if self.n_samples > 0 else 0
        iou = self.intersection / self.union if self.union > 0 else 1.0
        dice = (2 * self.intersection) / (self.true_sum + self.pred_sum) if (self.true_sum + self.pred_sum) > 0 else 1.0
        return {"RMSE": rmse, "L1": l1, "IoU": iou, "Dice": dice}

class NumpyMetrics:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.sum_se = 0.0
        self.sum_ae = 0.0
        self.intersection = 0
        self.union = 0
        self.true_sum = 0
        self.pred_sum = 0
        self.n_samples = 0

    def update(self, true: np.ndarray, pred: np.ndarray):
        """Accumulate metrics for true vs predicted arrays."""
        diff = true - pred
        self.sum_se += np.sum(diff * diff)
        self.sum_ae += np.sum(np.abs(diff))
        self.n_samples += diff.size

        true_binary = true > self.threshold
        pred_binary = pred > self.threshold
        self.intersection += np.sum(true_binary & pred_binary)
        self.union += np.sum(true_binary | pred_binary)
        self.true_sum += np.sum(true_binary)
        self.pred_sum += np.sum(pred_binary)

    def compute(self):
        """Return final metrics: RMSE, L1, IoU, Dice."""
        rmse = (self.sum_se / self.n_samples) ** 0.5 if self.n_samples > 0 else 0
        l1 = self.sum_ae / self.n_samples if self.n_samples > 0 else 0
        iou = self.intersection / self.union if self.union > 0 else 1.0
        dice = (2 * self.intersection) / (self.true_sum + self.pred_sum) if (self.true_sum + self.pred_sum) > 0 else 1.0
        return {"RMSE": rmse, "L1": l1, "IoU": iou, "Dice": dice}


