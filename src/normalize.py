import h5py, numpy as np, math

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
    return 2*(a - a0)/(a1 - a0) - 1

def denorm(y_norm: np.ndarray, stats: dict) -> np.ndarray:
    a = float(stats["rho_min"])
    b = float(stats["rho_max"])
    y_norm = np.clip(y_norm, -1.0, 1.0)
    return ((y_norm + 1.0) * 0.5) * (b - a) + a