import h5py, numpy as np, math

def compute_stats(h5_path, use_mask=False):
    gz_min, gz_max, rho_min, rho_max = math.inf, -math.inf, math.inf, -math.inf
    with h5py.File(h5_path, "r") as f:
        for s in f["samples"].values():
            gz = s["gz"][()]
            gz_min, gz_max = min(gz_min, gz.min()), max(gz_max, gz.max())
            tm = s["true_model"][()]
            vals = tm[s["ind_active"][()].astype(bool)] if use_mask else tm
            if vals.size:
                rho_min, rho_max = min(rho_min, vals.min()), max(rho_max, vals.max())
    return dict(gz_min=gz_min, gz_max=gz_max, rho_min=rho_min, rho_max=rho_max)