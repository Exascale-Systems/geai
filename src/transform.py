import torch

def make_transform(shape_cells, stats):
    nx, ny, nz = map(int, shape_cells)

    def norm(a, a0, a1): return 2*(a - a0)/(a1 - a0) - 1

    def to_tensors(sample):
        gz = norm(sample["gz"], stats["gz_min"], stats["gz_max"])
        tm = norm(sample["true_model"], stats["rho_min"], stats["rho_max"])
        x = torch.from_numpy(gz).float().view(1, ny, nx)
        y = torch.from_numpy(tm).float().reshape(nx, ny, nz).permute(2,1,0).contiguous()
        m = torch.from_numpy(sample["ind_active"]).bool().reshape(nx, ny, nz).permute(2,1,0).contiguous()
        return x, y, m, sample["seed"]

    return to_tensors   