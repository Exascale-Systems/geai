import torch

def make_transform(shape_cells):
    nx, ny, nz = map(int, shape_cells)

    def to_tensors(sample):
        x = torch.from_numpy(sample["gz"]).float().view(1, ny, nx)
        ind = torch.from_numpy(sample["ind_active"]).bool().reshape(nx, ny, nz)
        tm = torch.from_numpy(sample["true_model"]).reshape(nx, ny, nz)
        y = tm.permute(2, 1, 0).contiguous()
        mask = ind.permute(2, 1, 0).contiguous()
        return x, y, mask, sample["seed"]

    return to_tensors   