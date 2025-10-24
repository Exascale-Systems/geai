import numpy as np, torch
from pathlib import Path
from src.viz.samples import plot_topography, plot_gravity_measurements, plot_density_contrast_3D_voxels
from src.gen.gen import create_mesh
from src.io.hdf5_i import MasterReader
from src.load import MasterDataset
from src.transform import make_transform
from src.nn import GravInvNet
from src.normalize import compute_stats, denorm

def inspect_truth(h5_path: Path, seed_index: int = 1):
    """Read one sample and return stuff."""
    ds = MasterDataset(h5_path); nx, ny, nz = map(int, ds.shape_cells)
    with MasterReader(h5_path) as mr:
        seed = mr.list_seeds()[seed_index]
        s = mr.read(seed)
    rx, gz = s["receiver_locations"], s["gz"]
    ind = s["ind_active"].astype(bool)                 
    true = np.zeros_like(ind, float)
    true[ind] = s["true_model"]
    z_dom = nz * ds.hz[0]           
    mesh = create_mesh(np.flip(rx, 0), n_xy=nx, n_z=nz, z_dom=z_dom)
    return s, rx, gz, (nx, ny, nz), ind, true, mesh

@torch.no_grad()
def inspect_prediction(sample: dict, shape_cells, stats,device, net: GravInvNet):
    """Run net on sample and return prediction."""
    x, _, _, _ = make_transform(shape_cells, stats)(sample)        # (1,H,W)
    x = x.unsqueeze(0).to(device)                           # (1,1,H,W)
    net.eval()
    pred = net(x)[0]                                        # (Z,H,W)
    pred_full = pred.permute(2,1,0).cpu().numpy()           # (nx,ny,nz)
    return pred_full.reshape(-1)                            # (nx*ny*nz,)

def main():
    path = Path("data/master.h5")
    sample, rx, gz, shape, ind, true, mesh = inspect_truth(path, seed_index=0)
    # plot_topography(rx)
    # plot_gravity_measurements(rx, gz)
    plot_density_contrast_3D_voxels(mesh, ind, true > 0.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GravInvNet().to(device)
    ckpt_path = Path("checkpoints/overfit.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded trained model from {ckpt_path}")
    else:
        print("Using untrained model.")
    stats = compute_stats(str(path))
    pred_flat = inspect_prediction(sample, shape, stats, device, net)
    pred_phys_flat = denorm(pred_flat, stats)           # physical units (e.g., g/cc)
    block_flat = pred_phys_flat > 0.2
    plot_density_contrast_3D_voxels(mesh, ind, block_flat)

if __name__ == "__main__":
    main()