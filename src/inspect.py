import numpy as np, torch
from pathlib import Path
from src.plot import plot_topography, plot_gravity_measurements, plot_density_contrast_3D, plot_gravity_residuals, plot_density_slices, plot_density_slice_residuals
from src.gen.StructuralGeo_gen import create_mesh
from src.io.hdf5_i import MasterReader
from src.load import MasterDataset
from src.transform import make_transform
from src.nn import GravInvNet
from src.transform import *
from simpeg.potential_fields import gravity
from src.gen.StructuralGeo_gen import gravity_survey, init_model


def inspect_truth(h5_path: Path, seed_index: int = 0):
    """Read one sample and return stuff."""
    ds = MasterDataset(h5_path) 
    nx, ny, nz = map(int, ds.shape_cells)
    with MasterReader(h5_path) as mr:
        seed = mr.list_seeds()[seed_index]
        s = mr.read(seed)
    rx, gz = s["receiver_locations"], s["gz"]
    ind = s["ind_active"].astype(bool)                 
    true = np.zeros_like(ind, float)
    true[ind] = s["true_model"]
    mesh = create_mesh(bounds=((0, nx*ds.hx[0]), (0, ny*ds.hy[0]), (0, nz*ds.hz[0])), resolution=(nx, ny, nz))
    return s, rx, gz, (nx, ny, nz), ind, true, mesh

@torch.no_grad()
def inspect_prediction(sample: dict, shape_cells, stats,device, net: GravInvNet):
    """Run net on sample and return prediction."""
    x, _, _, _ = make_transform(shape_cells, stats)(sample) # (1,H,W)
    x = x.unsqueeze(0).to(device)                           # (1,1,H,W)
    net.eval()
    pred = net(x)[0]                                        # (Z,H,W)
    pred_full = pred.permute(2,1,0).cpu().numpy()           # (nx,ny,nz)
    return pred_full.reshape(-1)                            # (nx*ny*nz,)

def main():
    data = np.load("splits/sg.npz")
    va_indices = data["va"]
    tr_indices = data["tr"]
    path = Path("datasets/sg.h5")
    sample, rx, gz, shape, ind, true, mesh = inspect_truth(path, seed_index=tr_indices[3])
    # plot_topography(rx)
    # plot_gravity_measurements(rx, gz)
    # plot_density_contrast_3D(mesh, ind, true)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = GravInvNet().to(device)
    ckpt_path = Path("weights/sg.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded trained model from {ckpt_path}")
    else:
        print("Using untrained model.")
    stats = compute_stats(str(path))
    pred = inspect_prediction(sample, shape, stats, device, net)
    pred = denorm(pred, stats)
    pred = pred.numpy()           
    # plot_density_contrast_3D(mesh, ind, pred)
    _, survey = gravity_survey(rx, components=("gz",))
    _, _, model_map, _ = init_model(mesh, rx, 0)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )
    y = sim.dpred(pred)
    plot_gravity_measurements(rx, y)
    plot_gravity_residuals(rx, gz, y)
    plot_density_slices(mesh, ind, pred, slice_type='y')
    plot_density_slices(mesh, ind, true, slice_type='y')
    plot_density_slice_residuals(mesh, ind, true, pred, slice_type='y')

    l2 = np.mean((true - pred)**2)
    rmse = np.sqrt(l2)
    l1 = np.mean(np.abs(true - pred))
    
    # 3D IoU and Dice calculation (threshold at 0.1 g/cc)
    threshold = 0.1
    true_binary = true > threshold
    pred_binary = pred > threshold
    
    intersection = np.sum(true_binary & pred_binary)
    union = np.sum(true_binary | pred_binary)
    true_sum = np.sum(true_binary)
    pred_sum = np.sum(pred_binary)
    
    iou_3d = intersection / union if union > 0 else 1.0
    dice_3d = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
   
    print(f"RMSE: {rmse:.4f}")
    print(f"L1: {l1:.4f}")
    print(f"3D IoU: {iou_3d:.3f}")
    print(f"3D Dice: {dice_3d:.3f}")

if __name__ == "__main__":
    main()