import numpy as np
import torch
from pathlib import Path
from src.gen.StructuralGeo_gen import *
from src.io.hdf5_i import MasterReader
from src.load import MasterDataset
from src.transform import make_transform
from src.metrics import *
from src.nn import GravInvNet
from src.transform import *
from src.plot import *
from simpeg.potential_fields import gravity

def inspect_truth(h5_path: Path, seed_index: int = 0):
    """Load sample defined by seed from dataset and return:
        - seed
        - receiver locations
        - gravity values
        - shape
        - boolean array of active cells
        - true density
        - empty mesh 
     """
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
def inspect_prediction(sample: dict, shape_cells: tuple, stats: dict, net: GravInvNet, device: torch.device):
    """Forward pass (Evaluate) sample and return prediction and input x with same shape as y."""
    x, y, _, _ = make_transform(shape_cells, stats, (0.01, 0.95))(sample)
    x_input = x.unsqueeze(0).to(device)                          
    net.eval()
    pred = denorm(net(x_input)[0].permute(2,1,0).reshape(-1), stats, data_type="rho")    
    x_reshaped = denorm(x.squeeze(), stats, data_type="gz").cpu().numpy().flatten()
    return pred, x_reshaped

def main():
    data = np.load("splits/single_block.npz")
    tr_indices, va_indices = data["tr"], data["va"]
    path = Path("datasets/single_block.h5")
    sample, rx, gz, shape, ind, true, mesh = inspect_truth(h5_path=path, seed_index=va_indices[3])
    device = torch.device("cuda:7")
    net = GravInvNet().to(device)
    ckpt_path = Path("weights/single_block.pt")
    if ckpt_path.exists():
        state = torch.load(f=ckpt_path, map_location=device)
        net.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded trained model from {ckpt_path}")
    else:
        print("Using untrained model.")
    stats = compute_stats(str(path))
    pred, x = inspect_prediction(sample, shape, stats, net, device)
    _, survey = gravity_survey(rx, components=("gz",))
    _, _, model_map, _ = init_model(mesh, rx, 0)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )
    pred = pred.cpu().numpy()
    y = sim.dpred(pred)
    iou, dice = iou_dice(true, pred, 0.1)
   
    # plot_topography(rx)
    plot_gravity_measurements(rx, gz)
    plot_density_contrast_3D(mesh, ind, true)
    # plot_density_slices(mesh, ind, true, slice_type='y')
    # plot_gravity_residuals(rx, gz, x)

    plot_gravity_measurements(rx, x)
    plot_density_contrast_3D(mesh, ind, pred)
    # plot_density_slices(mesh, ind, pred, slice_type='y')
    plot_gravity_measurements(rx, y)

    # plot_gravity_residuals(rx, x, y)
    # plot_density_slice_residuals(mesh, ind, true, pred, slice_type='y')

    import matplotlib.pyplot as plt
    plt.hist(gz, bins=35)
    plt.show()

    plt.hist(x, bins=35)
    plt.show()

    plt.hist(y, bins=35)
    plt.show()

    print(gz.min(), gz.max())
    print(x.min(), x.max())
    print(y.min(), y.max())
    
    input("Press Enter to close all plots...")  # Keep plots open until user input
    print(f"RMSE: {rmse(true, pred):.4f}")
    print(f"L1: {l1(true, pred):.4f}")
    print(f"IoU: {iou:.3f}")
    print(f"Dice: {dice:.3f}")

if __name__ == "__main__":
    main()