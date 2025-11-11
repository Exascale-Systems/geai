device = "cuda:7"
idx = 3
threshold = 0.1

import numpy as np
import torch
from src.gen.StructuralGeo_gen import gravity_survey, init_model
from src.data import make_transform
from src.utils import denorm
from src.plot import *
from src.gen.StructuralGeo_gen import create_mesh
from src.utils import load_model, rmse, l1, iou_dice
from src.data import data_prep
from simpeg.potential_fields import gravity
from simpeg import data, inverse_problem, regularization, optimization, directives, inversion, data_misfit

def sample_nn(data_path: str = "single_block", split_path: str = "single_block", accuracy: float = 0.01, confidence: float = 0.95,
              split: str = "va", idx: int = idx, model_path: str = "models/single_block.pt", threshold: float = threshold, device: torch.device = device):
    """
    Neural network prediction sampling.
    """
    tr_ld, va_ld, stats = data_prep(data_path, split_path, 1, load_splits=True, transform=True, accuracy=accuracy, confidence=confidence)
    ds = va_ld.dataset if split == "va" else tr_ld.dataset
    net, device = load_model(model_path, device)
    net.eval()
    sum_se, sum_ae, n_samples = 0.0, 0.0, 0
    intersection, union, true_sum, pred_sum = 0, 0, 0, 0
    with torch.no_grad():
        x, tgt, _, _ = ds[idx]
        gz, tgt = x.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
        pred = net(gz)
        pred_denorm = denorm(pred, stats)
        tgt_denorm = denorm(tgt, stats)
        diff = tgt_denorm - pred_denorm
        sum_se += torch.sum(diff * diff).item()
        sum_ae += torch.sum(torch.abs(diff)).item()
        n_samples += diff.numel()
        true_binary = tgt_denorm > threshold
        pred_binary = pred_denorm > threshold
        intersection += torch.sum(true_binary & pred_binary).item()
        union += torch.sum(true_binary | pred_binary).item()
        true_sum += torch.sum(true_binary).item()
        pred_sum += torch.sum(pred_binary).item()
        rmse_val = (sum_se / n_samples) ** 0.5
        l1_val = sum_ae / n_samples
        iou_val = intersection / union if union > 0 else 1.0
        dice_val = (2 * intersection) / (true_sum + pred_sum) if (true_sum + pred_sum) > 0 else 1.0
    x = denorm(x.squeeze(), stats, data_type="gz").cpu().numpy().flatten() 
    pred = denorm(pred[0].permute(2,1,0).reshape(-1), stats, data_type="rho").cpu().numpy()    
    ds.dataset.transform = None
    g_idx = ds.indices[idx]
    sample_data = ds.dataset[g_idx]
    rx, gz = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
    ind = sample_data['ind_active'].cpu().numpy().astype(bool)
    true = np.zeros_like(ind, float)
    true[ind] = sample_data['true_model'].cpu().numpy()
    nx, ny, nz = map(int, ds.dataset.shape_cells)
    mesh = create_mesh(bounds=((0, nx*ds.dataset.hx[0]), (0, ny*ds.dataset.hy[0]), (0, nz*ds.dataset.hz[0])), resolution=(nx, ny, nz))
    _, survey = gravity_survey(rx, components=("gz",))
    _, _, model_map, _ = init_model(mesh, rx, 0)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )
    x_ = sim.dpred(pred)
    plot_gravity_measurements(rx, gz, title="True Gravity Data (gz)")
    plot_density_contrast_3D(mesh, ind, true)
    plot_gravity_measurements(rx, x, title="Input Gravity Data (x)")
    plot_density_contrast_3D(mesh, ind, pred)
    plot_gravity_measurements(rx, x_, title="Predicted Gravity Data (y)")
    input("Press Enter to close plots...")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"L1: {l1_val:.4f}")
    print(f"IoU: {iou_val:.3f}")
    print(f"Dice: {dice_val:.3f}")

def sample_bayesian(data_path: str = "data/single_block.h5", split_path: str = "splits/single_block.npz", 
              split: str = "val", idx: int = idx, threshold: float = threshold):
    """
    Bayesian inversion (SIMPEG) sampling.
    """
    dataset_name = data_path.split('/')[-1].replace('.h5', '')
    tr_ld, va_ld, stats = data_prep(dataset_name, dataset_name, 1, load_splits=True, transform=True)
    ds = va_ld.dataset if split == "va" else tr_ld.dataset
    sample_data = ds[idx]
    rx, gz = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
    ind = sample_data['ind_active'].cpu().numpy().astype(bool)
    true = np.zeros_like(ind, float)
    true[ind] = sample_data['true_model'].cpu().numpy()
    nx, ny, nz = map(int, ds.dataset.shape_cells)
    mesh = create_mesh(bounds=((0, nx*ds.dataset.hx[0]), (0, ny*ds.dataset.hy[0]), (0, nz*ds.dataset.hz[0])), resolution=(nx, ny, nz))
    _, survey = gravity_survey(rx, components=("gz",))
    _, _, model_map, _ = init_model(mesh, rx, 0)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )
    y = np.zeros(int(ind.sum())) 
    data_object = data.Data(survey, dobs=gz, noise_floor=1e-6)
    # L2 Norm: ||W(d_pred(m) - d_obs)||^2
    dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim) 
    reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map) # Tikhonov regularization (smooths it out) <-- lowkey unrealistic... good for macro density contrast
    # # Sparse (IRLS) regulatization (encourages sharp features) <-- more realistic
    # reg = regularization.Sparse(mesh, active_cells=ind, mapping=model_map)
    # reg.norms = [0, 2, 2, 2]
    # update_IRLS = directives.UpdateIRLS(
    #     f_min_change=1e-4,
    #     max_irls_iterations=30,
    #     irls_cooling_factor=1.5,
    #     misfit_tolerance=1e-2,
    # )
    # Gauss-Newton (Hessian Search)
    opt = optimization.ProjectedGNCG(
        maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, cg_maxiter=10, cg_atol=1e-3
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    save_iteration = directives.SaveOutputEveryIteration(on_disk=False)
    update_jacobi = directives.UpdatePreconditioner()
    target_misfit = directives.TargetMisfit(chifact=1)
    sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
    directives_list = [
        sensitivity_weights,
        starting_beta,
        beta_schedule,
        save_iteration,
        update_jacobi,
        target_misfit, # comment out for IRLS
    ]
    inv = inversion.BaseInversion(inv_prob, directives_list)
    y = inv.run(y)
    x_ = sim.dpred(y)
    iou, dice = iou_dice(true, y, threshold)
    plot_gravity_measurements(rx, gz, title="True Gravity Data (gz)")
    plot_density_contrast_3D(mesh, ind, true)
    plot_gravity_measurements(rx, data_object.dobs, title="Input Gravity Data (x)")
    plot_density_contrast_3D(mesh, ind, y)
    plot_gravity_measurements(rx, x_, title="Predicted Gravity Data (y)")
    input("Press Enter to close plots...")
    print(f"RMSE: {rmse(true, y):.4f}")
    print(f"L1: {l1(true, y):.4f}")
    print(f"IoU: {iou:.3f}")
    print(f"Dice: {dice:.3f}")

if __name__ == "__main__":
    sample_nn(data_path="single_block", split_path="single_block", accuracy=0.01, confidence=0.95,
              split="va", idx=3, model_path="models/single_block.pt", threshold=0.5)
    # sample_bayesian(data_path="single_block", split_path="single_block",
    #                 split="va", idx=3, threshold=0.5)