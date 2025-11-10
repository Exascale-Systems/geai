device = "cuda:7"
idx = 3
threshold = 0.1

import numpy as np
import torch
from src.gen.StructuralGeo_gen import gravity_survey, init_model
from src.transform import make_transform, denorm
from src.metrics import rmse, l1, iou_dice
from src.plot import *
from src.utils import load_model
from src.data import load_split_sample
from simpeg.potential_fields import gravity
from simpeg import data, inverse_problem, regularization, optimization, directives, inversion, data_misfit

@torch.no_grad()
def inspect_prediction(sample: dict, net: torch.nn.Module, device: torch.device, shape_cells: tuple, stats: dict, accuracy: float=0.01, 
                       confidence: float=0.95):
    """
    Forward pass (Evaluate) sample and return prediction and input x with same shape as y.
    """
    net.eval()
    x, _, _, _ = make_transform(shape_cells, stats, (accuracy, confidence))(sample)
    y = denorm(net(x.unsqueeze(0).to(device))[0].permute(2,1,0).reshape(-1), stats, data_type="rho").cpu().numpy()    
    x = denorm(x.squeeze(), stats, data_type="gz").cpu().numpy().flatten()
    return y, x

def sample_nn(data_path: str = "data/single_block.h5", split_path: str = "splits/single_block.npz", accuracy: float = 0.01, confidence: float = 0.95,
              split: str = "val", idx: int = idx, model_path: str = "models/single_block.pt", threshold: float = threshold, device: torch.device = device):
    """
    Neural network prediction sampling.
    """
    data, stats = load_split_sample(data_path, split_path, split, idx)
    sample, rx, gz, shape, ind, true, mesh = data, data['rx'], data['gz'], data['shape'], data['ind_active'], data['true_model'], data['mesh']
    net, device = load_model(model_path, device)
    pred, x = inspect_prediction(sample, net, device, shape, stats, accuracy, confidence)
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
    iou, dice = iou_dice(true, pred, threshold)
    plot_gravity_measurements(rx, gz, title="True Gravity Data (gz)")
    plot_density_contrast_3D(mesh, ind, true)
    plot_gravity_measurements(rx, x, title="Input Gravity Data (x)")
    plot_density_contrast_3D(mesh, ind, pred)
    plot_gravity_measurements(rx, x_, title="Predicted Gravity Data (y)")
    input("Press Enter to close plots...")
    print(f"RMSE: {rmse(true, pred):.4f}")
    print(f"L1: {l1(true, pred):.4f}")
    print(f"IoU: {iou:.3f}")
    print(f"Dice: {dice:.3f}")

def sample_bayesian(data_path: str = "data/single_block.h5", split_path: str = "splits/single_block.npz", 
              split: str = "val", idx: int = idx, threshold: float = threshold):
    """
    Bayesian inversion (SIMPEG) sampling.
    """
    sample_data, _ = load_split_sample(data_path, split_path, split, idx)
    rx, gz, ind, true, mesh = sample_data['rx'], sample_data['gz'], sample_data['ind_active'], sample_data['true_model'], sample_data['mesh']
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
        maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
    beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
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
    # sample_nn(data_path="data/single_block.h5", split_path="splits/single_block.npz", accuracy=0.01, confidence=0.95,
    #           split="val", idx=3, model_path="models/single_block.pt", threshold=0.5)
    sample_bayesian(data_path="data/single_block.h5", split_path="splits/single_block.npz",
                    split="val", idx=3, threshold=0.5)