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
from simpeg import (
    data,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    data_misfit,
)

def inspect_truth(h5_path: Path, seed_index: int = 0):
    """
    Load sample defined by seed from dataset and return:
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

def main():
    split_data = np.load("splits/single_block.npz")
    tr_indices, va_indices = split_data["tr"], split_data["va"]
    path = Path("datasets/single_block.h5")
    sample, rx, gz, shape, ind, true, mesh = inspect_truth(h5_path=path, seed_index=va_indices[3])
    _, survey = gravity_survey(rx, components=("gz",))
    _, _, model_map, _ = init_model(mesh, rx, 0)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )
    synthetic_model = np.zeros(int(ind.sum())) 
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
    recovered_model = inv.run(synthetic_model)
    y = sim.dpred(synthetic_model)
    iou, dice = iou_dice(true, synthetic_model, 0.1)
   
    plot_topography(rx)
    plot_gravity_measurements(rx, gz)
    plot_density_contrast_3D(mesh, ind, true)
    plot_density_slices(mesh, ind, true, slice_type='y')
    # plot_gravity_residuals(rx, gz, x)
    # plot_gravity_measurements(rx, x)
    plot_density_contrast_3D(mesh, ind, recovered_model)
    plot_density_slices(mesh, ind, recovered_model, slice_type='y')
    plot_gravity_measurements(rx, y)
    plot_gravity_residuals(rx, gz, y)
    # plot_density_slice_residuals(mesh, ind, true, pred, slice_type='y')

    # import matplotlib.pyplot as plt
    # plt.hist(gz, bins=35)
    # plt.show()

    # plt.hist(x, bins=35)
    # plt.show()

    # plt.hist(y, bins=35)
    # plt.show()

    # print(gz.min(), gz.max())
    # print(x.min(), x.max())
    # print(y.min(), y.max())
    
    input("Press Enter to close all plots...")  # Keep plots open until user input
    print(f"RMSE: {rmse(true, synthetic_model):.4f}")
    print(f"L1: {l1(true, synthetic_model):.4f}")
    print(f"IoU: {iou:.3f}")
    print(f"Dice: {dice:.3f}")

if __name__ == "__main__":
    main()