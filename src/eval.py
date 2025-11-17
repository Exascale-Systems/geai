import numpy as np
from scipy.stats import norm
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.gen.StructuralGeo_gen import gravity_survey, init_model
from src.plot import *
from src.gen.StructuralGeo_gen import create_mesh
from src.utils import load_model, denorm, TorchMetrics, NumpyMetrics
from src.data import data_prep
from simpeg.potential_fields import gravity
from simpeg import data, inverse_problem, regularization, optimization, directives, inversion, data_misfit

def eval_nn(net: torch.nn.Module, dl: torch.utils.data.DataLoader,  
              stats: dict, device: torch.device, idx: int = None, threshold: float = 0.1):
    """
    Neural network prediction sampling.
    """
    net.eval()
    metrics = TorchMetrics(stats, threshold)
    with torch.no_grad():
        if idx is None:
            desc = f"Evaluating NN"   
            iterator = tqdm(dl, desc=desc, leave=False, ncols=80)
            for x, true_y in iterator:
                x, true_y = x.to(device), true_y.to(device)
                metrics.update(net, x, true_y, None)  # No denorm needed since data not normalized
            results = metrics.compute()
            rmse_val, l1_val, iou_val, dice_val = results["RMSE"], results["L1"], results["IoU"], results["Dice"]
        else:
            ds = dl.dataset
            x, true_y, _, _ = ds[idx]
            x, true_y = x.unsqueeze(0).to(device), true_y.unsqueeze(0).to(device)
            metrics.update(net, x, true_y, None)
            results = metrics.compute()
            rmse_val, l1_val, iou_val, dice_val = results["RMSE"], results["L1"], results["IoU"], results["Dice"]
            pred_y = net(x)
            x = x.squeeze().cpu().numpy().flatten(order="F")  # denorm(x.squeeze(), stats, data_type="gz").cpu().numpy().flatten(order="F") 
            pred_y = pred_y[0].permute(2,1,0).reshape(-1).cpu().numpy().flatten(order="F")  # denorm(pred_y[0].permute(2,1,0).reshape(-1), stats, data_type="rho").cpu().numpy().flatten(order="F")
            plt.hist(pred_y, bins=50, alpha=0.5, label="Predicted"
                     )
            plt.show()
            ds.dataset.transform = None
            g_idx = ds.indices[idx]
            sample_data = ds.dataset[g_idx]
            rx, true_x = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
            ind = sample_data['ind_active'].cpu().numpy().astype(bool)
            true_y = np.zeros_like(ind, float)
            true_y[ind] = sample_data['true_model'].cpu().numpy()
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
            preq_x = sim.dpred(pred_y)
            plot_gravity_measurements(rx, true_x, title="True Gravity Data (gz)")
            plot_density_contrast_3D(mesh, ind, true_y)
            plot_gravity_measurements(rx, x, title="Input Gravity Data (x)")
            plot_density_contrast_3D(mesh, ind, pred_y)
            plot_gravity_measurements(rx, preq_x, title="Predicted Gravity Data (y)")
            input("Press Enter to close plots...")
    return {
        'rmse': rmse_val,
        'l1': l1_val,
        'iou': iou_val,
        'dice': dice_val,
        'n_samples': 1 if idx is not None else len(dl.dataset)
    }

def eval_hybrid(net: torch.nn.Module, dl: torch.utils.data.DataLoader, stats: dict, 
                device: torch.device, idx: int = None, threshold: float = 0.1, 
                max_samples: int = None, accuracy: float = 0.001, confidence: float = 0.95):
    """
    Hybrid NN-Bayesian inversion: Use NN prediction as initial model (m0) for SimPEG.
    """
    net.eval()
    z = (1.0 + confidence) / 2.0
    z = np.clip(z, 1e-10, 1 - 1e-10)
    ppf_z = norm.ppf(z)
    sigma = accuracy / ppf_z
    metrics = NumpyMetrics(threshold)
    if idx is None:
        desc = "Evaluating Hybrid"
        iterator = tqdm(dl, desc=desc, leave=False, ncols=80)        
        for local_idx, (x, true_y) in enumerate(iterator):
            if max_samples is not None and local_idx >= max_samples:
                break
            with torch.no_grad():
                x_tensor = x.to(device)
                nn_pred = net(x_tensor)
                nn_m0 = nn_pred[0].permute(2,1,0).reshape(-1).cpu().numpy().flatten(order="F")  # denorm(nn_pred[0].permute(2,1,0).reshape(-1), stats, data_type="rho").cpu().numpy().flatten(order="F")
            x = x.squeeze(0).cpu().numpy().flatten()  # denorm(x.squeeze(0), stats, "gz").cpu().numpy().flatten()
            true_y = true_y.squeeze(0).cpu().numpy().flatten(order="F")  # denorm(true_y.squeeze(0), stats, "rho").cpu().numpy().flatten(order="F")
            ds = dl.dataset.dataset
            ds.transform = None
            g_idx = dl.dataset.indices[local_idx]
            sample_data = ds[g_idx]
            rx, true_x = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
            ind = sample_data['ind_active'].cpu().numpy().astype(bool)
            mesh = create_mesh(bounds=((0, ds.shape_cells[0]*ds.hx[0]), 
                                    (0, ds.shape_cells[1]*ds.hy[0]), 
                                    (0, ds.shape_cells[2]*ds.hz[0])), 
                            resolution=ds.shape_cells)
            _, survey = gravity_survey(rx, components=("gz",))
            _, _, model_map, _ = init_model(mesh, rx, 0)
            sim = gravity.simulation.Simulation3DIntegral(
                survey=survey, mesh=mesh, rhoMap=model_map, active_cells=ind, engine="choclo"
                )
            data_object = data.Data(survey, dobs=x, noise_floor=sigma)
            dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
            reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map)
            opt = optimization.ProjectedGNCG(maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, cg_maxiter=10, cg_atol=1e-3)
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
                target_misfit,
            ]
            inv = inversion.BaseInversion(inv_prob, directives_list)
            pred_y = inv.run(nn_m0)
            pred_y = torch.from_numpy(pred_y).numpy()  # denorm(torch.from_numpy(pred_y), stats, "rho").numpy()
            metrics.update(true_y, pred_y)
    else:
        x, true_y, _, _ = dl.dataset[idx]
        with torch.no_grad():
            x_tensor = x.unsqueeze(0).to(device)
            nn_pred = net(x_tensor)
            nn_m0 = nn_pred[0].permute(2,1,0).reshape(-1).cpu().numpy().flatten(order="F")  # denorm(nn_pred[0].permute(2,1,0).reshape(-1), stats, data_type="rho").cpu().numpy().flatten(order="F")
        x = x.squeeze(0).cpu().numpy().flatten()  # denorm(x.squeeze(0), stats, "gz").cpu().numpy().flatten()
        true_y = true_y.squeeze(0).cpu().numpy().flatten(order="F")  # denorm(true_y.squeeze(0), stats, "rho").cpu().numpy().flatten(order="F")
        ds = dl.dataset.dataset
        ds.transform = None
        g_idx = dl.dataset.indices[idx]
        sample_data = ds[g_idx]
        rx, true_x = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
        ind = sample_data['ind_active'].cpu().numpy().astype(bool)
        mesh = create_mesh(bounds=((0, ds.shape_cells[0]*ds.hx[0]), 
                                (0, ds.shape_cells[1]*ds.hy[0]), 
                                (0, ds.shape_cells[2]*ds.hz[0])), 
                        resolution=ds.shape_cells)
        _, survey = gravity_survey(rx, components=("gz",))
        _, _, model_map, _ = init_model(mesh, rx, 0)
        sim = gravity.simulation.Simulation3DIntegral(
                survey=survey, mesh=mesh, rhoMap=model_map, active_cells=ind, engine="choclo"
            )
        data_object = data.Data(survey, dobs=x, noise_floor=sigma)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
        reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map)
        opt = optimization.ProjectedGNCG(maxIter=50, maxIterLS=20, cg_maxiter=10, cg_atol=1e-5)
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
            target_misfit,
        ]
        inv = inversion.BaseInversion(inv_prob, directives_list)
        pred_y = inv.run(nn_m0)
        metrics.update(true_y, pred_y)
        preq_x = sim.dpred(pred_y)
        plot_gravity_measurements(rx, true_x, title="True Gravity Data (gz)")
        plot_density_contrast_3D(mesh, ind, true_y)
        plot_gravity_measurements(rx, x, title="Input Gravity Data (x)")
        plot_density_contrast_3D(mesh, ind, pred_y)
        plot_gravity_measurements(rx, preq_x, title="Predicted Gravity Data (y)")
        input("Press Enter to close plots...")
    results = metrics.compute()
    rmse_val, l1_val, iou_val, dice_val = results["RMSE"], results["L1"], results["IoU"], results["Dice"]
    return {
        'rmse': rmse_val,
        'l1': l1_val,
        'iou': iou_val,
        'dice': dice_val,
        'n_samples': 1 if idx is not None else len(dl.dataset)
    }

def eval_bayesian(dl: torch.utils.data.DataLoader, stats: dict, 
                  idx: int = None, threshold: float = 0.1, max_samples: int = None,
                  accuracy: float = 0.001, confidence: float = 0.95):
    """
    Bayesian inversion (SIMPEG) sampling.
    """
    z = (1.0 + confidence) / 2.0
    z = np.clip(z, 1e-10, 1 - 1e-10)
    ppf_z = norm.ppf(z)
    sigma = accuracy / ppf_z
    metrics = NumpyMetrics(threshold)
    if idx is None:
        desc = "Evaluating Bayesian"
        iterator = tqdm(dl, desc=desc, leave=False, ncols=80)        
        for local_idx, (x, true_y) in enumerate(iterator):
            if max_samples is not None and local_idx >= max_samples:
                break
            x = x.squeeze(0).cpu().numpy().flatten()  # denorm(x.squeeze(0), stats, "gz").cpu().numpy().flatten()
            true_y = true_y.squeeze(0).cpu().numpy().flatten(order="F")  # denorm(true_y.squeeze(0), stats, "rho").cpu().numpy().flatten(order="F")
            ds = dl.dataset.dataset
            ds.transform = None
            g_idx = dl.dataset.indices[local_idx]
            sample_data = ds[g_idx]
            rx, true_x = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
            ind = sample_data['ind_active'].cpu().numpy().astype(bool)
            mesh = create_mesh(bounds=((0, ds.shape_cells[0]*ds.hx[0]), 
                                    (0, ds.shape_cells[1]*ds.hy[0]), 
                                    (0, ds.shape_cells[2]*ds.hz[0])), 
                            resolution=ds.shape_cells)
            _, survey = gravity_survey(rx, components=("gz",))
            _, _, model_map, _ = init_model(mesh, rx, 0)
            sim = gravity.simulation.Simulation3DIntegral(
                survey=survey, mesh=mesh, rhoMap=model_map, active_cells=ind, engine="choclo"
                )
            data_object = data.Data(survey, dobs=x, noise_floor=sigma)
            dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim) # L2 Norm: ||W(d_pred(m) - d_obs)||^2
            reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map) # Tikhonov regularization (smooths it out)
            # reg = regularization.Sparse(mesh, active_cells=ind, mapping=model_map) # Sparse (IRLS) regulatization (encourages sharp features)
            # reg.norms = [0, 2, 2, 2]
            # update_IRLS = directives.UpdateIRLS(
            #     f_min_change=1e-4,
            #     max_irls_iterations=30,
            #     irls_cooling_factor=1.5,
            #     misfit_tolerance=1e-2,
            # )
            opt = optimization.ProjectedGNCG(maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, cg_maxiter=10, cg_atol=1e-3) # Gauss-Newton (Hessian Search)
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
            y = np.zeros(int(ind.sum()))
            pred_y = inv.run(y)
            pred_y = torch.from_numpy(pred_y).numpy()  # denorm(torch.from_numpy(pred_y), stats, "rho").numpy()
            metrics.update(true_y, pred_y)
    else:
        x, true_y, _, _ = dl.dataset[idx]
        x = x.squeeze(0).cpu().numpy().flatten()  # denorm(x.squeeze(0), stats, "gz").cpu().numpy().flatten()
        true_y = true_y.squeeze(0).cpu().numpy().flatten(order="F")  # denorm(true_y.squeeze(0), stats, "rho").cpu().numpy().flatten(order="F")
        ds = dl.dataset.dataset
        ds.transform = None
        g_idx = dl.dataset.indices[idx]
        sample_data = ds[g_idx]
        rx, true_x = sample_data['receiver_locations'].cpu().numpy(), sample_data['gz'].cpu().numpy()
        ind = sample_data['ind_active'].cpu().numpy().astype(bool)
        mesh = create_mesh(bounds=((0, ds.shape_cells[0]*ds.hx[0]), 
                                (0, ds.shape_cells[1]*ds.hy[0]), 
                                (0, ds.shape_cells[2]*ds.hz[0])), 
                        resolution=ds.shape_cells)
        _, survey = gravity_survey(rx, components=("gz",))
        _, _, model_map, _ = init_model(mesh, rx, 0)
        y = np.zeros(int(ind.sum()))
        sim = gravity.simulation.Simulation3DIntegral(
                survey=survey, mesh=mesh, rhoMap=model_map, active_cells=ind, engine="choclo"
            )
        data_object = data.Data(survey, dobs=x, noise_floor=sigma)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim) # L2 Norm: ||W(d_pred(m) - d_obs)||^2
        reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map) # Tikhonov regularization (smooths it out)
        # reg = regularization.Sparse(mesh, active_cells=ind, mapping=model_map) # Sparse (IRLS) regulatization (encourages sharp features)
        # reg.norms = [0, 2, 2, 2]
        # update_IRLS = directives.UpdateIRLS(
        #     f_min_change=1e-4,
        #     max_irls_iterations=30,
        #     irls_cooling_factor=1.5,
        #     misfit_tolerance=1e-2,
        # )
        opt = optimization.ProjectedGNCG(maxIter=50, maxIterLS=20, cg_maxiter=10, cg_atol=1e-4) # Gauss-Newton (Hessian Search)
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
        pred_y = inv.run(y)
        metrics.update(true_y, pred_y)
        preq_x = sim.dpred(pred_y)
        plot_gravity_measurements(rx, true_x, title="True Gravity Data (gz)")
        plot_density_contrast_3D(mesh, ind, true_y)
        plot_gravity_measurements(rx, x, title="Input Gravity Data (x)")
        plot_density_contrast_3D(mesh, ind, pred_y)
        plot_gravity_measurements(rx, preq_x, title="Predicted Gravity Data (y)")
        # plt.hist(true_y, bins=50, alpha=0.5, label="True")
        # plt.hist(pred_y, bins=50, alpha=0.5, label="Predicted")
        # plt.legend()
        # plt.show()
        input("Press Enter to close plots...")
    results = metrics.compute()
    rmse_val, l1_val, iou_val, dice_val = results["RMSE"], results["L1"], results["IoU"], results["Dice"]
    return {
        'rmse': rmse_val,
        'l1': l1_val,
        'iou': iou_val,
        'dice': dice_val,
        'n_samples': 1 if idx is not None else len(dl.dataset)
    }

def eval(eval: str = "nn", split: str = "va", bs: int = 8 if eval == "nn" else 1, 
         idx: int = None, max_samples: int = None, accuracy: float = 0.001, confidence: float = 0.95):
    print(f"Eval: {eval}, Split: {split}, Index: {f'{max_samples if max_samples is not None else "All"}' if idx is None else idx}, Accuracy: {accuracy}, Confidence: {confidence}")
    tr_ld, va_ld, stats = data_prep(ds_name="single_block_v2", split_name="single_block_v2", bs=bs if eval == "nn" else 1, 
                                    accuracy=accuracy, confidence=confidence, load_splits=True, transform=True)
    dl = {"tr": tr_ld, "va": va_ld}.get(split)
    if dl is None:
        print("Invalid split name")
        return None 
    eval_funcs = {
        "nn": lambda: (lambda net, device: eval_nn(net=net, dl=dl, stats=stats, device=device, threshold=0.1, idx=idx))(*load_model(model_name="single_block_05", device="cuda:5")),
        "bayesian": lambda: eval_bayesian(dl=dl, stats=stats, idx=idx, threshold=0.012, max_samples=max_samples, accuracy=accuracy, confidence=confidence),
        "hybrid": lambda: (lambda net, device: eval_hybrid(net=net, dl=dl, stats=stats, device=device, threshold=0.1, idx=idx, max_samples=max_samples, accuracy=accuracy, confidence=confidence))(*load_model(model_name="single_block_05", device="cuda:5"))
    }
    if eval not in eval_funcs: 
        print("Invalid eval")
        return None
    metrics = eval_funcs[eval]()
    print(metrics)
    return metrics

if __name__ == "__main__":
    eval(eval="nn", split="va", idx=None, max_samples=None, accuracy=0.0001, confidence=0.95)
