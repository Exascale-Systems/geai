from typing import Any, cast

import numpy as np
import torch
from scipy.stats import norm
from simpeg import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    optimization,
    regularization,
)
from torch.utils.data import Subset
from tqdm import tqdm

from src.evaluation import (
    plot_density_contrast_3D,
    plot_gravity_measurements,
)
from src.evaluation.metrics import NumpyMetrics
from src.gen.core import sim_from_sample


def eval_hybrid(
    net: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    stats: dict,
    device: torch.device,
    idx: int | None = None,
    threshold: float = 0.1,
    max_samples: int | None = None,
    accuracy: float = 0.001,
    confidence: float = 0.95,
):
    """
    Hybrid NN-Bayesian inversion: Use NN prediction as initial model (m0) for SimPEG.
    """
    net.eval()
    z = (1.0 + confidence) / 2.0
    z = np.clip(z, 1e-10, 1 - 1e-10)
    ppf_z = norm.ppf(z)
    sigma = accuracy / ppf_z
    metrics = NumpyMetrics(threshold)

    if isinstance(dl.dataset, Subset):
        ds_dataset = dl.dataset.dataset
    else:
        ds_dataset = dl.dataset
    components = getattr(ds_dataset, "components", ("gz",))

    if idx is None:
        iterator = tqdm(dl, desc="Evaluating Hybrid", leave=False, ncols=80)
        for local_idx, (x, true_y) in enumerate(iterator):
            if max_samples is not None and local_idx >= max_samples:
                break
            with torch.no_grad():
                x_tensor = x.to(device)
                nn_pred = net(x_tensor)
                nn_m0 = (
                    nn_pred[0].permute(2, 1, 0).reshape(-1).cpu().numpy().flatten(order="F")
                )

            ds = cast(Subset, dl.dataset)
            ds_dataset = cast(Any, ds.dataset)
            old_transform = ds_dataset.transform
            ds_dataset.transform = None
            g_idx = ds.indices[local_idx]
            sample_data = ds_dataset[g_idx]
            ds_dataset.transform = old_transform

            x_obs_flat = x.squeeze(0).cpu().numpy().reshape(x.shape[1], -1).flatten()

            sim, mesh, survey, model_map, ind = sim_from_sample(
                sample_data,
                ds_dataset.shape_cells,
                (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
                components=components,
            )

            data_object = data.Data(survey, dobs=x_obs_flat, noise_floor=sigma)
            dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
            reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map)
            opt = optimization.ProjectedGNCG(
                maxIter=10, lower=-1.0, upper=1.0, maxIterLS=20, cg_maxiter=10, cg_atol=1e-3,
            )
            inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
            directives_list = [
                directives.UpdateSensitivityWeights(every_iteration=False),
                directives.BetaEstimate_ByEig(beta0_ratio=1e1),
                directives.BetaSchedule(coolingFactor=5, coolingRate=1),
                directives.UpdatePreconditioner(),
                directives.TargetMisfit(chifact=1),
            ]
            inv = inversion.BaseInversion(inv_prob, directives_list)
            pred_y = inv.run(nn_m0)
            pred_y = torch.from_numpy(pred_y).numpy()

            true_y_flat = true_y.squeeze(0).cpu().numpy().flatten(order="F")
            metrics.update(true_y_flat, pred_y)

    else:
        ds = cast(Subset, dl.dataset)
        x, true_y, _, _ = cast(tuple, ds[idx])

        with torch.no_grad():
            x_tensor = x.unsqueeze(0).to(device)
            nn_pred = net(x_tensor)
            nn_m0 = (
                nn_pred[0].permute(2, 1, 0).reshape(-1).cpu().numpy().flatten(order="F")
            )

        ds_dataset = cast(Any, ds.dataset)
        ds_dataset.transform = None
        g_idx = ds.indices[idx]
        sample_data = ds_dataset[g_idx]

        sim, mesh, survey, model_map, ind = sim_from_sample(
            sample_data,
            ds_dataset.shape_cells,
            (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
            components=components,
        )

        x_obs_flat = x.cpu().numpy().reshape(x.shape[0], -1).flatten()

        data_object = data.Data(survey, dobs=x_obs_flat, noise_floor=sigma)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
        reg = regularization.WeightedLeastSquares(mesh, active_cells=ind, mapping=model_map)
        opt = optimization.ProjectedGNCG(maxIter=50, maxIterLS=20, cg_maxiter=10, cg_atol=1e-5)
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        directives_list = [
            directives.UpdateSensitivityWeights(every_iteration=False),
            directives.BetaEstimate_ByEig(beta0_ratio=1e1),
            directives.BetaSchedule(coolingFactor=5, coolingRate=1),
            directives.UpdatePreconditioner(),
            directives.TargetMisfit(chifact=1),
        ]
        inv = inversion.BaseInversion(inv_prob, directives_list)
        pred_y = inv.run(nn_m0)
        metrics.update(true_y.cpu().numpy().flatten(order="F"), pred_y)

        preq_x = sim.dpred(pred_y)
        rx = sample_data["receiver_locations"].cpu().numpy()
        true_model_np = np.zeros_like(ind, float)
        true_model_np[ind] = sample_data["true_model"].cpu().numpy()

        n_rx = rx.shape[0]
        for i, comp in enumerate(components):
            true_grav = sample_data["gravity"][i].cpu().numpy()
            pred_grav_comp = preq_x[i * n_rx : (i + 1) * n_rx]
            plot_gravity_measurements(rx, true_grav, title=f"True Gravity Data ({comp})")
            plot_gravity_measurements(rx, pred_grav_comp, title=f"Predicted Gravity Data ({comp})")

        plot_density_contrast_3D(mesh, ind, true_model_np)
        plot_density_contrast_3D(mesh, ind, pred_y)
        input("Press Enter to close plots...")

    results = metrics.compute()
    return {
        "rmse": results["RMSE"],
        "l1": results["L1"],
        "iou": results["IoU"],
        "dice": results["Dice"],
        "n_samples": 1 if idx is not None else len(cast(Any, dl.dataset)),
    }
