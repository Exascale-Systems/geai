import numpy as np
from scipy.stats import norm
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.gen.simulation.factory import create_simulation_from_sample
from src.vis import (
    plot_gravity_measurements,
    plot_density_contrast_3D,
)
from src.training.metrics import TorchMetrics, NumpyMetrics
from src.data.dataset import data_prep
from src.modeling.networks import GravInvNet
from simpeg import (
    data,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
    data_misfit,
)
from torch.utils.data import Dataset, DataLoader, Subset
from typing import cast, Any, Union


def load_model(model_name=None, device="cpu", in_channels=1, checkpoint_path=None):
    device = torch.device(device)
    model = GravInvNet(in_channels=in_channels).to(device)

    # Determine which checkpoint to load
    if checkpoint_path:
        ckpt = checkpoint_path
    elif model_name:
        ckpt = f"checkpoints/{model_name}_final.pt"
    else:
        ckpt = "checkpoints/default_model_final.pt"

    if Path(ckpt).exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model", state))
    model.eval()
    return model, device


def eval_nn(
    net: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    stats: dict,
    device: torch.device,
    idx: int | None = None,
    threshold: float = 0.1,
    headless: bool = False,
):
    """
    Neural network prediction sampling.
    """
    net.eval()
    metrics = TorchMetrics(stats, threshold)

    # Retrieve components from dataset
    # dl.dataset is Subset, dl.dataset.dataset is MasterDataset
    if isinstance(dl.dataset, Subset):
        ds_dataset = dl.dataset.dataset
    else:
        ds_dataset = dl.dataset
    components = getattr(ds_dataset, "components", ("gz",))

    with torch.no_grad():
        if idx is None:
            desc = f"Evaluating NN"
            iterator = tqdm(dl, desc=desc, leave=False, ncols=80)
            for x, true_y in iterator:
                x, true_y = x.to(device), true_y.to(device)
                metrics.update(
                    net, x, true_y, None
                )  # No denorm needed since data not normalized
            results = metrics.compute()
            rmse_val, l1_val, iou_val, dice_val = (
                results["RMSE"],
                results["L1"],
                results["IoU"],
                results["Dice"],
            )
        else:
            ds = cast(Subset, dl.dataset)
            x, true_y, _, _ = cast(tuple, ds[idx])
            x, true_y = x.unsqueeze(0).to(device), true_y.unsqueeze(0).to(device)
            metrics.update(net, x, true_y, None)
            results = metrics.compute()
            rmse_val, l1_val, iou_val, dice_val = (
                results["RMSE"],
                results["L1"],
                results["IoU"],
                results["Dice"],
            )
            pred_y = net(x)

            # Prepare for plotting
            pred_y_np = (
                pred_y[0].permute(2, 1, 0).reshape(-1).cpu().numpy().flatten(order="F")
            )

            # Deactivate transform to get raw sample data
            ds_dataset = cast(Any, ds.dataset)
            ds_dataset.transform = None
            g_idx = ds.indices[idx]
            sample_data = ds_dataset[g_idx]

            # Use factory to create simulation
            sim, mesh, survey, model_map, ind = create_simulation_from_sample(
                sample_data,
                ds_dataset.shape_cells,
                (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
                components=components,
            )

            # Forward simulation for predicted gravity
            preq_x = sim.dpred(pred_y_np)

            # Extract data for plotting
            rx = sample_data["receiver_locations"].cpu().numpy()
            true_model_np = np.zeros_like(ind, float)
            true_model_np[ind] = sample_data["true_model"].cpu().numpy()

            print(
                f"Predicted density stats - min: {pred_y_np.min()}, max: {pred_y_np.max()}, mean: {pred_y_np.mean()}"
            )

            if not headless:
                # Plot each gravity component
                n_rx = rx.shape[0]
                for i, comp in enumerate(components):
                    true_grav = sample_data["gravity"][i].cpu().numpy()
                    pred_grav_comp = preq_x[i :: len(components)]

                    print(
                        f"\n>>> Showing True Gravity Data ({comp}) (close window to continue)..."
                    )
                    plot_gravity_measurements(
                        rx, true_grav, title=f"True Gravity Data ({comp})"
                    )

                    print(
                        f">>> Showing Predicted Gravity Data ({comp}) (close window to continue)..."
                    )
                    plot_gravity_measurements(
                        rx, pred_grav_comp, title=f"Predicted Gravity Data ({comp})"
                    )

                print(
                    "\n>>> Showing True Density Contrast 3D (close window to continue)..."
                )
                plot_density_contrast_3D(mesh, ind, true_model_np)

                print(
                    ">>> Showing Predicted Density Contrast 3D (close window to continue)..."
                )
                plot_density_contrast_3D(mesh, ind, pred_y_np)

            print("\nEvaluation complete!")

    return {
        "rmse": rmse_val,
        "l1": l1_val,
        "iou": iou_val,
        "dice": dice_val,
        "n_samples": 1 if idx is not None else len(cast(Any, dl.dataset)),
    }


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
    headless: bool = False,
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

    # Retrieve components
    if isinstance(dl.dataset, Subset):
        ds_dataset = dl.dataset.dataset
    else:
        ds_dataset = dl.dataset
    components = getattr(ds_dataset, "components", ("gz",))

    if idx is None:
        desc = "Evaluating Hybrid"
        iterator = tqdm(dl, desc=desc, leave=False, ncols=80)
        for local_idx, (x, true_y) in enumerate(iterator):
            if max_samples is not None and local_idx >= max_samples:
                break
            with torch.no_grad():
                x_tensor = x.to(device)
                nn_pred = net(x_tensor)
                nn_m0 = (
                    nn_pred[0]
                    .permute(2, 1, 0)
                    .reshape(-1)
                    .cpu()
                    .numpy()
                    .flatten(order="F")
                )

            # Get raw sample data with transform disabled
            ds = cast(Subset, dl.dataset)
            ds_dataset = cast(Any, ds.dataset)
            old_transform = ds_dataset.transform
            ds_dataset.transform = None
            g_idx = ds.indices[local_idx]
            sample_data = ds_dataset[g_idx]
            ds_dataset.transform = old_transform

            # Flatten observation tensor for SimPEG
            x_obs_flat = x.squeeze(0).cpu().numpy().reshape(x.shape[1], -1).flatten()

            sim, mesh, survey, model_map, ind = create_simulation_from_sample(
                sample_data,
                ds_dataset.shape_cells,
                (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
                components=components,
            )

            data_object = data.Data(survey, dobs=x_obs_flat, noise_floor=sigma)
            dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
            reg = regularization.WeightedLeastSquares(
                mesh, active_cells=ind, mapping=model_map
            )
            opt = optimization.ProjectedGNCG(
                maxIter=10,
                lower=-1.0,
                upper=1.0,
                maxIterLS=20,
                cg_maxiter=10,
                cg_atol=1e-3,
            )
            inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
            starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
            beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
            update_jacobi = directives.UpdatePreconditioner()
            target_misfit = directives.TargetMisfit(chifact=1)
            sensitivity_weights = directives.UpdateSensitivityWeights(
                every_iteration=False
            )
            directives_list = [
                sensitivity_weights,
                starting_beta,
                beta_schedule,
                update_jacobi,
                target_misfit,
            ]
            inv = inversion.BaseInversion(inv_prob, directives_list)
            pred_y = inv.run(nn_m0)
            pred_y = torch.from_numpy(pred_y).numpy()

            true_y_flat = true_y.squeeze(0).cpu().numpy().flatten(order="F")
            metrics.update(true_y_flat, pred_y)

    else:
        # Visualization Mode (Single Sample)
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

        sim, mesh, survey, model_map, ind = create_simulation_from_sample(
            sample_data,
            ds_dataset.shape_cells,
            (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
            components=components,
        )

        # Use x tensor (includes noise from transform)
        x_obs_flat = x.cpu().numpy().reshape(x.shape[0], -1).flatten()

        data_object = data.Data(survey, dobs=x_obs_flat, noise_floor=sigma)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
        reg = regularization.WeightedLeastSquares(
            mesh, active_cells=ind, mapping=model_map
        )
        opt = optimization.ProjectedGNCG(
            maxIter=50, maxIterLS=20, cg_maxiter=10, cg_atol=1e-5
        )
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
        beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
        update_jacobi = directives.UpdatePreconditioner()
        target_misfit = directives.TargetMisfit(chifact=1)
        sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
        directives_list = [
            sensitivity_weights,
            starting_beta,
            beta_schedule,
            update_jacobi,
            target_misfit,
        ]
        inv = inversion.BaseInversion(inv_prob, directives_list)
        pred_y = inv.run(nn_m0)
        metrics.update(true_y.cpu().numpy().flatten(order="F"), pred_y)

        if not headless:
            preq_x = sim.dpred(pred_y)

            rx = sample_data["receiver_locations"].cpu().numpy()
            true_model_np = np.zeros_like(ind, float)
            true_model_np[ind] = sample_data["true_model"].cpu().numpy()

            n_rx = rx.shape[0]
            for i, comp in enumerate(components):
                true_grav = sample_data["gravity"][i].cpu().numpy()
                pred_grav_comp = preq_x[i * n_rx : (i + 1) * n_rx]
                plot_gravity_measurements(
                    rx, true_grav, title=f"True Gravity Data ({comp})"
                )
                plot_gravity_measurements(
                    rx, pred_grav_comp, title=f"Predicted Gravity Data ({comp})"
                )

            plot_density_contrast_3D(mesh, ind, true_model_np)
            plot_density_contrast_3D(mesh, ind, pred_y)
            input("Press Enter to close plots...")

    results = metrics.compute()
    rmse_val, l1_val, iou_val, dice_val = (
        results["RMSE"],
        results["L1"],
        results["IoU"],
        results["Dice"],
    )
    return {
        "rmse": rmse_val,
        "l1": l1_val,
        "iou": iou_val,
        "dice": dice_val,
        "n_samples": 1 if idx is not None else len(cast(Any, dl.dataset)),
    }


def eval_bayesian(
    dl: torch.utils.data.DataLoader,
    stats: dict,
    idx: int | None = None,
    threshold: float = 0.1,
    max_samples: int | None = None,
    accuracy: float = 0.001,
    confidence: float = 0.95,
    headless: bool = False,
):
    """
    Bayesian inversion (SIMPEG) sampling.
    """
    z = (1.0 + confidence) / 2.0
    z = np.clip(z, 1e-10, 1 - 1e-10)
    ppf_z = norm.ppf(z)
    sigma = accuracy / ppf_z
    metrics = NumpyMetrics(threshold)

    # Retrieve components
    if isinstance(dl.dataset, Subset):
        ds_dataset = dl.dataset.dataset
    else:
        ds_dataset = dl.dataset
    components = getattr(ds_dataset, "components", ("gz",))

    if idx is None:
        desc = "Evaluating Bayesian"
        iterator = tqdm(dl, desc=desc, leave=False, ncols=80)
        for local_idx, (x, true_y) in enumerate(iterator):
            if max_samples is not None and local_idx >= max_samples:
                break

            ds = cast(Subset, dl.dataset)
            ds_dataset = cast(Any, ds.dataset)
            old_transform = ds_dataset.transform
            ds_dataset.transform = None
            g_idx = ds.indices[local_idx]
            sample_data = ds_dataset[g_idx]
            ds_dataset.transform = old_transform

            # x is tensor (C, Y, X). Need flat obs.
            x_obs_flat = x.squeeze(0).cpu().numpy().reshape(x.shape[1], -1).flatten()

            sim, mesh, survey, model_map, ind = create_simulation_from_sample(
                sample_data,
                ds_dataset.shape_cells,
                (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
                components=components,
            )

            data_object = data.Data(survey, dobs=x_obs_flat, noise_floor=sigma)
            dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
            reg = regularization.WeightedLeastSquares(
                mesh, active_cells=ind, mapping=model_map
            )
            opt = optimization.ProjectedGNCG(
                maxIter=10,
                lower=-1.0,
                upper=1.0,
                maxIterLS=20,
                cg_maxiter=10,
                cg_atol=1e-3,
            )
            inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
            starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
            beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
            update_jacobi = directives.UpdatePreconditioner()
            target_misfit = directives.TargetMisfit(chifact=1)
            sensitivity_weights = directives.UpdateSensitivityWeights(
                every_iteration=False
            )
            directives_list = [
                sensitivity_weights,
                starting_beta,
                beta_schedule,
                update_jacobi,
                target_misfit,
            ]
            inv = inversion.BaseInversion(inv_prob, directives_list)
            y = np.zeros(int(ind.sum()))
            pred_y = inv.run(y)
            pred_y = torch.from_numpy(pred_y).numpy()

            metrics.update(true_y.squeeze(0).cpu().numpy().flatten(order="F"), pred_y)
    else:
        # Visualization
        ds = cast(Subset, dl.dataset)
        ds_dataset = cast(Any, ds.dataset)
        ds_dataset.transform = None
        g_idx = ds.indices[idx]
        sample_data = ds_dataset[g_idx]

        sim, mesh, survey, model_map, ind = create_simulation_from_sample(
            sample_data,
            ds_dataset.shape_cells,
            (ds_dataset.hx, ds_dataset.hy, ds_dataset.hz),
            components=components,
        )

        # Use raw gravity data (no noise)
        x_obs_flat = sample_data["gravity"].cpu().numpy().flatten()

        y = np.zeros(int(ind.sum()))
        data_object = data.Data(survey, dobs=x_obs_flat, noise_floor=sigma)
        dmis = data_misfit.L2DataMisfit(data=data_object, simulation=sim)
        reg = regularization.WeightedLeastSquares(
            mesh, active_cells=ind, mapping=model_map
        )
        opt = optimization.ProjectedGNCG(
            maxIter=50, maxIterLS=20, cg_maxiter=10, cg_atol=1e-4
        )
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=1e1)
        beta_schedule = directives.BetaSchedule(coolingFactor=5, coolingRate=1)
        update_jacobi = directives.UpdatePreconditioner()
        target_misfit = directives.TargetMisfit(chifact=1)
        sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
        directives_list = [
            sensitivity_weights,
            starting_beta,
            beta_schedule,
            update_jacobi,
            target_misfit,
        ]
        inv = inversion.BaseInversion(inv_prob, directives_list)
        pred_y = inv.run(y)

        true_y = sample_data["true_model"].cpu().numpy()
        metrics.update(true_y, pred_y)

        if not headless:
            preq_x = sim.dpred(pred_y)

            rx = sample_data["receiver_locations"].cpu().numpy()
            true_model_np = np.zeros_like(ind, float)
            true_model_np[ind] = true_y

            n_rx = rx.shape[0]
            for i, comp in enumerate(components):
                true_grav = sample_data["gravity"][i].cpu().numpy()
                pred_grav_comp = preq_x[i * n_rx : (i + 1) * n_rx]
                plot_gravity_measurements(
                    rx, true_grav, title=f"True Gravity Data ({comp})"
                )
                plot_gravity_measurements(
                    rx, pred_grav_comp, title=f"Predicted Gravity Data ({comp})"
                )

            plot_density_contrast_3D(mesh, ind, true_model_np)
            plot_density_contrast_3D(mesh, ind, pred_y)
            input("Press Enter to close plots...")

    results = metrics.compute()
    rmse_val, l1_val, iou_val, dice_val = (
        results["RMSE"],
        results["L1"],
        results["IoU"],
        results["Dice"],
    )
    return {
        "rmse": rmse_val,
        "l1": l1_val,
        "iou": iou_val,
        "dice": dice_val,
        "n_samples": 1 if idx is not None else len(cast(Any, dl.dataset)),
    }


def _eval(
    eval: str = "nn",
    split: str = "va",
    bs: int = 8 if eval == "nn" else 1,
    idx: int | None = None,
    max_samples: int | None = None,
    accuracy: float | None = 0.001,
    confidence: float = 0.95,
    accuracy_loop: bool = False,
    components: tuple = ("gz",),
    checkpoint_path: Union[str, Path] = "checkpoints/default_model_final.pt",
    headless: bool = False,
):
    if accuracy is None:
        accuracy = 0.001

    # Resolve checkpoint path and validate
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    # Extract model name from checkpoint path (e.g., "my_model_best" from "checkpoints/my_model_best.pt")
    model_name_display = ckpt_path.stem

    tr_ld, va_ld, stats = data_prep(
        ds_name="single_block_v2",
        split_name=(
            "single_block_v2_all_comps" if len(components) > 1 else "single_block_v2"
        ),
        bs=bs if eval == "nn" else 1,
        accuracy=accuracy,
        confidence=confidence,
        load_splits=True,
        transform=True,
        components=components,
    )
    dl = {"tr": tr_ld, "va": va_ld}.get(split)
    print(
        f"Eval: {eval}, Split: {split}, Index: {f'{max_samples if max_samples is not None else "All"}' if idx is None else idx}, Accuracy: {accuracy}, Confidence: {confidence}, Components: {components}, Model: {model_name_display}"
    )
    if dl is None:
        print("Invalid split name")
        return None

    def run_nn():
        model, device = load_model(
            device="cuda", in_channels=len(components), checkpoint_path=str(ckpt_path)
        )
        return eval_nn(
            net=model, dl=dl, stats=stats, device=device, threshold=0.1, idx=idx, headless=headless
        )

    def run_hybrid():
        model, device = load_model(model_name="single_block_05", device="cuda")
        return eval_hybrid(
            net=model,
            dl=dl,
            stats=stats,
            device=device,
            threshold=0.1,
            idx=idx,
            max_samples=max_samples,
            accuracy=accuracy,
            confidence=confidence,
            headless=headless,
        )

    eval_funcs = {
        "nn": run_nn,
        "bayesian": lambda: eval_bayesian(
            dl=dl,
            stats=stats,
            idx=idx,
            threshold=0.012,
            max_samples=max_samples,
            accuracy=accuracy,
            confidence=confidence,
            headless=headless,
        ),
        "hybrid": run_hybrid,
    }
    if eval not in eval_funcs:
        print("Invalid eval")
        return None
    if accuracy_loop:
        import csv

        accuracies = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        with open("accuracy_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["accuracy", "rmse", "l1", "iou", "dice", "n_samples"])
            for acc in accuracies:
                print(f"Running accuracy: {acc}")
                metrics = _eval(
                    eval=eval,
                    split=split,
                    bs=bs,
                    idx=idx,
                    max_samples=max_samples,
                    accuracy=acc,
                    confidence=confidence,
                    accuracy_loop=False,
                    components=components,
                    headless=headless,
                )
                if metrics:
                    writer.writerow(
                        [
                            acc,
                            metrics["rmse"],
                            metrics["l1"],
                            metrics["iou"],
                            metrics["dice"],
                            metrics["n_samples"],
                        ]
                    )
        print("Done. Results saved to accuracy_results.csv")
        return None
    metrics = eval_funcs[eval]()
    print(metrics)
    return metrics


if __name__ == "__main__":
    _eval(
        eval="nn",
        split="va",
        idx=None,
        max_samples=None,
        accuracy=None,
        confidence=0.95,
        accuracy_loop=True,
    )
