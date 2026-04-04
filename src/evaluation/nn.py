from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
from typing import Any, cast

import matplotlib.pyplot as plt
from torch.utils.data import Subset

from src.evaluation import (
    plot_density_contrast_3D,
    plot_gravity_measurements,
)
from src.evaluation.metrics import TorchMetrics
from src.gen.core import sim_from_sample
from src.nn.unet import GravInvNet


def load_model(model_name, device="cpu", in_channels=1, checkpoint_path=None):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    model = GravInvNet(in_channels=in_channels).to(device)
    path = Path(checkpoint_path) if checkpoint_path else Path(f"checkpoints/{model_name}_final.pt")
    if path.exists():
        state = torch.load(path, map_location=device)
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

    if isinstance(dl.dataset, Subset):
        ds_dataset = dl.dataset.dataset
    else:
        ds_dataset = dl.dataset
    components = getattr(ds_dataset, "components", ("gz",))

    with torch.no_grad():
        if idx is None:
            iterator = tqdm(dl, desc="Evaluating NN", leave=False, ncols=80)
            for x, true_y in iterator:
                x, true_y = x.to(device), true_y.to(device)
                metrics.update(net, x, true_y, None)
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

            pred_y_np = pred_y[0].permute(2, 1, 0).reshape(-1).cpu().numpy().flatten(order="F")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(pred_y_np, bins=50, alpha=0.5, label="Predicted")
            ax.set_xlabel("Density Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Predicted Density Distribution")
            ax.legend()
            if not headless:
                print("\n>>> Showing density histogram (close window to continue)...")
                plt.show(block=True)
            plt.close()

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

            preq_x = sim.dpred(pred_y_np)

            rx = sample_data["receiver_locations"].cpu().numpy()
            true_model_np = np.zeros_like(ind, float)
            true_model_np[ind] = sample_data["true_model"].cpu().numpy()

            for i, comp in enumerate(components):
                true_grav = sample_data["gravity"][i].cpu().numpy()
                pred_grav_comp = preq_x[i :: len(components)]

                print(f"\n>>> Showing True Gravity Data ({comp}) (close window to continue)...")
                plot_gravity_measurements(rx, true_grav, title=f"True Gravity Data ({comp})")

                print(f">>> Showing Predicted Gravity Data ({comp}) (close window to continue)...")
                plot_gravity_measurements(rx, pred_grav_comp, title=f"Predicted Gravity Data ({comp})")

            print("\n>>> Showing True Density Contrast 3D (close window to continue)...")
            plot_density_contrast_3D(mesh, ind, true_model_np)

            print(">>> Showing Predicted Density Contrast 3D (close window to continue)...")
            plot_density_contrast_3D(mesh, ind, pred_y_np)

            print(
                f"Predicted density stats - min: {pred_y_np.min()}, max: {pred_y_np.max()}, mean: {pred_y_np.mean()}"
            )
            print("\nEvaluation complete!")

    return {
        "rmse": rmse_val,
        "l1": l1_val,
        "iou": iou_val,
        "dice": dice_val,
        "n_samples": 1 if idx is not None else len(cast(Any, dl.dataset)),
    }
