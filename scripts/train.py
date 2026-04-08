import json
from pathlib import Path

import dvc.api

from src.data.dataset import data_prep
from src.nn.engine import train_model
from src.nn.unet import GravInvNet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", default=None, help="Override train.model_name")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.max_epochs")
    parser.add_argument("--bs", type=int, default=None, help="Override data.batch_size")
    parser.add_argument("--ds-name", default=None, help="Override data.ds_name")
    parser.add_argument("--device", default=None, help="Override train.device")
    args = parser.parse_args()

    params = dvc.api.params_show()
    train_p = params["train"]
    data_p = params["data"]

    model_name = args.model or train_p["model_name"]
    ds_name = args.ds_name or data_p["ds_name"]
    max_epochs = args.epochs or train_p["max_epochs"]
    batch_size = args.bs or train_p["batch_size"]
    n_samples = train_p.get("n_samples")
    train_split = train_p.get("train_split", 0.8)
    device = args.device or train_p["device"]

    components = tuple(train_p["components"])
    loss_function = train_p["experiments"].get(model_name, {}).get("loss_function", "mse")

    tr_ld, va_ld, stats = data_prep(
        ds_name=ds_name,
        split_name=data_p["split_name"],
        bs=batch_size,
        load_splits=True,
        transform=True,
        accuracy=train_p["noise"]["accuracy"],
        confidence=train_p["confidence"],
        components=components,
        n_samples=n_samples,
        train_split=train_split,
    )

    config = {
        "device": device,
        "lr": train_p["lr"],
        "wd": train_p["wd"],
        "max_epochs": max_epochs,
        "min_loss": train_p["min_loss"],
        "eval_interval": train_p["eval_interval"],
        "components": components,
        "model_name": model_name,
        "loss_function": loss_function,
    }

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    net = GravInvNet(in_channels=len(components), model_name=model_name)

    resume_path = Path(f"checkpoints/best.pt")
    if resume_path.exists():
        import torch
        state = torch.load(resume_path, map_location="cpu")
        net.load_state_dict(state["model"])
        config["start_epoch"] = state.get("epoch", 0) + 1
        config["best_val_loss"] = state.get("best_val_loss", float("inf"))
        config["_resume_optimizer"] = state.get("optimizer")
        config["_resume_scaler"] = state.get("scaler")
        remaining = config["max_epochs"] - config["start_epoch"]
        print(f"Resuming from {resume_path} (epoch {config['start_epoch']}, {remaining} epochs remaining)")

    final_metrics = train_model(net, tr_ld, va_ld, stats, config)

    metrics_path = Path(f"metrics/{model_name}_train.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
