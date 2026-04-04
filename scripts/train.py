import json
from pathlib import Path

import dvc.api

from src.data.dataset import data_prep
from src.nn.engine import train_model
from src.nn.unet import GravInvNet

REQUIRED_PARAMS = {
    "data": ["ds_name", "split_name", "batch_size"],
    "train": ["lr", "wd", "max_epochs", "min_loss", "eval_interval", "device",
              "components", "confidence", "noise", "experiments", "model_name"],
}


def validate_params(params, model_name):
    for section, keys in REQUIRED_PARAMS.items():
        if section not in params:
            raise KeyError(f"params.yaml missing section: '{section}'")
        for key in keys:
            if key not in params[section]:
                raise KeyError(f"params.yaml missing '{section}.{key}'")
    if model_name not in params["train"]["experiments"]:
        raise KeyError(f"params.yaml missing 'train.experiments.{model_name}'")
    if "accuracy" not in params["train"]["noise"]:
        raise KeyError("params.yaml missing 'train.noise.accuracy'")


def main(model_name=None):
    params = dvc.api.params_show()

    if model_name is None:
        model_name = params["train"].get("model_name", "default_model")

    validate_params(params, model_name)

    data_params = params["data"]
    train_params = params["train"]
    loss_function = train_params["experiments"][model_name]["loss_function"]

    components = tuple(train_params["components"])
    accuracy = train_params["noise"]["accuracy"]
    confidence = train_params["confidence"]

    tr_ld, va_ld, stats = data_prep(
        ds_name=data_params["ds_name"],
        split_name=data_params["split_name"],
        bs=data_params["batch_size"],
        load_splits=True,
        transform=True,
        accuracy=accuracy,
        confidence=confidence,
        components=components,
    )

    config = {
        "device": train_params["device"],
        "lr": train_params["lr"],
        "wd": train_params["wd"],
        "max_epochs": train_params["max_epochs"],
        "min_loss": train_params["min_loss"],
        "eval_interval": train_params["eval_interval"],
        "components": components,
        "model_name": model_name,
        "loss_function": loss_function,
    }

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    net = GravInvNet(in_channels=len(components), model_name=model_name)
    final_metrics = train_model(net, tr_ld, va_ld, stats, config)

    metrics_path = Path(f"metrics/{model_name}_train.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs="?", default=None, help="Model/experiment name")
    args = parser.parse_args()
    main(args.model_name)
