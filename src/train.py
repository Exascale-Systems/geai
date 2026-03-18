import dvc.api

from src.data.dataset import data_prep
from src.training.engine import train_model
from src.modeling.networks import GravInvNet


def main(model_name="default_model"):
    # Load params from params.yaml
    params = dvc.api.params_show()

    data_params = params["data"]
    train_params = params["train"]

    # Use params with CLI overrides for per-stage values
    config = {
        "device": train_params["device"],
        "lr": train_params["lr"],
        "wd": train_params["wd"],
        "max_epochs": train_params["max_epochs"],
        "min_loss": train_params["min_loss"],
        "eval_interval": train_params["eval_interval"],
        "components": tuple(train_params["components"]),
        "model_name": model_name,
    }

    # Find accuracy for this model from noise_levels
    noise_level = next(
        (nl for nl in train_params["noise_levels"] if nl["name"] == model_name),
        None
    )
    accuracy = noise_level["accuracy"] if noise_level else 0.5
    confidence = train_params["confidence"]

    num_components = len(config["components"])
    batch_size = data_params["batch_size"]
    split_name = data_params["split_name"]

    tr_ld, va_ld, stats = data_prep(
        ds_name=data_params["ds_name"],
        split_name=split_name,
        bs=batch_size,
        load_splits=True,
        transform=True,
        accuracy=accuracy,
        confidence=confidence,
        components=config["components"],
    )

    net = GravInvNet(in_channels=num_components, model_name=model_name)

    train_model(net, tr_ld, va_ld, stats, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train gravity inversion model")
    parser.add_argument(
        "model_name", nargs="?", default="default_model", help="Model name"
    )

    args = parser.parse_args()
    main(model_name=args.model_name)
