import dvc.api

from src.data.dataset import data_prep
from src.training.engine import train_model
from src.modeling.networks import GravInvNet


def main(model_name):
    params = dvc.api.params_show()

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

    net = GravInvNet(in_channels=len(components), model_name=model_name)
    train_model(net, tr_ld, va_ld, stats, config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Model/experiment name")
    args = parser.parse_args()
    main(args.model_name)
