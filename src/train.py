import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.data.dataset import data_prep
from src.training.engine import train_model
from src.modeling.networks import GravInvNet


def main(model_name="default_model", accuracy=0.5, confidence=0.95):
    config = {
        "device": "cuda",
        "lr": 3e-4,
        "wd": 0.0,
        "max_epochs": 50,
        "min_loss": 1e-6,
        "eval_interval": 1,
        "components": ("gx", "gy", "gz"),
        "model_name": model_name,
    }

    num_components = len(config["components"])
    batch_size = 32
    split_name = "single_block_v2"

    tr_ld, va_ld, stats = data_prep(
        ds_name="single_block_v2",
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
    parser.add_argument(
        "--accuracy", type=float, default=0.5, help="Measurement accuracy (noise level)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="Measurement confidence"
    )

    args = parser.parse_args()
    main(model_name=args.model_name, accuracy=args.accuracy, confidence=args.confidence)
