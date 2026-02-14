import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.data.dataset import data_prep
from src.training.engine import train_model
from src.modeling.networks import GravInvNet


def main():
    config = {
        "device": "cuda",
        "lr": 3e-4,
        "wd": 0.0,
        "max_epochs": 2,
        "min_loss": 1e-6,
        "eval_interval": 1,
        "components": ("gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"),
    }

    num_components = len(config["components"])
    batch_size = 32
    split_name = "single_block_v2_all_comps"

    accuracy = 5e-1
    confidence = 0.95

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

    net = GravInvNet(in_channels=num_components)

    train_model(net, tr_ld, va_ld, stats, config)


if __name__ == "__main__":
    main()
