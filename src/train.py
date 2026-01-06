from src.data.dataset import data_prep
from src.training.engine import train_model
from src.modeling.networks import GravInvNet


def main():
    config = {
        "device": "cuda:5",
        "lr": 3e-4,
        "wd": 0.0,
        "max_epochs": 200,
        "min_loss": 1e-6,
        "eval_interval": 10,
    }

    accuracy = 5e-1
    confidence = 0.95

    # Load Data
    tr_ld, va_ld, stats = data_prep(
        ds_name="single_block_v2",
        split_name="single_block_v2",
        bs=8,
        load_splits=True,
        transform=True,
        accuracy=accuracy,
        confidence=confidence,
    )

    # Init Model
    net = GravInvNet()

    # Train
    train_model(net, tr_ld, va_ld, stats, config)


if __name__ == "__main__":
    main()
