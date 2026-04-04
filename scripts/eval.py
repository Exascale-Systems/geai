import csv
import json
import logging
from pathlib import Path

import dvc.api
import torch

from src.data.dataset import data_prep
from src.evaluation.hybrid import eval_hybrid
from src.evaluation.nn import eval_nn, load_model
from src.evaluation.simpeg import eval_bayesian

logger = logging.getLogger(__name__)


def _eval(
    eval: str = "nn",
    split: str = "va",
    bs: int | None = None,
    idx: int | None = None,
    max_samples: int | None = None,
    accuracy: float | None = 0.001,
    confidence: float = 0.95,
    accuracy_loop: bool = False,
    components: tuple = ("gz",),
    model_name: str = "model",
    checkpoint_path: str | None = None,
    headless: bool = False,
    threshold: float = 0.1,
    ds_name: str = "single_block_v2",
    split_name: str | None = None,
):
    if accuracy is None:
        accuracy = 0.001
    if bs is None:
        bs = 8 if eval == "nn" else 1
    if split_name is None:
        split_name = f"{ds_name}_all_comps" if len(components) > 1 else ds_name

    tr_ld, va_ld, stats = data_prep(
        ds_name=ds_name,
        split_name=split_name,
        bs=bs if eval == "nn" else 1,
        accuracy=accuracy,
        confidence=confidence,
        load_splits=True,
        transform=True,
        components=components,
    )
    dl = {"tr": tr_ld, "va": va_ld}.get(split)
    logger.info(
        "Eval: %s | Split: %s | Index: %s | Accuracy: %s | Confidence: %s | Components: %s",
        eval, split,
        idx if idx is not None else (max_samples if max_samples is not None else "All"),
        accuracy, confidence, components,
    )
    if dl is None:
        logger.error("Invalid split name: %s", split)
        return None

    def run_nn():
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model, device = load_model(
            model_name=model_name, device=_device, in_channels=len(components),
            checkpoint_path=checkpoint_path,
        )
        return eval_nn(
            net=model, dl=dl, stats=stats, device=device, threshold=threshold,
            idx=idx, headless=headless,
        )

    def run_hybrid():
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model, device = load_model(
            model_name=model_name, device=_device, in_channels=len(components),
            checkpoint_path=checkpoint_path,
        )
        return eval_hybrid(
            net=model, dl=dl, stats=stats, device=device, threshold=threshold,
            idx=idx, max_samples=max_samples, accuracy=accuracy, confidence=confidence,
        )

    eval_funcs = {
        "nn": run_nn,
        "bayesian": lambda: eval_bayesian(
            dl=dl, stats=stats, idx=idx, threshold=threshold,
            max_samples=max_samples, accuracy=accuracy, confidence=confidence,
        ),
        "hybrid": run_hybrid,
    }
    if eval not in eval_funcs:
        logger.error("Invalid eval mode: %s. Choose from: %s", eval, list(eval_funcs))
        return None

    if accuracy_loop:
        accuracies = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        with open("accuracy_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["accuracy", "rmse", "l1", "iou", "dice", "n_samples"])
            for acc in accuracies:
                logger.info("Running accuracy: %s", acc)
                metrics = _eval(
                    eval=eval, split=split, bs=bs, idx=idx, max_samples=max_samples,
                    accuracy=acc, confidence=confidence, accuracy_loop=False,
                    components=components, ds_name=ds_name, split_name=split_name,
                )
                if metrics:
                    writer.writerow([
                        acc, metrics["rmse"], metrics["l1"],
                        metrics["iou"], metrics["dice"], metrics["n_samples"],
                    ])
        logger.info("Done. Results saved to accuracy_results.csv")
        return None

    metrics = eval_funcs[eval]()
    logger.info("Results: %s", metrics)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("model", nargs="?", default=None, help="Model name")
    parser.add_argument("--headless", action="store_true", help="Run without displaying plots")
    args = parser.parse_args()

    params = dvc.api.params_show()
    train_params = params["train"]
    eval_params = params["eval"]

    model_name = args.model or train_params.get("model_name", "default_model")
    headless = args.headless

    components = tuple(train_params["components"])
    confidence = train_params["confidence"]
    accuracy = train_params["noise"]["accuracy"]

    output_dir = Path(eval_params["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{model_name}.json"
    threshold = eval_params.get("threshold", 0.1)

    checkpoint_path = f"checkpoints/{model_name}_final.pt"

    metrics = _eval(
        eval="nn",
        split="va",
        idx=None,
        max_samples=None,
        accuracy=accuracy,
        confidence=confidence,
        accuracy_loop=False,
        components=components,
        checkpoint_path=checkpoint_path,
        headless=headless,
        threshold=threshold,
        ds_name=params["data"]["ds_name"],
        split_name=params["data"]["split_name"],
    )

    if metrics:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
