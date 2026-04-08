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
    mode: str = "nn",
    split: str = "va",
    idx: int | None = None,
    max_samples: int | None = None,
    accuracy: float = 0.001,
    confidence: float = 0.95,
    components: tuple = ("gz",),
    model_path: str | None = None,
    headless: bool = False,
    threshold: float = 0.1,
    ds_name: str = "single_block_v2",
    split_name: str | None = None,
    inversion: dict | None = None,
    hybrid: dict | None = None,
):
    if split_name is None:
        split_name = f"{ds_name}_all_comps" if len(components) > 1 else ds_name

    bs = 8 if mode == "nn" else 1
    tr_ld, va_ld, stats = data_prep(
        ds_name=ds_name,
        split_name=split_name,
        bs=bs,
        accuracy=accuracy,
        confidence=confidence,
        load_splits=True,
        transform=True,
        components=components,
    )
    dl = {"tr": tr_ld, "va": va_ld}.get(split)
    if dl is None:
        logger.error("Invalid split: %s", split)
        return None

    logger.info("Mode: %s | Split: %s | Idx: %s | Components: %s", mode, split, idx, components)

    if mode in ("nn", "hybrid"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, device = load_model(model_path=model_path, device=device, in_channels=len(components))

    if mode == "nn":
        return eval_nn(net=model, dl=dl, stats=stats, device=device,
                       threshold=threshold, idx=idx, headless=headless)
    elif mode == "hybrid":
        return eval_hybrid(net=model, dl=dl, stats=stats, device=device,
                           threshold=threshold, idx=idx, max_samples=max_samples,
                           accuracy=accuracy, confidence=confidence,
                           inv_params=inversion or {}, hybrid=hybrid or {})
    elif mode == "bayesian":
        return eval_bayesian(dl=dl, stats=stats, idx=idx, threshold=threshold,
                             max_samples=max_samples, accuracy=accuracy, confidence=confidence,
                             inv_params=inversion or {})
    else:
        raise ValueError(f"Invalid eval mode: {mode!r}. Choose from: nn, bayesian, hybrid")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--model", default=None, help="Override eval.model_name")
    parser.add_argument("--mode", default=None, choices=["nn", "bayesian", "hybrid"], help="Override eval.mode")
    parser.add_argument("--split", default=None, choices=["tr", "va"], help="Override eval.split")
    parser.add_argument("--idx", type=int, default=None, help="Sample index (enables per-sample plots)")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    params = dvc.api.params_show()
    train_p = params["train"]
    eval_p = params["eval"]
    data_p = params["data"]

    model_name = args.model or eval_p["model_name"]
    run_name = eval_p.get("run_name") or model_name
    mode = args.mode or eval_p["mode"]
    split = args.split or eval_p["split"]

    output_dir = Path(eval_p["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _eval(
        mode=mode,
        split=split,
        idx=args.idx,
        accuracy=eval_p["noise"]["accuracy"],
        confidence=eval_p["confidence"],
        components=tuple(train_p["components"]),
        model_path=f"models/{model_name}.pt",
        headless=args.headless,
        threshold=eval_p["threshold"],
        ds_name=data_p["ds_name"],
        split_name=data_p["split_name"],
        inversion=eval_p.get("inversion", {}),
        hybrid=eval_p.get("hybrid", {}),
    )

    if metrics:
        output_path = output_dir / f"eval_{run_name}.json"
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
