import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import dvc.api

from src.training.evaluation import _eval

if __name__ == "__main__":
    import argparse

    # Load components from params instead of hardcoding
    params = dvc.api.params_show()
    components = tuple(params["train"]["components"])
    confidence = params["train"]["confidence"]

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("model", nargs="?", default="default_model", help="Model name or checkpoint path")
    parser.add_argument("--headless", action="store_true", help="Run without displaying plots")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file for metrics")
    args = parser.parse_args()

    # If arg doesn't have .pt extension, assume it's a model name and resolve to checkpoint
    if not args.model.endswith(".pt"):
        checkpoint_path = f"checkpoints/{args.model}_final.pt"
    else:
        checkpoint_path = args.model

    metrics = _eval(
        eval="nn",
        split="va",
        idx=0,
        max_samples=None,
        accuracy=None,
        confidence=confidence,
        accuracy_loop=False,
        components=components,
        checkpoint_path=checkpoint_path,
        headless=args.headless,
    )

    if args.output and metrics:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")
