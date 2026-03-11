import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import dvc.api

from src.training.evaluation import _eval

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("model", nargs="?", default="default_model", help="Model name")
    args = parser.parse_args()

    model_name = args.model

    # Load params from params.yaml
    params = dvc.api.params_show()
    train_params = params["train"]
    eval_params = params["eval"]

    components = tuple(train_params["components"])
    confidence = train_params["confidence"]

    # Find accuracy for this model from noise_levels
    noise_level = next(
        (nl for nl in train_params["noise_levels"] if nl["name"] == model_name),
        None
    )
    accuracy = noise_level["accuracy"] if noise_level else 0.5

    headless = eval_params["headless"]
    output_path = f"{eval_params['output_dir']}/eval_{accuracy}.json"
    threshold = eval_params.get("threshold", 0.1)

    checkpoint_path = f"checkpoints/{model_name}_final.pt"

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
        headless=headless,
        threshold=threshold,
    )

    if metrics:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
