import json

import dvc.api

from src.training.evaluation import _eval

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("model", nargs="?", default="default_model", help="Model name")
    parser.add_argument("--headless", action="store_true", help="Run without displaying plots")
    args = parser.parse_args()

    model_name = args.model
    headless = args.headless

    # Load params from params.yaml
    params = dvc.api.params_show()
    train_params = params["train"]
    eval_params = params["eval"]

    components = tuple(train_params["components"])
    confidence = train_params["confidence"]
    accuracy = train_params["noise"]["accuracy"]

    output_path = f"{eval_params['output_dir']}/eval_{model_name}.json"
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
    )

    if metrics:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")
