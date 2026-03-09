import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.training.evaluation import _eval

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "default_model"

    # If arg doesn't have .pt extension, assume it's a model name and resolve to checkpoint
    if not arg.endswith(".pt"):
        checkpoint_path = f"checkpoints/{arg}_final.pt"
    else:
        checkpoint_path = arg

    _eval(
        eval="nn",
        split="va",
        idx=0,
        max_samples=None,
        accuracy=None,
        confidence=0.95,
        accuracy_loop=False,
        components=("gx", "gy", "gz"),
        checkpoint_path=checkpoint_path,
    )
