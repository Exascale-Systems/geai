import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.training.evaluation import _eval

if __name__ == "__main__":
    _eval(
        eval="nn",
        split="va",
        idx=0,
        max_samples=None,
        accuracy=None,
        confidence=0.95,
        accuracy_loop=False,
        components=("gx", "gy", "gz"),
    )
