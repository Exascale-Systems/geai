import time
import numpy as np
import matplotlib.pyplot as plt

from src.gen.batch import generate_batch

def profile_scaling(
    n_values=(16, 32, 64),
    batch_size=3,
    base_seed=0,
    out_dir="src/gen/profiler/logs",
    ):
    times = []
    for n in n_values:
        print(f"\n=== Running n_per_axis = {n} ===")
        t0 = time.perf_counter()
        generate_batch(
            out_path=f"{out_dir}/master_n{n}.h5",
            n_xy=n,
            batch_size=batch_size,
            base_seed=base_seed,
        )
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        print(f"→ time = {dt:.2f} s")
    np.savez(f"{out_dir}/scaling_results.npz", n_values=n_values, times=times)
    plt.figure()
    plt.plot(n_values, times, "o-", label="Runtime")
    plt.xlabel("n_per_axis")
    plt.ylabel("Time per batch (s)")
    plt.title("Scaling: runtime vs receiver grid resolution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scaling_curve.png", dpi=150)
    plt.show()
    n = np.array(n_values, dtype=float)
    t = np.array(times, dtype=float)
    coeffs = np.polyfit(np.log(n), np.log(t), 1)
    print(f"\nApprox. scaling law: time ∝ n^{coeffs[0]:.2f}")


if __name__ == "__main__":
    profile_scaling()
