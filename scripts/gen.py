import argparse

import dvc.api

from src.gen.batch import generate_batch as generate_batch_blocks
from src.gen.structuralgeo.batch import generate_batch as generate_batch_sg

# dvc.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset")
    parser.add_argument("--generator", default=None, choices=["blocks", "structuralgeo"], help="Override gen.generator")
    parser.add_argument("--out-path", default=None, help="Override gen.out_path")
    parser.add_argument("--ds-size", type=int, default=None, help="Override gen.ds_size")
    args = parser.parse_args()

    params = dvc.api.params_show()
    g = params["gen"]

    generator = args.generator or g["generator"]
    out_path = args.out_path or g["out_path"]
    ds_size = args.ds_size or g["ds_size"]

    components = tuple(g["components"])

    if generator == "structuralgeo":
        generate_batch_sg(
            out_path=out_path,
            ds_size=ds_size,
            bounds=tuple(tuple(b) for b in g["bounds"]),
            resolution=tuple(g["resolution"]),
            components=components,
        )
    elif generator == "blocks":
        generate_batch_blocks(
            out_path=out_path,
            ds_size=ds_size,
            x_dom=g["x_dom"],
            y_dom=g["y_dom"],
            z_dom=g["z_dom"],
            n_xy=g["n_xy"],
            n_z=g["n_z"],
            n_blocks=g["n_blocks"],
            size_frac=(g["size_frac_min"], g["size_frac_max"]),
            density_range=(g["density_min"], g["density_max"]),
            components=components,
        )
    else:
        raise ValueError(f"Unknown generator: {generator!r}. Choose 'blocks' or 'structuralgeo'.")
