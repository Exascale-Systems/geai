import dvc.api

from src.gen.batch import generate_batch

if __name__ == "__main__":
    params = dvc.api.params_show()
    g = params["gen"]

    generate_batch(
        out_path=g["out_path"],
        ds_size=g["ds_size"],
        x_dom=g["x_dom"],
        y_dom=g["y_dom"],
        z_dom=g["z_dom"],
        n_xy=g["n_xy"],
        n_z=g["n_z"],
        n_blocks=g["n_blocks"],
        size_frac=(g["size_frac_min"], g["size_frac_max"]),
        density_range=(g["density_min"], g["density_max"]),
    )
