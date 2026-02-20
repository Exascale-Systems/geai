from . import generators
from .simulator import GravitySimulator


class GravitySimBuilder:
    def __init__(self):
        self.topo_xyz = None
        self.mesh = None
        self.ind_active = None
        self.true_model = None
        self.model_map = None
        self.blocks_mask = None

        # Defaults
        self.topo_config = {}
        self.mesh_config = {}
        self.model_config = {}

    def set_topography(
        self, x_dom=1.6e3, y_dom=1.6e3, dx=50, dy=50, fbm_amp=0.0, seed=0, **kwargs
    ):
        self.topo_xyz = generators.create_topo(
            x_dom=x_dom, y_dom=y_dom, dx=dx, dy=dy, fbm_amp=fbm_amp, seed=seed, **kwargs
        )
        # Store config for creating mesh if needed
        self.topo_config = {"x_dom": x_dom, "y_dom": y_dom}
        return self

    def set_mesh(self, n_xy=32, n_z=16, z_dom=800.0):
        if self.topo_xyz is None:
            raise ValueError("Topography must be set before creating mesh.")
        self.mesh = generators.create_mesh(
            self.topo_xyz, n_xy=n_xy, n_z=n_z, z_dom=z_dom
        )
        self.mesh_config = {"n_xy": n_xy, "n_z": n_z, "z_dom": z_dom}

        # Auto-init model if mesh changes
        self.ind_active, _, self.model_map, self.true_model = generators.init_model(
            self.mesh, self.topo_xyz
        )
        return self

    def add_blocks(
        self,
        n_blocks=1,
        size_frac_range=(0.05, 0.30),
        density_range=(0.0, 1.0),
        seed=0,
        enforce_nonoverlap=False,
        **kwargs,
    ):
        if self.true_model is None:
            raise ValueError("Model not initialized. Call set_mesh() first.")

        self.true_model, self.blocks_mask = generators.add_random_blocks(
            mesh=self.mesh,
            ind_active=self.ind_active,
            model=self.true_model,
            n_blocks=n_blocks,
            size_frac_range=size_frac_range,
            density_range=density_range,
            seed=seed,
            enforce_nonoverlap=enforce_nonoverlap,
            **kwargs,
        )
        return self

    def build(self):
        if self.mesh is None:
            raise ValueError(
                "Incomplete simulation definition. Call set_topography() and set_mesh()."
            )

        return GravitySimulator(
            mesh=self.mesh,
            ind_active=self.ind_active,
            model_map=self.model_map,
            true_model=self.true_model,
            blocks_mask=self.blocks_mask,
        )
