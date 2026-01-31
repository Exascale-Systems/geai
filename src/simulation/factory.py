import numpy as np
from src.simulation.survey import gravity_survey
from src.simulation.generators import init_model, create_mesh_from_bounds
import src.simulation.simulator as gravity_sim
from simpeg.potential_fields import gravity


def create_simulation_from_sample(sample_data, shape_cells, h, components=("gz",)):
    """
    Create simulation for a specific sample.

    Args:
        sample_data: Dictionary containing 'receiver_locations' and 'ind_active'.
        shape_cells: Tuple (nx, ny, nz).
        h: Tuple/List of (hx, hy, hz) cell spacings.
        components: Tuple of gravity components.

    Returns:
        tuple: (sim, mesh, survey, model_map, ind_active)
    """
    rx = sample_data["receiver_locations"].cpu().numpy()
    ind = sample_data["ind_active"].cpu().numpy().astype(bool)

    nx, ny, nz = shape_cells
    hx, hy, hz = h

    mesh = create_mesh_from_bounds(
        bounds=(
            (0, nx * hx[0]),
            (0, ny * hy[0]),
            (0, nz * hz[0]),
        ),
        resolution=(nx, ny, nz),
    )

    _, survey = gravity_survey(rx, components=components)
    _, _, model_map, _ = init_model(mesh, rx, 0)

    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )

    return sim, mesh, survey, model_map, ind
