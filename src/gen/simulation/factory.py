import numpy as np
from .survey import gravity_survey
from .generators import init_model, create_mesh_from_bounds
from . import simulator as gravity_sim
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

    # Build the survey directly from stored receiver locations to preserve ordering.
    # Calling gravity_survey(rx) would reverse the grid because it uses rx[-1] and
    # rx[0] as bounds, which are swapped relative to the original generation order.
    rx_obj = gravity.receivers.Point(rx, components=list(components))
    src = gravity.sources.SourceField(receiver_list=[rx_obj])
    survey = gravity.survey.Survey(src)
    _, _, model_map, _ = init_model(mesh, rx, 0)

    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=model_map,
        active_cells=ind,
        engine="choclo",
    )

    return sim, mesh, survey, model_map, ind
