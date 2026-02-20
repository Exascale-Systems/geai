import numpy as np
from scipy.interpolate import LinearNDInterpolator
from simpeg.potential_fields import gravity


def gravity_survey(
    topo_xyz,
    n_per_axis=32,
    components=("gz",),
):
    """
    Build receiver grid at topography surface; returns (receiver_locations, survey).
    """
    rx_xy = (
        np.array(
            np.meshgrid(
                np.linspace(topo_xyz[-1, 0], topo_xyz[0, 0], n_per_axis),
                np.linspace(topo_xyz[-1, 1], topo_xyz[0, 1], n_per_axis),
                indexing="xy",
            )
        )
        .reshape(2, -1)
        .T
    )
    fun = LinearNDInterpolator(topo_xyz[:, :2], topo_xyz[:, 2], fill_value=np.nan)
    z = fun(rx_xy)
    if np.isnan(z).any():
        msg = f"{np.isnan(z).sum()} receivers outside convex hull; values set to NaN."
        raise ValueError(msg)
    receiver_locations = np.c_[rx_xy, z]
    rx = gravity.receivers.Point(receiver_locations, components=list(components))
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    return receiver_locations, survey
