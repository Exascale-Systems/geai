import numpy as np
import pyvista as pv


# ── private helpers ────────────────────────────────────────────────────────────

def _density_volume(mesh, ind_active, density_values):
    """Map sparse active-cell density values onto the full mesh volume (Fortran order)."""
    full = np.zeros(mesh.nC, dtype=float)
    full[ind_active] = density_values
    return full.reshape(mesh.shape_cells, order="F")


def _mesh_edges(mesh):
    return (
        mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])],
        mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])],
        mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])],
    )


def _build_density_grid(mesh, ind_active, density_values):
    """Return a PyVista RectilinearGrid with 'density' cell data."""
    density_vol = _density_volume(mesh, ind_active, density_values)
    grid = pv.RectilinearGrid(*_mesh_edges(mesh))
    grid.cell_data["density"] = density_vol.flatten(order="F").astype(float)
    return grid


def _scalar_bar(title, position_x=0.94, label_font_size=14):
    return {
        "title": title,
        "vertical": True,
        "position_x": position_x,
        "position_y": 0.15,
        "width": 0.04,
        "height": 0.7,
        "label_font_size": label_font_size,
    }


# ── public API ─────────────────────────────────────────────────────────────────

def plot_topography(topo_xyz):
    """Create a 3D scatter plot of topography points."""
    pts = pv.PolyData(topo_xyz)
    p = pv.Plotter()
    p.add_points(pts, render_points_as_spheres=True, point_size=8)
    p.show_bounds(grid="front", xtitle="X", ytitle="Y", ztitle="Z")
    p.show()


def plot_density_contrast_3D(mesh, ind_active, density_values):
    """
    Interactive 3D density plot with a movable clipping plane.

    Shows a semi-transparent active-earth volume as context and solid density-anomaly
    voxels in front. Drag the clip-plane widget to section through the anomaly and
    reveal interior structure. Use the checkbox (bottom-left) to toggle the slice on/off.
    """
    full_active = np.zeros(mesh.nC, dtype=bool)
    full_active[ind_active] = True
    active_vol = full_active.reshape(mesh.shape_cells, order="F")

    grid = _build_density_grid(mesh, ind_active, density_values)
    grid.cell_data["active"] = active_vol.flatten(order="F").astype(np.uint8)

    dens_thresh = grid.threshold(value=0.01, scalars="density", invert=False)

    _mesh_kwargs = dict(
        scalars="density",
        cmap="plasma",
        show_edges=True,
        line_width=0.2,
        clim=(0, 4),
        scalar_bar_args=_scalar_bar("Density (g/cc)"),
    )

    p = pv.Plotter()
    p.add_mesh(grid, scalars="active", opacity=0.4, show_edges=False, show_scalar_bar=False)

    # Unclipped actor shown when slice is off
    full_actor = p.add_mesh(dens_thresh, **_mesh_kwargs)
    full_actor.SetVisibility(False)

    # Clipped actor shown when slice is on (default)
    clip_actor = p.add_mesh_clip_plane(dens_thresh, **_mesh_kwargs)

    def _toggle_slice(enabled):
        clip_actor.SetVisibility(enabled)
        full_actor.SetVisibility(not enabled)
        p.add_text(
            "Slice: ON" if enabled else "Slice: OFF",
            position=(70, 18), font_size=12, color="white", name="slice_label",
        )

    p.add_checkbox_button_widget(
        _toggle_slice, value=True,
        position=(10, 10), size=50,
        color_on="#00cc44", color_off="#cc3300",
    )
    p.add_text("Slice: ON", position=(70, 18), font_size=12, color="white", name="slice_label")

    p.show_bounds(
        grid="back", location="outer", ticks="both",
        xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)", font_size=12,
    )
    p.add_text("Drag plane to section • Density Contrast (g/cc)", font_size=11, position="upper_edge")
    p.camera_position = "iso"
    p.camera.azimuth += 270
    p.show()


def plot_gravity_measurements(
    receiver_locations,
    dpred,
    title="Gravity Anomaly",
    units="$mgal$",
    show_points=True,
    cmap="bwr",
    ncontour=30,
):
    """Plot a 2D gravity anomaly map with survey stations."""
    pts2d = np.c_[receiver_locations[:, :2], np.zeros(len(receiver_locations))]
    surf = pv.PolyData(pts2d).delaunay_2d()
    v_max = float(np.max(np.abs(dpred)))
    surf["dpred"] = np.asarray(dpred).ravel()
    p = pv.Plotter()
    p.add_mesh(
        surf,
        scalars="dpred",
        cmap=cmap,
        clim=(-v_max, v_max),
        scalar_bar_args=_scalar_bar("", position_x=0.90, label_font_size=16),
    )
    if ncontour and ncontour > 0:
        p.add_mesh(surf.contour(ncontour, scalars="dpred"), line_width=1, show_scalar_bar=False)
    if show_points:
        p.add_points(pv.PolyData(pts2d), color="black", point_size=6, render_points_as_spheres=True)
    p.add_text(title, position="upper_edge", font_size=12)
    p.show_bounds(grid="front", xtitle="x (m)", ytitle="y (m)", ztitle="z (m)")
    p.enable_parallel_projection()
    p.view_xy()
    p.show()

# ── Additional ─────────────────────────────────────────────────────────────────

def plot_gravity_residuals(
    receiver_locations,
    dobs,
    dpred,
    title="Gravity Residuals",
    show_points=True,
    cmap="RdBu_r",
    ncontour=20,
):
    """Plot gravity residuals (observed - predicted) to assess model fit."""
    residuals = np.asarray(dobs) - np.asarray(dpred)
    pts2d = np.c_[receiver_locations[:, :2], np.zeros(len(receiver_locations))]
    surf = pv.PolyData(pts2d).delaunay_2d()
    v_max = float(np.max(np.abs(residuals)))
    surf["residuals"] = residuals.ravel()

    p = pv.Plotter()
    p.add_mesh(
        surf,
        scalars="residuals",
        cmap=cmap,
        clim=(-v_max, v_max),
        scalar_bar_args=_scalar_bar("Residuals (mgal)", position_x=0.90, label_font_size=16),
    )
    if ncontour and ncontour > 0:
        p.add_mesh(surf.contour(ncontour, scalars="residuals"), line_width=1, show_scalar_bar=False)
    if show_points:
        p.add_points(pv.PolyData(pts2d), color="black", point_size=6, render_points_as_spheres=True)
    rms = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((np.asarray(dobs) - np.mean(dobs)) ** 2))
    p.add_text(
        f"RMS: {rms:.3f} mgal\nMax: {np.max(np.abs(residuals)):.3f} mgal\nR2: {r2:.3f}",
        position="lower_right", font_size=10,
    )
    p.add_text(title, position="upper_edge", font_size=12)
    p.show_bounds(grid="front", xtitle="x (m)", ytitle="y (m)", ztitle="z (m)")
    p.enable_parallel_projection()
    p.view_xy()
    p.show()


def plot_density_slices(
    mesh, ind_active, density_values, slice_type="y", slice_indices=None, cmap="plasma"
):
    """Plot 2D slices of the density field through the mesh."""
    density_vol = _density_volume(mesh, ind_active, density_values)
    centers = [mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z]

    if slice_indices is None:
        ax = {"x": 0, "y": 1, "z": 2}[slice_type.lower()]
        slice_indices = [mesh.shape_cells[ax] // 2]

    slice_configs = {
        "x": lambda i: (density_vol[i, :, :], np.meshgrid(centers[1], centers[2], indexing="ij"), f"X = {centers[0][i]:.1f} m", "Y (m)", "Z (m)"),
        "y": lambda i: (density_vol[:, i, :], np.meshgrid(centers[0], centers[2], indexing="ij"), f"Y = {centers[1][i]:.1f} m", "X (m)", "Z (m)"),
        "z": lambda i: (density_vol[:, :, i], np.meshgrid(centers[0], centers[1], indexing="ij"), f"Z = {centers[2][i]:.1f} m", "X (m)", "Y (m)"),
    }

    p = pv.Plotter(shape=(1, len(slice_indices)))
    v_max = float(np.max(density_values)) if len(density_values) > 0 else 1.0
    for col, i in enumerate(slice_indices):
        p.subplot(0, col)
        slice_data, (X, Y), slice_title, xlabel, ylabel = slice_configs[slice_type.lower()](i)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        surf = pv.PolyData(pts).delaunay_2d()
        surf["density"] = slice_data.ravel()
        p.add_mesh(surf, scalars="density", cmap=cmap, clim=(0, v_max),
                   scalar_bar_args=_scalar_bar("Density (g/cc)", position_x=0.90))
        p.add_text(slice_title, position="upper_edge", font_size=12)
        p.show_bounds(grid="front", xtitle=xlabel, ytitle=ylabel)
        p.enable_parallel_projection()
        p.view_xy()
    p.show()


def plot_density_slice_residuals(
    mesh,
    ind_active,
    density_true,
    density_pred,
    slice_type="y",
    slice_indices=None,
    cmap="RdBu_r",
):
    """Plot 2D slices of density residuals (true - predicted) with per-slice metrics."""
    true_vol = _density_volume(mesh, ind_active, density_true)
    pred_vol = _density_volume(mesh, ind_active, density_pred)
    residual_vol = true_vol - pred_vol
    centers = [mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z]

    if slice_indices is None:
        ax = {"x": 0, "y": 1, "z": 2}[slice_type.lower()]
        slice_indices = [mesh.shape_cells[ax] // 2]

    slice_configs = {
        "x": lambda i: (residual_vol[i, :, :], np.meshgrid(centers[1], centers[2], indexing="ij"), f"X = {centers[0][i]:.1f} m", "Y (m)", "Z (m)"),
        "y": lambda i: (residual_vol[:, i, :], np.meshgrid(centers[0], centers[2], indexing="ij"), f"Y = {centers[1][i]:.1f} m", "X (m)", "Z (m)"),
        "z": lambda i: (residual_vol[:, :, i], np.meshgrid(centers[0], centers[1], indexing="ij"), f"Z = {centers[2][i]:.1f} m", "X (m)", "Y (m)"),
    }

    p = pv.Plotter(shape=(1, len(slice_indices)))
    for col, i in enumerate(slice_indices):
        p.subplot(0, col)
        residual_slice, (X, Y), slice_title, xlabel, ylabel = slice_configs[slice_type.lower()](i)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        surf = pv.PolyData(pts).delaunay_2d()
        surf["residuals"] = residual_slice.ravel()
        v_max = float(np.max(np.abs(residual_slice))) if np.any(residual_slice != 0) else 1.0
        p.add_mesh(surf, scalars="residuals", cmap=cmap, clim=(-v_max, v_max),
                   scalar_bar_args=_scalar_bar("Residuals (g/cc)", position_x=0.90))
        p.add_text(slice_title, position="upper_edge", font_size=12)
        p.show_bounds(grid="front", xtitle=xlabel, ytitle=ylabel)
        p.enable_parallel_projection()
        p.view_xy()
    p.show()
