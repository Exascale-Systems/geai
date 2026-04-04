import numpy as np
import pyvista as pv


def plot_topography(topo_xyz):
    """
    Create a 3D scatter plot of topography points.
    """
    pts = pv.PolyData(topo_xyz)
    p = pv.Plotter()
    p.add_points(pts, render_points_as_spheres=True, point_size=8)
    p.show_bounds(grid="front", xtitle="X", ytitle="Y", ztitle="Z")
    p.show()


def plot_density_contrast_3D(mesh, ind_active, density_values):
    """
    Plot 3D voxel grid showing:
      - semi-transparent active (earth) volume
      - solid blocks overlay for density anomalies
    """
    full_active = np.zeros(mesh.nC, dtype=bool)
    full_active[ind_active] = True
    full_density = np.zeros(mesh.nC, dtype=float)
    full_density[ind_active] = density_values
    active_vol = full_active.reshape(mesh.shape_cells, order="F")
    density_vol = full_density.reshape(mesh.shape_cells, order="F")
    x_edges = mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])]
    y_edges = mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])]
    z_edges = mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])]
    grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    grid.cell_data["active"] = active_vol.flatten(order="F").astype(np.uint8)
    grid.cell_data["density"] = density_vol.flatten(order="F").astype(float)
    dens_thresh = grid.threshold(value=0.01, scalars="density", invert=False)
    p = pv.Plotter()
    p.add_mesh(
        grid, scalars="active", opacity=0.4, show_edges=False, show_scalar_bar=False
    )
    p.add_mesh(
        dens_thresh,
        scalars="density",
        cmap="plasma",
        show_edges=True,
        line_width=0.2,
        clim=(0, 4),
        scalar_bar_args={
            "title": "",
            "vertical": True,
            "position_x": 0.94,
            "position_y": 0.15,
            "width": 0.04,
            "height": 0.7,
            "label_font_size": 14,
        },
    )
    p.show_bounds(
        grid="back",
        location="outer",
        ticks="both",
        xtitle="X (m)",
        ytitle="Y (m)",
        ztitle="Z (m)",
        font_size=12,
    )
    p.add_text("Density Contrast (g/cc)", font_size=12, position="upper_edge")
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
    """
    Plot a 2D gravity anomaly map with survey stations.
    """
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
        scalar_bar_args={
            "title": "",
            "vertical": True,
            "position_x": 0.90,
            "position_y": 0.15,
            "width": 0.04,
            "height": 0.7,
            "label_font_size": 16,
        },
    )
    if ncontour and ncontour > 0:
        p.add_mesh(
            surf.contour(ncontour, scalars="dpred"), line_width=1, show_scalar_bar=False
        )
    if show_points:
        p.add_points(
            pv.PolyData(pts2d),
            color="black",
            point_size=6,
            render_points_as_spheres=True,
        )
    p.add_text(title, position="upper_edge", font_size=12)
    p.show_bounds(grid="front", xtitle="x (m)", ytitle="y (m)", ztitle="z (m)")
    p.enable_parallel_projection()
    p.view_xy()
    p.show()


def plot_gravity_residuals(
    receiver_locations,
    dobs,
    dpred,
    title="Gravity Residuals",
    show_points=True,
    cmap="RdBu_r",
    ncontour=20,
):
    """
    Plot gravity residuals (observed - predicted) to assess model fit.
    """
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
        scalar_bar_args={
            "title": "Residuals (mgal)",
            "vertical": True,
            "position_x": 0.90,
            "position_y": 0.15,
            "width": 0.04,
            "height": 0.7,
            "label_font_size": 16,
        },
    )

    if ncontour and ncontour > 0:
        p.add_mesh(
            surf.contour(ncontour, scalars="residuals"),
            line_width=1,
            show_scalar_bar=False,
        )
    if show_points:
        p.add_points(
            pv.PolyData(pts2d),
            color="black",
            point_size=6,
            render_points_as_spheres=True,
        )
    rms = np.sqrt(np.mean(residuals**2))
    r2 = 1 - (np.sum(residuals**2) / np.sum((np.asarray(dobs) - np.mean(dobs)) ** 2))
    stats_text = (
        f"RMS: {rms:.3f} mgal\nMax: {np.max(np.abs(residuals)):.3f} mgal\nR2: {r2:.3f}"
    )
    p.add_text(stats_text, position="lower_right", font_size=10)
    p.add_text(title, position="upper_edge", font_size=12)
    p.show_bounds(grid="front", xtitle="x (m)", ytitle="y (m)", ztitle="z (m)")
    p.enable_parallel_projection()
    p.view_xy()
    p.show()


def plot_density_slices(
    mesh, ind_active, density_values, slice_type="y", slice_indices=None, cmap="plasma"
):
    """
    Plot 2D slices of the density field through the mesh.

    Parameters:
    -----------
    mesh : discretize mesh
        The mesh object
    ind_active : array_like
        Indices of active cells
    density_values : array_like
        Density values for active cells
    slice_type : str
        'x', 'y', or 'z' for slice direction
    slice_indices : list or None
        Specific indices to slice. If None, uses middle slices
    cmap : str
        Colormap for density values
    """
    full_density = np.zeros(mesh.nC, dtype=float)
    full_density[ind_active] = density_values
    density_vol = full_density.reshape(mesh.shape_cells, order="F")
    centers = [mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z]

    if slice_indices is None:
        idx = {"x": 0, "y": 1, "z": 2}[slice_type.lower()]
        slice_indices = [mesh.shape_cells[idx] // 2]
    p = pv.Plotter(shape=(1, len(slice_indices)))
    slice_configs = {
        "x": lambda idx: (
            density_vol[idx, :, :],
            np.meshgrid(centers[1], centers[2], indexing="ij"),
            f"X = {centers[0][idx]:.1f} m",
            "Y (m)",
            "Z (m)",
        ),
        "y": lambda idx: (
            density_vol[:, idx, :],
            np.meshgrid(centers[0], centers[2], indexing="ij"),
            f"Y = {centers[1][idx]:.1f} m",
            "X (m)",
            "Z (m)",
        ),
        "z": lambda idx: (
            density_vol[:, :, idx],
            np.meshgrid(centers[0], centers[1], indexing="ij"),
            f"Z = {centers[2][idx]:.1f} m",
            "X (m)",
            "Y (m)",
        ),
    }

    for i, idx in enumerate(slice_indices):
        p.subplot(0, i)
        slice_data, (X, Y), slice_title, xlabel, ylabel = slice_configs[
            slice_type.lower()
        ](idx)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        surf = pv.PolyData(pts).delaunay_2d()
        surf["density"] = slice_data.ravel()
        v_max = float(np.max(density_values)) if len(density_values) > 0 else 1.0
        scalar_bar_args = {
            "title": "Density (g/cc)",
            "vertical": True,
            "position_x": 0.90,
            "position_y": 0.15,
            "width": 0.04,
            "height": 0.7,
            "label_font_size": 14,
        }
        p.add_mesh(
            surf,
            scalars="density",
            cmap=cmap,
            clim=(0, v_max),
            scalar_bar_args=scalar_bar_args,
        )
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
    """
    Plot 2D slices of density residuals (true - predicted) with MSE and L1 metrics.

    Parameters:
    -----------
    mesh : discretize mesh
        The mesh object
    ind_active : array_like
        Indices of active cells
    density_true : array_like
        True density values for active cells
    density_pred : array_like
        Predicted density values for active cells
    slice_type : str
        'x', 'y', or 'z' for slice direction
    slice_indices : list or None
        Specific indices to slice. If None, uses middle slices
    cmap : str
        Colormap for residuals
    """
    full_true, full_pred = (
        np.zeros(mesh.nC, dtype=float),
        np.zeros(mesh.nC, dtype=float),
    )
    full_true[ind_active], full_pred[ind_active] = density_true, density_pred
    true_vol, pred_vol = (
        full_true.reshape(mesh.shape_cells, order="F"),
        full_pred.reshape(mesh.shape_cells, order="F"),
    )
    residual_vol = true_vol - pred_vol
    centers = [mesh.cell_centers_x, mesh.cell_centers_y, mesh.cell_centers_z]

    if slice_indices is None:
        idx = {"x": 0, "y": 1, "z": 2}[slice_type.lower()]
        slice_indices = [mesh.shape_cells[idx] // 2]
    p = pv.Plotter(shape=(1, len(slice_indices)))
    slice_configs = {
        "x": lambda idx: (
            residual_vol[idx, :, :],
            true_vol[idx, :, :],
            pred_vol[idx, :, :],
            np.meshgrid(centers[1], centers[2], indexing="ij"),
            f"X = {centers[0][idx]:.1f} m",
            "Y (m)",
            "Z (m)",
        ),
        "y": lambda idx: (
            residual_vol[:, idx, :],
            true_vol[:, idx, :],
            pred_vol[:, idx, :],
            np.meshgrid(centers[0], centers[2], indexing="ij"),
            f"Y = {centers[1][idx]:.1f} m",
            "X (m)",
            "Z (m)",
        ),
        "z": lambda idx: (
            residual_vol[:, :, idx],
            true_vol[:, :, idx],
            pred_vol[:, :, idx],
            np.meshgrid(centers[0], centers[1], indexing="ij"),
            f"Z = {centers[2][idx]:.1f} m",
            "X (m)",
            "Y (m)",
        ),
    }

    for i, idx in enumerate(slice_indices):
        p.subplot(0, i)
        residual_slice, true_slice, pred_slice, (X, Y), slice_title, xlabel, ylabel = (
            slice_configs[slice_type.lower()](idx)
        )
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        surf = pv.PolyData(pts).delaunay_2d()
        surf["residuals"] = residual_slice.ravel()
        v_max = (
            float(np.max(np.abs(residual_slice)))
            if np.any(residual_slice != 0)
            else 1.0
        )
        scalar_bar_args = {
            "title": "Residuals (g/cc)",
            "vertical": True,
            "position_x": 0.90,
            "position_y": 0.15,
            "width": 0.04,
            "height": 0.7,
            "label_font_size": 14,
        }
        p.add_mesh(
            surf,
            scalars="residuals",
            cmap=cmap,
            clim=(-v_max, v_max),
            scalar_bar_args=scalar_bar_args,
        )
        p.add_text(slice_title, position="upper_edge", font_size=12)
        p.show_bounds(grid="front", xtitle=xlabel, ytitle=ylabel)
        p.enable_parallel_projection()
        p.view_xy()
    p.show()
