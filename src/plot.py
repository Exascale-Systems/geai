import numpy as np
import os
import pyvista as pv

def plot_topography(topo_xyz):
    """
    Create a 3D scatter plot of topography points.
    """
    pts = pv.PolyData(topo_xyz)
    p = pv.Plotter()
    p.add_points(pts, render_points_as_spheres=True, point_size=8)
    p.show_bounds(grid='front', xtitle='X', ytitle='Y', ztitle='Z')
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
    active_vol  = full_active.reshape(mesh.shape_cells, order="F")
    density_vol = full_density.reshape(mesh.shape_cells, order="F")
    x_edges = mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])]
    y_edges = mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])]
    z_edges = mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])]
    grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
    grid.cell_data["active"]  = active_vol.flatten(order="F").astype(np.uint8)
    grid.cell_data["density"] = density_vol.flatten(order="F").astype(float)
    dens_thresh = grid.threshold(value=0.5, scalars="density", invert=False)
    p = pv.Plotter()
    p.add_mesh(grid, scalars="active", opacity=0.4, show_edges=False, show_scalar_bar=False)
    p.add_mesh(
        dens_thresh,
        scalars="density",
        cmap="plasma",
        show_edges=True,
        line_width=0.2,
        clim=(0,4),
        scalar_bar_args={
            "title": "",
            "vertical": True,        
            "position_x": 0.94,      
            "position_y": 0.15,      
            "width": 0.04,          
            "height": 0.7,           
            "label_font_size": 14,
        }
    )
    p.show_bounds(
        grid="back",
        location="outer",
        ticks="both",
        xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
        font_size=12
    )
    p.add_text("Density Contrast (g/cc)", font_size=12, position="upper_edge")
    p.camera_position = "iso"
    p.camera.azimuth += 270
    p.show()

def plot_gravity_measurements(receiver_locations, dpred, title="Gravity Anomaly (Z-component)",
                              units="$mgal$", show_points=True, cmap="bwr", ncontour=30):
    """
    Plot a 2D gravity anomaly map with survey stations.
    """
    xy = receiver_locations[:, :2]
    pts2d = np.c_[xy, np.zeros(len(xy))]
    v_max = float(np.max(np.abs(dpred)))
    cloud = pv.PolyData(pts2d)
    surf = cloud.delaunay_2d()
    surf["dpred"] = np.asarray(dpred).ravel()
    p = pv.Plotter()
    p.add_mesh(
        surf, scalars="dpred", cmap=cmap, clim=(-v_max, v_max),
        scalar_bar_args={
            "title": "",
            "vertical": True,        
            "position_x": 0.90,      
            "position_y": 0.15,      
            "width": 0.04,          
            "height": 0.7,           
            "label_font_size": 16,
        }
    )
    if ncontour and ncontour > 0:
        p.add_mesh(surf.contour(ncontour, scalars="dpred"), line_width=1, show_scalar_bar=False)
    if show_points:
        p.add_points(cloud, color="black", point_size=6, render_points_as_spheres=True)
    p.add_text(title, position="upper_edge", font_size=12)
    p.show_bounds(grid="front", xtitle="x (m)", ytitle="y (m)", ztitle="z (m)")
    p.enable_parallel_projection()
    p.view_xy()
    p.show()




