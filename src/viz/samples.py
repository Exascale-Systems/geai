import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from simpeg import maps
from simpeg.utils import plot2Ddata
if os.environ.get("DISPLAY"):
    mpl.use("Qt5Agg") 
import pyvista as pv


def plot_topography(topo_xyz):
    """
    Create a 3D scatter plot of topography points.
    """ 
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(topo_xyz[:,0], topo_xyz[:,1], topo_xyz[:,2], 
           c=topo_xyz[:,2], cmap='terrain', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
def plot_density_contrast_2D(mesh, ind_active, true_model):
    """
    Plot a 2D density contrast model slice with colorbar.
    """
    fig = plt.figure(figsize=(7, 5))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)
    ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
    mesh.plot_slice(
        plotting_map * true_model,
        normal="Y",
        ax=ax1,
        ind=int(mesh.shape_cells[1] / 2), 
        grid=True,
        clim=(np.min(true_model), np.max(true_model)),
        pcolor_opts={"cmap": "viridis"},
    )
    ax1.set_title("Model slice at y = 0 m")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
    )
    cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)
    plt.show()

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
        clim=(0,5),
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
    fig = plt.figure(figsize=(7, 5))
    v_max = np.max(np.abs(dpred))
    ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
    plot2Ddata(
        receiver_locations,
        dpred,
        clim=(-v_max, v_max),
        ax=ax1,
        ncontour=ncontour,
        contourOpts={"cmap": cmap},
    )
    ax1.set_title(title)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    if show_points:
        xy = receiver_locations[:, :2]
        ax1.scatter(
            xy[:, 0], xy[:, 1],
            s=18, marker='o', c='k',
            edgecolors='w', linewidths=0.4,
            zorder=3, label='Stations'
        )
        ax1.legend(loc='upper right', frameon=True)
        ax1.set_aspect('equal', adjustable='box')
    ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.85])
    norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=getattr(mpl.cm, cmap), format="%.1e"
    )
    if units:
        cbar.set_label(units, rotation=270, labelpad=15, size=12)
    plt.show()



