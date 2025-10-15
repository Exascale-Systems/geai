import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from simpeg import maps
from simpeg.utils import plot2Ddata


def plot_topography(topo_xyz):
    """
    Create a 3D scatter plot of topography points.

    Parameters
    ----------
    topo_xyz : ndarray, shape (N,3)
        Point cloud of topography.
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
    Plot a density contrast model slice with colorbar.

    Parameters
    ----------
    mesh : discretize.TensorMesh
        The mesh object used in the simulation.
    ind_active : numpy.ndarray (bool)
        Boolean mask for active cells.
    true_model : numpy.ndarray
        The density contrast model (active cell values).
    """

    fig = plt.figure(figsize=(7, 5))
    plotting_map = maps.InjectActiveCells(mesh, ind_active, np.nan)

    # Main slice plot
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

    # Colorbar
    ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
    norm = mpl.colors.Normalize(vmin=np.min(true_model), vmax=np.max(true_model))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
    )
    cbar.set_label("$g/cm^3$", rotation=270, labelpad=15, size=12)

    plt.show()


def plot_density_contrast_3D(mesh, ind_active, blocks_mask):
    """
    Create a 3D scatter plot of the density contrast model.
    Parameters
    ----------
    mesh : discretize.TensorMesh
        The mesh object used in the simulation.
    ind_active : numpy.ndarray (bool)
        Boolean mask for active cells.
    blocks_mask : numpy.ndarray (bool)
        Boolean mask for cells occupied by density blocks.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    CCa = mesh.gridCC[ind_active]
    ax.scatter(CCa[~blocks_mask,0], CCa[~blocks_mask,1], CCa[~blocks_mask,2], s=1, alpha=0.04)
    ax.scatter(CCa[blocks_mask,0],   CCa[blocks_mask,1],   CCa[blocks_mask,2],   s=1)
    plt.show()


def plot_density_contrast_3D_voxels(mesh, ind_active, blocks_mask):
    """
    Plot 3D voxel grid showing:
      - semi-transparent active (earth) volume
      - solid blocks overlay for density anomalies
    """
    # --- 1. Full-size masks ---
    full_active = np.zeros(mesh.nC, dtype=bool)
    full_active[ind_active] = True

    full_blocks = np.zeros(mesh.nC, dtype=bool)
    full_blocks[ind_active] = blocks_mask

    # --- 2. Reshape to 3D (Fortran order) ---
    active_vol = full_active.reshape(mesh.shape_cells, order="F")
    block_vol  = full_blocks.reshape(mesh.shape_cells, order="F")

    # --- 3. Compute voxel edge grids ---
    x_edges = mesh.x0[0] + np.r_[0.0, np.cumsum(mesh.h[0])]
    y_edges = mesh.x0[1] + np.r_[0.0, np.cumsum(mesh.h[1])]
    z_edges = mesh.x0[2] + np.r_[0.0, np.cumsum(mesh.h[2])]
    X, Y, Z = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")

    # --- 4. Plot ---
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    # Base active domain (semi-transparent gray)
    ax.voxels(
        X, Y, Z, active_vol,
        facecolors="lightgray",
        edgecolor="none",
        alpha=0.30
    )

    # Overlay blocks (solid blue)
    ax.voxels(
        X, Y, Z, block_vol,
        facecolors="tab:blue",
        edgecolor="k",
        linewidth=0.2,
        alpha=0.9
    )

    # --- 5. Formatting ---
    ax.set_box_aspect((np.sum(mesh.h[0]), np.sum(mesh.h[1]), np.sum(mesh.h[2])))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Density Blocks Overlay on Active Volume")
    plt.tight_layout()
    plt.show()


def plot_gravity_measurements(receiver_locations, dpred, title="Gravity Anomaly (Z-component)",
                         units="$mgal$", show_points=True, cmap="bwr", ncontour=30):
    """
    Plot a 2D gravity anomaly map with survey stations.

    Parameters
    ----------
    receiver_locations : (N, 3) array
        Station coordinates (x, y, z).
    dpred : array_like
        Predicted data values at the stations.
    title : str, optional
        Title for the plot.
    units : str, optional
        Units for the colorbar.
    show_points : bool, optional
        If True, overlay station locations as dots.
    cmap : str, optional
        Matplotlib colormap name.
    ncontour : int, optional
        Number of contour levels.
    """

    fig = plt.figure(figsize=(7, 5))

    v_max = np.max(np.abs(dpred))

    # Main plot
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

    # Overlay survey stations
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

    # Colorbar
    ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.85])
    norm = mpl.colors.Normalize(vmin=-v_max, vmax=v_max)
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=getattr(mpl.cm, cmap), format="%.1e"
    )
    if units:
        cbar.set_label(units, rotation=270, labelpad=15, size=12)

    plt.show()



