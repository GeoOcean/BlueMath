from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bluemath_tk.core.plotting.scatter import plot_scatters_in_triangle
from bluemath_tk.waves.series import waves_dispersion


def animate_case_propagation(
    case_dataset: pd.DataFrame,
    depth: np.ndarray,
    tini: int = 0,
    tend: int = 30,
    tstep: int = 2,
    figsize: Tuple[int, int] = (15, 5),
) -> animation.FuncAnimation:
    """
    Animate the propagation of swash for a single case.

    This function creates an animation showing the wave propagation over time,
    displaying the water level and bathymetry at different time steps.

    Parameters
    ----------
    case_dataset : pd.DataFrame
        Dataset containing wave propagation data with columns 'Xp' (cross-shore distance)
        and 'Watlev' (water level elevation), indexed by time ('Tsec').
    depth : np.ndarray
        Bathymetry depth values (positive downward).
    tini : int, optional
        Initial time step for animation. Default is 0.
    tend : int, optional
        Final time step for animation. Default is 30.
    tstep : int, optional
        Time step increment for animation frames. Default is 2.
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in inches. Default is (15, 5).

    Returns
    -------
    animation.FuncAnimation
        Matplotlib animation object showing wave propagation over time.

    Notes
    -----
    The animation shows:
    - Bathymetry in wheat color (land)
    - Water level in light blue color
    - Time progression in the title
    - Cross-shore distance on x-axis
    - Elevation on y-axis
    """

    fig, ax = plt.subplots(1, figsize=figsize)

    # Init animation
    def init():
        return []

    # Función de actualización de la animación
    def update(frame):
        x = case_dataset["Xp"].values
        x_lim = [min(x), max(x)]
        y_lim = [-10, 6]
        ax.clear()

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xlabel("Cross-shore Distance (m)", fontsize=12)
        ax.set_ylabel("Elevation (m)", fontsize=12)

        # bathymetry
        ax.fill_between(
            # np.arange(len(depth)),
            x,
            -depth,
            y_lim[0],
            fc="wheat",
            zorder=2,
        )

        # waves
        elev = case_dataset.isel(Tsec=frame)["Watlev"].values
        ax.fill_between(
            # np.arange(len(depth)),
            x,
            -depth,
            elev,
            fc="deepskyblue",
            alpha=0.5,
            zorder=1,
        )
        ax.set_title("Time : {0} s".format(frame), fontsize=12)

        return []

    # Crear animación
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(tini, tend, tstep), init_func=init, blit=True
    )
    plt.close()

    # Mostrar animación
    return ani


def plot_depthfile(
    depthfile: str,
    meshProfilefile: str,
    ax: Optional[plt.Axes] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot bathymetry data from depth file.

    This function reads bathymetry data from files and creates a visualization
    showing the underwater topography with water and land areas.

    Parameters
    ----------
    depthfile : str
        Path to the depth file containing bathymetry data.
        Expected format: text file with depth values.
    meshProfilefile : str
        Path to the mesh profile file containing x-coordinates.
        Expected format: text file with coordinate data.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. Default is None, which creates a new figure.
    xlim : Tuple[float, float], optional
        X-axis limits as (xmin, xmax). Default is None, which uses full range of data.

    Returns
    -------
    None
        The function modifies the provided axes or creates a new plot.

    Notes
    -----
    The plot shows:
    - Water area in light blue (deepskyblue)
    - Land area in wheat color
    - Depth values on y-axis (negative values)
    - Distance on x-axis
    """

    depth = np.loadtxt(depthfile)[2, :] * -1
    x = np.loadtxt(meshProfilefile)[2, :]

    if not ax:
        fig, ax = plt.subplots(1, figsize=(11, 3))

    ax.fill_between(
        x,
        -depth[0],
        np.zeros((len(depth))),
        facecolor="deepskyblue",
        alpha=0.5,
        zorder=1,
    )
    ax.fill_between(
        x,
        np.zeros((len(depth))) - depth[0],
        -depth,
        facecolor="wheat",
        alpha=1,
        zorder=2,
    )

    if not xlim:
        ax.set_xlim(x[0], x[-1])

    ax.set_xlim(xlim)
    ax.set_ylim(-depth[0], None)
    ax.set_ylabel("Depth (m)")
    ax.set_xlabel("Distance (m)")

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scatters_Tp(
    df_centroids: pd.DataFrame,
    df_lhs_data: pd.DataFrame,
    scatter_points_thick: int = 10,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot peak wave period (Tp) scatter plots in triangular format.

    This function calculates the peak wave period (Tp) from significant wave height (Hs)
    and wave steepness (Hs_L0) using the deep water wave dispersion relation, then creates
    scatter plots comparing different wave and vegetation parameters.

    Parameters
    ----------
    df_centroids : pd.DataFrame
        DataFrame containing centroid data with wave and vegetation parameters.
        Expected columns: 'Hs', 'Hs_L0', 'WL'
    df_lhs_data : pd.DataFrame
        DataFrame containing Latin Hypercube Sampling data with the same parameters.
        Expected columns: 'Hs', 'Hs_L0', 'WL'
    scatter_points_thick : int, optional
        Size of scatter points. Default is 10.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        A tuple containing (fig, axes) from the triangular scatter plot

    Notes
    -----
    The peak wave period is calculated using the formula:
    Tp = sqrt((Hs * 2 * π) / (g * Hs_L0))

    Where:
    - Hs: Significant wave height (m)
    - Hs_L0: Wave steepness (dimensionless)
    - g: Gravitational acceleration (9.806 m/s²)

    The function creates scatter plots with:
    - Blue points: LHS data
    - Red points: Centroid data
    """

    df_centroids["Tp"] = np.sqrt(
        (df_centroids["Hs"].values * 2 * np.pi) / (9.806 * df_centroids["Hs_L0"])
    )
    df_lhs_data["Tp"] = np.sqrt(
        (df_lhs_data["Hs"].values * 2 * np.pi) / (9.806 * df_lhs_data["Hs_L0"])
    )
    df_centroids = df_centroids.drop(columns=["Hs_L0"])
    df_lhs_data = df_lhs_data.drop(columns=["Hs_L0"])
    df_centroids = df_centroids[["Hs", "Tp", "WL"]]
    df_lhs_data = df_lhs_data[["Hs", "Tp", "WL"]]

    fig, axes = plot_scatters_in_triangle(
        dataframes=[df_lhs_data, df_centroids],
        s=scatter_points_thick,
        data_colors=["blue", "red"],
    )
    fig.set_size_inches(7, 7)

    return fig, axes


def get_real_scenarios_from_dataset(
    lhs_dataset: pd.DataFrame, h0: float
) -> pd.DataFrame:
    """
    Filter the LHS dataset to obtain physically realistic scenarios.

    This function filters out scenarios that are not physically realistic based on
    the wave dispersion relation and wave theory constraints.

    Parameters
    ----------
    lhs_dataset : pd.DataFrame
        DataFrame containing the Latin Hypercube Sampling data with columns:
        'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'
    h0 : float
        Water depth (m) used for wave dispersion calculations.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only physically realistic scenarios.

    Notes
    -----
    The function applies the following physical constraints:
    - kh < 1: Ensures waves are not in deep water (k*h < 1)
    - Tp > 7: Ensures minimum wave period of 7 seconds
    - h/L < 0.5: Ensures waves are not in shallow water (h/L < 0.5)

    Where:
    - k: Wavenumber (2π/L)
    - h: Water depth
    - L: Wavelength
    - Tp: Peak wave period
    """

    # Calculation of the wave length and other parameters to avoid unphysical values
    df_centroids = lhs_dataset.copy()
    df_centroids["Tp"] = np.sqrt(
        (df_centroids["Hs"].values * 2 * np.pi) / (9.806 * df_centroids["Hs_L0"])
    )
    df_centroids["L"] = [waves_dispersion(i, h0)[0] for i in df_centroids["Tp"]]
    df_centroids["h/L"] = h0 / df_centroids["L"]
    df_centroids["kh"] = (2 * np.pi / df_centroids["L"]) * h0
    df_centroids = df_centroids.loc[(df_centroids["kh"] < 1) & (df_centroids["Tp"] > 7)]
    df_centroids = df_centroids.loc[(df_centroids["h/L"] < 0.5)]
    df_dataset = lhs_dataset.loc[df_centroids.index]

    return df_dataset
