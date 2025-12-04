import ipywidgets as widgets
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.plotting.scatter import plot_scatters_in_triangle
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.rbf import RBF
from bluemath_tk.waves.series import waves_dispersion
from ipywidgets import interact


def animate_case_propagation(
    case_dataset, depth, tini=0, tend=30, tstep=2, figsize=(15, 5)
):
    """
    Function to animate the propagation of the swash for a single case
    """

    fig, ax = plt.subplots(1, figsize=figsize)

    # Init animation
    def init():
        return []

    # Función de actualización de la animación
    def update(frame):
        y_lim = [-10, 6]
        ax.clear()

        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.set_xlim(0, 600)
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xlabel("Cross-shore Distance (m)", fontsize=12)
        ax.set_ylabel("Elevation (m)", fontsize=12)

        # bathymetry
        ax.fill_between(
            np.arange(len(depth)),
            -depth,
            y_lim[0],
            fc="wheat",
            zorder=2,
        )

        # waves
        elev = case_dataset.isel(Tsec=frame)["Watlev"].values
        ax.fill_between(
            np.arange(len(depth)),
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


def show_graph_for_different_parameters(pca: PCA, rbf: RBF, lhs_parameters, depthfile):
    """
    Show graph for different parameters
    """

    # Function to update the plot based on widget input
    def update_plot(Hs, Hs_L0, Wv, hv, Nv):
        data = {"Hs": Hs, "Hs_L0": Hs_L0, "Wv": Wv, "hv": hv, "Nv": Nv}
        df_dataset_single_case = pd.DataFrame([data])

        # Spatial Reconstruction
        predicted_hs = rbf.predict(dataset=df_dataset_single_case)
        predicted_hs_ds = xr.Dataset(
            {
                "PCs": (["case_num", "n_component"], predicted_hs.values),
            },
            coords={
                "case_num": [0],
                "n_component": np.arange(len(pca.pcs_df.columns)),
            },
        )
        ds_output_all = pca.inverse_transform(PCs=predicted_hs_ds)

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(11, 3))
        plot_depthfile(depthfile=depthfile, ax=ax)
        ds_output_all["Hs"].sel(case_num=0).plot(x="Xp", ax=ax, color="k")
        ax.set_title("")
        depth = np.loadtxt(depthfile)

        # Improved vegetation visualization using fill_between
        veg_start = 400 - int(Wv)
        veg_end = 400

        n_stems = int(Nv / 10)  # Number of stems based on density
        stem_positions = np.linspace(veg_start, veg_end - 1, n_stems).astype(int)
        for x in stem_positions:
            y_base = -depth[x]
            ax.plot(
                [x, x],
                [y_base, y_base + hv],
                color="green",
                linewidth=2,
                alpha=0.8,
                zorder=4,
            )

        ax.set_ylim(-12, 6)
        ax.set_xlim(0, 600)
        ax.grid(True)

        # ax.set_title(
        #    f"Reconstructed Hs for Hs: {hs}, Hs_L0: {hs_l0}, wl: {wl}, vegetation height: {vegetation_height} and plants density: {plants_density}"
        # )

    # variables_to_analyse_in_metamodel = ["Hs", "Hs_L0", "WL","vegetation_height","plants_density"]
    # lhs_parameters = {
    #     "num_dimensions": 5,
    #     "num_samples": 10000,
    #     "dimensions_names": variables_to_analyse_in_metamodel,
    #     "lower_bounds": [0.5, 0.005, 0, 0, 0],
    #     "upper_bounds": [2, 0.05, 1, 1.5, 1000],
    # }
    i = 0
    parameters = {}
    # Create widgets for each parameter
    for param in lhs_parameters["dimensions_names"]:
        min = lhs_parameters["lower_bounds"][i]
        max = lhs_parameters["upper_bounds"][i]
        step = (max - min) / 10
        value = np.random.uniform(min, max)
        parameters[param] = widgets.FloatSlider(
            value=value,
            min=min,
            max=max,
            step=step,
            description=param,
            continuous_update=False,
        )
        i = i + 1

    # Using interact to link widgets to the function
    return interact(
        update_plot,
        Hs=parameters["Hs"],
        Hs_L0=parameters["Hs_L0"],
        Wv=parameters["Wv"],
        hv=parameters["hv"],
        Nv=parameters["Nv"],
    )


def show_graph_for_all_vegetations(pca: PCA, rbf: RBF, depth, hs=2.0, hs_l0=0.02):
    """
    Show graph for all vegetations
    """

    # Create dataframe
    df_dataset_same_hs = pd.DataFrame(
        data={
            "Hs": np.repeat(hs, 100),
            "Hs_L0": np.repeat(hs_l0, 100),
            "VegetationHeight": np.linspace(0, 1.5, 100),
        }
    )

    # Spatial Reconstruction
    predicted_hs = rbf.predict(dataset=df_dataset_same_hs)

    predicted_hs_ds = xr.Dataset(
        {
            "PCs": (["case_num", "n_component"], predicted_hs.values),
        },
        coords={
            "case_num": range(100),
            "n_component": np.arange(len(pca.pcs_df.columns)),
        },
    )

    # Get reconstructed Hs
    ds_output_all = pca.inverse_transform(PCs=predicted_hs_ds)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create a colormap and a normalization based on vegetation values
    norm = colors.Normalize(vmin=0, vmax=1.5)
    cmap = cm.get_cmap("YlGn", 100)

    for i, case_num in enumerate(ds_output_all["case_num"].values):
        vegetation = df_dataset_same_hs["VegetationHeight"].iloc[case_num]
        color = cmap(norm(vegetation))
        ds_output_all["Hs"].isel(case_num=case_num).plot(
            x="Xp",
            # label=f'Case {case} (Veg={vegetation:.2f})',
            color=color,
            ax=ax,
        )

    # Add colorbar for vegetation scale
    smp = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smp.set_array([])
    cbar = plt.colorbar(smp, ax=ax)
    cbar.set_label("Vegetation")

    # bathymetry
    # ax.fill_between(
    #     np.arange(len(depth)),
    #     np.ones(len(depth)) * depth[-1],
    #     -depth,
    #     fc="wheat",
    #     zorder=2,
    # )
    # ax.plot(
    #     np.arange(int(pp.swash_proj.np_ini), int(pp.swash_proj.np_fin)),
    #     np.repeat(-2.5, int(pp.swash_proj.np_fin - pp.swash_proj.np_ini)),
    #     color="darkgreen",
    #     linewidth=10,
    # )
    ax.set_ylim(-7, 4)
    ax.set_xlim(400, 1160)
    ax.grid(True)

    ax.set_title(
        f"Reconstructed Hs for Hs: {hs}, Hs_L0: {hs_l0} and different vegetation heights"
    )


def plot_depthfile(depthfile, ax=None, xlim=None, dxinp=1):
    """
    Plot bathymetry data including friction or vegetation area in case active commands

    Parameters
    ----------
    depthfile : str
        Path to the depth file.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    xlim : tuple, optional
        x-axis limits.
    dxinp : float, optional
        Increment in x direction.
    """

    depth = np.loadtxt(depthfile)
    x = np.arange(0, len(depth) * dxinp, dxinp)

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
    # ax.plot(
    #     x, -depth,
    #     color = 'grey',
    #     zorder = 3,
    # )

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


def plot_scatters_Tp(df_centroids, df_lhs_data, scatter_points_thick=10):
    """
    Plot peak wave period (Tp) scatter plots in triangular format.

    This function calculates the peak wave period (Tp) from significant wave height (Hs)
    and wave steepness (Hs_L0) using the deep water wave dispersion relation, then creates
    scatter plots comparing different wave and vegetation parameters.

    Parameters
    ----------
    df_centroids : pandas.DataFrame
        DataFrame containing centroid data with wave and vegetation parameters.
        Expected columns: 'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'
    df_lhs_data : pandas.DataFrame
        DataFrame containing Latin Hypercube Sampling data with the same parameters.
        Expected columns: 'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'
    scatter_points_thick : int

    Returns
    -------
    tuple
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

    # Note: The following lines should use the function parameters instead of 'mda'

    df_centroids["Tp"] = np.sqrt(
        (df_centroids["Hs"].values * 2 * np.pi) / (9.806 * df_centroids["Hs_L0"])
    )
    df_lhs_data["Tp"] = np.sqrt(
        (df_lhs_data["Hs"].values * 2 * np.pi) / (9.806 * df_lhs_data["Hs_L0"])
    )
    df_centroids = df_centroids.drop(columns=["Hs_L0"])
    df_lhs_data = df_lhs_data.drop(columns=["Hs_L0"])
    df_centroids = df_centroids[["Hs", "Tp", "Wv", "hv", "Nv"]]
    df_lhs_data = df_lhs_data[["Hs", "Tp", "Wv", "hv", "Nv"]]

    fig, axes = plot_scatters_in_triangle(
        dataframes=[df_lhs_data, df_centroids],
        s=scatter_points_thick,
        data_colors=["blue", "red"],
    )
    fig.set_size_inches(8, 8)


def get_real_scenarios_from_dataset(lhs_dataset, h0):
    """
    Function to filter the LHS dataset to obtain real scenarios.
    It filters out scenarios that are not physically realistic based on the wave dispersion relation.

    Parameters
    ----------
    lhs_dataset : pandas.DataFrame
        DataFrame containing the Latin Hypercube Sampling data with columns:
        'Hs', 'Hs_L0', 'Wv', 'hv', 'Nv'

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only physically realistic scenarios.
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
