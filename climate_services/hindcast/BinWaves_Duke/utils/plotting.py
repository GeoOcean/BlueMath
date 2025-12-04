from typing import Any, Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import wavespectra
import xarray as xr
from bluemath_tk.core.operations import get_uv_components
from bluemath_tk.core.plotting.colors import colormap_spectra
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde


def safe_gaussian_kde(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """
    Create a Gaussian KDE that handles NaN values properly.

    Parameters
    ----------
    x_data : np.ndarray
        First dataset (e.g., buoy data)
    y_data : np.ndarray
        Second dataset (e.g., model data)

    Returns
    -------
    np.ndarray
        KDE values for the data points, or uniform values if KDE fails
    """

    try:
        # Stack the data and remove NaN values
        combined_data = np.vstack([x_data, y_data])

        # Create mask for finite values only
        finite_mask = np.isfinite(combined_data).all(axis=0)

        if finite_mask.sum() < 2:  # Need at least 2 points for KDE
            print(
                "Warning: Insufficient finite data points for KDE, using uniform coloring"
            )
            return np.ones_like(x_data)

        # Filter to finite values only
        finite_data = combined_data[:, finite_mask]

        # Create KDE with finite data only
        kde = gaussian_kde(finite_data)

        # Evaluate KDE on all original data points, but only where both are finite
        result = np.ones_like(x_data)  # Default to uniform
        finite_xy_mask = np.isfinite(x_data) & np.isfinite(y_data)

        if finite_xy_mask.sum() > 0:
            finite_combined = np.vstack(
                [x_data[finite_xy_mask], y_data[finite_xy_mask]]
            )
            result[finite_xy_mask] = kde(finite_combined)

        return result

    except Exception as e:
        print(f"Warning: KDE creation failed ({e}), using uniform coloring")
        return np.ones_like(x_data)


def axplot_spectrum(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vmax: Optional[float] = None,
    ylim: float = 0.49,
    cmap: str = "magma",
    plot_center: bool = False,
) -> Any:
    """
    Plot spectrum in given polar axes.

    Parameters
    ----------
    ax : Axes
        Input axes (polar)
    x : np.ndarray
        Spectrum directions
    y : np.ndarray
        Spectrum frequency
    z : np.ndarray
        Spectrum energy
    vmax : float, optional
        Maximum value for color scale. If None, calculated from data
    ylim : float, optional
        Y-axis limit, by default 0.49
    cmap : str, optional
        Colormap name, by default 'magma'
    plot_center : bool, optional
        Whether to plot center point, by default False

    Returns
    -------
    Any
        Matplotlib pcolormesh object
    """

    # fix coordinates for pcolormesh
    x1 = np.append(x, x[0])
    if plot_center:
        y1 = np.append(0, y)
    else:
        y1 = np.append(y, y[-1])
    z1 = z

    # If vmax is not provided, calculate it from the data for an optimal color scale
    if vmax is None:
        vmax = np.nanmax(np.sqrt(z1))
        if vmax == 0:  # Handle cases with no energy
            vmax = 0.1

    # polar pcolormesh
    p1 = ax.pcolormesh(
        x1,
        y1,
        np.sqrt(z1),
        vmin=0,
        vmax=vmax,
    )

    # polar axes configuration
    p1.set_cmap(cmap)
    ax.set_theta_zero_location("N", offset=0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)

    return p1


def Plot_spectrum(
    sp: Union[xr.Dataset, xr.DataArray],
    time_ix: int = 0,
    average: bool = False,
    ylim: float = 0.49,
    figsize: List[int] = [8, 8],
    title: str = "",
) -> Figure:
    """
    Plot superpoint spectrum at a time index or time average.

    Parameters
    ----------
    sp : Union[xr.Dataset, xr.DataArray]
        Spectrum dataset
    time_ix : int, optional
        Time index, instant to plot, by default 0
    average : bool, optional
        True to plot energy average, by default False
    ylim : float, optional
        Y-axis limit, by default 0.49
    figsize : List[int], optional
        Figure size, by default [8, 8]
    title : str, optional
        Title of the plot, by default ''

    Returns
    -------
    Figure
        Matplotlib figure object

    Raises
    ------
    ValueError
        If direction or frequency coordinates are not found
    TypeError
        If input is not an xarray Dataset or DataArray
    """

    # Detect direction coordinate name
    dir_coord_name = None
    if "dir" in sp.coords:
        dir_coord_name = "dir"
    elif "direction" in sp.coords:
        dir_coord_name = "direction"
    else:
        raise ValueError(
            f"Direction coordinate not found. Available coordinates: {list(sp.coords.keys())}"
        )

    # Detect frequency coordinate name
    freq_coord_name = None
    if "freq" in sp.coords:
        freq_coord_name = "freq"
    elif "frequency" in sp.coords:
        freq_coord_name = "frequency"
    else:
        raise ValueError(
            f"Frequency coordinate not found. Available coordinates: {list(sp.coords.keys())}"
        )

    # Handle both Dataset and DataArray inputs for 'sp' to make the function more flexible
    if isinstance(sp, xr.Dataset):
        # If it's a dataset, assume the data is in the 'efth' variable
        data_values = sp.efth.values
    elif isinstance(sp, xr.DataArray):
        # If it's a data array, use its values directly
        data_values = sp.values
    else:
        raise TypeError(
            f"Input 'sp' must be an xarray Dataset or DataArray, not {type(sp)}"
        )

    # superpoint spectrum energy (time index or time average)
    if not average:
        z = data_values[time_ix, :, :]
        ttl = title + " - time: {0}".format(sp.time[time_ix].values)

    else:
        # time average
        z = np.nanmean(data_values, axis=0)
        ttl = title

    # generate figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    # TODO: Check why the super point isn't doing the angle correction amd if the efth also need to be corrected
    # sp[dir_coord_name] = fix_dir(sp[dir_coord_name])
    # plot spectrum
    axplot_spectrum(
        ax,
        np.deg2rad(sp[dir_coord_name].values),
        sp[freq_coord_name].values,
        z,
        ylim=ylim,
    )
    ax.set_title(ttl, fontsize=14)

    return fig


def detect_coordinate_system(bathy: Union[xr.DataArray, xr.Dataset]) -> Dict[str, Any]:
    """
    Detect the coordinate system and extract coordinate variables from bathymetry data.

    Parameters
    ----------
    bathy : Union[xr.DataArray, xr.Dataset]
        Input bathymetry data with coordinates. Expected to have either:
        - lon/lat coordinates (geographic)
        - x/y coordinates (UTM)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - is_geographic: bool, whether the coordinates are geographic
        - x_coord: str, name of x coordinate
        - y_coord: str, name of y coordinate
        - proj: cartopy.crs projection or None
        - transform: cartopy.crs transform or None
    """

    # Determine coordinate system based on coordinate names
    coord_names = list(bathy.coords)
    is_geographic = any(name in ["lon", "longitude"] for name in coord_names) and any(
        name in ["lat", "latitude"] for name in coord_names
    )

    # Get coordinate variables
    if is_geographic:
        x_coord = next(name for name in coord_names if name in ["lon", "longitude"])
        y_coord = next(name for name in coord_names if name in ["lat", "latitude"])
        proj = ccrs.PlateCarree()
        transform = ccrs.PlateCarree()
    else:
        x_coord = next(
            name for name in coord_names if name in ["x", "X", "cx", "easting"]
        )
        y_coord = next(
            name for name in coord_names if name in ["y", "Y", "cy", "northing"]
        )
        proj = None
        transform = None

    return {
        "is_geographic": is_geographic,
        "x_coord": x_coord,
        "y_coord": y_coord,
        "proj": proj,
        "transform": transform,
    }


def plot_selected_bathy(
    bathy: xr.DataArray,
    utm_zone: Optional[int] = None,
    buoys: Optional[Dict[str, Tuple[float, float]]] = None,
    ax: Optional[Axes] = None,
) -> None:
    """
    Plot bathymetry data in either UTM or geographic coordinates.

    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data with coordinates. Expected to have either:
        - lon/lat coordinates (geographic)
        - x/y coordinates (UTM)
    utm_zone : int, optional
        UTM zone number if data is in UTM coordinates
    buoys : Dict[str, Tuple[float, float]], optional
        Dictionary of buoy names and their coordinates
        Example: {'NDBC-41001': (x1, y1), 'NDBC-41002': (x2, y2)}
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.

    Raises
    ------
    ValueError
        If UTM zone is not provided for UTM coordinates
    """

    # Get coordinate variables
    coords = detect_coordinate_system(bathy)
    is_geographic = coords["is_geographic"]
    x_coord = coords["x_coord"]
    y_coord = coords["y_coord"]
    proj = coords["proj"]
    transform = coords["transform"]

    # If UTM and no zone provided, raise error
    if not is_geographic and utm_zone is None:
        raise ValueError("UTM zone must be provided for UTM coordinates")
    elif not is_geographic:
        proj = ccrs.UTM(zone=utm_zone)
        transform = proj

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5), subplot_kw={"projection": proj})
        created_fig = True

    # Plot bathymetry contours
    bathy.plot.contourf(
        x=x_coord,
        y=y_coord,
        ax=ax,
        levels=[10, 25, 50, 100, 200, 300, 500, 1000],
        cmap="Blues",
        transform=transform,
    )

    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=20)

    # Plot buoys if provided
    if buoys is not None:
        # Separate coordinates and names
        x_buoys = [coord[0] for coord in buoys.values()]
        y_buoys = [coord[1] for coord in buoys.values()]

        # Plot all buoys
        ax.scatter(
            x_buoys, y_buoys, c="darkred", transform=transform, label="Available Buoys"
        )

        for name, (x, y) in buoys.items():
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(3, 3),  # Small offset from point
                textcoords="offset points",
                fontsize=8,  # Smaller font size
                color="darkred",
                transform=transform,
                ha="left",  # Horizontal alignment
                va="bottom",  # Vertical alignment
            )

        # Add legend
        ax.legend(loc="upper right")  # Specify legend location

    # Set appropriate extent if geographic
    if is_geographic:
        x_min, x_max = bathy[x_coord].min().item(), bathy[x_coord].max().item()
        y_min, y_max = bathy[y_coord].min().item(), bathy[y_coord].max().item()
        ax.set_extent([x_min, x_max, y_min, y_max], crs=transform)

    ax.set_aspect("auto")

    if created_fig:
        plt.show()


def plot_cases_grid(
    data: xr.DataArray,
    cases_to_plot: List[int] = [0, 320, 615],
    colors_to_plot: List[str] = ["green", "orange", "purple"],
    num_directions: int = 24,
    num_frequencies: int = 29,
) -> None:
    """
    Plot all cases in a grid and selected cases with colored borders.

    Parameters
    ----------
    data : xr.DataArray
        Data array containing case data
    cases_to_plot : List[int], optional
        List of case numbers to highlight, by default [0, 320, 615]
    colors_to_plot : List[str], optional
        List of colors for highlighted cases, by default ["green", "orange", "purple"]
    num_directions : int, optional
        Number of directions, by default 24
    num_frequencies : int, optional
        Number of frequencies, by default 29
    """

    # Plot all cases in a grid
    fig, axes = plt.subplots(
        ncols=num_frequencies, nrows=num_directions, figsize=(29, 15)
    )
    for i, ax in enumerate(axes.flat):
        try:
            ax.pcolor(
                (
                    data.sel(case_num=i)
                    .isel(Xp=slice(None, None, 3), Yp=slice(None, None, 3))
                    .values
                ),
                cmap="RdBu_r",
                vmin=0,
                vmax=2,
            )
        except Exception as e:
            print(e)
    for i, ax in enumerate(axes.flat):
        ax.set_aspect("equal")
        ax.set_title("")
        ax.axis("off")
    fig.tight_layout()
    # Set texts in left part of grid and top part
    fig.text(
        0, 0.5, "Directions", ha="center", va="center", rotation="vertical", fontsize=20
    )
    fig.text(0.5, 1, "Frequencies", ha="center", va="center", fontsize=20)
    # Plot selected cases in a grid
    fig_sel, axes_sel = plt.subplots(
        ncols=len(cases_to_plot), nrows=1, figsize=(5 * len(cases_to_plot), 4)
    )
    for ax, ax_sel, case_to_plot, color_to_plot in zip(
        axes.flat[cases_to_plot], axes_sel.flat, cases_to_plot, colors_to_plot
    ):
        try:
            data.sel(case_num=case_to_plot).plot(
                ax=ax_sel,
                cmap="RdBu_r",
                vmin=0,
                vmax=2,
                add_colorbar=True,
                cbar_kwargs={"orientation": "horizontal", "shrink": 0.8},
            )
            ax_sel.set_aspect("equal")
            ax_sel.set_title("")
            # Remove ticks and labels
            ax_sel.set_xticks([])
            ax_sel.set_yticks([])
            ax_sel.set_xticklabels([])
            ax_sel.set_yticklabels([])
            ax_sel.set_xlabel("")
            ax_sel.set_ylabel("")
            # Set axis of color to indicate it is plotted
            ax_sel.spines["top"].set_color(color_to_plot)
            ax_sel.spines["top"].set_linewidth(2)
            ax_sel.spines["right"].set_color(color_to_plot)
            ax_sel.spines["right"].set_linewidth(2)
            ax_sel.spines["bottom"].set_color(color_to_plot)
            ax_sel.spines["bottom"].set_linewidth(2)
            ax_sel.spines["left"].set_color(color_to_plot)
            ax_sel.spines["left"].set_linewidth(2)
            # Set axis of color to indicate it is plotted
            ax.axis("on")
            # Remove ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Set axis of color to indicate it is plotted
            ax.spines["top"].set_color(color_to_plot)
            ax.spines["top"].set_linewidth(2)
            ax.spines["right"].set_color(color_to_plot)
            ax.spines["right"].set_linewidth(2)
            ax.spines["bottom"].set_color(color_to_plot)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_color(color_to_plot)
            ax.spines["left"].set_linewidth(2)
        except Exception as e:
            print(e)
    fig_sel.tight_layout()


def plot_case_variables(
    data: xr.Dataset,
    step: int = 10,
    vmin_hs: float = 0,
    vmax_hs: float = 1.5,
    vmin_tm: float = 0,
    vmax_tm: float = 20,
    vmin_dir: float = 0,
    vmax_dir: float = 360,
) -> None:
    """
    Plot the significant wave height, mean wave period, and wave direction from the dataset.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing the wave data with variables 'Hsig', 'Tm02', and 'Dir'
    step : int, optional
        Step size for quiver plot, by default 10
    vmin_hs : float, optional
        Minimum value for significant wave height, by default 0
    vmax_hs : float, optional
        Maximum value for significant wave height, by default 1.5
    vmin_tm : float, optional
        Minimum value for mean wave period, by default 0
    vmax_tm : float, optional
        Maximum value for mean wave period, by default 20
    vmin_dir : float, optional
        Minimum value for wave direction, by default 0
    vmax_dir : float, optional
        Maximum value for wave direction, by default 360
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    data["Hsig"].plot(
        ax=axes[0],
        cbar_kwargs={"label": "Hsig [m]", "orientation": "horizontal", "shrink": 0.7},
        cmap="RdBu_r",
        vmin=vmin_hs,
        vmax=vmax_hs,
    )
    data["Tm02"].plot(
        ax=axes[1],
        cbar_kwargs={"label": "Tm02 [s]", "orientation": "horizontal", "shrink": 0.7},
        cmap="magma",
        vmin=vmin_tm,
        vmax=vmax_tm,
    )
    data["Dir"].plot(
        ax=axes[2],
        cbar_kwargs={"label": "Dir [deg]", "orientation": "horizontal", "shrink": 0.7},
        cmap="twilight",
        vmin=vmin_dir,
        vmax=vmax_dir,
    )

    dir_u, dir_v = get_uv_components(data["Dir"].values)
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        step = step
        ax.quiver(
            data["Xp"][::step],
            data["Yp"][::step],
            -dir_u[::step, ::step],
            -dir_v[::step, ::step],
            color="grey",
            scale=25,
        )
    fig.tight_layout()


def create_text_with_metrics(array1: np.ndarray, array2: np.ndarray) -> str:
    """
    Create a text with metrics comparing two arrays, handling NaN values.
    Includes dispersion coefficients commonly used in wave validation.

    Parameters
    ----------
    array1 : np.ndarray
        Observed data (e.g., buoy measurements)
    array2 : np.ndarray
        Model data (e.g., BinWaves predictions)

    Returns
    -------
    str
        Formatted text string with metrics (MAE, RMSE, R², SI, Bias)
    """

    # Create mask for finite values in both arrays
    finite_mask = np.isfinite(array1) & np.isfinite(array2)

    if finite_mask.sum() < 2:  # Need at least 2 points for correlation
        return "MAE: nan\nRMSE: nan\nR²: nan\nSI: nan\nBias: nan"

    # Filter to finite values only
    obs_clean = array1[finite_mask]  # observations (buoy)
    model_clean = array2[finite_mask]  # model (BinWaves)

    # Calculate standard metrics
    mae = np.mean(np.abs(obs_clean - model_clean))
    rmse = np.sqrt(np.mean((obs_clean - model_clean) ** 2))
    r2 = np.corrcoef(obs_clean, model_clean)[0, 1] ** 2

    # Calculate dispersion coefficients
    bias = np.mean(model_clean - obs_clean)  # Mean bias (positive = model overestimate)
    scatter_index = rmse / np.mean(obs_clean)  # Normalized RMSE (SI)

    # Create text with additional metrics
    text = (
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R²: {r2:.2f}\n"
        f"SI: {scatter_index:.2f}\n"
        f"Bias: {bias:.2f}"
    )

    return text


def plot_wave_series(
    buoy_data: wavespectra.SpecArray,
    binwaves_data: wavespectra.SpecArray,
    offshore_data: wavespectra.SpecArray,
    times: np.ndarray,
    save_plots: bool = False,
    save_path: str = "wave_validation_plots",
    title_prefix: str = "Wave Validation",
    dpi: int = 300,
    format: str = "png",
) -> Tuple[Figure, np.ndarray, Figure, np.ndarray]:
    """
    Plot wave series comparison between buoy, BinWaves, and offshore data.

    Parameters
    ----------
    buoy_data : wavespectra.SpecArray
        Buoy wave data
    binwaves_data : wavespectra.SpecArray
        BinWaves reconstructed data
    offshore_data : wavespectra.SpecArray
        Offshore wave data
    times : np.ndarray
        Time array for plotting
    save_plots : bool, optional
        Whether to save the plots to files, by default False
    save_path : str, optional
        Base path for saving plots, by default "wave_validation_plots"
    title_prefix : str, optional
        Prefix for plot titles, by default "Wave Validation"
    dpi : int, optional
        DPI for saved plots, by default 300
    format : str, optional
        Format for saved plots, by default "png"

    Returns
    -------
    Tuple[Figure, np.ndarray, Figure, np.ndarray]
        (fig1, axes1, fig2, axes2) - Both figure and axes objects
    """

    buoy_color = "lightcoral"
    binwaves_color = "royalblue"
    # offshore_color = "gold"

    # Create first figure: Time series plots
    fig1, axes1 = plt.subplots(3, 1, figsize=(20, 10))

    # Plot time series
    buoy_data["Hs_Buoy"].plot(ax=axes1[0], label="Buoy", c=buoy_color, alpha=0.8, lw=1)
    buoy_data["Tp_Buoy"].plot(ax=axes1[1], label="Buoy", c=buoy_color, alpha=0.8, lw=1)
    axes1[2].scatter(
        times,
        buoy_data["Dir_Buoy"].values,
        c=buoy_color,
        label="Buoy",
        alpha=0.8,
        s=1,
    )
    binwaves_data.hs().plot(
        ax=axes1[0], label="BinWaves", c=binwaves_color, alpha=0.8, lw=1
    )
    binwaves_data.tp().plot(
        ax=axes1[1], label="BinWaves", c=binwaves_color, alpha=0.8, lw=1
    )
    axes1[2].scatter(
        times,
        binwaves_data.dpm().values,
        c=binwaves_color,
        label="BinWaves",
        alpha=0.8,
        s=1,
    )
    # offshore_data.hs().plot(
    #     ax=axes1[0], label="Offshore", c=offshore_color, alpha=0.5, lw=1
    # )
    # offshore_data.tp().plot(
    #     ax=axes1[1], label="Offshore", c=offshore_color, alpha=0.5, lw=1
    # )
    # axes1[2].scatter(
    #     times,
    #     offshore_data.dpm().values,
    #     c=offshore_color,
    #     label="Offshore",
    #     alpha=0.8,
    #     s=1,
    # )

    # Set labels and title for time series
    fig1.suptitle(f"{title_prefix} - Time Series", fontsize=16, fontweight="bold")
    axes1[0].set_ylabel("Hs [m]")
    axes1[0].legend()
    axes1[1].set_ylabel("T [s] - tp")
    axes1[2].set_ylabel("Dir [°] - dm")
    for ax in axes1:
        ax.set_title("")
        ax.grid(True, alpha=0.3)

    # Create second figure: Scatter plots
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    # Hs scatter plot
    hs_colors = safe_gaussian_kde(
        buoy_data["Hs_Buoy"].values, binwaves_data.hs().values
    )
    axes2[0].scatter(
        buoy_data["Hs_Buoy"].values,
        binwaves_data.hs().values,
        s=1,
        c=hs_colors,
        cmap="turbo",
    )
    axes2[0].text(
        5,
        0.5,
        create_text_with_metrics(
            buoy_data["Hs_Buoy"].values, binwaves_data.hs().values
        ),
        color="darkred",
    )
    axes2[0].plot([0, 7], [0, 7], c="darkred", linestyle="--")
    axes2[0].set_xlabel("Hs - Buoy [m]")
    axes2[0].set_ylabel("Hs - BinWaves [m]")
    axes2[0].set_xlim([0, 7])
    axes2[0].set_ylim([0, 7])

    # Tp scatter plot
    tp_colors = safe_gaussian_kde(
        buoy_data["Tp_Buoy"].values, binwaves_data.tp().values
    )
    axes2[1].scatter(
        buoy_data["Tp_Buoy"].values,
        binwaves_data.tp().values,
        s=1,
        c=tp_colors,
        cmap="turbo",
        label="Tp",
    )
    axes2[1].text(
        15,
        1.25,
        create_text_with_metrics(
            buoy_data["Tp_Buoy"].values, binwaves_data.tp().values
        ),
        color="darkred",
    )
    axes2[1].plot([0, 20], [0, 20], c="darkred", linestyle="--")
    axes2[1].set_xlabel("Tp - Buoy [s]")
    axes2[1].set_ylabel("Tp - BinWaves [s]")
    axes2[1].set_xlim([0, 20])
    axes2[1].set_ylim([0, 20])

    # Direction scatter plot
    dpm_colors = safe_gaussian_kde(
        buoy_data["Dir_Buoy"].values, binwaves_data.dpm().values
    )
    axes2[2].scatter(
        buoy_data["Dir_Buoy"].values,
        binwaves_data.dpm().values,
        s=1,
        c=dpm_colors,
        cmap="turbo",
        label="Dpm",
    )
    axes2[2].text(
        250,
        25,
        create_text_with_metrics(
            buoy_data["Dir_Buoy"].values, binwaves_data.dpm().values
        ),
        color="darkred",
    )
    axes2[2].plot([0, 360], [0, 360], c="darkred", linestyle="--")
    axes2[2].set_xlabel("Dir - Buoy [°]")
    axes2[2].set_ylabel("Dir - BinWaves [°]")
    axes2[2].set_xlim([0, 360])
    axes2[2].set_ylim([0, 360])

    # Set title for scatter plots
    fig2.suptitle(f"{title_prefix} - Scatter Plots", fontsize=16, fontweight="bold")

    # Format scatter plots
    for ax in axes2:
        ax.set_aspect("equal")
        # Delete top and right axis
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Set axis color and ticks to darkred
        ax.spines["left"].set_color("darkred")
        ax.spines["bottom"].set_color("darkred")
        ax.yaxis.label.set_color("darkred")
        ax.xaxis.label.set_color("darkred")
        ax.tick_params(axis="x", colors="darkred")
        ax.tick_params(axis="y", colors="darkred")

    # Save plots if requested
    if save_plots:
        import os

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save time series plot
        timeseries_filename = os.path.join(
            save_path, f"timeseries_{title_prefix}.{format}"
        )
        fig1.savefig(timeseries_filename, dpi=dpi, bbox_inches="tight")
        print(f"Time series plot saved as: {timeseries_filename}")

        # Save scatter plot
        scatter_filename = os.path.join(save_path, f"scatter_{title_prefix}.{format}")
        fig2.savefig(scatter_filename, dpi=dpi, bbox_inches="tight")
        print(f"Scatter plot saved as: {scatter_filename}")

    return fig1, axes1, fig2, axes2


def create_white_zero_colormap(cmap_name: str = "Spectral") -> LinearSegmentedColormap:
    """
    Create a colormap with white at zero, and selected colormap for positive values.

    Parameters
    ----------
    cmap_name : str, optional
        Name of the base colormap to use, by default "Spectral"

    Returns
    -------
    LinearSegmentedColormap
        Custom colormap with white at zero
    """

    # Get the base colormap
    base_cmap = plt.cm.get_cmap(cmap_name)

    # Create a colormap list with white at the beginning
    colors_list = [(0.0, (1.0, 1.0, 1.0, 1.0))]  # White at the start for zero

    # Add colors from the original colormap for positive values
    for i in np.linspace(0, 1, 100):
        colors_list.append((0.01 + 0.99 * i, base_cmap(i)))

    # Create the custom colormap
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        f"white_{cmap_name}", colors_list
    )

    return custom_cmap


def create_custom_bathy_cmap() -> LinearSegmentedColormap:
    """
    Create a custom colormap for bathymetry visualization.

    Returns
    -------
    LinearSegmentedColormap
        Custom bathymetry colormap with blue to brown color scheme
    """

    # Define your colors
    custom_colors = [
        "#4a84b5",
        "#5493c8",
        "#5fa9d1",
        "#74c3dc",
        "#8ed7e8",
        "#a0e2ef",
        "#b7f1eb",
        "#c8ebd8",
        "#d7e8c3",
        "#e2e5a5",
        "#f4cda0",
        "#f1e2c6",
    ]
    # Create the custom colormap
    custom_cmap = colors.LinearSegmentedColormap.from_list(
        "custom_bathy_cmap", custom_colors
    )
    return custom_cmap


def plot_spectrum_in_coastline(
    bathy: xr.DataArray,
    reconstructed_onshore_spectra: xr.Dataset,
    reconstruction_kps: xr.Dataset,  # TODO: This is not used
    offshore_spectra: xr.Dataset,
    time_to_plot: str,
    sites_for_spectrum: List[int],
) -> Tuple[Figure, Axes]:
    """
    Plot gridded graph with wave spectra visualization.
    Handles both geographic (lat/lon) and Cartesian (UTM) coordinates.

    Parameters
    ----------
    bathy : xr.DataArray
        Bathymetry data
    reconstructed_onshore_spectra : xr.Dataset
        Reconstructed onshore wave spectra
    reconstruction_kps : xr.Dataset
        Reconstruction key points (currently unused)
    offshore_spectra : xr.Dataset
        Offshore wave spectra
    time_to_plot : str
        Time string to plot
    sites_for_spectrum : List[int]
        List of site indices for spectrum plotting

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and axes objects

    Raises
    ------
    Exception
        If there's an error in plotting the spectrum
    """

    try:
        # Print shapes for debugging

        # list(reconstruction_kps.coords)
        time_slice = reconstructed_onshore_spectra.sel(
            time=time_to_plot, method="nearest"
        )

        # Use the utility function
        coords = detect_coordinate_system(bathy)
        is_geographic = coords["is_geographic"]
        x_coord = coords["x_coord"]
        y_coord = coords["y_coord"]
        proj = coords["proj"]
        transform = coords["transform"]

        # Create figure with proper projection if geographic
        if is_geographic:
            fig, ax = plt.subplots(figsize=(15, 6), subplot_kw={"projection": proj})
            ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
        else:
            fig, ax = plt.subplots(figsize=(15, 6))

        # Plot bathymetry as a contour
        plot_kwargs = {
            "ax": ax,
            "levels": [0, -10, -25, -50, -100, -200, -500, -1000],
            "cmap": "Blues_r",
            "add_colorbar": False,
        }
        if is_geographic:
            plot_kwargs.update({"x": x_coord, "y": y_coord, "transform": transform})
        bathy.plot.contourf(**plot_kwargs)

        if is_geographic:
            # Use reconstructed_onshore_spectra coordinates since they match the sites
            x_vals = reconstructed_onshore_spectra.coord_x.values
            y_vals = reconstructed_onshore_spectra.coord_y.values
        else:
            x_vals = reconstructed_onshore_spectra.coord_x.values
            y_vals = reconstructed_onshore_spectra.coord_y.values

        # Calculate Hs directly from the time slice
        hs_values = time_slice.kp.spec.hs().values

        scatter_kwargs = {
            "c": hs_values,
            "cmap": colormap_spectra(),
            "s": 20,
        }
        if is_geographic:
            scatter_kwargs["transform"] = transform

        phs = ax.scatter(x_vals, y_vals, **scatter_kwargs)
        plt.colorbar(phs).set_label("Hs [m]")

        for site in sites_for_spectrum:
            try:
                # Get site coordinates
                if is_geographic:
                    lon = reconstructed_onshore_spectra.coord_x.values[site]
                    lat = reconstructed_onshore_spectra.coord_y.values[site]
                else:
                    lon = reconstructed_onshore_spectra.coord_x.values[site]
                    lat = reconstructed_onshore_spectra.coord_y.values[site]

                # Create inset with explicit size relative to data coordinates
                inset_width = (
                    bathy[x_coord].max() - bathy[x_coord].min()
                ) * 0.1  # 10% of plot width
                axin = ax.inset_axes(
                    [
                        lon - inset_width / 2,
                        lat - inset_width / 2,
                        inset_width,
                        inset_width,
                    ],
                    transform=ax.transData,
                    projection="polar",
                )

                # Mark the site location
                site_scatter_kwargs = {"c": "black", "marker": "*", "s": 100}
                if is_geographic:
                    site_scatter_kwargs["transform"] = transform
                ax.scatter(lon, lat, **site_scatter_kwargs)

                # Get and plot the spectrum
                spectrum = time_slice.isel(site=site).kp
                if not np.all(np.isnan(spectrum)):
                    _pcm = axin.pcolormesh(
                        np.deg2rad(reconstructed_onshore_spectra.dir.values),
                        reconstructed_onshore_spectra.freq.values,
                        np.sqrt(spectrum),
                        cmap=colormap_spectra(),
                    )
                    axin.set_theta_zero_location("N", offset=0)
                    axin.set_theta_direction(-1)
                    axin.axis("off")
                else:
                    print(f"Warning: NaN values found in spectrum for site {site}")
            except Exception as e:
                print(f"Error plotting site {site}: {str(e)}")
                continue

        # Set reasonable axis limits based on bathymetry extent
        if is_geographic:
            ax.set_extent(
                [
                    float(bathy[x_coord].min()),
                    float(bathy[x_coord].max()),
                    float(bathy[y_coord].min()),
                    float(bathy[y_coord].max()),
                ],
                crs=transform,
            )
            ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        else:
            ax.set_xlim([bathy[x_coord].min(), bathy[x_coord].max()])
            ax.set_ylim([bathy[y_coord].min(), bathy[y_coord].max()])
            ax.grid(True, linestyle="--", alpha=0.5)

        return fig, ax

    except Exception as e:
        print(f"Error in plot_spectrum_in_coastline: {str(e)}")
        raise
