from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ── Figure style ──────────────────────────────────────────────────────────────
# One palette for the whole project so every figure in the paper reads as part of
# the same system. The three categorical hues were validated for colour-vision
# deficiency (worst adjacent pair dE 11.6 protan / 18.7 tritan) and for contrast
# against a white surface. Do not cycle or extend this list: a fourth category
# means a different chart, not a new hue.

COLORS: Dict[str, str] = {
    "data": "#9AA5B1",  # the full cloud of cases — recessive by design
    "primary": "#3A6EA5",  # blue    · the dataset / PCA
    "selected": "#D1495B",  # crimson · cases selected by MDA
    "test": "#E9A23B",  # amber   · cases held out for validation
    "grid": "#DDE2E7",
    "ink": "#2B3137",
    "muted": "#6B7580",
}

#: Perceptually uniform, colour-vision-safe sequential map for magnitudes.
SEQUENTIAL_CMAP = "viridis"


def set_paper_style() -> None:
    """
    Apply a clean, publication-oriented Matplotlib style.

    Notes
    -----
    Keeps axes and grid recessive so the data marks carry the figure. Call once
    at the top of a notebook; every plotting function in this module assumes it
    but does not depend on it.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "text.color": COLORS["ink"],
        }
    )


def plot_wind_domain(
    ds: xr.Dataset,
    u_var: str = "u10",
    v_var: str = "v10",
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the mean wind field over the ERA5 domain.

    Parameters
    ----------
    ds : xr.Dataset
        Wind dataset with dimensions (time, lat, lon) containing the zonal and
        meridional components.
    u_var : str, optional
        Name of the zonal wind component. Default is 'u10'.
    v_var : str, optional
        Name of the meridional wind component. Default is 'v10'.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    Axes
        The axes containing the plot.

    Notes
    -----
    Colour encodes the time-mean wind speed (a magnitude, hence a sequential
    map); arrows show the time-mean vector, which is weaker than the mean speed
    wherever the direction varies. Grid nodes are marked so the coarseness of
    the ERA5 grid is visible rather than hidden by interpolation.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    u_mean = ds[u_var].mean(dim="time")
    v_mean = ds[v_var].mean(dim="time")
    speed_mean = np.sqrt(ds[u_var] ** 2 + ds[v_var] ** 2).mean(dim="time")

    mesh = ax.pcolormesh(
        ds.lon,
        ds.lat,
        speed_mean,
        cmap=SEQUENTIAL_CMAP,
        shading="nearest",
    )
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("mean wind speed (m/s)")
    cbar.outline.set_visible(False)

    lon_grid, lat_grid = np.meshgrid(ds.lon, ds.lat)
    ax.quiver(
        lon_grid,
        lat_grid,
        u_mean,
        v_mean,
        color="white",
        width=0.004,
        scale=30,
    )
    ax.scatter(
        lon_grid,
        lat_grid,
        s=9,
        color="white",
        edgecolors=COLORS["ink"],
        linewidths=0.5,
        zorder=3,
    )

    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.set_title(
        f"ERA5 10 m wind · {ds.sizes['time']:,} times · "
        f"{ds.sizes['lat']}x{ds.sizes['lon']} grid"
    )
    ax.grid(False)

    return ax


def plot_wind_eofs(
    pca: Any,
    num_eofs: int = 4,
    u_var: str = "u10",
    v_var: str = "v10",
    ncols: int = 2,
    axes: Optional[Sequence[Axes]] = None,
) -> Figure:
    """
    Plot the EOFs of a wind field as vector patterns.

    Parameters
    ----------
    pca : bluemath_tk.datamining.pca.PCA
        A fitted PCA model whose `eofs` contain the zonal and meridional
        loadings on a (n_component, lat, lon) grid.
    num_eofs : int, optional
        Number of EOFs to plot. Silently clipped to the number of components
        actually retained. Default is 4.
    u_var : str, optional
        Name of the zonal wind variable in the EOFs. Default is 'u10'.
    v_var : str, optional
        Name of the meridional wind variable in the EOFs. Default is 'v10'.
    ncols : int, optional
        Number of panel columns. Default is 2.
    axes : sequence of Axes, optional
        Axes to plot on, one per EOF. If None, a new figure is created.

    Returns
    -------
    Figure
        The figure containing the panels.

    Raises
    ------
    ValueError
        If the EOFs do not contain both wind components.

    Notes
    -----
    `PCA.plot_eofs` draws one scalar panel per variable and, when `map_center`
    is given, on an Orthographic (whole-globe) projection. For a domain of a
    degree or so that renders the study area as an invisible dot, and its
    `coastlines()` call needs to download Natural Earth data. This function
    plots the domain at its own extent instead, and — more usefully — treats
    each EOF as the *vector* pattern it is: colour is the magnitude of the
    loading and arrows show its direction, so u and v are read together
    rather than as two unrelated maps.

    The sign of an EOF is arbitrary: every arrow in a panel may flip together
    without changing the decomposition. Read the pattern, not the direction.
    """

    eofs = pca.eofs
    missing = [var for var in (u_var, v_var) if var not in eofs.data_vars]
    if missing:
        raise ValueError(
            f"EOFs are missing {missing}. Found: {list(eofs.data_vars)}"
        )

    available = eofs.sizes["n_component"]
    num_eofs = min(num_eofs, available)
    evr = np.asarray(getattr(pca, "explained_variance_ratio", []), dtype=float)

    nrows = int(np.ceil(num_eofs / ncols))
    if axes is None:
        fig, axes_grid = plt.subplots(
            nrows,
            ncols,
            figsize=(5.0 * ncols, 3.8 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        axes = axes_grid.flat
    else:
        axes = np.atleast_1d(np.asarray(axes, dtype=object)).flat
        fig = next(iter(np.atleast_1d(np.asarray(axes, dtype=object)))).get_figure()

    axes = list(axes)
    lon_grid, lat_grid = np.meshgrid(eofs.lon, eofs.lat)

    # A shared colour scale, so panel-to-panel differences in amplitude are
    # real rather than an artefact of per-panel normalisation.
    magnitudes = [
        np.sqrt(
            eofs[u_var].isel(n_component=k) ** 2 + eofs[v_var].isel(n_component=k) ** 2
        )
        for k in range(num_eofs)
    ]
    vmax = float(max(m.max() for m in magnitudes))

    mesh = None
    for k in range(num_eofs):
        ax = axes[k]
        mesh = ax.pcolormesh(
            eofs.lon,
            eofs.lat,
            magnitudes[k],
            cmap=SEQUENTIAL_CMAP,
            vmin=0.0,
            vmax=vmax,
            shading="nearest",
        )
        ax.quiver(
            lon_grid,
            lat_grid,
            eofs[u_var].isel(n_component=k),
            eofs[v_var].isel(n_component=k),
            color="white",
            width=0.006,
            scale=2.0,
        )

        title = f"EOF {k + 1}"
        if k < len(evr):
            title += f"  ·  {evr[k]:.1%} of variance"
        ax.set_title(title)
        # Only the outer panels carry axis labels; the grids are shared.
        if k >= num_eofs - ncols:
            ax.set_xlabel("longitude (deg E)")
        if k % ncols == 0:
            ax.set_ylabel("latitude (deg N)")
        ax.grid(False)

    for ax in axes[num_eofs:]:
        ax.set_visible(False)

    cbar = fig.colorbar(
        mesh, ax=axes[:num_eofs], pad=0.02, fraction=0.03, aspect=30
    )
    cbar.set_label("loading magnitude")
    cbar.outline.set_visible(False)


    return fig


def plot_explained_variance(
    explained_variance_ratio: Union[np.ndarray, Sequence[float]],
    num_pcs: Optional[int] = None,
    target: Optional[float] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the variance explained by each principal component.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray or sequence of float
        Fraction of variance explained by each component, in order.
    num_pcs : int, optional
        Number of components to show. If None, all are shown.
    target : float, optional
        Cumulative variance threshold to mark with a reference line, e.g. 0.95.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.

    Returns
    -------
    Axes
        The axes containing the plot.

    Notes
    -----
    Individual and cumulative variance share a single 0-100 % axis, so the two
    series are directly comparable. Values are labelled on the bars rather than
    read off the grid.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    evr = np.asarray(explained_variance_ratio, dtype=float)
    if num_pcs is not None:
        evr = evr[:num_pcs]
    ranks = np.arange(1, len(evr) + 1)
    cumulative = np.cumsum(evr)

    ax.bar(
        ranks,
        evr * 100,
        color=COLORS["primary"],
        width=0.6,
        label="individual",
        zorder=2,
    )
    ax.plot(
        ranks,
        cumulative * 100,
        color=COLORS["selected"],
        linewidth=2,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="cumulative",
        zorder=3,
    )

    for rank, value in zip(ranks, evr * 100):
        if value >= 2.0:
            ax.text(
                rank,
                value + 1.5,
                f"{value:.0f}",
                ha="center",
                fontsize=8,
                color=COLORS["muted"],
            )

    if target is not None:
        ax.axhline(
            target * 100,
            color=COLORS["muted"],
            linestyle=(0, (4, 3)),
            linewidth=1,
            zorder=1,
        )
        ax.text(
            ranks[-1],
            target * 100 + 1.5,
            f"{target:.0%}",
            ha="right",
            fontsize=8,
            color=COLORS["muted"],
        )

    ax.set_xticks(ranks)
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance explained (%)")
    ax.set_ylim(0, 105)
    ax.set_title("PCA of the ERA5 wind field")
    ax.legend(loc="center right")

    return ax


def plot_mda_selection(
    pcs_df: pd.DataFrame,
    selected_idx: Union[List[int], np.ndarray],
    test_idx: Optional[Union[List[int], np.ndarray]] = None,
    pc_pairs: Sequence[Tuple[int, int]] = ((1, 2), (1, 3)),
    label_first: int = 0,
    axes: Optional[Sequence[Axes]] = None,
) -> Figure:
    """
    Plot the MDA-selected cases against the full cloud in principal-component space.

    Parameters
    ----------
    pcs_df : pd.DataFrame
        Principal components indexed by time, with columns 'PC1', 'PC2', ...
    selected_idx : list of int or np.ndarray
        Positional indices of the selected cases.
    test_idx : list of int or np.ndarray, optional
        Positional indices of cases held out for validation.
    pc_pairs : sequence of tuple of int, optional
        Component pairs to plot, 1-based. Default is ((1, 2), (1, 3)).
    label_first : int, optional
        Annotate the first N selected cases with their MDA rank. Default is 0
        (no annotation). Useful for small selections; unreadable above ~20.
    axes : sequence of Axes, optional
        Axes to plot on, one per pair. If None, a new figure is created.

    Returns
    -------
    Figure
        The figure containing the plots.

    Raises
    ------
    ValueError
        If the number of provided axes does not match the number of pairs.

    Notes
    -----
    The full cloud is deliberately recessive: the question the figure answers is
    whether the selection reaches the edges of the cloud, which is where the
    extreme sea states live.
    """

    if axes is None:
        fig, axes = plt.subplots(
            1, len(pc_pairs), figsize=(5.5 * len(pc_pairs), 4.8)
        )
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_1d(np.asarray(axes, dtype=object))
        if len(axes) != len(pc_pairs):
            raise ValueError(
                f"Got {len(axes)} axes for {len(pc_pairs)} component pairs"
            )
        fig = axes[0].get_figure()

    selected_idx = np.asarray(selected_idx, dtype=int)

    for ax, (i, j) in zip(axes, pc_pairs):
        x, y = pcs_df[f"PC{i}"].values, pcs_df[f"PC{j}"].values

        ax.scatter(
            x, y, s=6, color=COLORS["data"], alpha=0.35, linewidths=0, zorder=1,
            label=f"all cases (n={len(pcs_df):,})",
        )
        if test_idx is not None and len(test_idx) > 0:
            test_idx = np.asarray(test_idx, dtype=int)
            ax.scatter(
                x[test_idx], y[test_idx], s=55, marker="s",
                color=COLORS["test"], edgecolors="white", linewidths=1.0,
                zorder=2, label=f"test (n={len(test_idx)})",
            )
        ax.scatter(
            x[selected_idx], y[selected_idx], s=70,
            color=COLORS["selected"], edgecolors="white", linewidths=1.0,
            zorder=3, label=f"MDA selected (n={len(selected_idx)})",
        )

        for rank in range(min(label_first, len(selected_idx))):
            k = selected_idx[rank]
            ax.annotate(
                str(rank + 1),
                (x[k], y[k]),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=8,
                fontweight="semibold",
                color=COLORS["selected"],
            )

        ax.set_xlabel(f"PC{i}")
        ax.set_ylabel(f"PC{j}")
        ax.axhline(0, color=COLORS["grid"], linewidth=0.8, zorder=0)
        ax.axvline(0, color=COLORS["grid"], linewidth=0.8, zorder=0)

    # A figure-level legend below the panels: with the cloud filling the axes
    # there is no reliable empty corner for an inset legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "MDA case selection in PC space", y=1.0, fontsize=11, fontweight="semibold"
    )
    fig.tight_layout()

    return fig


def plot_selection_coverage(
    wind_df: pd.DataFrame,
    selected_idx: Union[List[int], np.ndarray],
    test_idx: Optional[Union[List[int], np.ndarray]] = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    Plot the selected cases on a wind speed / direction polar diagram.

    Parameters
    ----------
    wind_df : pd.DataFrame
        Domain-averaged wind indexed by time, with columns 'wspd' (m/s) and
        'wdir' (degrees, meteorological convention).
    selected_idx : list of int or np.ndarray
        Positional indices of the selected cases.
    test_idx : list of int or np.ndarray, optional
        Positional indices of cases held out for validation.
    ax : Axes, optional
        Polar axes to plot on. If None, a new polar figure is created.

    Returns
    -------
    Axes
        The polar axes containing the plot.

    Notes
    -----
    This is a sanity check on the selection, not the selection criterion: MDA
    works on the principal components of the full field, so a selection can be
    well spread here and still miss spatial patterns, or the reverse. North is
    up and angles increase clockwise, matching the meteorological convention.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

    theta = np.deg2rad(wind_df["wdir"].values)
    radius = wind_df["wspd"].values
    selected_idx = np.asarray(selected_idx, dtype=int)

    ax.scatter(theta, radius, s=5, color=COLORS["data"], alpha=0.3,
               linewidths=0, zorder=1, label=f"all cases (n={len(wind_df):,})")
    if test_idx is not None and len(test_idx) > 0:
        test_idx = np.asarray(test_idx, dtype=int)
        ax.scatter(theta[test_idx], radius[test_idx], s=55, marker="s",
                   color=COLORS["test"], edgecolors="white", linewidths=1.0,
                   zorder=2, label=f"test (n={len(test_idx)})")
    ax.scatter(theta[selected_idx], radius[selected_idx], s=70,
               color=COLORS["selected"], edgecolors="white", linewidths=1.0,
               zorder=3, label=f"MDA selected (n={len(selected_idx)})")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(112.5)
    for label in ax.get_yticklabels():
        # The radial labels sit on top of the cloud; a surface-coloured backing
        # keeps them readable without hiding data.
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.set_title("Coverage in wind speed and direction", pad=18)
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.08))

    return ax


#: Colormap per wave variable. Hsig and Tm02 are magnitudes, so they get
#: perceptually uniform sequential maps; Dir is an angle and needs a *cyclic*
#: map, or 359 deg and 1 deg would read as opposite extremes.
WAVE_CMAPS: Dict[str, str] = {
    "Hsig": "viridis",
    "Tm02": "cividis",
    "Dir": "twilight_shifted",
}


def plot_wave_examples(
    ds: xr.Dataset,
    variables: Optional[List[str]] = None,
    time_indices: Optional[List[int]] = None,
    vmin_dict: Optional[Dict[str, float]] = None,
    vmax_dict: Optional[Dict[str, float]] = None,
    wind: Optional[xr.Dataset] = None,
    wind_step: int = 14,
    wind_scale: float = 350.0,
    mask_wind: bool = True,
) -> Figure:
    """
    Plot a grid of wave fields, one row per time and one column per variable.

    Parameters
    ----------
    ds : xr.Dataset
        Wave fields with dimensions (lat, lon, time). Select a single tide
        level before calling.
    variables : list of str, optional
        Variables to show. Default is ['Hsig', 'Tm02', 'Dir'].
    time_indices : list of int, optional
        Time indices to show. Default is [0, 1, 2].
    vmin_dict : dict, optional
        Lower colour limit per variable, e.g. {'Hsig': 0}.
    vmax_dict : dict, optional
        Upper colour limit per variable, e.g. {'Hsig': 0.3}.
    wind : xr.Dataset, optional
        High-resolution wind with 'M' (speed) and 'Dir' (direction) on a
        (time, lat, lon) grid. When given, an extra column is added showing the
        wind that forced each case: speed in colour, direction as arrows.
        Use the key 'wind' in `vmin_dict` / `vmax_dict` to fix its colour scale.
    wind_step : int, optional
        Draw one arrow every `wind_step` grid cells. Default is 14.
    wind_scale : float, optional
        Matplotlib quiver scale; larger means shorter arrows. Default is 350.
    mask_wind : bool, optional
        Hide the wind over land, using the same mask as the wave fields, so the
        columns read as one figure. Default is True. Set to False to see the
        forcing over the whole domain.

    Returns
    -------
    Figure
        The figure containing the panels.

    Notes
    -----
    Each variable keeps one colour scale across all rows, so a field really is
    comparable between times. Set the limits explicitly through `vmin_dict` and
    `vmax_dict` when comparing runs, otherwise they come from the data shown.

    The wind column is interpolated onto the wave grid, both so the panels line
    up and because that is what the model actually saw: the wrapper interpolates
    the wind to the bathymetry grid before writing `wind_file.dat`. The arrows
    are built the same way the wrapper builds them
    (``u = -sin(Dir) * M``, ``v = -cos(Dir) * M``), so with `Dir` in the
    nautical convention they point where the wind blows *towards*.
    """

    variables = variables or ["Hsig", "Tm02", "Dir"]
    time_indices = time_indices or [0, 1, 2]
    vmin_dict, vmax_dict = vmin_dict or {}, vmax_dict or {}

    ds = ds.transpose("lat", "lon", "time")
    n_rows = len(time_indices)
    n_cols = len(variables) + (1 if wind is not None else 0)

    if wind is not None:
        missing = [var for var in ("M", "Dir") if var not in wind.data_vars]
        if missing:
            raise ValueError(
                f"wind is missing {missing}. Found: {list(wind.data_vars)}"
            )
        # Match each wave time to the wind field that forced it, then move the
        # wind onto the wave grid so the columns share axes.
        wind = (
            wind.sel(time=ds["time"].values, method="nearest")
            .assign_coords(time=ds["time"].values)
            .interp(lat=ds["lat"], lon=ds["lon"], method="linear")
        )
        wind_rad = np.deg2rad(wind["Dir"])
        wind = wind.assign(
            u=-np.sin(wind_rad) * wind["M"],
            v=-np.cos(wind_rad) * wind["M"],
        )
        if mask_wind:
            sea = ds[variables[0]].notnull().any(dim="time")
            wind = wind.where(sea)
        wind_limits = (
            vmin_dict.get("wind", 0.0),
            vmax_dict.get("wind", float(wind["M"].isel(time=time_indices).max())),
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.6 * n_cols, 3.6 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    # One colour scale per variable, shared by every row.
    limits = {}
    for var in variables:
        data = ds[var].isel(time=time_indices)
        limits[var] = (
            vmin_dict.get(var, float(data.min())),
            vmax_dict.get(var, float(data.max())),
        )

    for i, t in enumerate(time_indices):
        for j, var in enumerate(variables):
            ax = axes[i][j]
            vmin, vmax = limits[var]
            mesh = ax.pcolormesh(
                ds["lon"],
                ds["lat"],
                ds[var].isel(time=t),
                cmap=WAVE_CMAPS.get(var, SEQUENTIAL_CMAP),
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            if i == 0:
                cbar = fig.colorbar(
                    mesh, ax=axes[:, j].tolist(), pad=0.02, fraction=0.04, aspect=40
                )
                cbar.set_label(var)
                cbar.outline.set_visible(False)
            if j == 0:
                stamp = str(ds["time"].isel(time=t).values)[:16].replace("T", " ")
                ax.set_ylabel(f"{stamp}\nlatitude (deg N)")
            if i == n_rows - 1:
                ax.set_xlabel("longitude (deg E)")
            ax.grid(False)

        if wind is None:
            continue

        # ── Wind column ──────────────────────────────────────────────────────
        j = len(variables)
        ax = axes[i][j]
        wind_t = wind.isel(time=t)

        mesh = ax.pcolormesh(
            ds["lon"],
            ds["lat"],
            wind_t["M"],
            cmap="Oranges",
            shading="auto",
            vmin=wind_limits[0],
            vmax=wind_limits[1],
        )
        step = wind_step
        ax.quiver(
            ds["lon"][::step],
            ds["lat"][::step],
            wind_t["u"][::step, ::step],
            wind_t["v"][::step, ::step],
            color=COLORS["ink"],
            width=0.005,
            scale=wind_scale,
        )
        if i == 0:
            cbar = fig.colorbar(
                mesh, ax=axes[:, j].tolist(), pad=0.02, fraction=0.04, aspect=40
            )
            cbar.set_label("wind speed (m/s)")
            cbar.outline.set_visible(False)
        if i == n_rows - 1:
            ax.set_xlabel("longitude (deg E)")
        ax.grid(False)

    return fig


def plot_kfold_curves(
    curves: Dict[str, pd.DataFrame],
    units: Optional[Dict[str, str]] = None,
    chosen: Optional[int] = None,
    axes: Optional[Sequence[Axes]] = None,
) -> Figure:
    """
    Plot the K-Fold error against the number of training cases.

    Parameters
    ----------
    curves : dict of str to pd.DataFrame
        One entry per variable. Each frame is indexed by the number of training
        cases and has the columns 'mean' and 'std'.
    units : dict of str to str, optional
        Unit label per variable, e.g. {'Hsig_rescaled': 'm'}.
    chosen : int, optional
        Number of cases to mark with a vertical line, once decided.
    axes : sequence of Axes, optional
        Axes to plot on, one per variable. If None, a new figure is created.

    Returns
    -------
    Figure
        The figure containing the panels.

    Raises
    ------
    ValueError
        If the number of provided axes does not match the number of variables.

    Notes
    -----
    One panel per variable rather than one shared axis: the variables are in
    metres, seconds and degrees, and putting them on a common scale would only
    make the largest one legible.

    The shaded band is +/- 1 standard deviation across folds. Where it is wide
    compared with the gap between two case counts, those two are not really
    distinguishable — which is usually the point at which adding cases stops
    paying for itself.
    """

    units = units or {}
    variables = list(curves)

    if axes is None:
        fig, axes_grid = plt.subplots(
            1, len(variables), figsize=(4.8 * len(variables), 3.8),
            squeeze=False, layout="constrained",
        )
        axes = list(axes_grid[0])
    else:
        axes = list(np.atleast_1d(np.asarray(axes, dtype=object)))
        if len(axes) != len(variables):
            raise ValueError(
                f"Got {len(axes)} axes for {len(variables)} variables"
            )
        fig = axes[0].get_figure()

    for ax, var in zip(axes, variables):
        curve = curves[var]
        n_cases = curve.index.values

        ax.fill_between(
            n_cases,
            curve["mean"] - curve["std"],
            curve["mean"] + curve["std"],
            color=COLORS["primary"],
            alpha=0.18,
            linewidth=0,
            label="±1 std across folds",
        )
        ax.plot(
            n_cases,
            curve["mean"],
            color=COLORS["primary"],
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="K-Fold mean",
        )

        if chosen is not None:
            ax.axvline(
                chosen, color=COLORS["selected"], linestyle=(0, (4, 3)),
                linewidth=1.5, zorder=1,
            )
            ax.text(
                chosen, ax.get_ylim()[1], f" {chosen} cases",
                color=COLORS["selected"], fontsize=8, va="top",
            )

        unit = units.get(var, "")
        ax.set_title(f"{var}{f'  ({unit})' if unit else ''}")
        ax.set_xlabel("training cases")
        ax.set_ylabel(f"RMSE{f' ({unit})' if unit else ''}")
        ax.set_xticks(n_cases)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="upper right")
    fig.suptitle("RBF error vs training-set size", fontsize=11,
                 fontweight="semibold")

    return fig


def plot_reconstruction_maps(
    truth: xr.DataArray,
    predicted: xr.DataArray,
    times: Sequence[Any],
    var_label: str = "",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    error_limit: Optional[float] = None,
) -> Figure:
    """
    Compare the modelled and reconstructed fields side by side, one row per time.

    Parameters
    ----------
    truth : xr.DataArray
        The field produced by SWAN, with dimensions (lat, lon, time).
    predicted : xr.DataArray
        The field rebuilt from the RBF prediction, same dimensions.
    times : sequence
        Time stamps to show, one row each.
    var_label : str, optional
        Label for the colour bar, e.g. 'Hsig (m)'.
    cmap : str, optional
        Colormap for the two field columns. Defaults to the project sequential map.
    vmin, vmax : float, optional
        Colour limits shared by the SWAN and RBF columns. Taken from the data
        when not given.
    error_limit : float, optional
        Symmetric limit for the error column. Taken from the data when not given.

    Returns
    -------
    Figure
        The figure containing the panels.

    Notes
    -----
    The two field columns share one scale, so any visible difference between
    them is real. The error column is diverging around zero with a neutral
    midpoint, which is what separates 'too high' from 'too low' at a glance —
    a sequential map here would hide the sign.
    """

    truth = truth.transpose("lat", "lon", "time")
    predicted = predicted.transpose("lat", "lon", "time")

    if vmin is None:
        vmin = float(min(truth.min(), predicted.min()))
    if vmax is None:
        vmax = float(max(truth.max(), predicted.max()))
    if error_limit is None:
        error_limit = float(np.nanmax(np.abs((predicted - truth).values)))

    fig, axes = plt.subplots(
        len(times), 3, figsize=(13.5, 3.6 * len(times)),
        squeeze=False, sharex=True, sharey=True, layout="constrained",
    )

    field_mesh = error_mesh = None
    for i, t in enumerate(times):
        panels = [
            (truth.sel(time=t), cmap or SEQUENTIAL_CMAP, vmin, vmax),
            (predicted.sel(time=t), cmap or SEQUENTIAL_CMAP, vmin, vmax),
            (predicted.sel(time=t) - truth.sel(time=t), "RdBu_r",
             -error_limit, error_limit),
        ]
        for j, (data, cm, lo, hi) in enumerate(panels):
            ax = axes[i][j]
            mesh = ax.pcolormesh(
                truth["lon"], truth["lat"], data,
                cmap=cm, shading="auto", vmin=lo, vmax=hi,
            )
            if j < 2:
                field_mesh = mesh
            else:
                error_mesh = mesh
            if i == 0:
                ax.set_title(["SWAN", "RBF reconstruction", "RBF − SWAN"][j])
            if j == 0:
                stamp = str(np.datetime64(t))[:16].replace("T", " ")
                ax.set_ylabel(f"{stamp}\nlatitude (deg N)")
            if i == len(times) - 1:
                ax.set_xlabel("longitude (deg E)")
            ax.grid(False)

    bar = fig.colorbar(field_mesh, ax=axes[:, :2].ravel().tolist(), pad=0.02,
                       fraction=0.03, aspect=40)
    bar.set_label(var_label)
    bar.outline.set_visible(False)
    bar = fig.colorbar(error_mesh, ax=axes[:, 2].ravel().tolist(), pad=0.02,
                       fraction=0.05, aspect=40)
    bar.set_label(f"error ({var_label})" if var_label else "error")
    bar.outline.set_visible(False)

    return fig


def plot_gauge_validation(
    observed: pd.Series,
    predicted: pd.Series,
    var_label: str = "",
    window: Optional[Tuple[Any, Any]] = None,
    axes: Optional[Sequence[Axes]] = None,
) -> Figure:
    """
    Compare a reconstructed series against tide-gauge observations.

    Parameters
    ----------
    observed : pd.Series
        Observed values indexed by time.
    predicted : pd.Series
        Reconstructed values indexed by the same times.
    var_label : str, optional
        Axis label, e.g. 'Hs (m)'.
    window : tuple, optional
        (start, end) to show in the time-series panel. Defaults to the busiest
        stretch of the record.
    axes : sequence of Axes, optional
        Two axes: scatter and time series. If None, a new figure is created.

    Returns
    -------
    Figure
        The figure containing both panels.

    Raises
    ------
    ValueError
        If fewer than two axes are provided.
    """

    paired = pd.concat(
        {"observed": observed, "predicted": predicted}, axis=1
    ).dropna()

    if axes is None:
        fig, axes_grid = plt.subplots(
            1, 2, figsize=(13, 4.4),
            gridspec_kw={"width_ratios": [1, 2]}, layout="constrained",
        )
        axes = list(axes_grid)
    else:
        axes = list(np.atleast_1d(np.asarray(axes, dtype=object)))
        if len(axes) < 2:
            raise ValueError("Two axes are required: scatter and time series")
        fig = axes[0].get_figure()

    # ── Scatter ──────────────────────────────────────────────────────────────
    ax = axes[0]
    limit = float(max(paired.max()))
    ax.scatter(
        paired["observed"], paired["predicted"],
        s=8, alpha=0.25, color=COLORS["primary"], linewidths=0,
    )
    ax.plot([0, limit], [0, limit], color=COLORS["muted"],
            linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(f"observed {var_label}")
    ax.set_ylabel(f"reconstructed {var_label}")

    error = paired["predicted"] - paired["observed"]
    stats = (
        f"n = {len(paired):,}\n"
        f"r = {paired['observed'].corr(paired['predicted']):.2f}\n"
        f"RMSE = {np.sqrt((error ** 2).mean()):.3f}\n"
        f"bias = {error.mean():+.3f}"
    )
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va="top", fontsize=9,
            color=COLORS["muted"])

    # ── Time series ──────────────────────────────────────────────────────────
    ax = axes[1]
    if window is None:
        # The densest month of the record, so the panel shows a continuous run
        # rather than isolated points scattered over years.
        counts = paired.resample("30D").size()
        start = counts.idxmax()
        window = (start, start + pd.Timedelta(days=30))
    shown = paired.loc[window[0]:window[1]]

    # Break the line across gaps in the record: joining across a missing day
    # would draw a straight segment that reads as data.
    gaps = shown.index.to_series().diff() > pd.Timedelta(hours=2)
    shown = shown.copy()
    shown[gaps.values] = np.nan

    ax.plot(shown.index, shown["observed"], color=COLORS["ink"], linewidth=1.4,
            marker="o", markersize=3, label="tide gauge")
    ax.plot(shown.index, shown["predicted"], color=COLORS["selected"],
            linewidth=1.4, marker="o", markersize=3, label="RBF")
    ax.set_ylabel(var_label)
    ax.set_xlabel("")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=25)

    return fig


def plot_error_chain(
    observed: pd.Series,
    swan: pd.Series,
    rbf: pd.Series,
    var_label: str = "",
    axes: Optional[Sequence[Axes]] = None,
) -> Figure:
    """
    Show where the error enters the chain: gauge, SWAN and metamodel.

    Parameters
    ----------
    observed : pd.Series
        Tide-gauge observations indexed by time.
    swan : pd.Series
        SWAN output at the gauge point, same times.
    rbf : pd.Series
        Metamodel reconstruction at the gauge point, same times.
    var_label : str, optional
        Axis label, e.g. 'Hsig (m)'.
    axes : sequence of Axes, optional
        Three axes. If None, a new figure is created.

    Returns
    -------
    Figure
        The figure containing the three panels.

    Raises
    ------
    ValueError
        If fewer than three axes are provided.

    Notes
    -----
    The panels are ordered so the reading is forced: the metamodel can only ever
    reproduce SWAN, so its agreement with SWAN (third panel) is what measures the
    metamodel, while the gap between SWAN and the gauge (first panel) is the
    ceiling that no amount of training data can lift. Comparing the metamodel to
    the gauge on its own conflates the two.
    """

    paired = pd.concat(
        {"observed": observed, "swan": swan, "rbf": rbf}, axis=1
    ).dropna()

    if axes is None:
        fig, axes_grid = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="constrained")
        axes = list(axes_grid)
    else:
        axes = list(np.atleast_1d(np.asarray(axes, dtype=object)))
        if len(axes) < 3:
            raise ValueError("Three axes are required")
        fig = axes[0].get_figure()

    limit = float(paired.max().max()) * 1.05
    panels = [
        ("observed", "swan", "tide gauge", "SWAN",
         "the physical model", COLORS["primary"]),
        ("observed", "rbf", "tide gauge", "metamodel",
         "the end result", COLORS["test"]),
        ("swan", "rbf", "SWAN", "metamodel",
         "the metamodel alone", COLORS["selected"]),
    ]

    for ax, (xk, yk, xl, yl, title, colour) in zip(axes, panels):
        x, y = paired[xk], paired[yk]
        ax.scatter(x, y, s=14, alpha=0.45, color=colour, linewidths=0)
        ax.plot([0, limit], [0, limit], color=COLORS["muted"],
                linestyle=(0, (4, 3)), linewidth=1)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", "box")
        ax.set_xlabel(f"{xl} {var_label}")
        ax.set_ylabel(f"{yl} {var_label}")
        ax.set_title(title)

        error = y - x
        ax.text(
            0.05, 0.95,
            f"r = {x.corr(y):.2f}\n"
            f"RMSE = {np.sqrt((error ** 2).mean()):.3f}\n"
            f"bias = {error.mean():+.3f}",
            transform=ax.transAxes, va="top", fontsize=9, color=COLORS["muted"],
        )

    fig.suptitle(
        f"Where the error comes from  ·  n = {len(paired)}",
        fontsize=11, fontweight="semibold",
    )

    return fig


def _draw_coastline(coastline: Any, ax: Axes, extent: Sequence[float], **kwargs) -> None:
    """
    Draw a coastline clipped to the axes extent.

    Parameters
    ----------
    coastline : geopandas.GeoDataFrame
        Coastline geometries in the same coordinates as the axes.
    ax : Axes
        Axes to draw on.
    extent : sequence of float
        (lon_min, lon_max, lat_min, lat_max) to clip to.
    **kwargs
        Passed to `GeoDataFrame.plot`, e.g. color and linewidth.

    Notes
    -----
    Plotted with `.plot()`, not `.boundary.plot()`. A coastline is stored as a
    MultiLineString, and the boundary of a line is its end points: calling
    `.boundary` here draws a handful of invisible dots instead of the coast.

    Clipping first matters for speed — the Europe-wide file holds some 75,000
    line segments and only a handful fall inside a bay.
    """

    import shapely

    clipped = coastline.clip(
        shapely.geometry.box(extent[0], extent[2], extent[1], extent[3])
    )
    if not clipped.empty:
        clipped.plot(ax=ax, **kwargs)


def plot_case_study(
    wave: xr.DataArray,
    wind: xr.Dataset,
    series: pd.DataFrame,
    times: Sequence[Any],
    orthophoto: Optional[np.ndarray] = None,
    coastline: Optional[Any] = None,
    var_label: str = "Hsig (m)",
    wave_clim: Optional[Tuple[float, float]] = None,
    wind_clim: Tuple[float, float] = (0.0, 15.0),
    wave_cmap: Optional[Any] = None,
    wind_cmap: Any = "Oranges",
    time_labels: Sequence[str] = ("quiet", "storm"),
) -> Figure:
    """
    Case-study figure: regional wind, local wave response, and the time series.

    Parameters
    ----------
    wave : xr.DataArray
        Reconstructed wave field with dimensions (lat, lon, time).
    wind : xr.Dataset
        ERA5 winds with 'u10' and 'v10' on a (time, lat, lon) grid, covering a
        wider area than the wave domain.
    series : pd.DataFrame
        Time series to plot underneath, one column per line, indexed by time.
    times : sequence
        Two time stamps: one quiet, one energetic. They get one row each and are
        marked on the series.
    orthophoto : np.ndarray, optional
        Aerial image, stretched to the wave domain bounds.
    coastline : geopandas.GeoDataFrame, optional
        Coastline in lon/lat, drawn over both map columns.
    var_label : str, optional
        Colour-bar label for the wave panels.
    wave_clim : tuple of float, optional
        Colour limits for the wave panels. Taken from the shown times if absent.
    wind_clim : tuple of float, optional
        Colour limits for the wind speed panels. Default is (0, 15) m/s.
    wave_cmap : str or Colormap, optional
        Colormap for the wave panels. Defaults to the project sequential map.
        Pass `bluemath_tk.core.plotting.colors.colormap_spectra()` to match the
        house style of the earlier figures.
    wind_cmap : str or Colormap, optional
        Colormap for the wind panels. Default is 'Oranges'.
    time_labels : sequence of str, optional
        Names for the two time stamps. They label the vertical markers on the
        series and are colour-matched to the map rows.

    Returns
    -------
    Figure
        The assembled figure.

    Raises
    ------
    ValueError
        If `times` does not contain exactly two time stamps.

    Notes
    -----
    The two map columns deliberately cover different areas: the wind is the
    regional forcing on the ERA5 grid, the waves are the local response inside
    the bay, and the bay is outlined on the wind panels so the scale difference
    is visible rather than implied.
    """

    if len(times) != 2:
        raise ValueError("Exactly two time stamps are required")

    wave = wave.transpose("lat", "lon", "time")
    bay = [
        float(wave.lon.min()), float(wave.lon.max()),
        float(wave.lat.min()), float(wave.lat.max()),
    ]
    if wave_clim is None:
        shown = wave.sel(time=list(times)).values
        wave_clim = (0.0, float(np.nanpercentile(shown, 99)))

    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75], hspace=0.28, wspace=0.18)

    speed = np.sqrt(wind["u10"] ** 2 + wind["v10"] ** 2)
    wind_mesh = wave_mesh = None

    # The markers and frames are annotation, not data, so they wear a text
    # token rather than a series hue: colouring them from the palette would
    # collide with the lines they sit on top of. The labels carry the identity.
    row_colours = [COLORS["muted"], COLORS["muted"]]

    for row, t in enumerate(times):
        # ── Regional wind ────────────────────────────────────────────────────
        ax = fig.add_subplot(grid[row, 0])
        snap = wind.sel(time=t)
        wind_mesh = ax.pcolormesh(
            wind.lon, wind.lat, speed.sel(time=t),
            cmap=wind_cmap, shading="nearest",
            vmin=wind_clim[0], vmax=wind_clim[1],
        )
        lon_grid, lat_grid = np.meshgrid(wind.lon, wind.lat)
        ax.quiver(lon_grid, lat_grid, snap["u10"], snap["v10"],
                  color=COLORS["ink"], width=0.005, scale=200)
        ax.add_patch(plt.Rectangle(
            (bay[0], bay[2]), bay[1] - bay[0], bay[3] - bay[2],
            fill=False, edgecolor=COLORS["selected"], linewidth=1.8, zorder=5,
        ))
        wind_extent = [
            float(wind.lon.min()), float(wind.lon.max()),
            float(wind.lat.min()), float(wind.lat.max()),
        ]
        if coastline is not None:
            _draw_coastline(coastline, ax, wind_extent,
                            color=COLORS["ink"], linewidth=0.7)
        ax.set_xlim(wind_extent[0], wind_extent[1])
        ax.set_ylim(wind_extent[2], wind_extent[3])
        stamp = str(np.datetime64(t))[:16].replace("T", " ")
        ax.set_title(f"{time_labels[row]} · {stamp}", color=row_colours[row])
        ax.set_ylabel("latitude (deg N)")
        for spine in ax.spines.values():
            spine.set_edgecolor(row_colours[row])
            spine.set_linewidth(1.6)
        ax.grid(False)

        # ── Local waves ──────────────────────────────────────────────────────
        ax = fig.add_subplot(grid[row, 1])
        if orthophoto is not None:
            # The image is stretched onto the model bounds: it carries no
            # georeference of its own, so this assumes it frames that domain.
            ax.imshow(orthophoto, extent=bay, origin="upper", aspect="auto")
        wave_mesh = ax.pcolormesh(
            wave.lon, wave.lat, wave.sel(time=t),
            cmap=wave_cmap or SEQUENTIAL_CMAP, shading="auto",
            vmin=wave_clim[0], vmax=wave_clim[1], alpha=0.85,
        )
        if coastline is not None:
            _draw_coastline(coastline, ax, bay, color="white", linewidth=1.0)
        ax.set_xlim(bay[0], bay[1])
        ax.set_ylim(bay[2], bay[3])
        ax.set_title("reconstructed wave field", color=row_colours[row])
        for spine in ax.spines.values():
            spine.set_edgecolor(row_colours[row])
            spine.set_linewidth(1.6)
        ax.grid(False)

        if row == 1:
            for axis in fig.axes[-2:]:
                axis.set_xlabel("longitude (deg E)")

    bar = fig.colorbar(wind_mesh, ax=[fig.axes[0], fig.axes[2]],
                       fraction=0.04, pad=0.02, aspect=30)
    bar.set_label("wind speed (m/s)")
    bar.outline.set_visible(False)
    bar = fig.colorbar(wave_mesh, ax=[fig.axes[1], fig.axes[3]],
                       fraction=0.04, pad=0.02, aspect=30)
    bar.set_label(var_label)
    bar.outline.set_visible(False)

    # ── Time series ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(grid[2, :])
    palette = [COLORS["ink"], COLORS["selected"], COLORS["primary"]]
    for column, colour in zip(series.columns, palette):
        ax.plot(series.index, series[column], color=colour, linewidth=1.4,
                label=column)
    top = series.max().max()
    span = series.index[-1] - series.index[0]
    for t, label, colour in zip(times, time_labels, row_colours):
        stamp = pd.Timestamp(t)
        ax.axvline(stamp, color=colour, linestyle=(0, (4, 3)),
                   linewidth=1.6, zorder=1)
        # Flip the label inwards near the right edge, or it is clipped.
        near_end = (stamp - series.index[0]) / span > 0.8
        ax.annotate(
            f"{label}\n{stamp:%d %b %H:%M}",
            xy=(stamp, top),
            xytext=(-5 if near_end else 5, -2),
            textcoords="offset points",
            color=COLORS["ink"], fontsize=9, fontweight="semibold",
            va="top", ha="right" if near_end else "left",
        )
    ax.set_ylim(top=top * 1.30)
    ax.set_ylabel(var_label)
    ax.legend(loc="upper center", ncol=len(series.columns),
              bbox_to_anchor=(0.5, 1.16))
    ax.tick_params(axis="x", rotation=20)

    return fig


def plot_selection_match(
    scores: pd.DataFrame,
    pcs_df: pd.DataFrame,
    target_days: pd.DatetimeIndex,
    rebuilt_days: pd.DatetimeIndex,
    daily_wind: pd.Series,
    chance: Optional[float] = None,
    reference: Optional[Tuple[float, float]] = None,
) -> Figure:
    """
    Compare a rebuilt MDA selection with the days that were actually simulated.

    Parameters
    ----------
    scores : pd.DataFrame
        One row per recipe tried, indexed by a short label, with the columns
        'evr' (explained variance ratio) and 'hits' (days recovered).
    pcs_df : pd.DataFrame
        Daily principal components of the ERA5 wind, indexed by day, with the
        columns 'PC1', 'PC2', ...
    target_days : pd.DatetimeIndex
        The days that were actually simulated — the set being reverse-engineered.
    rebuilt_days : pd.DatetimeIndex
        The days picked by the best recipe.
    daily_wind : pd.Series
        Daily mean wind speed (m/s) over the domain, indexed by day.
    chance : float, optional
        Expected number of hits from a random draw of the same size, drawn as a
        reference line on the first panel.
    reference : tuple of float, optional
        Range of hits that two independent runs of the same procedure recover
        from each other, drawn as a band on the first panel. This is the
        ceiling: a rebuilt selection cannot agree with the original more than
        the procedure agrees with itself.

    Returns
    -------
    Figure
        The figure containing the three panels.

    Notes
    -----
    Three questions, one per panel: how many days each recipe recovers, whether
    the two selections occupy the same region of PC space, and whether they
    sample the same range of wind speeds. A recipe can score badly on the first
    and still be the right family of method, which is what the other two panels
    are for.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    n_target = len(target_days)

    # ── 1. how many days each recipe recovers ────────────────────────────────
    ax = axes[0]
    labels = list(scores.index)
    positions = np.arange(len(labels))
    best = scores["hits"].idxmax()
    colors = [COLORS["selected"] if label == best else COLORS["primary"]
              for label in labels]

    hits_values = scores["hits"].astype(int).values
    ax.barh(positions, hits_values, color=colors, height=0.68)
    for y, hits in zip(positions, hits_values):
        ax.text(hits + n_target * 0.012, y, f"{hits}  ({100 * hits / n_target:.0f}%)",
                va="center", fontsize=8, color=COLORS["ink"])

    if reference is not None:
        ax.axvspan(reference[0], reference[1], color=COLORS["ink"], alpha=0.10,
                   linewidth=0, zorder=0)
        ax.text(np.mean(reference), -0.72, "MDA vs itself", fontsize=8,
                color=COLORS["ink"], ha="center", va="bottom")

    if chance is not None:
        ax.axvline(chance, color=COLORS["muted"], linestyle=(0, (4, 3)),
                   linewidth=1.2, zorder=3)
        ax.text(chance, -0.72, "chance", fontsize=8,
                color=COLORS["muted"], ha="center", va="bottom")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, n_target)
    ax.set_ylim(len(labels) - 0.5, -1.0)
    ax.set_xlabel(f"days recovered out of {n_target}")
    ax.set_title("Agreement with the simulated days")
    ax.grid(axis="y", visible=False)

    # ── 2. the same region of PC space? ──────────────────────────────────────
    ax = axes[1]
    x, y = pcs_df["PC1"].values, pcs_df["PC2"].values
    in_target = pcs_df.index.isin(target_days)
    in_rebuilt = pcs_df.index.isin(rebuilt_days)

    ax.scatter(x, y, s=4, color=COLORS["data"], alpha=0.45, linewidths=0,
               zorder=1, label=f"all days (n={len(pcs_df):,})")
    ax.scatter(x[in_rebuilt], y[in_rebuilt], s=26, marker="s",
               color=COLORS["test"], edgecolors="white", linewidths=0.5,
               zorder=2, label=f"rebuilt MDA (n={in_rebuilt.sum()})")
    ax.scatter(x[in_target], y[in_target], s=26, color=COLORS["selected"],
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"actually simulated (n={in_target.sum()})")

    ax.axhline(0, color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Position in wind PC space")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92,
              borderpad=0.5, handletextpad=0.4)

    # ── 3. the same wind speeds? ─────────────────────────────────────────────
    ax = axes[2]

    def ecdf(values):
        v = np.sort(np.asarray(values, dtype=float))
        return v, np.arange(1, len(v) + 1) / len(v)

    for label, days, color, width in (
        (f"all days (n={len(daily_wind):,})", None, COLORS["data"], 1.6),
        (f"rebuilt MDA (n={len(rebuilt_days)})", rebuilt_days, COLORS["test"], 1.8),
        (f"actually simulated (n={n_target})", target_days, COLORS["selected"], 1.8),
    ):
        series = daily_wind if days is None else daily_wind.loc[
            daily_wind.index.isin(days)]
        values, prob = ecdf(series.values)
        ax.plot(values, prob, color=color, linewidth=width, label=label)

    ax.set_xlabel("daily mean wind speed (m/s)")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Wind speeds sampled")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Rebuilding the 365 simulated days", fontsize=12,
                 fontweight="semibold", y=1.04)

    return fig


def plot_wind_wave_frame(
    wind: xr.Dataset,
    wave: xr.Dataset,
    time: Any,
    orthophoto: Optional[np.ndarray] = None,
    coastline: Optional[Any] = None,
    wind_clim: Tuple[float, float] = (0.0, 15.0),
    wave_clim: Tuple[float, float] = (0.0, 0.7),
    wave_var: str = "Hsig_rescaled",
    dir_var: Optional[str] = "Dir",
    wave_cmap: Any = None,
    wind_cmap: Any = "Oranges",
    dir_step: int = 18,
    var_label: str = "Hsig (m)",
    tide_state: Optional[pd.DataFrame] = None,
    figsize: Tuple[float, float] = (12.5, 5.4),
) -> Figure:
    """
    One frame of the early-warning animation: forcing on the left, response on
    the right.

    Parameters
    ----------
    wind : xr.Dataset
        ERA5 winds with 'u10' and 'v10' on a (time, lat, lon) grid.
    wave : xr.Dataset
        Reconstructed wave field on the bay grid, with `wave_var` and optionally
        `dir_var`.
    time : datetime-like
        The instant to draw. Both datasets are selected at it.
    orthophoto : np.ndarray, optional
        Aerial image, stretched onto the bay bounds.
    coastline : geopandas.GeoDataFrame, optional
        Coastline in lon/lat, drawn on both panels.
    wind_clim, wave_clim : tuple of float, optional
        Colour limits. Fix these across frames — letting them float per frame
        makes an animation where the colours mean something different in every
        image.
    wave_var : str, optional
        Variable to shade on the right panel. Default is 'Hsig_rescaled'.
    dir_var : str, optional
        Wave direction in degrees, drawn as arrows. None to skip them.
    wave_cmap, wind_cmap : str or Colormap, optional
        Colormaps. `wave_cmap` defaults to the project sequential map.
    dir_step : int, optional
        Draw one direction arrow every N grid cells. Default is 18.
    var_label : str, optional
        Colour-bar label for the wave panel.
    tide_state : pd.DataFrame, optional
        Indexed by time, with columns 'tide' and 'case' as returned by
        `utils.tide.nearest_level`. When given, the title says which tide level
        the frame was reconstructed at — worth showing, because the model
        changes underneath as the tide moves.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    Figure
        The frame, ready to be saved.

    Notes
    -----
    The two panels cover very different areas on purpose — 0.25° ERA5 nodes over
    Cantabria against a bay 12 km across — so the bay is outlined on the wind
    panel rather than left implied.
    """

    snap_wind = wind.sel(time=time)
    # The wave models return (lon, lat) or (lat, lon) depending on how they were
    # stacked; pcolormesh needs the second.
    snap_wave = wave.sel(time=time).transpose(..., "lat", "lon")

    bay = [
        float(wave.lon.min()), float(wave.lon.max()),
        float(wave.lat.min()), float(wave.lat.max()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    # ── forcing ──────────────────────────────────────────────────────────────
    ax = axes[0]
    speed = np.sqrt(snap_wind["u10"] ** 2 + snap_wind["v10"] ** 2)
    mesh = ax.pcolormesh(wind.lon, wind.lat, speed, cmap=wind_cmap,
                         shading="nearest", vmin=wind_clim[0], vmax=wind_clim[1])
    lon_grid, lat_grid = np.meshgrid(wind.lon, wind.lat)
    ax.quiver(lon_grid, lat_grid, snap_wind["u10"], snap_wind["v10"],
              color=COLORS["ink"], width=0.006, scale=130,
              pivot="middle", zorder=4)
    ax.add_patch(plt.Rectangle(
        (bay[0], bay[2]), bay[1] - bay[0], bay[3] - bay[2],
        fill=False, edgecolor=COLORS["selected"], linewidth=2.0, zorder=5))

    extent = [float(wind.lon.min()), float(wind.lon.max()),
              float(wind.lat.min()), float(wind.lat.max())]
    if coastline is not None:
        _draw_coastline(coastline, ax, extent, color=COLORS["ink"], linewidth=0.7)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title("ERA5 wind")
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.grid(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label("wind speed (m/s)")
    bar.outline.set_visible(False)

    # ── response ─────────────────────────────────────────────────────────────
    ax = axes[1]
    if orthophoto is not None:
        ax.imshow(orthophoto, extent=bay, origin="upper", aspect="auto")
    mesh = ax.pcolormesh(wave.lon, wave.lat, snap_wave[wave_var],
                         cmap=wave_cmap or SEQUENTIAL_CMAP, shading="auto",
                         vmin=wave_clim[0], vmax=wave_clim[1], alpha=0.85)

    if dir_var is not None and dir_var in snap_wave:
        radians = np.deg2rad(snap_wave[dir_var].values[::dir_step, ::dir_step])
        ax.quiver(wave.lon.values[::dir_step], wave.lat.values[::dir_step],
                  -np.sin(radians), -np.cos(radians), color="white",
                  width=0.004, scale=35, alpha=0.9)

    if coastline is not None:
        _draw_coastline(coastline, ax, bay, color="white", linewidth=1.0)
    ax.set_xlim(bay[0], bay[1])
    ax.set_ylim(bay[2], bay[3])
    ax.set_title("reconstructed wave field")
    ax.set_xlabel("longitude (deg E)")
    ax.grid(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(var_label)
    bar.outline.set_visible(False)

    stamp = str(np.datetime64(pd.Timestamp(time)))[:16].replace("T", " ")
    if tide_state is not None:
        row = tide_state.loc[pd.Timestamp(time)]
        stamp += f"    ·    tide {row['tide']:+.2f} m  →  {row['case']}"
    fig.suptitle(stamp, fontsize=12, fontweight="semibold")

    return fig


def create_gif(
    plot_function,
    frames: Sequence[Any],
    output_path: str,
    frame_arg: str = "time",
    fps: float = 2.0,
    dpi: int = 110,
    keep_frames: bool = False,
    **plot_kwargs,
) -> str:
    """
    Render one figure per frame and assemble them into an animated GIF.

    Parameters
    ----------
    plot_function : callable
        Called once per frame as ``plot_function(**plot_kwargs, **{frame_arg: f})``
        and expected to return a Figure.
    frames : sequence
        Values passed one at a time, usually timestamps.
    output_path : str
        Where to write the GIF. Parent directories are created.
    frame_arg : str, optional
        Name of the argument that varies. Default is 'time'.
    fps : float, optional
        Frames per second. Default is 2.
    dpi : int, optional
        Resolution of each frame. Default is 110.
    keep_frames : bool, optional
        Keep the individual PNGs next to the GIF. Default is False.
    **plot_kwargs
        Passed through to `plot_function` unchanged on every frame.

    Returns
    -------
    str
        The path written.

    Notes
    -----
    Every frame is closed after saving, so a long animation does not accumulate
    open figures. Colour limits are not managed here: pass fixed ones through
    `plot_kwargs`, or the colours will mean something different in each frame.
    """

    import os
    import shutil
    import tempfile

    from PIL import Image

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_dir = (
        f"{os.path.splitext(output_path)[0]}_frames" if keep_frames
        else tempfile.mkdtemp()
    )
    os.makedirs(frames_dir, exist_ok=True)

    paths = []
    for index, frame in enumerate(frames):
        figure = plot_function(**plot_kwargs, **{frame_arg: frame})
        path = os.path.join(frames_dir, f"frame_{index:03d}.png")
        figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        paths.append(path)

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
    images[0].save(output_path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return output_path


def plot_climatology_map(
    field: xr.DataArray,
    cells: Optional[pd.DataFrame] = None,
    orthophoto: Optional[np.ndarray] = None,
    label: str = "Hsig p99 (m)",
    cmap: Any = None,
    ax: Optional[Axes] = None,
) -> Axes:
    """
    A climatological field over the bay, with the warning points on top.

    Parameters
    ----------
    field : xr.DataArray
        A statistic of the reconstruction — a percentile, a mean — on the bay
        grid.
    cells : pd.DataFrame, optional
        From `utils.ews.snap_to_sea`. Drawn as labelled markers.
    orthophoto : np.ndarray, optional
        Aerial image, stretched onto the field's bounds.
    label : str, optional
        Colour-bar label.
    cmap : str or Colormap, optional
        Defaults to the project sequential map.
    ax : Axes, optional
        Axes to draw on. A new figure is made if absent.

    Returns
    -------
    Axes

    Notes
    -----
    The labels are placed alternately left and right of their marker. With eight
    points inside twelve kilometres, anything else overlaps.
    """

    field = field.transpose("lat", "lon")
    bounds = [
        float(field.lon.min()), float(field.lon.max()),
        float(field.lat.min()), float(field.lat.max()),
    ]

    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 8))

    if orthophoto is not None:
        ax.imshow(orthophoto, extent=bounds, origin="upper", aspect="auto")

    mesh = ax.pcolormesh(field.lon, field.lat, field, cmap=cmap or SEQUENTIAL_CMAP,
                         shading="auto", alpha=0.85)
    bar = ax.get_figure().colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(label)
    bar.outline.set_visible(False)

    if cells is not None:
        for position, (place, row) in enumerate(cells.iterrows()):
            ax.plot(row.cell_lon, row.cell_lat, "o", markersize=9,
                    markerfacecolor="white", markeredgecolor=COLORS["ink"],
                    markeredgewidth=1.6, zorder=5)
            side = 1 if position % 2 == 0 else -1
            ax.annotate(
                place,
                (row.cell_lon, row.cell_lat),
                xytext=(9 * side, 6),
                textcoords="offset points",
                ha="left" if side > 0 else "right",
                fontsize=9, fontweight="semibold", color="white", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=COLORS["ink"],
                          edgecolor="none", alpha=0.75),
            )

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_xlabel("longitude (deg E)")
    ax.set_ylabel("latitude (deg N)")
    ax.grid(False)

    return ax


def plot_alert_timeline(
    levels: pd.DataFrame,
    series: pd.DataFrame,
    colors: Dict[str, str],
    order: Sequence[str],
    tide: Optional[pd.Series] = None,
    wind: Optional[pd.Series] = None,
    highlight: Optional[Any] = None,
    var_label: str = "Hsig (m)",
) -> Figure:
    """
    The week as a warning calendar: one row per place, one cell per hour.

    Parameters
    ----------
    levels : pd.DataFrame
        Warning level per hour and place, from `utils.ews.alert_levels`.
    series : pd.DataFrame
        The wave heights behind them, same shape.
    colors : dict
        Level name to colour.
    order : sequence of str
        Level names, quietest first, for the legend.
    tide : pd.Series, optional
        Predicted tide, drawn underneath.
    wind : pd.Series, optional
        Wind speed, drawn underneath with the tide.
    highlight : datetime-like, optional
        An hour to mark on every panel — usually the worst one.
    var_label : str, optional
        Label for the wave axis.

    Returns
    -------
    Figure

    Notes
    -----
    The calendar answers "when and where", the panel below answers "why": the
    interesting cases are the ones where the warning follows the tide rather
    than the wind, and putting the two under the same time axis is what makes
    that visible.
    """

    import matplotlib.dates as mdates

    places = list(levels.columns)
    times = pd.DatetimeIndex(levels.index)
    # broken_barh works in the axis's own units, which for a date axis is days
    # since the epoch, so the hours have to be converted rather than passed raw.
    step = (times[1] - times[0]) / np.timedelta64(1, "D")

    has_context = tide is not None or wind is not None
    fig, axes = plt.subplots(
        2 if has_context else 1, 1,
        figsize=(13, 0.42 * len(places) + (3.4 if has_context else 1.2)),
        sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [len(places) * 0.42, 2.6]} if has_context else None,
    )
    axes = np.atleast_1d(axes)
    ax = axes[0]

    for row, place in enumerate(places):
        for level in order:
            hours = times[levels[place] == level]
            if len(hours) == 0:
                continue
            ax.broken_barh(
                [(mdates.date2num(t), step) for t in hours], (row - 0.4, 0.8),
                facecolors=colors[level], edgecolors="none",
            )

    ax.xaxis_date()
    ax.set_yticks(range(len(places)))
    ax.set_yticklabels(places, fontsize=9)
    ax.set_ylim(len(places) - 0.5, -0.5)
    ax.set_title("Warning level by location", pad=26)
    ax.grid(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[level])
               for level in order]
    ax.legend(handles, order, loc="lower center", ncol=len(order),
              bbox_to_anchor=(0.5, 1.14), frameon=False)

    if has_context:
        context = axes[1]
        if tide is not None:
            context.plot(tide.index, tide.values, color=COLORS["primary"],
                         linewidth=1.6, label="tide (m)")
            context.axhline(0, color=COLORS["grid"], linewidth=0.8)
            context.set_ylabel("tide (m)")
        if wind is not None:
            twin = context.twinx()
            twin.plot(wind.index, wind.values, color=COLORS["muted"],
                      linewidth=1.4, linestyle=(0, (4, 2)), label="wind (m/s)")
            twin.set_ylabel("wind speed (m/s)", color=COLORS["muted"])
            twin.grid(False)
        context.set_xlabel("")
        context.legend(loc="upper left", fontsize=8, frameon=False)

    if highlight is not None:
        for panel in axes:
            panel.axvline(pd.Timestamp(highlight), color=COLORS["ink"],
                          linestyle=(0, (3, 2)), linewidth=1.4, zorder=6)
        axes[-1].annotate(
            f"{pd.Timestamp(highlight):%d %b %H:%M}",
            xy=(pd.Timestamp(highlight), axes[-1].get_ylim()[1]),
            xytext=(5, -10), textcoords="offset points", fontsize=8,
            fontweight="semibold", color=COLORS["ink"], va="top",
        )

    axes[-1].tick_params(axis="x", rotation=20)

    return fig


def plot_alert_maps(
    wave: xr.Dataset,
    levels: pd.DataFrame,
    cells: pd.DataFrame,
    times: Sequence[Any],
    colors: Dict[str, str],
    order: Sequence[str],
    orthophoto: Optional[np.ndarray] = None,
    wave_var: str = "Hsig_rescaled",
    wave_clim: Tuple[float, float] = (0.0, 0.8),
    wave_cmap: Any = None,
    tide_state: Optional[pd.DataFrame] = None,
    columns: int = 4,
    var_label: str = "Hsig (m)",
    wave_dir_var: Optional[str] = None,
) -> Figure:
    """
    A grid of maps through the episode, each with the warning at every place.

    Parameters
    ----------
    wave : xr.Dataset
        Reconstructed fields, with `wave_var`.
    levels : pd.DataFrame
        Warning level per hour and place.
    cells : pd.DataFrame
        From `utils.ews.snap_to_sea`.
    times : sequence
        The hours to draw, usually every three.
    colors : dict
        Level name to colour.
    order : sequence of str
        Level names, quietest first, for the legend.
    orthophoto : np.ndarray, optional
        Aerial image behind each panel.
    wave_var : str, optional
        Field to shade.
    wave_clim : tuple of float, optional
        Colour limits, fixed across panels.
    wave_cmap : str or Colormap, optional
        Colormap for the field.
    tide_state : pd.DataFrame, optional
        Adds the tide level to each panel title.
    columns : int, optional
        Panels per row. Default is 4.
    var_label : str, optional
        Colour-bar label.

    Returns
    -------
    Figure

    Notes
    -----
    The field and the markers say different things on purpose: the field is the
    wave height everywhere, the markers are what that height *means* at each
    place given its own history. A panel can be uniformly pale and still carry
    an orange marker where the local climatology is quiet.
    """

    times = list(times)
    rows = int(np.ceil(len(times) / columns))
    field = wave[wave_var].transpose("time", "lat", "lon")
    
    p90_val = float(np.nanpercentile(field.values, 90))
    _vmax = 1.0 if p90_val > 0.45 else 0.5
    wave_clim = (wave_clim[0], _vmax)   
    
    bounds = [
        float(field.lon.min()), float(field.lon.max()),
        float(field.lat.min()), float(field.lat.max()),
    ]

    fig, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 3.1 * rows),
        squeeze=False, layout="constrained",
    )

    mesh = None
    for index, when in enumerate(times):
        ax = axes[index // columns][index % columns]
        if orthophoto is not None:
            ax.imshow(orthophoto, extent=bounds, origin="upper", aspect="auto")
        mesh = ax.pcolormesh(
            field.lon, field.lat, field.sel(time=when),
            cmap=wave_cmap or SEQUENTIAL_CMAP, shading="auto",
            vmin=wave_clim[0], vmax=wave_clim[1], alpha=0.8,
        )
        

        if wave_dir_var is not None and wave_dir_var in wave:
            # field ya está transpuesto a (time, lat, lon) — usar como referencia
            hs_sl  = field.sel(time=when)                          # DataArray (lat, lon)
            dir_sl = wave[wave_dir_var].transpose("time", "lat", "lon").sel(time=when)
            lon_g  = field.lon.values; lat_g = field.lat.values
            step   = max(1, min(len(lon_g), len(lat_g)) // 10)
            lo = lon_g[::step]; la = lat_g[::step]
            lo2, la2 = np.meshgrid(lo, la)
            dq  = dir_sl.values[::step, ::step]   # (lat_sub, lon_sub)
            hsq = hs_sl.values[::step, ::step]
            wet = np.isfinite(hsq) & (hsq > 0.01)
            # Dir SWAN = dirección de procedencia → invertir para mostrar propagación
            u_q = np.where(wet, -np.sin(np.deg2rad(dq)), np.nan)
            v_q = np.where(wet, -np.cos(np.deg2rad(dq)), np.nan)
            ax.quiver(lo2, la2, u_q, v_q,
                      scale=15, width=0.005, color="black", alpha=0.65, zorder=6)
                        
        for place, row in cells.iterrows():
            level = levels.loc[pd.Timestamp(when), place]
            ax.plot(row.cell_lon, row.cell_lat, "o", markersize=10,
                    markerfacecolor=colors[level], markeredgecolor="white",
                    markeredgewidth=1.4, zorder=5)

        title = f"{pd.Timestamp(when):%d %b %H:%M}"
        if tide_state is not None:
            title += f"  ·  {tide_state.loc[pd.Timestamp(when), 'case']}"
        ax.set_title(title, fontsize=9)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for index in range(len(times), rows * columns):
        axes[index // columns][index % columns].axis("off")

    bar = fig.colorbar(mesh, ax=axes, fraction=0.02, pad=0.01, aspect=40)
    bar.set_label(var_label)
    bar.outline.set_visible(False)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                          markerfacecolor=colors[level], markeredgecolor="white")
               for level in order]
    fig.legend(handles, order, loc="upper center", ncol=len(order),
               bbox_to_anchor=(0.5, 1.05), frameon=False)

    return fig
