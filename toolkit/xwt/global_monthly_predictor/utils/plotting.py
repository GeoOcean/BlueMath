"""
Plots of the predictor preprocessing steps and of the PCA (EOFs).
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from .preprocessing import polynomial_trend
from . import PURPLE, GREEN, ORANGE, CMAP_DIV, CMAP_DIV2


def plot_global_maps(das, cmaps=None, titles=None, figsize=None, central_longitude=180):
    """Plot a list of 2D (lat, lon) DataArrays side by side on PlateCarree maps."""
    n = len(das)
    cmaps = cmaps or ["magma"] * n
    fig, axs = plt.subplots(
        1, n, figsize=figsize or (11 * n, 4),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_longitude)},
        squeeze=False,
    )
    for ax, da, cmap in zip(axs[0], das, cmaps):
        da.plot(ax=ax, cmap=cmap, transform=ccrs.PlateCarree())
        ax.coastlines()
    if titles:
        for ax, t in zip(axs[0], titles):
            ax.set_title(t)
    return fig, axs[0]


def plot_spatial_mean_trend(means, variables=("sst", "msl"), labels=("SST [°C]", "MSL [hPa]"),
                            deg=2, figsize=(13, 5)):
    """Spatial-mean time series with the fitted polynomial trend."""
    fig, axs = plt.subplots(len(variables), 1, figsize=figsize)
    for ax, var, label in zip(axs, variables, labels):
        y = means[var].values
        ax.plot(means.time, y, "-", color="black")
        ax.plot(means.time, polynomial_trend(y, deg), color=ORANGE, linewidth=2.5)
        ax.set_ylabel(label)
        ax.set_xlabel("Time")
    return fig, axs


def plot_preprocessing_steps(steps, variables=("sst", "msl"), point=None, figsize=(13, 8)):
    """
    Compare the successive preprocessing steps at a grid point or as a global mean.

    steps : dict {label: Dataset}, e.g. {'Original': ds, 'Detrended': ds_det}
    point : (ilon, ilat) grid indices; if None the spatial mean is plotted.
    """
    colors = [PURPLE, ORANGE, GREEN]
    alphas = [0.5, 1, 1]
    fig, axs = plt.subplots(len(variables), 1, figsize=figsize)
    for ax, var in zip(axs, variables):
        for i, (label, ds) in enumerate(steps.items()):
            if point is None:
                da = ds[var].mean(dim=["longitude", "latitude"])
            else:
                da = ds[var].isel(longitude=point[0], latitude=point[1])
            da.plot(ax=ax, color=colors[i], alpha=alphas[i], label=f"{label} Monthly Data")
        ax.set_xlim(da.time.min().values, da.time.max().values)
        ax.grid(":", color="gray", alpha=0.5)
        ax.legend()
    return fig, axs


def plot_eofs_scaled(pca, var, num_eofs=10, cmap=CMAP_DIV, vmin=-0.8, vmax=0.8,
                     cbar_label="", ncols=2, central_longitude=180):
    """
    EOF maps scaled by the std of their PC (physical units), with the
    explained variance of each mode in the title.
    """
    std_pcs = pca.pcs.PCs.std(dim="time")
    exp_variance = pca.explained_variance / np.sum(pca.explained_variance) * 100.0

    nrows = int(np.ceil(num_eofs / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(10 * ncols, 5 * nrows),
        subplot_kw={"projection": ccrs.Mollweide(central_longitude=central_longitude)},
    )
    axs = np.atleast_1d(axs).ravel()

    for k, ax in enumerate(axs[:num_eofs]):
        data = pca.eofs.isel(n_component=k).squeeze() * std_pcs.isel(n_component=k).squeeze()
        im = ax.pcolormesh(data.longitude, data.latitude, data[var],
                           transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.coastlines()
        ax.gridlines()
        ax.set_title(f"EOF {k + 1}, EV = {float(exp_variance[k]):.1f}%", fontsize=17)
    for ax in axs[num_eofs:]:
        ax.set_visible(False)

    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.colorbar(im, ax=axs, orientation="horizontal", label=cbar_label, shrink=0.5, pad=0.05)
    return fig, axs


def plot_explained_variance(explained_variance_ratio, n_show=100, marks=(50, 70, 90),
                            figsize=(16, 5)):
    """Variance explained by each PC (bars) and cumulative (line), marking some thresholds."""
    ev = np.asarray(explained_variance_ratio)[:n_show] * 100
    cum = np.cumsum(np.asarray(explained_variance_ratio)) * 100
    x = np.arange(1, len(ev) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, ev, color=PURPLE, alpha=0.7)
    ax.set_xlabel("PC", fontsize=17)
    ax.set_ylabel("Explained variance [%]", fontsize=17, color=PURPLE)
    ax.set_xlim(0, len(ev) + 1)

    ax1 = ax.twinx()
    ax1.plot(x, cum[:len(ev)], "-", color="black", linewidth=2)
    ax1.set_ylabel("Cumulative [%]", fontsize=17)
    ax1.set_ylim(0, 100)
    for m in marks:
        n = int(np.searchsorted(cum, m)) + 1
        if n <= len(ev):
            ax1.axvline(n, color="grey", linestyle="--")
            ax1.text(n + 0.5, m - 6, f"{m}%: {n} PCs", fontsize=14)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_mode_maps(maps, var, modes, evr=None, cmap=CMAP_DIV, vmax=1, ncols=2, title="Mode",
                   cbar_label="Correlation", central_longitude=180):
    """
    Maps of several modes (DataArray/Dataset with dims mode, latitude, longitude).
    `maps[var]` is plotted for each mode in `modes`; `evr` (per mode) is shown in the title.
    """
    nrows = int(np.ceil(len(modes) / ncols))
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(10 * ncols, 4.5 * nrows),
        subplot_kw={"projection": ccrs.Mollweide(central_longitude=central_longitude)},
    )
    axs = np.atleast_1d(axs).ravel()
    for ax, m in zip(axs, modes):
        da = maps[var].sel(mode=m)
        im = ax.pcolormesh(da.longitude, da.latitude, da, transform=ccrs.PlateCarree(),
                           cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.coastlines()
        ax.gridlines()
        t = f"{title} {m}"
        if evr is not None:
            t += f", EV = {100 * float(evr.sel(mode=m)):.1f}%"
        ax.set_title(t, fontsize=17)
    for ax in axs[len(modes):]:
        ax.set_visible(False)
    plt.subplots_adjust(wspace=0.05, hspace=0.12)
    plt.colorbar(im, ax=axs, orientation="horizontal", label=cbar_label, shrink=0.5, pad=0.04)
    return fig, axs


def plot_mode_comparison(maps_a, maps_b, var_a, var_b, modes, titles=("EOF", "RPC"),
                         evr_a=None, evr_b=None, cmap=CMAP_DIV, vmax=1, central_longitude=180):
    """Side by side maps (rows: modes; left: maps_a[var_a], right: maps_b[var_b])."""
    fig, axs = plt.subplots(
        len(modes), 2, figsize=(20, 4.2 * len(modes)),
        subplot_kw={"projection": ccrs.Mollweide(central_longitude=central_longitude)},
        squeeze=False,
    )
    for row, m in enumerate(modes):
        for col, (maps, var, evr, t) in enumerate([(maps_a, var_a, evr_a, titles[0]),
                                                    (maps_b, var_b, evr_b, titles[1])]):
            ax = axs[row, col]
            da = maps[var].sel(mode=m)
            im = ax.pcolormesh(da.longitude, da.latitude, da, transform=ccrs.PlateCarree(),
                               cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.coastlines()
            label = f"{t} {m}"
            if evr is not None:
                label += f", EV = {100 * float(evr.sel(mode=m)):.1f}%"
            ax.set_title(label, fontsize=17)
    plt.subplots_adjust(wspace=0.05, hspace=0.12)
    plt.colorbar(im, ax=axs, orientation="horizontal", label="Correlation", shrink=0.5, pad=0.03)
    return fig, axs


def plot_mode_timeseries(Z, time, n=9, label="PC", window=12, figsize=(20, 14)):
    """Time series of the first `n` (standardized) modes with a centred running mean."""
    import pandas as pd

    fig, axs = plt.subplots(n, 1, figsize=figsize, sharex=True)
    for k, ax in enumerate(np.atleast_1d(axs)):
        s = pd.Series(Z[:, k], index=time)
        ax.plot(s.index, s.values, color="grey", linewidth=0.7)
        ax.plot(s.index, s.rolling(window, center=True).mean(), color=ORANGE, linewidth=2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{label} {k + 1}")
        ax.grid(alpha=0.3)
    axs[0].set_title(f"Monthly values (grey) and {window}-month running mean (orange)")
    plt.tight_layout()
    return fig, axs
