"""
Loading of monthly climate indices and correlation with the PCs.

The indices are downloaded and stored in a single NetCDF by
00b_update_climate_indices.ipynb (see utils/download_indices.py).
Each index is described in INDICES, so every index goes through the same functions.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from . import PURPLE, GREEN, ORANGE, CMAP_DIV


# anomaly : base period to remove the mean (None = index is already an anomaly)
# rolling : centred rolling mean in months (None = monthly values)
# plot_thr: (upper, lower) thresholds shown in the index plot
# corr_thr: (positive, negative) correlation thresholds to select PCs
# color, marker : style in the summary plots (dict order = summary order)
INDICES = {
    "NAO": dict(plot_thr=(1, -1), corr_thr=(0.2, -0.2), color="#6A3D9A", marker="s"),
    "AO":  dict(plot_thr=(0.5, -0.5), corr_thr=(0.2, -0.2), color="#1B9E77", marker="o"),
    "AMO": dict(plot_thr=(0.3, -0.3), corr_thr=(0.3, -0.2), color="#E66101", marker="D"),
    "SCA": dict(plot_thr=(0.5, -0.3), corr_thr=(0.25, -0.2), color="#C51B7D", marker="X"),
    "OLR": dict(plot_thr=(0.5, -0.5), corr_thr=(0.2, -0.3), color="#8C510A", marker="p"),
    "N34": dict(anomaly=("1991-01-01", "2020-12-31"), rolling=3,
                plot_thr=(0.7, -0.7), corr_thr=(0.2, -0.21), color="#66A61E", marker="^"),
    "PDO": dict(plot_thr=(1, -1), corr_thr=(0.2, -0.3), color="#D9A441", marker="*"),
    "SOI": dict(plot_thr=(1, -1), corr_thr=(0.25, -0.3), color="#B2ABD2", marker="<"),
    "PNA": dict(plot_thr=(1, -1), corr_thr=(0.25, -0.25), color="#984EA3", marker="8"),
    "SAM": dict(plot_thr=(1, -1), corr_thr=(0.3, -0.3), color="#35978F", marker="v"),
    "ELI": dict(plot_thr=(170, 157), corr_thr=(0.2, -0.2), color="#FDB863", marker="h"),
    "RONI": dict(plot_thr=(0.5, -0.5), corr_thr=(0.2, -0.2), color="#5E3C99", marker="P"),
}

# variable name of each index in the saved PCs + indices NetCDF
INDEX_VARS = {n: "scand" if n == "SCA" else n.lower() for n in INDICES}

PC_COLORS = ["#6A3D9A", "#1B9E77", "#E66101", "#D9A441", "#984EA3", "#66A61E", "#8C510A",
             "#B2ABD2", "#FDB863", "#5E3C99", "#A6DBA0", "#C51B7D", "#35978F", "#E08214",
             "#4D9221", "#7B3294"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_index(name, p_indices):
    """Read an index from the indices NetCDF into a Dataset with variable 'index'."""
    cfg = INDICES[name]
    with xr.open_dataset(p_indices) as ds:
        index = ds[name].dropna("time").load().to_dataset(name="index")

    if cfg.get("anomaly"):
        index["index"] = index["index"] - index["index"].sel(time=slice(*cfg["anomaly"])).mean()
    if cfg.get("rolling"):
        index = index.rolling(time=cfg["rolling"], center=True).mean()

    return index


# ---------------------------------------------------------------------------
# Detrending
# ---------------------------------------------------------------------------

# degree of the polynomial trend removed from the indices (same parabola as the SST / MSL
# fields in notebook 01); None keeps the indices as published
DETREND_DEG = 2


def detrend_index(da, deg=DETREND_DEG):
    """
    Remove a polynomial trend in time from an index (DataArray with 'time'), fitted on its
    non-NaN months. The result has zero mean. deg=None returns the index unchanged.
    """
    if deg is None:
        return da
    y = da.values.astype(float)
    m = np.isfinite(y)
    x = (da.time.values - da.time.values[0]) / np.timedelta64(1, "D") / 365.25
    x = x - x[m].mean()
    out = np.full_like(y, np.nan)
    out[m] = y[m] - np.polyval(np.polyfit(x[m], y[m], deg), x[m])
    return da.copy(data=out)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlate_pcs_index(pcs, index, detrend_deg=DETREND_DEG):
    """
    Pearson correlation of every PC with the index over the common (non-NaN) months.
    The index is first detrended over those months (`detrend_index`, degree `detrend_deg`;
    None to keep the trend), as the PCs come from detrended fields.

    pcs   : DataArray (time, n_pcs)
    index : Dataset with variable 'index'
    Returns (corr array of length n_pcs, Dataset with 'PCs' and the (detrended) 'index').
    """
    t = np.intersect1d(pcs.time.values, index.time.values)
    ds = xr.Dataset({"PCs": pcs.sel(time=t), "index": index["index"].sel(time=t)})
    ds = ds.dropna(dim="time")
    ds["index"] = detrend_index(ds["index"], detrend_deg)
    corr = np.array([np.corrcoef(ds.PCs.isel(n_pcs=i), ds["index"])[0, 1]
                     for i in range(ds.sizes["n_pcs"])])
    return corr, ds


def linear_regression(X, y, verbose=True):
    """
    Multiple linear regression of y on the columns of X (rows with NaN are ignored).
    Returns (correlation between observed and reconstructed y, reconstructed y, model).
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    corr = np.corrcoef(y, y_pred)[0, 1]
    if verbose:
        print(f"Multiple R  = {corr:.3f}")
        print(f"R^2         = {reg.score(X, y):.3f}")
        print(f"Intercept   = {reg.intercept_:.3f}")
    return corr, y_pred, reg


def regress_index_on_pcs(ds, verbose=True):
    """Regression of the index on all the PCs (ds: output of correlate_pcs_index)."""
    return linear_regression(ds.PCs.values, ds["index"].values, verbose)


def regression_corr_by_npcs(pcs_indices, n_pcs_list, verbose=False):
    """
    Correlation of the regression of each index on the first n PCs, for every n in n_pcs_list.

    pcs_indices : Dataset saved by save_pcs_indices
    Returns DataArray (n_pcs_used, climate_index).
    """
    names = [str(n) for n in pcs_indices.index_name.values]
    corr = np.full((len(n_pcs_list), len(names)), np.nan)
    for i, n_pcs in enumerate(n_pcs_list):
        X = pcs_indices.PCs.isel(n_pcs=slice(0, n_pcs)).values
        for j, name in enumerate(names):
            corr[i, j] = linear_regression(X, pcs_indices[INDEX_VARS[name]].values, verbose)[0]
    return xr.DataArray(corr, dims=("n_pcs_used", "climate_index"),
                        coords={"n_pcs_used": list(n_pcs_list), "climate_index": names})


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_index(index, name=" ", thresholds=(0.5, -0.5), figsize=(19, 6)):
    """Index time series highlighting the values above / below the thresholds."""
    up, low = thresholds
    t, v = index.time.values, index["index"].values
    s_up, s_low = v > up, v < low

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, v, ".-", color="grey", linewidth=2, markersize=8, alpha=0.5, label=name)
    ax.plot(t[s_up], v[s_up], ".", color=ORANGE, markersize=12, label="Upper limit")
    ax.plot(t[s_low], v[s_low], ".", color=PURPLE, markersize=12, label="Lower limit")
    ax.axhline(up, color=ORANGE, linewidth=2, label="Upper limit threshold")
    ax.axhline(0, color="grey", linestyle="--", linewidth=2)
    ax.axhline(low, color=PURPLE, linewidth=2, label="Lower limit threshold")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time", fontsize=19)
    ax.set_ylabel(name, fontsize=19)
    ax.grid(color="lightgrey")
    ax.legend(ncol=5, fontsize=16)
    return fig, ax


def plot_correlation_by_pc(corr, name, figsize=(15, 5), label="PC"):
    """Correlation of each PC with the index."""
    n = np.arange(len(corr))
    fig, ax = plt.subplots(figsize=figsize)
    lim = np.nanmax(np.abs(corr))
    ax.scatter(n + 1, corr, s=130, c=corr, marker="*", cmap=CMAP_DIV, vmin=-lim, vmax=lim,
               edgecolor="grey", linewidth=0.3)
    ax.axhline(0, color="black", linewidth=3, zorder=-1)
    ax.set_xlim(n[0], n[-1])
    ax.set_xlabel(f"{label}s", fontsize=19)
    ax.set_ylabel(f"Correlation {name}", fontsize=19)
    ax.grid()
    return fig, ax


def plot_correlated_pcs(ds, corr, name, threshold, figsize=(25, 5), label="PC"):
    """
    Time series of the PCs whose correlation exceeds `threshold`
    (corr > threshold if positive, corr < threshold if negative) against the index.
    """
    sel = np.where(corr > threshold)[0] if threshold > 0 else np.where(corr < threshold)[0]
    t = ds.time.values

    fig, ax = plt.subplots(figsize=figsize)
    for k, ipc in enumerate(sel):
        ax.plot(t, ds.PCs.isel(n_pcs=ipc).values, linewidth=2,
                color=PC_COLORS[k % len(PC_COLORS)],
                label=f"{label} {ipc + 1} - corr {np.round(corr[ipc], 2)}")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(label)
    if len(sel):
        ax.legend(fontsize=17, ncol=5, loc=2)

    ax1 = ax.twinx()
    ax1.plot(t, ds["index"].values, ".-", color="black", markersize=16, linewidth=4,
             alpha=0.4, label=f"{name} INDEX")
    ax1.legend(fontsize=17, loc=1)
    return fig, ax


def plot_index_reconstruction(ds, y_pred, corr, name, figsize=(20, 5)):
    """Observed index vs index reconstructed from the PCs (regression)."""
    t = ds.time.values
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, ds["index"].values, ".-", color="black", linewidth=3, alpha=0.5,
            label=f"{name} observed")
    ax1 = ax.twinx()
    ax1.plot(t, y_pred, "-", color=ORANGE, linewidth=2, label=f"{name} reconstructed")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel(f"{name} index")
    fig.legend(fontsize=17, loc="upper left", bbox_to_anchor=(0.12, 0.88))
    ax.set_title(f"{name} index - Linear Regression corr={corr:.2f}", fontsize=19)
    ax.grid()
    return fig, ax


def analyze_index(name, pcs, p_indices, label="PC", detrend_deg=DETREND_DEG):
    """
    Load the index, plot it, correlate it with each PC, plot the correlated PCs
    and reconstruct the index from all the PCs with a multiple linear regression.
    The first plot shows the index as published; the rest use the index detrended
    over the PC period (degree `detrend_deg`, None to keep the trend).

    Returns dict with 'corr' (per PC), 'corr_regression' and 'data' (PCs + index).
    """
    cfg = INDICES[name]
    index = load_index(name, p_indices)
    plot_index(index, name=name, thresholds=cfg["plot_thr"])

    corr, ds = correlate_pcs_index(pcs, index, detrend_deg)
    plot_correlation_by_pc(corr, name, label=label)
    plot_correlated_pcs(ds, corr, name, cfg["corr_thr"][0], label=label)
    plot_correlated_pcs(ds, corr, name, cfg["corr_thr"][1], label=label)

    corr_reg, y_pred, _ = regress_index_on_pcs(ds)
    plot_index_reconstruction(ds, y_pred, corr_reg, name)
    plt.show()
    return {"corr": corr, "corr_regression": corr_reg, "data": ds}


def save_pcs_indices(results, p_out, explained_variance_ratio=None):
    """
    Save the PCs, every index (one variable per index, see INDEX_VARS), the
    regression correlation of each index and (optionally) the explained variance
    ratio of all the PCA components in a single NetCDF.
    """
    names = [n for n in INDICES if n in results]
    ds = xr.merge([results[n]["data"].rename(index=INDEX_VARS[n]) for n in names])
    ds["index_correlation"] = (("climate_index"), [results[n]["corr_regression"] for n in names])
    ds["index_name"] = (("climate_index"), names)
    if explained_variance_ratio is not None:
        ds["explained_variance_ratio"] = (("n_component"), np.asarray(explained_variance_ratio))
    ds.to_netcdf(p_out)
    return ds


def add_indices(pcs_indices, names, p_indices, detrend_deg=DETREND_DEG, overwrite=False):
    """
    Add to a Dataset saved by save_pcs_indices the indices in `names` (index variable,
    index_correlation and index_name), read from `p_indices` and processed as in
    analyze_index (detrended with degree `detrend_deg`, regression on all the PCs).
    By default only the indices not yet in the Dataset are added; overwrite=True reloads
    also the existing ones (e.g. to change `detrend_deg`). Returns a new Dataset (the file
    is not modified).
    """
    if not overwrite:
        names = [n for n in names if INDEX_VARS[n] not in pcs_indices]
    if not names:
        return pcs_indices

    out = pcs_indices.copy()
    current = list(pcs_indices.index_name.values.astype(str))
    corrs = dict(zip(current, pcs_indices.index_correlation.values))
    for n in names:
        _, ds = correlate_pcs_index(pcs_indices.PCs, load_index(n, p_indices), detrend_deg)
        corrs[n] = regress_index_on_pcs(ds, verbose=False)[0]
        out[INDEX_VARS[n]] = ds["index"].reindex(time=pcs_indices.time)
        if n not in current:
            current.append(n)

    return out.assign(
        index_correlation=("climate_index", np.array([corrs[n] for n in current])),
        index_name=("climate_index", np.array(current)),
    )

def _style_index_axis(ax, names, ymin=0):
    ax.set_ylim(ymin, 1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, fontsize=19)
    ax.set_xlabel("Climate Index", fontsize=21, labelpad=20)
    ax.set_ylabel("Correlation", fontsize=19, labelpad=20)
    ax.tick_params(axis="y", labelsize=19)
    ax.grid(color="lightgrey", linestyle="--", linewidth=1)
    for side in ax.spines.values():
        side.set_visible(False)


def plot_regression_summary(corr_regression, figsize=(20, 6)):
    """
    Correlation between each observed index and its reconstruction from the PCs.
    corr_regression : dict {index name: correlation}
    """
    names = [n for n in INDICES if n in corr_regression]
    fig, ax = plt.subplots(figsize=figsize)
    for i, n in enumerate(names):
        ax.plot(i, corr_regression[n], marker=INDICES[n]["marker"],
                color=INDICES[n]["color"], ms=20)
    _style_index_axis(ax, names)
    return fig, ax


def plot_regression_comparison(corr, colors=None, ylabel="Correlation", figsize=(20, 6)):
    """
    Same as plot_regression_summary for several sets of components (side by side, one colour per set;
    the marker is the one of each index). corr : DataFrame (index x set label).
    """
    from matplotlib.lines import Line2D

    names = [n for n in INDICES if n in corr.index]
    labels = list(corr.columns)
    colors = colors or [PURPLE, GREEN, ORANGE, "#D9A441", "grey"][:len(labels)]
    w = 0.7 / len(labels)
    fig, ax = plt.subplots(figsize=figsize)
    for k, (label, c) in enumerate(zip(labels, colors)):
        dx = (k - (len(labels) - 1) / 2) * w
        for i, n in enumerate(names):
            ax.plot(i + dx, corr.loc[n, label], marker=INDICES[n]["marker"], color=c, ms=14,
                    markeredgecolor="black", markeredgewidth=0.5)
    _style_index_axis(ax, names, ymin=min(0, np.floor(np.nanmin(corr.values) * 10) / 10))
    ax.set_ylabel(ylabel, fontsize=19, labelpad=20)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="black",
                      markersize=13, label=l) for l, c in zip(labels, colors)]
    ax.legend(handles=handles, ncols=len(labels), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=15, frameon=False)
    return fig, ax


def plot_regression_vs_npcs(corr, figsize=(20, 6)):
    """
    Regression correlation of each index using different numbers of PCs
    (darker = more PCs). corr : output of regression_corr_by_npcs.
    """
    from matplotlib.lines import Line2D

    names = [str(n) for n in corr.climate_index.values]
    n_pcs_list = corr.n_pcs_used.values
    alpha = np.linspace(0.2, 0.9, len(n_pcs_list) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(n_pcs_list)):
        for j, n in enumerate(names):
            ax.plot(j, corr.values[i, j], marker=INDICES[n]["marker"],
                    color=INDICES[n]["color"], ms=20, alpha=alpha[i])
    _style_index_axis(ax, names, ymin=min(0, np.floor(np.nanmin(corr.values) * 10) / 10))
    ax.axhline(0, color="black", linewidth=1)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=12,
                      alpha=alpha[i], label=f"{n} PCs") for i, n in enumerate(n_pcs_list)]
    ax.legend(handles=handles, ncols=len(n_pcs_list), title="Nº PCs", loc="lower right",
              fontsize=15, title_fontsize=16, frameon=True)
    return fig, ax


def plot_correlation_summary(corrs, mode="abs", xmax=None, explained_variance_ratio=None,
                             ev_lines=(50, 70), figsize=(25, 10), s=(100, 200), label="PC"):
    """
    Correlation of every PC with every index. For each PC the index with the
    highest correlation is highlighted.

    corrs : dict {index name: corr array}
    mode  : 'abs' (max |corr|), 'pos' (max corr), 'neg' (min corr) or 'both' (pos + neg)
    explained_variance_ratio : if given, vertical lines mark the PCs reaching `ev_lines` %
    s     : marker size of (all points, highlighted points)
    """
    names = [n for n in INDICES if n in corrs]
    matrix = np.vstack([corrs[n] for n in names])
    x = np.arange(matrix.shape[1]) + 1

    best = {
        "abs": [np.argmax(np.abs(matrix), axis=0)],
        "pos": [np.argmax(matrix, axis=0)],
        "neg": [np.argmin(matrix, axis=0)],
        "both": [np.argmax(matrix, axis=0), np.argmin(matrix, axis=0)],
    }[mode]

    fig, ax = plt.subplots(figsize=figsize)
    for iv, name in enumerate(names):
        cfg = INDICES[name]
        v = matrix[iv]
        ax.scatter(x, v, s=s[0], c="lightgrey", edgecolor="grey", marker=cfg["marker"])
        imax = np.concatenate([np.where(b == iv)[0] for b in best])
        ax.scatter(x[imax], v[imax], s=s[1], c=cfg["color"], edgecolor=cfg["color"],
                   alpha=0.8, marker=cfg["marker"], label=name, zorder=3)

    if explained_variance_ratio is not None:
        cum_ev = np.cumsum(explained_variance_ratio) * 100
        for ev in ev_lines:
            ax.vlines(np.sum(cum_ev < ev) + 1, -1, 1, color="black", linestyle="--",
                      linewidth=3, label=f"{ev}% variance")

    ax.axhline(0, color="black", linewidth=3, zorder=-1)
    ax.set_xlim(0, xmax if xmax is not None else x[-1] + 1)
    ax.legend(ncol=len(names), fontsize=17)
    ax.set_xlabel(f"{label}s", fontsize=19)
    ax.set_ylabel("Correlation Indexes", fontsize=19)
    ax.tick_params(axis="both", labelsize=17)
    ax.grid()
    return fig, ax
