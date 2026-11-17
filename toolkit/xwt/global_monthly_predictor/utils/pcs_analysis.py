"""
Analysis of how well the PCs reproduce the climate indices.

All functions take the Dataset saved by `climate_indices.save_pcs_indices`
(PCs + one variable per index + regression correlation).
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from .climate_indices import INDICES, INDEX_VARS
from . import PURPLE, GREEN, ORANGE, CMAP_DIV, APPROACH_COLORS as _APPROACH_COLORS


SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}


def _index_names(pcs_indices):
    return [str(n) for n in pcs_indices.index_name.values]


def _valid(X, y):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[mask], y[mask]


def estimate_explained_variance(pcs_indices, p_predictor):
    """
    Explained variance ratio of the stored PCs, computed as var(PC) / total variance of
    the PCA predictor (saved by notebook 01). Used when the PCs file does not include
    'explained_variance_ratio' (files created before it was added).
    """
    pred = xr.open_dataset(p_predictor)
    total = sum(float(pred[v].var("time", ddof=1).sum()) for v in pred.data_vars)
    pc_var = pcs_indices.PCs.var("time", ddof=1).values
    return xr.DataArray(pc_var / total, dims="n_component")


# ---------------------------------------------------------------------------
# Skill metrics
# ---------------------------------------------------------------------------

def cv_regression_corr(X, y, n_splits=5):
    """
    Cross-validated correlation of a multiple linear regression.

    The samples are split in `n_splits` contiguous blocks in time (no shuffling, so
    autocorrelated neighbouring months do not leak between training and test sets).
    Each block is predicted with the model fitted on the others.
    """
    X, y = _valid(X, y)
    y_cv = np.empty_like(y)
    for train, test in KFold(n_splits=n_splits, shuffle=False).split(X):
        y_cv[test] = LinearRegression().fit(X[train], y[train]).predict(X[test])
    return np.corrcoef(y, y_cv)[0, 1]


def noise_regression_corr(n_samples, n_predictors, n_rep=100, seed=0):
    """
    In-sample correlation obtained by regressing a series on `n_predictors` random
    (white noise) predictors. Returns (mean, 95th percentile) over `n_rep` repetitions:
    the skill expected by chance alone.
    """
    rng = np.random.default_rng(seed)
    r = np.empty(n_rep)
    for k in range(n_rep):
        X = rng.normal(size=(n_samples, n_predictors))
        y = rng.normal(size=n_samples)
        r[k] = np.corrcoef(y, LinearRegression().fit(X, y).predict(X))[0, 1]
    return r.mean(), np.percentile(r, 95)


def skill_vs_npcs(pcs_indices, n_pcs_list, n_splits=5, n_rep=100):
    """
    For each index and number of PCs: in-sample correlation, cross-validated
    correlation and the chance level (white-noise predictors, mean and 95th pct.).
    Returns a Dataset with dims (n_pcs_used, climate_index).
    """
    names = _index_names(pcs_indices)
    out = {k: np.full((len(n_pcs_list), len(names)), np.nan)
           for k in ["insample", "cv", "noise_mean", "noise_p95"]}
    noise_cache = {}

    for i, n_pcs in enumerate(n_pcs_list):
        X_all = pcs_indices.PCs.isel(n_pcs=slice(0, n_pcs)).values
        for j, name in enumerate(names):
            X, y = _valid(X_all, pcs_indices[INDEX_VARS[name]].values)
            out["insample"][i, j] = np.corrcoef(y, LinearRegression().fit(X, y).predict(X))[0, 1]
            out["cv"][i, j] = cv_regression_corr(X, y, n_splits)
            key = (len(y), n_pcs)
            if key not in noise_cache:
                noise_cache[key] = noise_regression_corr(len(y), n_pcs, n_rep)
            out["noise_mean"][i, j], out["noise_p95"][i, j] = noise_cache[key]

    coords = {"n_pcs_used": list(n_pcs_list), "climate_index": names}
    return xr.Dataset({k: (("n_pcs_used", "climate_index"), v) for k, v in out.items()},
                      coords=coords)


def seasonal_skill(pcs_indices, n_pcs=20, n_splits=5):
    """Cross-validated regression correlation computed separately for each season."""
    names = _index_names(pcs_indices)
    months = pcs_indices.time.dt.month.values
    X_all = pcs_indices.PCs.isel(n_pcs=slice(0, n_pcs)).values
    corr = np.full((len(SEASONS), len(names)), np.nan)
    for i, m in enumerate(SEASONS.values()):
        sel = np.isin(months, m)
        for j, name in enumerate(names):
            corr[i, j] = cv_regression_corr(X_all[sel], pcs_indices[INDEX_VARS[name]].values[sel],
                                            n_splits)
    return xr.DataArray(corr, dims=("season", "climate_index"),
                        coords={"season": list(SEASONS), "climate_index": names})


def pc_index_correlation(pcs_indices, n_pcs=30):
    """Correlation of each index with each of the first `n_pcs` PCs (common months)."""
    names = _index_names(pcs_indices)
    n_pcs = min(n_pcs, pcs_indices.sizes["n_pcs"])
    corr = np.full((len(names), n_pcs), np.nan)
    for j, name in enumerate(names):
        y = pcs_indices[INDEX_VARS[name]].values
        mask = np.isfinite(y)
        for k in range(n_pcs):
            corr[j, k] = np.corrcoef(pcs_indices.PCs.values[mask, k], y[mask])[0, 1]
    return xr.DataArray(corr, dims=("climate_index", "pc"),
                        coords={"climate_index": names, "pc": np.arange(1, n_pcs + 1)})


def top_pcs_table(pcs_indices, n_pcs=100, k=3, label="PC"):
    """For each index, the `k` PCs with the highest |correlation| (among the first `n_pcs`)."""
    corr = pc_index_correlation(pcs_indices, n_pcs)
    rows = {}
    for name in corr.climate_index.values:
        c = corr.sel(climate_index=name).values
        order = np.argsort(-np.abs(c))[:k]
        rows[str(name)] = [f"{label}{p + 1} ({c[p]:+.2f})" for p in order]
    return pd.DataFrame(rows, index=[f"#{i + 1}" for i in range(k)]).T


def index_correlation_matrix(pcs_indices):
    """Correlation between the indices (each pair over their common months)."""
    names = _index_names(pcs_indices)
    df = pd.DataFrame({n: pcs_indices[INDEX_VARS[n]].values for n in names})
    return df.corr(min_periods=24)


def forward_selection(X, y, max_m, n_splits=5):
    """
    Greedy forward selection of predictors: at each step the column of X that most
    increases the in-sample R^2 is added. Returns (selected columns, in-sample corr,
    cross-validated corr) for 1..max_m selected predictors.
    """
    X, y = _valid(X, y)
    selected, r_in, r_cv = [], [], []
    for _ in range(min(max_m, X.shape[1])):
        best, best_r2 = None, -np.inf
        for j in set(range(X.shape[1])) - set(selected):
            cols = selected + [j]
            r2 = LinearRegression().fit(X[:, cols], y).score(X[:, cols], y)
            if r2 > best_r2:
                best, best_r2 = j, r2
        selected.append(best)
        r_in.append(np.sqrt(max(best_r2, 0)))
        r_cv.append(cv_regression_corr(X[:, selected], y, n_splits))
    return selected, np.array(r_in), np.array(r_cv)


def forward_selection_skill(pcs_indices, n_pool, max_m=10, n_splits=5):
    """
    Forward selection for every index, choosing among the first `n_pool` components.
    Returns Dataset (m, climate_index) with 'insample', 'cv' and 'selected'
    (component number, 1-based), plus 'cv_all' (climate_index): cross-validated
    skill using the whole pool.
    """
    names = _index_names(pcs_indices)
    X = pcs_indices.PCs.isel(n_pcs=slice(0, n_pool)).values
    shape = (min(max_m, n_pool), len(names))
    r_in, r_cv, sel = np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, int)
    cv_all = np.full(len(names), np.nan)
    for j, name in enumerate(names):
        y = pcs_indices[INDEX_VARS[name]].values
        s, r_in[:, j], r_cv[:, j] = forward_selection(X, y, max_m, n_splits)
        sel[:, j] = np.array(s) + 1
        cv_all[j] = cv_regression_corr(X, y, n_splits)
    coords = {"m": np.arange(1, shape[0] + 1), "climate_index": names}
    return xr.Dataset({"insample": (("m", "climate_index"), r_in),
                       "cv": (("m", "climate_index"), r_cv),
                       "selected": (("m", "climate_index"), sel),
                       "cv_all": (("climate_index",), cv_all)}, coords=coords)


def n_needed(fs, frac=0.9):
    """
    Number of selected components needed to reach `frac` of the full-pool cross-validated
    skill (NaN if not reached or if the full-pool skill is not positive).
    """
    target = frac * fs.cv_all
    reached = fs.cv >= target
    n = reached.argmax("m") + 1
    return n.where(reached.any("m") & (fs.cv_all > 0))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_skill_vs_npcs(skill, ncols=5, figsize=(22, 8)):
    """One panel per index: in-sample and cross-validated skill vs number of PCs, with chance level."""
    names = [str(n) for n in skill.climate_index.values]
    n = skill.n_pcs_used.values
    nrows = int(np.ceil(len(names) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    ymin = min(0, np.floor(float(skill.cv.min()) * 10) / 10)

    for ax, name in zip(axs, names):
        s = skill.sel(climate_index=name)
        color = INDICES[name]["color"]
        ax.fill_between(n, 0, s.noise_p95, color="lightgrey", label="Chance (95th pct.)")
        ax.plot(n, s.noise_mean, ":", color="grey", label="Chance (mean)")
        ax.plot(n, s.insample, "-o", color=color, label="In-sample")
        ax.plot(n, s.cv, "--s", color="black", alpha=0.7, label="Cross-validated")
        ax.set_title(name, fontsize=17)
        ax.set_ylim(ymin, 1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.3)
    for ax in axs[len(names):]:
        ax.set_visible(False)
    for ax in axs[-ncols:]:
        ax.set_xlabel("Number of PCs")
    for ax in axs[::ncols]:
        ax.set_ylabel("Correlation")
    axs[0].legend(fontsize=12, loc="upper left")
    plt.tight_layout()
    return fig, axs


def plot_pc_index_heatmap(corr, figsize=(20, 5), vmax=0.8, label="PC", ax=None):
    """Heatmap of the correlation between each index and each PC."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    fig = ax.figure
    im = ax.pcolormesh(corr.pc, np.arange(corr.sizes["climate_index"]), corr.values,
                       cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, shading="nearest")
    ax.set_yticks(np.arange(corr.sizes["climate_index"]))
    ax.set_yticklabels(corr.climate_index.values, fontsize=16)
    ax.set_xticks(corr.pc.values)
    ax.set_xlabel(label, fontsize=17)
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, label="Correlation", pad=0.01)
    return fig, ax


def plot_annotated_matrix(df, title="", cmap=CMAP_DIV, vmin=-1, vmax=1, fmt="{:.2f}",
                          figsize=(9, 7)):
    """Annotated heatmap of a 2D table (pandas DataFrame)."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df.values, cmap=cmap, vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    ax.set_xticks(np.arange(df.shape[1]))
    long_labels = max(len(str(c)) for c in df.columns) > 5
    ax.set_xticklabels(df.columns, fontsize=15, rotation=35 if long_labels else 0,
                       ha="right" if long_labels else "center")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=15)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = df.values[i, j]
            if np.isfinite(v):
                r, g, b, _ = cm((v - vmin) / (vmax - vmin))
                dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=13,
                        color="white" if dark else "black")
    ax.set_title(title, fontsize=17)
    plt.colorbar(im, ax=ax, shrink=0.8)
    return fig, ax


# groups of indices used to order the correlation matrix (indices not listed go at the end)
INDEX_GROUPS = {
    "ENSO": ["N34", "RONI", "ELI", "OLR", "SOI"],
    "Pacific": ["PDO", "PNA"],
    "N. Atlantic": ["NAO", "AO", "SCA", "AMO"],
    "S. Hem.": ["SAM"],
}


def plot_index_correlation(corr, groups=INDEX_GROUPS, cmap=CMAP_DIV, figsize=(8, 6.8),
                           title="Correlation between indices", strong=0.5):
    """
    Lower-triangle correlation matrix between indices, ordered and separated by groups.
    Correlations with |r| >= `strong` are written in bold, |r| < 0.1 in grey.
    """
    names = list(corr.index)
    order, bounds, glabels = [], [], []
    for g, members in groups.items():
        m = [n for n in members if n in names]
        if m:
            glabels.append((g, len(order), len(order) + len(m)))
            order += m
            bounds.append(len(order))
    rest = [n for n in names if n not in order]
    if rest:
        glabels.append(("Other", len(order), len(order) + len(rest)))
        order += rest
    corr = corr.loc[order, order]
    n = len(order)

    # lower triangle without diagonal: rows 1..n-1, columns 0..n-2
    vals = corr.values.copy()
    vals[np.triu_indices(n)] = np.nan
    vals = vals[1:, :-1]
    rows, cols = order[1:], order[:-1]

    cm = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cm, vmin=-1, vmax=1)
    # white grid between cells
    ax.set_xticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n - 1, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(n - 1):
        for j in range(n - 1):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            r, g, b, _ = cm((v + 1) / 2)
            dark = 0.299 * r + 0.587 * g + 0.114 * b < 0.5
            color = "white" if dark else ("0.55" if abs(round(v, 2)) < 0.1 else "black")
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "\u2212."),
                    ha="center", va="center", fontsize=10.5, color=color,
                    fontweight="bold" if abs(v) >= strong else "normal")

    # separators between groups (only inside the triangle)
    for b in bounds[:-1]:
        ax.plot([b - 0.5, b - 0.5], [b - 1.5, n - 1.5], color="0.3", lw=1.2)   # vertical
        ax.plot([-0.5, b - 0.5], [b - 1.5, b - 1.5], color="0.3", lw=1.2)      # horizontal

    ax.set_xticks(np.arange(n - 1))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(n - 1))
    ax.set_yticklabels(rows)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 1.5, -0.5)

    # group names along the diagonal
    for g, i0, i1 in glabels:
        ax.text((i0 + i1) / 2 - 0.5, (i0 + i1) / 2 - 1.5, g, ha="center", va="bottom",
                fontsize=11.5, fontstyle="italic", color="0.3", rotation=-45, rotation_mode="anchor")

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02, ticks=np.linspace(-1, 1, 5))
    cb.set_label("Correlation")
    cb.outline.set_visible(False)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    return fig, ax


def plot_forward_selection(fs_dict, ncols=5, figsize=(22, 8)):
    """
    Cross-validated skill vs number of selected components for several approaches
    (fs_dict: {label: output of forward_selection_skill}), one panel per index.
    The horizontal lines give the skill using the whole pool.
    """
    styles = [("-o", PURPLE), ("-s", GREEN), ("-^", ORANGE)]
    first = next(iter(fs_dict.values()))
    names = [str(n) for n in first.climate_index.values]
    nrows = int(np.ceil(len(names) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    ymin = min(0, np.floor(min(float(f.cv.min()) for f in fs_dict.values()) * 10) / 10)
    for ax, name in zip(axs, names):
        for (label, fs), (st, color) in zip(fs_dict.items(), styles):
            f = fs.sel(climate_index=name)
            ax.plot(f.m, f.cv, st, color=color, label=label)
            ax.axhline(float(f.cv_all), color=color, linestyle=":", linewidth=1.5)
        ax.set_title(name, fontsize=17)
        ax.set_ylim(ymin, 1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(alpha=0.3)
    for ax in axs[len(names):]:
        ax.set_visible(False)
    for ax in axs[-ncols:]:
        ax.set_xlabel("Number of selected components")
    for ax in axs[::ncols]:
        ax.set_ylabel("Cross-validated corr.")
    axs[0].legend(fontsize=13, loc="lower right")
    plt.tight_layout()
    return fig, axs


def plot_grouped_bars(df, ylabel="", figsize=(16, 5), colors=(PURPLE, GREEN, ORANGE),
                      ylim=None):
    """Grouped bar plot of a DataFrame (rows: groups on the x axis, columns: bars)."""
    x = np.arange(len(df.index))
    w = 0.8 / df.shape[1]
    fig, ax = plt.subplots(figsize=figsize)
    for k, col in enumerate(df.columns):
        ax.bar(x + (k - (df.shape[1] - 1) / 2) * w, df[col].values, w, label=col,
               color=colors[k % len(colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=17)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=15)
    return fig, ax


# ---------------------------------------------------------------------------
# Decision plots (PCA vs RPCA)
# ---------------------------------------------------------------------------

APPROACH_COLORS = _APPROACH_COLORS


def plot_dumbbell(values, m_list, xlabel="Cross-validated correlation", figsize=(18, 6)):
    """
    One panel per number of components m: for each index, PCA and RPCA values joined by
    a line (green line: RPCA better, purple line: PCA better).
    values : dict {m: DataFrame (index x ['PCA', 'RPCA'])}
    """
    fig, axs = plt.subplots(1, len(m_list), figsize=figsize, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, m_list):
        df = values[m]
        y = np.arange(len(df))
        for k, (a, b) in enumerate(zip(df["PCA"], df["RPCA"])):
            ax.plot([a, b], [k, k], color=GREEN if b > a else PURPLE, linewidth=3,
                    alpha=0.6, zorder=1)
        for col in ["PCA", "RPCA"]:
            ax.scatter(df[col], y, s=120, color=APPROACH_COLORS[col], label=col, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(df.index, fontsize=16)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{m} component{'s' if m > 1 else ''}", fontsize=17)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
    axs[0].legend(fontsize=15, loc="lower left")
    plt.tight_layout()
    return fig, axs


def plot_group_curves(curves, groups, xlabel="Number of selected components", figsize=(18, 5)):
    """
    Mean cross-validated skill of each group of indices vs number of components.
    curves : dict {label: DataArray (m, climate_index)}  (label containing 'RPCA' -> red, else blue)
    groups : dict {group name: list of indices}
    """
    fig, axs = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
    axs = np.atleast_1d(axs)
    styles = ["-", "--", ":", "-."]
    for ax, (gname, names) in zip(axs, groups.items()):
        n_r = n_p = 0
        for label, cv in curves.items():
            is_r = "RPCA" in label
            st = styles[(n_r if is_r else n_p) % len(styles)]
            n_r, n_p = n_r + is_r, n_p + (not is_r)
            mean = cv.sel(climate_index=names).mean("climate_index")
            ax.plot(cv.m, mean, st, marker="o", linewidth=2.5, label=label,
                    color=APPROACH_COLORS["RPCA" if is_r else "PCA"])
        ax.set_title(f"{gname}\n({', '.join(names)})", fontsize=16)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Mean cross-validated correlation")
    axs[-1].legend(fontsize=14, loc="lower right")
    plt.tight_layout()
    return fig, axs


def plot_decision_dashboard(criteria, labels=("PCA", "RPCA"), colors=None, figsize=(20, 4.5)):
    """
    One small panel per criterion with the value of each approach.
    criteria : dict {title: (value_1, ..., value_n, higher_is_better)}, values in the order of `labels`
    The winner of each criterion is outlined in black and the title says which is better.
    """
    colors = colors or [APPROACH_COLORS.get(l, "grey")
                        for l in labels]
    fig, axs = plt.subplots(1, len(criteria), figsize=figsize)
    axs = np.atleast_1d(axs)
    for ax, (title, v) in zip(axs, criteria.items()):
        vals, higher = list(v[:-1]), v[-1]
        win = int(np.argmax(vals) if higher else np.argmin(vals))
        bars = ax.bar(list(labels), vals, color=colors, alpha=0.8)
        bars[win].set_edgecolor("black")
        bars[win].set_linewidth(3)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=14)
        ax.set_title(f"{title}\n({'higher' if higher else 'lower'} is better)", fontsize=14)
        ax.set_ylim(min(0, min(vals) * 1.25), max(vals) * 1.25)
        ax.tick_params(axis="x", labelsize=14, rotation=20)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# Frequency analysis of the PCs (seasonal / interannual / decadal)
# ---------------------------------------------------------------------------

# period bands in years: [min, max)
BANDS = {"Intra-annual (< 1 yr)": (0, 1),
         "Interannual (1-8 yr)": (1, 8),
         "Decadal (> 8 yr)": (8, np.inf)}
BAND_COLORS = [GREEN, PURPLE, ORANGE]
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def regular_monthly_pcs(pcs_indices, n_pcs=100):
    """
    First `n_pcs` PCs on a regular monthly axis (needed for the Fourier analysis).
    Missing months (anomalous months removed in notebook 01) are filled by linear
    interpolation. Returns (DataArray (time, pc), number of filled months).
    """
    pcs = pcs_indices.PCs.isel(n_pcs=slice(0, n_pcs))
    pcs = pcs.assign_coords(time=pcs.time.dt.floor("D"))
    t = pd.date_range(pcs.time.values[0], pcs.time.values[-1], freq="MS")
    reg = pcs.reindex(time=t)
    n_filled = int(reg.isel(n_pcs=0).isnull().sum())
    reg = reg.interpolate_na("time").rename(n_pcs="pc")
    return reg.assign_coords(pc=np.arange(1, reg.sizes["pc"] + 1)), n_filled


def periodogram(x, dt=1 / 12):
    """
    One-sided periodogram normalised so that its sum is the variance of `x`.
    Returns (frequencies [cycles / year], power) without the zero frequency.
    """
    x = np.asarray(x, float) - np.nanmean(x)
    p = np.abs(np.fft.rfft(x)) ** 2 / len(x) ** 2
    p[1:] *= 2
    if len(x) % 2 == 0:
        p[-1] /= 2
    f = np.fft.rfftfreq(len(x), dt)
    return f[1:], p[1:]


def ar1_spectrum(r1, f, dt=1 / 12):
    """Shape of the AR(1) (red noise) spectrum with lag-1 autocorrelation `r1`."""
    return (1 - r1 ** 2) / (1 - 2 * r1 * np.cos(2 * np.pi * f * dt) + r1 ** 2)


def _lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    return np.sum(x[1:] * x[:-1]) / np.sum(x * x)


def _band_masks(f, bands):
    period = 1 / f
    return {b: (period >= lo) & (period < hi) for b, (lo, hi) in bands.items()}


def band_variance(pcs, bands=BANDS):
    """
    Fraction of the variance of each PC in each period band (Parseval on the
    periodogram), and the fraction expected for a red noise (AR(1)) process with the
    same lag-1 autocorrelation.
    pcs : DataArray (time, pc) on a regular monthly axis (see `regular_monthly_pcs`)
    Returns Dataset (pc, band) with 'fraction', 'red_noise' and 'ratio', plus 'r1' (pc).
    """
    frac = np.zeros((pcs.sizes["pc"], len(bands)))
    red = np.zeros_like(frac)
    r1 = np.zeros(pcs.sizes["pc"])
    for k in range(pcs.sizes["pc"]):
        x = pcs.isel(pc=k).values
        f, p = periodogram(x)
        r1[k] = _lag1(x)
        s_red = ar1_spectrum(r1[k], f)
        for j, m in enumerate(_band_masks(f, bands).values()):
            frac[k, j] = p[m].sum() / p.sum()
            red[k, j] = s_red[m].sum() / s_red.sum()
    dims = ("pc", "band")
    return xr.Dataset({"fraction": (dims, frac), "red_noise": (dims, red),
                       "ratio": (dims, frac / red), "r1": ("pc", r1)},
                      coords={"pc": pcs.pc.values, "band": list(bands)})


def field_band_variance(bv, explained_variance_ratio):
    """
    Variance of the SST + MSL field explained by the first n PCs in each band
    (cumulative over n). The EOFs are orthonormal, so the variance of the field in a
    band is the sum of the variance of the PCs in that band.
    Returns DataArray (pc, band) in fraction of the total field variance.
    """
    evr = xr.DataArray(np.asarray(explained_variance_ratio)[:bv.sizes["pc"]], dims="pc",
                       coords={"pc": bv.pc})
    return (bv.fraction * evr).cumsum("pc")


def smoothed_spectrum(x, n_smooth=7, dt=1 / 12):
    """
    Periodogram smoothed with a Daniell (running mean) filter of `n_smooth` frequencies,
    with the AR(1) red noise spectrum and its 95% confidence level (chi2 with
    2 * n_smooth degrees of freedom). The power is in variance per unit frequency.
    Returns Dataset with 'power', 'red_noise' and 'red_noise_95' (freq [cycles / year]).
    """
    from scipy.stats import chi2
    f, p = periodogram(x, dt)
    df = f[0]
    # running mean (fewer frequencies at the edges instead of padding with zeros)
    kernel = np.ones(n_smooth)
    p_s = (np.convolve(p, kernel, mode="same") / np.convolve(np.ones_like(p), kernel, mode="same")
           / df)
    red = ar1_spectrum(_lag1(x), f, dt)
    red *= p.sum() / df / red.sum()
    dof = 2 * n_smooth
    return xr.Dataset({"power": ("freq", p_s), "red_noise": ("freq", red),
                       "red_noise_95": ("freq", red * chi2.ppf(0.95, dof) / dof)},
                      coords={"freq": f})


def band_filter(x, lo, hi, dt=1 / 12):
    """Component of `x` with periods in [lo, hi) years (ideal Fourier filter)."""
    x = np.asarray(x, float)
    fx = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), dt)
    with np.errstate(divide="ignore"):
        period = 1 / f
    fx[~((period >= lo) & (period < hi))] = 0
    return np.fft.irfft(fx, len(x))


def seasonal_std(pcs):
    """
    Standard deviation of each PC by calendar month divided by its overall standard
    deviation (> 1: the PC is more active in that month). Returns DataFrame (pc, month).
    """
    ratio = pcs.groupby("time.month").std("time") / pcs.std("time")
    df = ratio.T.to_pandas()
    df.index = [f"PC{p}" for p in df.index]
    df.columns = MONTH_LABELS
    return df


def seasonal_std_significance(pcs, n_rep=500, seed=0):
    """
    95th percentile of the ratio max(monthly std) / min(monthly std) obtained when the
    calendar months are randomly shuffled by year blocks (keeps the autocorrelation).
    Returns (observed ratio per PC, 95th percentile per PC) as DataArrays.
    """
    rng = np.random.default_rng(seed)
    month = pcs.time.dt.month.values
    year = pcs.time.dt.year.values

    def _ratio(m):
        s = np.array([pcs.values[m == mm].std(0) for mm in range(1, 13)])
        return s.max(0) / s.min(0)

    obs = _ratio(month)
    null = np.empty((n_rep, pcs.sizes["pc"]))
    for r in range(n_rep):
        # circular shift of the calendar month inside each year
        shift = rng.integers(0, 12, size=year.max() - year.min() + 1)[year - year.min()]
        null[r] = _ratio((month - 1 + shift) % 12 + 1)
    p95 = np.percentile(null, 95, axis=0)
    return (xr.DataArray(obs, dims="pc", coords={"pc": pcs.pc}),
            xr.DataArray(p95, dims="pc", coords={"pc": pcs.pc}))


def plot_band_fractions(bv, n_show=30, figsize=(20, 9)):
    """
    Top: fraction of the variance of each PC in each period band (stacked bars), with the
    fraction expected for red noise (markers). Bottom: ratio observed / red noise.
    """
    bv = bv.isel(pc=slice(0, n_show))
    pcs = bv.pc.values
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    bottom = np.zeros(len(pcs))
    red_bottom = np.zeros(len(pcs))
    for b, color in zip(bv.band.values, BAND_COLORS):
        v = bv.fraction.sel(band=b).values
        ax.bar(pcs, v, bottom=bottom, color=color, alpha=0.8, label=b)
        bottom += v
        red_bottom += bv.red_noise.sel(band=b).values
        if b != bv.band.values[-1]:
            ax.plot(pcs, red_bottom, "_", color="black", markersize=14, markeredgewidth=2.5)
    ax.plot([], [], "_", color="black", markersize=14, markeredgewidth=2.5,
            label="Red noise (band limits)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of variance")
    ax.legend(fontsize=13, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="y", alpha=0.3)

    w = 0.8 / bv.sizes["band"]
    for j, (b, color) in enumerate(zip(bv.band.values, BAND_COLORS)):
        ax2.bar(pcs + (j - 1) * w, bv.ratio.sel(band=b).values, w, color=color, alpha=0.8)
    ax2.axhline(1, color="black", linewidth=1)
    ax2.set_yscale("log", base=2)
    ax2.set_ylabel("Observed / red noise")
    ax2.set_xlabel("PC")
    ax2.set_xticks(pcs)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig, (ax, ax2)


def plot_field_band_variance(fbv, figsize=(14, 5)):
    """Cumulative variance of the SST + MSL field explained by the first n PCs, per band."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(fbv.pc, *[100 * fbv.sel(band=b) for b in fbv.band.values],
                 labels=list(fbv.band.values), colors=BAND_COLORS, alpha=0.8)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Explained variance of the field [%]")
    ax.set_xlim(1, int(fbv.pc[-1]))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig, ax


def plot_pc_spectra(pcs, n_show=12, ncols=4, n_smooth=7, bands=BANDS, figsize=(22, 12)):
    """
    Smoothed spectrum of the first `n_show` PCs vs period, with the red noise spectrum
    and its 95% level. Peaks above the dashed line are significant. Shaded: bands.
    """
    nrows = int(np.ceil(n_show / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axs = np.atleast_1d(axs).ravel()
    for k, ax in enumerate(axs[:n_show]):
        s = smoothed_spectrum(pcs.isel(pc=k).values, n_smooth)
        period = 1 / s.freq.values
        for (lo, hi), color in zip(bands.values(), BAND_COLORS):
            ax.axvspan(max(lo, period.min()), min(hi, period.max()), color=color, alpha=0.08)
        ax.plot(period, s.power, color="black", linewidth=1.8)
        ax.plot(period, s.red_noise, ":", color="grey")
        ax.plot(period, s.red_noise_95, "--", color=ORANGE, linewidth=1.5)
        sig = s.power > s.red_noise_95
        ax.plot(period[sig], s.power[sig], "o", color=ORANGE, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"PC{int(pcs.pc[k])}", fontsize=16)
        ax.grid(alpha=0.3, which="both")
    for ax in axs[n_show:]:
        ax.set_visible(False)
    for ax in axs[-ncols:]:
        ax.set_xlabel("Period [years]")
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8", "16", "32"])
    for ax in axs[::ncols]:
        ax.set_ylabel("Power")
    axs[0].plot([], [], ":", color="grey", label="Red noise")
    axs[0].plot([], [], "--", color=ORANGE, label="Red noise 95%")
    axs[0].legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    return fig, axs


def plot_band_decomposition(pcs, pc_list=(1, 2, 3, 4), bands=BANDS, figsize=(20, 12)):
    """Time series of some PCs (grey) with their components in each band."""
    fig, axs = plt.subplots(len(pc_list), 1, figsize=figsize, sharex=True)
    axs = np.atleast_1d(axs)
    t = pcs.time.values
    for ax, p in zip(axs, pc_list):
        x = pcs.sel(pc=p).values
        ax.plot(t, x, color="lightgrey", linewidth=1, label="PC")
        for (b, (lo, hi)), color in zip(bands.items(), BAND_COLORS):
            if lo == 0:
                continue
            ax.plot(t, band_filter(x, lo, hi), color=color, linewidth=2.2, label=b)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"PC{p}")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=12, loc="upper left", ncol=3)
    plt.tight_layout()
    return fig, axs
