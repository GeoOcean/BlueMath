"""
Selection of the PCs of a VAR model by their cross-validated forecast skill.

The PCA orders the modes by variance, but a VAR needs modes that are predictable or that help
to predict the others. The PCs are added one by one (greedy forward selection), each time the
one that most increases the cross-validated forecast skill of the VAR.

All the functions work with standardized PCs Z (time x modes, unit variance) and their
variances lam (field variance of each mode). With X ~ Z L^T and orthogonal loadings, the
field squared error of a component error e is sum_k lam_k e_k^2, so field-space skill only
needs lam.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import PURPLE, GREEN, ORANGE


# ---------------------------------------------------------------------------
# Cross-validated forecast skill of a VAR
# ---------------------------------------------------------------------------

def cv_forecast(Z, horizon=6, n_splits=5, valid=None, ridge=0.0, lags=1):
    """
    Cross-validated VAR(lags) forecasts: the record is split in `n_splits` contiguous blocks;
    the VAR is fitted on the other blocks and forecasts of 1..horizon months are issued from
    every origin of the held-out block (origin and target inside the block), iterating the
    model (companion matrix powers).
    Returns (forecast, observed) arrays (n_origin, horizon, k) and a boolean valid mask
    (n_origin, horizon) with the targets that are valid months.
    """
    n, k = Z.shape
    p = lags
    valid = np.ones(n, bool) if valid is None else valid
    # state at time t: [z_t, z_{t-1}, ..., z_{t-p+1}]
    S = np.hstack([np.roll(Z, j, axis=0) for j in range(p)])
    ok_state = np.array([t >= p - 1 and valid[t - p + 1:t + 1].all() for t in range(n)])
    blocks = np.array_split(np.arange(n), n_splits)
    fc, ob, ok = [], [], []
    for b in blocks:
        in_test = np.zeros(n, bool)
        in_test[b] = True
        t = np.arange(p - 1, n - 1)
        # regression z_{t+1} = A s_t: origin state and target outside the test block
        m = ok_state[t] & valid[t + 1] & ~in_test[t + 1] & ~np.array([in_test[i - p + 1:i + 1].any() for i in t])
        X, Y = S[t[m]], Z[t[m] + 1]
        A = np.linalg.solve(X.T @ X + ridge * np.eye(k * p), X.T @ Y).T       # (k, k p)
        C = np.vstack([A, np.eye(k * (p - 1), k * p)]) if p > 1 else A          # companion (k p, k p)
        powers = [np.linalg.matrix_power(C, h)[:k] for h in range(1, horizon + 1)]
        for t0 in b[p - 1:len(b) - horizon]:
            if not ok_state[t0]:
                continue
            fc.append(np.array([P @ S[t0] for P in powers]))
            ob.append(Z[t0 + 1:t0 + horizon + 1])
            ok.append(valid[t0 + 1:t0 + horizon + 1])
    return np.array(fc), np.array(ob), np.array(ok)


def field_skill(fc, obs, lam, total_var, valid=None):
    """
    Fraction of the total field variance predicted at each lead (see var_analysis.field_predicted_fraction),
    for standardized components with field variances lam (diagonal G).
    """
    valid = np.ones(fc.shape[:2], bool) if valid is None else valid
    w = valid.astype(float)
    sse = ((fc - obs) ** 2 * lam).sum(-1)
    sso = (obs ** 2 * lam).sum(-1)
    return ((sso * w).sum(0) - (sse * w).sum(0)) / w.sum(0) / total_var


def component_r2(fc, obs, valid=None):
    """Out-of-sample R^2 (1 - MSE / variance) of each component at each lead -> (horizon, k)."""
    valid = np.ones(fc.shape[:2], bool) if valid is None else valid
    w = valid[..., None].astype(float)
    mse = (((fc - obs) ** 2) * w).sum(0) / w.sum(0)
    var = ((obs ** 2) * w).sum(0) / w.sum(0)
    return 1 - mse / var


def var_score(Z, cols, lam, total_var, leads, horizon, n_splits=5, valid=None, lags=1):
    """Mean (over `leads`, 1-based) cross-validated field variance predicted by a VAR(lags) on `cols`."""
    if max(leads) > horizon:
        raise ValueError(f"leads {tuple(leads)} go beyond the forecast horizon ({horizon} months)")
    fc, ob, ok = cv_forecast(Z[:, cols], horizon, n_splits, valid, lags=lags)
    fs = field_skill(fc, ob, lam[cols], total_var, ok)
    return fs[np.asarray(leads) - 1].mean(), fs


def forward_select_var(Z, lam, total_var, pool, max_sel=20, leads=(1, 2, 3), horizon=6,
                       n_splits=5, valid=None, start=None, verbose=True, lags=1):
    """
    Greedy forward selection of the PCs of a VAR(lags): at each step the PC of `pool` that most
    increases the cross-validated field variance predicted (mean over `leads`) is added.
    A PC can enter because it carries much variance, because it is very persistent, or
    because it improves the forecast of the PCs already selected (dynamical coupling).
    Returns DataFrame (one row per step) with the selected PC (1-based), the score and its gain,
    and the skill at each lead.
    """
    selected = list(start or [])
    rows, prev = [], (var_score(Z, selected, lam, total_var, leads, horizon, n_splits, valid, lags)[0]
                      if selected else 0.0)
    for step in range(max_sel - len(selected)):
        best, best_score, best_fs = None, -np.inf, None
        for j in pool:
            if j in selected:
                continue
            s, fs = var_score(Z, selected + [j], lam, total_var, leads, horizon, n_splits, valid, lags)
            if s > best_score:
                best, best_score, best_fs = j, s, fs
        selected.append(best)
        rows.append({"n": len(selected), "PC": best + 1, "score": best_score, "gain": best_score - prev,
                     **{f"lead {h + 1}": v for h, v in enumerate(best_fs)}})
        prev = best_score
        if verbose:
            print(f"{len(selected):3d}: + PC{best + 1:<4d} score = {100 * best_score:.2f}%  "
                  f"(gain {100 * rows[-1]['gain']:+.2f}%)")
    return pd.DataFrame(rows).set_index("n")


def skill_first_n(Z, lam, total_var, n_list, leads=(1, 2, 3), horizon=6, n_splits=5, valid=None, lags=1):
    """Same score as forward_select_var for the first n PCs (variance order)."""
    return pd.Series({n: var_score(Z, list(range(n)), lam, total_var, leads, horizon, n_splits, valid, lags)[0]
                      for n in n_list}, name="first n")


def choose_n(sel, min_gain=0.001):
    """Number of selected PCs: last step whose gain is >= min_gain (fraction of total variance)."""
    ok = np.where(sel["gain"].values >= min_gain)[0]
    return int(sel.index[ok[-1]]) if len(ok) else int(sel.index[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pc_predictability(diag, n_show=None, figsize=(20, 8)):
    """
    Per PC: variance and predictable variance (variance x cross-validated R^2) at leads 1 and 3
    months (top), and cross-validated R^2 (bottom). diag: DataFrame indexed by PC number.
    """
    d = diag.iloc[:n_show] if n_show else diag
    x = d.index.values
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    axs[0].bar(x, d["variance [%]"], color="lightgrey", edgecolor="grey", label="Variance")
    axs[0].bar(x - 0.2, d["predictable var. lead 1 [%]"], 0.4, color=ORANGE, label="Predictable, lead 1")
    axs[0].bar(x + 0.2, d["predictable var. lead 3 [%]"], 0.4, color=PURPLE, label="Predictable, lead 3")
    axs[0].set_ylabel("% of total variance")
    axs[0].set_yscale("log")
    axs[0].legend(ncol=3)
    axs[1].bar(x - 0.2, d["CV R2 lead 1"], 0.4, color=ORANGE, label="lead 1")
    axs[1].bar(x + 0.2, d["CV R2 lead 3"], 0.4, color=PURPLE, label="lead 3")
    axs[1].axhline(0, color="black", linewidth=0.8)
    axs[1].set_ylabel("Cross-validated R²")
    axs[1].set_xlabel("PC")
    axs[1].set_xticks(x)
    axs[1].legend(ncol=2)
    for ax in axs:
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    return fig, axs


def plot_selection(sel, first_n, captured_sel=None, captured_first=None, n_chosen=None, figsize=(18, 5)):
    """Cross-validated skill vs number of PCs: greedy selection vs first n PCs (and variance captured)."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    ax = axs[0]
    ax.plot(sel.index, 100 * sel["score"], "-o", color=ORANGE, label="Selected (greedy)")
    ax.plot(first_n.index, 100 * first_n.values, "-s", color=PURPLE, label="First n PCs")
    if n_chosen:
        ax.axvline(n_chosen, color="black", linestyle="--", linewidth=1, label=f"chosen n = {n_chosen}")
    ax.set_xlabel("Number of PCs in the VAR")
    ax.set_ylabel("CV field variance predicted [%]")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axs[1]
    ax.bar(sel.index, 100 * sel["gain"], color=ORANGE, alpha=0.8)
    for n, pc in zip(sel.index, sel["PC"]):
        ax.text(n, 100 * sel.loc[n, "gain"], f"{pc}", ha="center", va="bottom", fontsize=10, rotation=90)
    ax.set_xlabel("Step (label: PC added)")
    ax.set_ylabel("Gain [% of total variance]")
    ax.grid(alpha=0.3, axis="y")
    if captured_sel is not None:
        ax2 = axs[0].twinx()
        ax2.plot(captured_sel.index, 100 * captured_sel.values, ":", color=ORANGE)
        ax2.plot(captured_first.index, 100 * captured_first.values, ":", color=PURPLE)
        ax2.set_ylabel("Variance captured [%] (dotted)")
    plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------------
# Climate indices
# ---------------------------------------------------------------------------

def components_indices(Z, time, p_indices, names=None):
    """
    Dataset with the components (time, n_pcs) and every climate index aligned to their months,
    in the format of save_pcs_indices (so the functions of pcs_analysis can be used).
    """
    import xarray as xr
    from .climate_indices import INDICES, INDEX_VARS, load_index

    names = names or list(INDICES)
    ds = xr.Dataset({"PCs": (("time", "n_pcs"), Z)},
                    coords={"time": time, "n_pcs": np.arange(Z.shape[1])})
    for n in names:
        ds[INDEX_VARS[n]] = load_index(n, p_indices)["index"].reindex(time=ds.time)
    ds["index_name"] = ("climate_index", names)
    return ds


def index_skill(datasets, n_splits=5, insample=False):
    """
    Regression correlation of each index on each set of components: cross-validated, or in-sample
    (as index_correlation in notebooks 01 / 03) with `insample`.
    datasets : {label: output of components_indices}. Returns DataFrame (index x label).
    """
    from .climate_indices import INDEX_VARS, linear_regression
    from .pcs_analysis import cv_regression_corr

    out = {}
    for label, ds in datasets.items():
        names = [str(n) for n in ds.index_name.values]
        X = ds.PCs.values
        out[label] = {n: (linear_regression(X, ds[INDEX_VARS[n]].values, verbose=False)[0] if insample
                          else cv_regression_corr(X, ds[INDEX_VARS[n]].values, n_splits)) for n in names}
    return pd.DataFrame(out)


def best_single_component(datasets, prefix=None):
    """
    Highest |correlation| between each index and a single component of each set, and which one.
    prefix : {label: list of component names}. Returns (DataFrame of |corr|, DataFrame of names).
    """
    from .pcs_analysis import pc_index_correlation

    corr, which = {}, {}
    for label, ds in datasets.items():
        c = pc_index_correlation(ds, n_pcs=ds.sizes["n_pcs"])
        k = np.abs(c).argmax("pc")
        names = (prefix or {}).get(label, [f"{j + 1}" for j in range(ds.sizes["n_pcs"])])
        corr[label] = pd.Series(np.abs(c).max("pc").values, index=c.climate_index.values)
        which[label] = pd.Series([f"{names[int(i)]} ({float(c.values[j, int(i)]):+.2f})"
                                  for j, i in enumerate(k.values)], index=c.climate_index.values)
    return pd.DataFrame(corr), pd.DataFrame(which)
