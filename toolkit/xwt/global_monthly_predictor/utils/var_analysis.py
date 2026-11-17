"""
Vector autoregressive (VAR) models of the monthly (rotated) PCs: fitting, rolling
forecasts, stochastic simulation and comparison of simulated vs observed statistics.

Components are standardized (unit variance). Field-space metrics use the loadings L
(points x modes): the field anomaly reconstructed from components z is z L^T, so the
field squared error of a component error e is e G e^T with G = L^T L.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf

from .pcs_analysis import cv_regression_corr
from . import PURPLE, GREEN, ORANGE, CMAP_DIV, APPROACH_COLORS



def color_of(label):
    """Colour of an approach from a label containing its name (longest match first)."""
    for key in sorted(APPROACH_COLORS, key=len, reverse=True):
        if key in label:
            return APPROACH_COLORS[key]
    return "grey"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def components_frame(Z, time, n, prefix="PC", names=None):
    """
    First `n` components (columns of Z) as a DataFrame on a regular monthly time axis.
    Months removed in the preprocessing are filled by linear interpolation
    (a VAR needs regular time steps). Returns (DataFrame, filled dates).
    """
    df = pd.DataFrame(Z[:, :n], index=pd.to_datetime(time),
                      columns=names or [f"{prefix}{k + 1}" for k in range(n)])
    full = pd.date_range(df.index[0], df.index[-1], freq="MS")
    missing = full.difference(df.index)
    return df.reindex(full).interpolate("linear"), missing


def sunspots_monthly(p_file):
    """Monthly mean sunspot number and its time derivative (exogenous forcing)."""
    df = xr.open_dataset(p_file).to_dataframe().dropna()
    return df.resample("MS").mean()


# ---------------------------------------------------------------------------
# Fit and forecast
# ---------------------------------------------------------------------------

def fit_var(df, lags=1, exog=None, trend="n"):
    """Fit a VAR(lags) with optional exogenous variables (aligned DataFrame)."""
    return VAR(df, exog=exog).fit(maxlags=lags, trend=trend)


def order_selection(df, maxlags=12, exog=None):
    """Information criteria (AIC, BIC, HQIC, FPE) for lags 0..maxlags and the selected order."""
    sel = VAR(df, exog=exog).select_order(maxlags=maxlags, trend="n")
    ic = pd.DataFrame({k: v for k, v in sel.ics.items()})
    ic.index.name = "lags"
    return ic, sel.selected_orders


def rolling_forecast(results, df, start, horizon=12, exog=None):
    """
    Forecasts from every origin between `start` (first forecast target, position) and the
    end, with observed values as initial conditions (model parameters kept fixed).
    Returns (forecast, observed, persistence) arrays (n_origin, horizon, k) and the
    target dates (n_origin, horizon).
    """
    p, y = results.k_ar, df.values
    ex = None if exog is None else np.asarray(exog)
    fc, obs, pers, dates = [], [], [], []
    for i in range(max(start, p), len(y) - horizon + 1):
        kw = {} if ex is None else {"exog_future": ex[i:i + horizon]}
        fc.append(results.forecast(y[i - p:i], steps=horizon, **kw))
        obs.append(y[i:i + horizon])
        pers.append(np.repeat(y[i - 1][None], horizon, axis=0))
        dates.append(df.index[i:i + horizon].values)
    return np.array(fc), np.array(obs), np.array(pers), np.array(dates)


def lead_correlation(fc, obs, valid=None):
    """
    Correlation between forecast and observation for each lead and component (H, k).
    valid : boolean (n_origin, H), targets used in the evaluation (e.g. not interpolated)
    """
    H, k = fc.shape[1:]
    valid = np.ones(fc.shape[:2], bool) if valid is None else valid
    return np.array([[np.corrcoef(fc[valid[:, h], h, j], obs[valid[:, h], h, j])[0, 1]
                      for j in range(k)] for h in range(H)])


def field_predicted_fraction(fc, obs, G, total_var, valid=None):
    """
    Fraction of the total field variance predicted at each lead:
    (mean field variance of the observations - mean field squared error) / total variance.
    0 = climatology forecast; the maximum is the variance captured by the components.
    valid : boolean (n_origin, H), targets used in the evaluation
    """
    valid = np.ones(fc.shape[:2], bool) if valid is None else valid
    e = fc - obs
    sse = np.einsum("tha,ab,thb->th", e, G, e)
    sso = np.einsum("tha,ab,thb->th", obs, G, obs)
    w = valid.astype(float)
    return ((sso * w).sum(0) - (sse * w).sum(0)) / w.sum(0) / total_var


def insample_r2(results, df, exclude=None):
    """R2 of the one-step fitted values for each component (dates in `exclude` are ignored)."""
    fv = results.fittedvalues
    if exclude is not None:
        fv = fv.loc[fv.index.difference(exclude)]
    obs = df.loc[fv.index]
    return 1 - ((obs - fv) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _deterministic(results):
    """Rows of the parameter matrix that are not lags (trend / exogenous terms)."""
    params = results.params
    det = [r for r in params.index if not (r.startswith("L") and "." in r and r[1:r.index(".")].isdigit())]
    return params.loc[det]


def simulate(results, n_steps, n_sims=100, seed=0, burn=120, initial_values=None, exog=None):
    """
    Stochastic simulations y_t = sum_i A_i y_{t-i} + B x_t + e_t, e_t ~ N(0, Sigma_u).

    initial_values : (p, k) array; if None the simulation starts from zeros with a burn-in
    exog           : (n_steps, n_exog) array of exogenous values (same names/order as in the fit)
    Returns array (n_sims, n_steps, k).
    """
    rng = np.random.default_rng(seed)
    A, p, k = results.coefs, results.k_ar, results.neqs
    chol = np.linalg.cholesky(np.asarray(results.sigma_u))
    det = _deterministic(results)
    burn = 0 if initial_values is not None else burn

    x = np.zeros((burn + n_steps, len(det)))
    for j, name in enumerate(det.index):
        if name == "const":
            x[:, j] = 1
        elif exog is not None:
            col = np.asarray(exog)[:, list(exog.columns).index(name)] if hasattr(exog, "columns") \
                else np.asarray(exog)[:, j]
            x[burn:, j] = col
            x[:burn, j] = col[0]
    B = det.values.T  # (k, n_det)

    out = np.empty((n_sims, n_steps, k))
    for s in range(n_sims):
        y = np.zeros((p + burn + n_steps, k))
        if initial_values is not None:
            y[:p] = initial_values
        eps = rng.standard_normal((burn + n_steps, k)) @ chol.T
        for t in range(burn + n_steps):
            yt = eps[t] + (B @ x[t] if len(det) else 0)
            for i in range(p):
                yt = yt + A[i] @ y[p + t - 1 - i]
            y[p + t] = yt
        out[s] = y[p + burn:]
    return out


# ---------------------------------------------------------------------------
# Statistics of observed vs simulated series
# ---------------------------------------------------------------------------

def acf_matrix(x, nlags=24):
    """ACF (lags 0..nlags) of each column of x (T, k) -> (nlags + 1, k)."""
    return np.array([acf(x[:, j], nlags=nlags, fft=True) for j in range(x.shape[1])]).T


def lowfreq_std(x, window=60):
    """Std of the centred running mean (default 5 years) of each column."""
    return pd.DataFrame(x).rolling(window, center=True).mean().std().values


def simulation_stats(obs, sims, nlags=24, window=60):
    """
    Compare observed (T, k) and simulated (n_sims, T, k) series.
    Returns dict with std, ACF and low-frequency std (obs and per simulation).
    """
    return {
        "std_obs": obs.std(axis=0),
        "std_sim": sims.std(axis=1),
        "acf_obs": acf_matrix(obs, nlags),
        "acf_sim": np.array([acf_matrix(s, nlags) for s in sims]),
        "lf_obs": lowfreq_std(obs, window),
        "lf_sim": np.array([lowfreq_std(s, window) for s in sims]),
    }


def simulation_scores(st):
    """Scalar fidelity scores (lower is better): mean |ACF error| (lags 1..), |log std ratio|, |log low-freq ratio|."""
    acf_err = np.abs(np.median(st["acf_sim"], axis=0)[1:] - st["acf_obs"][1:]).mean()
    std_err = np.abs(np.log(np.median(st["std_sim"], axis=0) / st["std_obs"])).mean()
    lf_err = np.abs(np.log(np.median(st["lf_sim"], axis=0) / st["lf_obs"])).mean()
    return {"ACF error": acf_err, "Std error": std_err, "Low-freq. std error": lf_err}


# ---------------------------------------------------------------------------
# Climate indices from components
# ---------------------------------------------------------------------------

def index_models(df, indices, index_vars, exclude=None, n_splits=5):
    """
    Linear regression of each index on the VAR components (observed months only;
    dates in `exclude`, e.g. interpolated months, are ignored).
    indices : Dataset with one variable per index (time)
    Returns dict {name: (model, cv corr)}.
    """
    out = {}
    for name, var in index_vars.items():
        if var not in indices:
            continue
        y = indices[var].reindex(time=df.index).values.copy()
        if exclude is not None:
            y[df.index.isin(exclude)] = np.nan
        X = df.values
        m = np.isfinite(y)
        out[name] = (LinearRegression().fit(X[m], y[m]), cv_regression_corr(X, y, n_splits))
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pvalues(results_dict, alpha=0.05, figsize=(14, 5)):
    """p-values of the VAR coefficients for several models side by side (dots: p < alpha)."""
    fig, axs = plt.subplots(1, len(results_dict), figsize=figsize)
    axs = np.atleast_1d(axs)
    for ax, (label, res) in zip(axs, results_dict.items()):
        pv = res.pvalues
        im = ax.imshow(pv.values, cmap="Purples_r", vmin=0, vmax=1, aspect="auto")
        yy, xx = np.where(pv.values < alpha)
        ax.scatter(xx, yy, color=ORANGE, edgecolor="white", s=40)
        ax.set_xticks(range(pv.shape[1]))
        ax.set_xticklabels(pv.columns, rotation=90)
        ax.set_yticks(range(pv.shape[0]))
        ax.set_yticklabels(pv.index, fontsize=11)
        n_sig = (pv.values < alpha).sum()
        ax.set_title(f"{label}: {n_sig}/{pv.size} with p < {alpha} (dots)", fontsize=15)
        ax.set_xlabel("Equation (dependent variable)")
    plt.colorbar(im, ax=axs, label="p-value", shrink=0.8)
    return fig, axs


def plot_coef_matrices(results_dict, lag=1, vmax=None, figsize=(14, 5)):
    """Lag-`lag` coefficient matrices A (row: equation, column: predictor) side by side."""
    fig, axs = plt.subplots(1, len(results_dict), figsize=figsize)
    axs = np.atleast_1d(axs)
    mats = {k: r.coefs[lag - 1] for k, r in results_dict.items()}
    vmax = vmax or max(np.abs(m).max() for m in mats.values())
    for ax, (label, A) in zip(axs, mats.items()):
        names = results_dict[label].names
        im = ax.imshow(A, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=11)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f"{n}(t-{lag})" for n in names], rotation=90)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([f"{n}(t)" for n in names])
        diag = np.trace(np.abs(A)) / np.abs(A).sum()
        ax.set_title(f"{label}: A{lag}  (diagonal weight {100 * diag:.0f}%)", fontsize=15)
    plt.colorbar(im, ax=axs, shrink=0.8)
    return fig, axs


def plot_fit(df, results, n_show=None, color=GREEN, figsize=None, exclude=None):
    """Observed vs one-step fitted values for each component (time series + scatter)."""
    cols = df.columns[:n_show] if n_show else df.columns
    fig, axs = plt.subplots(len(cols), 2, figsize=figsize or (14, 2.2 * len(cols)),
                            gridspec_kw={"width_ratios": [3, 1]}, squeeze=False)
    r2 = insample_r2(results, df, exclude)
    for i, c in enumerate(cols):
        obs, fv = df.loc[results.fittedvalues.index, c], results.fittedvalues[c]
        axs[i, 0].plot(obs, color="black", linewidth=1, label="obs")
        axs[i, 0].plot(fv, color=color, linewidth=1, label="fitted")
        axs[i, 0].set_ylabel(c)
        axs[i, 1].scatter(obs, fv, s=5, alpha=0.5, color=color)
        lim = [min(obs.min(), fv.min()), max(obs.max(), fv.max())]
        axs[i, 1].plot(lim, lim, "k", linewidth=1)
        axs[i, 1].set_title(f"R2 = {r2[c]:.2f}", fontsize=13)
    axs[0, 0].legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    return fig, axs


def plot_lead_skill(curves, ylabel, baselines=None, figsize=(9, 5), ax=None):
    """Skill vs lead time. curves / baselines: dict {label: array (H,)}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for label, v in curves.items():
        color = color_of(label)
        ax.plot(np.arange(1, len(v) + 1), v, "-o", color=color, linewidth=2.5, label=label)
    for label, v in (baselines or {}).items():
        color = color_of(label)
        ax.plot(np.arange(1, len(v) + 1), v, ":", color=color, linewidth=2, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Lead time [months]", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=13)
    return ax.figure, ax


def plot_blind_forecast(df_train, df_blind, fc_det, sims, label="PC", q=(0.05, 0.95), n_examples=3,
                        color=GREEN, figsize=None):
    """Blind period: observations, deterministic forecast and simulation envelope (reference style)."""
    cols = df_blind.columns
    fig, axs = plt.subplots(len(cols), 1, figsize=figsize or (14, 1.8 * len(cols)), sharex=True)
    axs = np.atleast_1d(axs)
    x0 = df_train.index[-36:]
    for j, (ax, c) in enumerate(zip(axs, cols)):
        lo, hi = np.quantile(sims[:, :, j], q, axis=0)
        ax.plot(x0, df_train[c].values[-36:], color="black", linewidth=1.5)
        ax.fill_between(df_blind.index, lo, hi, color=color, alpha=0.3,
                        label=f"sim {int(100 * q[0])}-{int(100 * q[1])}%")
        for s in range(n_examples):
            ax.plot(df_blind.index, sims[s, :, j], color=color, linewidth=0.8, alpha=0.8)
        ax.plot(df_blind.index, fc_det[:, j], "--", color="dimgrey", linewidth=2, label="deterministic forecast")
        ax.plot(df_blind.index, df_blind[c].values, color="black", linewidth=1.5, label="obs")
        ax.axvline(df_blind.index[0], color="k", linestyle=":")
        ax.set_ylabel(c)
    axs[0].legend(loc="upper left", fontsize=11, ncol=3)
    plt.tight_layout()
    return fig, axs


def plot_acf_comparison(stats_dict, names_dict, q=(0.05, 0.95), ncols=5, figsize=None):
    """Observed ACF (black) vs simulated ACF (median and band) for each component."""
    n = max(len(v) for v in names_dict.values())
    fig, axs = plt.subplots(len(stats_dict), n, figsize=figsize or (3.4 * n, 3 * len(stats_dict)),
                            sharex=True, sharey=True, squeeze=False)
    for r, (label, st) in enumerate(stats_dict.items()):
        color = color_of(label)
        lags = np.arange(st["acf_obs"].shape[0])
        for j, name in enumerate(names_dict[label]):
            ax = axs[r, j]
            lo, hi = np.quantile(st["acf_sim"][:, :, j], q, axis=0)
            ax.fill_between(lags, lo, hi, color=color, alpha=0.3)
            ax.plot(lags, np.median(st["acf_sim"][:, :, j], axis=0), color=color, linewidth=2, label="sim")
            ax.plot(lags, st["acf_obs"][:, j], "k-", linewidth=1.5, label="obs")
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set_title(name, fontsize=14)
            ax.grid(alpha=0.3)
        axs[r, 0].set_ylabel(f"{label}\nACF")
    for ax in axs[-1]:
        ax.set_xlabel("Lag [months]")
    axs[0, 0].legend(fontsize=12)
    plt.tight_layout()
    return fig, axs


def plot_spectra(obs_dict, sims_dict, names_dict, nperseg=256, q=(0.05, 0.95), figsize=None):
    """Power spectra (Welch) of observed (black) and simulated (band) components."""
    n = max(len(v) for v in names_dict.values())
    fig, axs = plt.subplots(len(obs_dict), n, figsize=figsize or (3.4 * n, 3 * len(obs_dict)),
                            sharex=True, sharey=True, squeeze=False)
    for r, label in enumerate(obs_dict):
        color = color_of(label)
        obs, sims = obs_dict[label], sims_dict[label]
        for j, name in enumerate(names_dict[label]):
            ax = axs[r, j]
            f, po = welch(obs[:, j], fs=12, nperseg=nperseg)
            ps = np.array([welch(s[:, j], fs=12, nperseg=nperseg)[1] for s in sims])
            lo, hi = np.quantile(ps, q, axis=0)
            ax.fill_between(f[1:], lo[1:], hi[1:], color=color, alpha=0.3)
            ax.loglog(f[1:], np.median(ps, axis=0)[1:], color=color, linewidth=2, label="sim")
            ax.loglog(f[1:], po[1:], "k-", linewidth=1.2, label="obs")
            ax.set_title(name, fontsize=14)
            ax.grid(alpha=0.3, which="both")
        axs[r, 0].set_ylabel(f"{label}\nPSD")
    for ax in axs[-1]:
        ax.set_xlabel("Frequency [cycles/year]")
    axs[0, 0].legend(fontsize=12)
    plt.tight_layout()
    return fig, axs


def plot_stat_ratio(stats_dict, key, names_dict, ylabel, figsize=(16, 4.5)):
    """Ratio simulated / observed of a statistic ('std' or 'lf') per component (boxplot over simulations)."""
    fig, axs = plt.subplots(1, len(stats_dict), figsize=figsize, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (label, st) in zip(axs, stats_dict.items()):
        color = color_of(label)
        ratio = st[f"{key}_sim"] / st[f"{key}_obs"]
        b = ax.boxplot(ratio, patch_artist=True, showfliers=False)
        for patch in b["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(names_dict[label]) + 1))
        ax.set_xticklabels(names_dict[label])
        ax.axhline(1, color="black", linestyle="--")
        ax.set_title(label, fontsize=16)
        ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel(ylabel)
    plt.tight_layout()
    return fig, axs
