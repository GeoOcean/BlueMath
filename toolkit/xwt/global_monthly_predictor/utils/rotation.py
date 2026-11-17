"""
PCA and varimax-rotated PCA (RPCA) of the SST + MSL predictor.

Notation (X: n_time x n_points, centred predictor):
    X = U S V^T                      (SVD)
    lambda_k = S_k^2 / (n - 1)       (variance of mode k)
    Z = U sqrt(n - 1)                standardized PCs (unit variance)
    L = V sqrt(lambda)               loadings, X ~ Z L^T
Varimax finds an orthogonal matrix R that simplifies the loadings of the first k modes:
    L_r = L R,   Z_r = Z R           (X ~ Z_r L_r^T, same subspace and total variance)
"""

import numpy as np
import xarray as xr

VARS = ("sst_detrend", "msl_detrend")


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def stack_predictor(ds, variables=VARS):
    """Stack the predictor variables into a (time, point) matrix, dropping points with NaN."""
    blocks, index = [], []
    for v in variables:
        a = ds[v].transpose("time", "latitude", "longitude").values
        a = a.reshape(a.shape[0], -1)
        valid = np.all(np.isfinite(a), axis=0)
        blocks.append(a[:, valid])
        index.append(np.where(valid)[0])
    return np.hstack(blocks), index


def pca_svd(ds, variables=VARS):
    """
    PCA of the predictor via SVD.
    Returns dict with standardized PCs 'Z', loadings 'L', variances 'lam',
    explained variance ratio 'evr', grid-point std 'std' and the stacking 'index'.
    """
    X, index = stack_predictor(ds, variables)
    X = X - X.mean(axis=0)
    n = X.shape[0]
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    lam = S ** 2 / (n - 1)
    return {
        "Z": U * np.sqrt(n - 1),
        "L": Vt.T * np.sqrt(lam),
        "lam": lam,
        "evr": lam / lam.sum(),
        "std": X.std(axis=0, ddof=1),
        "index": index,
        "time": ds.time.values,
    }


# ---------------------------------------------------------------------------
# Varimax
# ---------------------------------------------------------------------------

def varimax(L, normalize=True, max_iter=1000, tol=1e-8):
    """
    Varimax rotation matrix R (k x k) for the loadings L (p x k).
    With `normalize` (Kaiser normalization) each row is scaled to unit length
    during the rotation, so all grid points weigh the same.
    """
    L = np.asarray(L, dtype=float)
    if normalize:
        h = np.sqrt((L ** 2).sum(axis=1, keepdims=True))
        h[h == 0] = 1
        L = L / h
    p, k = L.shape
    R = np.eye(k)
    d = 0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - Lr @ np.diag((Lr ** 2).sum(axis=0)) / p))
        R = u @ vt
        d_old, d = d, s.sum()
        if d_old and d / d_old < 1 + tol:
            break
    return R


def rotate(pca, n_rot, normalize=True):
    """
    Varimax rotation of the first `n_rot` modes.
    Rotated modes are sorted by explained variance and their sign is set so that
    the largest loading is positive.
    Returns dict with rotated standardized PCs 'Zr', loadings 'Lr', 'evr',
    and the rotation matrix 'R' (unrotated mode x rotated mode).
    """
    return rotate_modes(pca, list(range(n_rot)), normalize)


def rotate_modes(pca, modes, normalize=True):
    """
    Varimax rotation of a selection of modes (0-based column positions, e.g. the PCs
    selected for a VAR). Same output as `rotate`; the rows of 'R' follow `modes`.
    """
    L, Z = pca["L"][:, modes], pca["Z"][:, modes]
    n_rot = len(modes)
    R = varimax(L, normalize=normalize)
    Lr, Zr = L @ R, Z @ R

    var = (Lr ** 2).sum(axis=0)
    order = np.argsort(-var)
    Lr, Zr, R, var = Lr[:, order], Zr[:, order], R[:, order], var[order]

    sign = np.sign(Lr[np.abs(Lr).argmax(axis=0), np.arange(n_rot)])
    Lr, Zr, R = Lr * sign, Zr * sign, R * sign

    return {"Zr": Zr, "Lr": Lr, "R": R, "evr": var / pca["lam"].sum()}


# ---------------------------------------------------------------------------
# Maps and metrics
# ---------------------------------------------------------------------------

def loadings_to_maps(L, pca, ds, variables=VARS, as_correlation=True):
    """
    Unstack loadings (p x k) into maps (mode, latitude, longitude) per variable.
    With `as_correlation` the loadings are divided by the grid-point std, i.e. the
    maps show the correlation between each grid point and the (standardized) mode.
    """
    if as_correlation:
        L = L / pca["std"][:, None]
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    out, i0 = {}, 0
    for v, idx in zip(variables, pca["index"]):
        m = np.full((L.shape[1], nlat * nlon), np.nan)
        m[:, idx] = L[i0:i0 + len(idx)].T
        i0 += len(idx)
        out[v] = (("mode", "latitude", "longitude"), m.reshape(-1, nlat, nlon))
    return xr.Dataset(out, coords={"mode": np.arange(1, L.shape[1] + 1),
                                   "latitude": ds.latitude.values,
                                   "longitude": ds.longitude.values})


def participation_ratio(L):
    """
    Fraction of grid points that effectively contribute to each pattern,
    (sum l^2)^2 / (p * sum l^4): 1 = uniform over all points, small = localized.
    """
    L2 = L ** 2
    return L2.sum(axis=0) ** 2 / (L.shape[0] * (L2 ** 2).sum(axis=0))


def build_rpca_dataset(pca, rot, ds, n_rot):
    """Dataset with rotated and unrotated PCs, correlation maps, explained variance and R."""
    maps_r = loadings_to_maps(rot["Lr"], pca, ds)
    maps_u = loadings_to_maps(pca["L"][:, :n_rot], pca, ds)
    out = xr.Dataset(
        {
            "RPCs": (("time", "mode"), rot["Zr"]),
            "PCs": (("time", "mode"), pca["Z"][:, :n_rot]),
            "rpc_sst": maps_r["sst_detrend"], "rpc_msl": maps_r["msl_detrend"],
            "eof_sst": maps_u["sst_detrend"], "eof_msl": maps_u["msl_detrend"],
            "evr_rotated": (("mode",), rot["evr"]),
            "evr_unrotated": (("mode",), pca["evr"][:n_rot]),
            "pr_rotated": (("mode",), participation_ratio(rot["Lr"])),
            "pr_unrotated": (("mode",), participation_ratio(pca["L"][:, :n_rot])),
            "rotation_matrix": (("mode_unrotated", "mode"), rot["R"]),
        },
        coords={"time": pca["time"], "mode": np.arange(1, n_rot + 1),
                "mode_unrotated": np.arange(1, n_rot + 1),
                "latitude": ds.latitude.values, "longitude": ds.longitude.values},
    )
    out.attrs["n_rot"] = n_rot
    out.attrs["description"] = ("Varimax rotation (Kaiser normalized) of the first n_rot modes. "
                                "PCs/RPCs standardized (unit variance). Maps: correlation between "
                                "grid-point anomaly and mode.")
    return out


def pcs_dataarray(Z, time, n=None):
    """PCs as DataArray (time, n_pcs) as expected by utils.climate_indices."""
    Z = Z if n is None else Z[:, :n]
    return xr.DataArray(Z, dims=("time", "n_pcs"),
                        coords={"time": time, "n_pcs": np.arange(Z.shape[1])})


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def _loadings(X, n_modes, rotated):
    """Loadings of the first n_modes (optionally varimax-rotated) of a (time, point) matrix."""
    X = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    lam = S ** 2 / (X.shape[0] - 1)
    L = Vt[:n_modes].T * np.sqrt(lam[:n_modes])
    return L @ varimax(L) if rotated else L


def split_half_stability(ds, n_modes, rotated, variables=VARS):
    """
    Reproducibility of the spatial patterns: the analysis is repeated with the first and
    the second half of the period, and each full-period pattern is matched with the most
    similar pattern of each half (|congruence coefficient|, i.e. uncentred spatial correlation).
    Returns array (n_modes,) with the mean of the two halves (1 = identical pattern).
    """
    X, _ = stack_predictor(ds, variables)
    L_full = _loadings(X, n_modes, rotated)
    if rotated:   # same order as `rotate`: by explained variance
        L_full = L_full[:, np.argsort(-(L_full ** 2).sum(axis=0))]
    half = X.shape[0] // 2
    out = []
    for Xh in (X[:half], X[half:]):
        L_h = _loadings(Xh, n_modes, rotated)
        a = L_full / np.linalg.norm(L_full, axis=0)
        b = L_h / np.linalg.norm(L_h, axis=0)
        out.append(np.abs(a.T @ b).max(axis=1))
    return np.mean(out, axis=0)
