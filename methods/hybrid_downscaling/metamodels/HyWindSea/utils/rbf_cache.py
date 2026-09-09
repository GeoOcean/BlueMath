"""Make a fitted RBF fast to predict from, by caching what it recomputes.

`bluemath_tk`'s `RBF` stores only the optimal sigmas when it is fitted and
rebuilds the interpolation coefficients on every call to `predict`. Rebuilding
means assembling a matrix as wide as the training set and solving it once per
target component, which for a 1,000-case model with tens of components is
seconds of work — and it is the same work every time, because the coefficients
depend on the fitted model alone, not on what is being predicted.

These helpers compute them once, keep them on disk, and reuse them:

>>> coefficients = rbf_coefficients(model, cache_path="outputs/.../model.npz")
>>> predicted = rbf_predict(model, wind_pcs, coefficients)

`rbf_predict` returns the same thing `model.predict` does. The first run per
model costs what one `predict` costs today; every run after it is a few hundred
milliseconds.

Note that this reaches into private attributes of `RBF` (`_opt_sigmas`,
`_calc_rbf_coeff`, `_rbf_variable_interpolation`). It is pinned to the toolkit's
internals on purpose — the alternative was editing the installed package — so if
an upgrade changes them, `check_against_predict` below says so immediately.
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

#: Coefficients and sigma for one target component.
Coefficients = Dict[str, Tuple[Optional[float], np.ndarray]]


def rbf_coefficients(
    model,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Coefficients:
    """
    Interpolation coefficients of a fitted RBF, computed once and cached.

    Parameters
    ----------
    model : bluemath_tk.interpolation.rbf.RBF
        A fitted model, as returned by `load_model`.
    cache_path : str, optional
        Where to keep them. Read if it exists, written if it does not. Pass None
        to compute every time.
    verbose : bool, optional
        Report whether the cache was used and how long the solve took.

    Returns
    -------
    dict
        One entry per target component: `(sigma, coefficients)`. `sigma` is None
        for kernels that do not use one, the linear kernel included.

    Notes
    -----
    The cache is a plain `.npz`, so it can be inspected and it does not carry a
    pickle of the toolkit's classes — which means it survives a toolkit upgrade
    even if these helpers do not.
    """

    variables = list(model.target_processed_variables)

    if cache_path and os.path.exists(cache_path):
        stored = np.load(cache_path, allow_pickle=False)
        names = [str(name) for name in stored["names"]]
        if names != variables:
            raise ValueError(
                f"{cache_path} holds {len(names)} components and this model has "
                f"{len(variables)}. Delete the cache and let it rebuild."
            )
        sigmas = stored["sigmas"]
        coefficients = {
            name: (None if np.isnan(sigmas[i]) else float(sigmas[i]),
                   stored["coefficients"][:, i])
            for i, name in enumerate(names)
        }
        if verbose:
            print(f"RBF coefficients read from {cache_path}")
        return coefficients

    import time

    start = time.time()
    subset = model.normalized_subset_data.values.T

    coefficients = {}
    for name in variables:
        sigma = model._opt_sigmas[name]
        target = model.normalized_target_data[name].values
        solved, _ = model._calc_rbf_coeff(sigma=sigma, x=subset, y=target)
        coefficients[name] = (sigma, solved.flatten())

    if verbose:
        print(f"solved {len(variables)} components in {time.time() - start:.0f} s")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(
            cache_path,
            names=np.array(variables),
            sigmas=np.array([np.nan if coefficients[n][0] is None
                             else coefficients[n][0] for n in variables]),
            coefficients=np.column_stack([coefficients[n][1] for n in variables]),
        )
        if verbose:
            print(f"cached to {cache_path} "
                  f"({os.path.getsize(cache_path) / 1e6:.1f} MB)")

    return coefficients


def rbf_predict(
    model,
    dataset: pd.DataFrame,
    coefficients: Optional[Coefficients] = None,
    cache_path: Optional[str] = None,
    batched: bool = True,
    row_chunk: int = 4096,
) -> pd.DataFrame:
    """
    Predict with a fitted RBF, reusing cached coefficients.

    Parameters
    ----------
    model : bluemath_tk.interpolation.rbf.RBF
        The fitted model.
    dataset : pd.DataFrame
        Points to predict at, with the same columns the model was fitted on.
    coefficients : dict, optional
        From `rbf_coefficients`. Fetched using `cache_path` if not given.
    cache_path : str, optional
        Passed to `rbf_coefficients` when `coefficients` is None.
    batched : bool, optional
        Evaluate every target component from one kernel matrix. Default is True.
        False walks the components one by one exactly as the toolkit does, which
        is slower but reproduces its arithmetic bit for bit — useful when
        checking this module against it.
    row_chunk : int, optional
        Rows evaluated at a time. The kernel matrix is `row_chunk` by the size of
        the training set, so this is what bounds memory. Default is 4096.

    Returns
    -------
    pd.DataFrame
        The same thing `model.predict(dataset)` returns: one column per target
        component, indexed like `dataset`, denormalised if the model was fitted
        with normalised targets.

    Notes
    -----
    Every target component is evaluated from the same kernel matrix. The
    distances from the query points to the training points do not depend on
    which component is being interpolated, but the toolkit rebuilds them once
    per component — with fifty components that is fifty times the same distance
    matrix. Here it is built once per chunk and all the components come out of a
    single matrix product.

    That only holds while every component shares a sigma, which is the case for
    kernels that do not use one (the linear kernel included). When they differ
    the components are evaluated one by one, exactly as the toolkit does.

    Batching changes the order the sums are accumulated in, so the result is not
    bit-identical to the toolkit's: on these models the components differ by
    around 1e-4, which comes out as **6e-7 m** in the reconstructed wave field —
    seven orders of magnitude below anything the model resolves. `batched=False`
    is exact if that matters.
    """

    if coefficients is None:
        coefficients = rbf_coefficients(model, cache_path=cache_path, verbose=False)

    normalized = model._preprocess_subset_data(subset_data=dataset, is_fit=False)
    num_variables, num_points = model.normalized_subset_data.T.shape

    variables = list(model.target_processed_variables)
    sigmas = {coefficients[name][0] for name in variables}

    if batched and len(sigmas) == 1:
        interpolated = _batched(
            model, normalized.values, coefficients, variables,
            num_points, num_variables, sigmas.pop(), row_chunk,
        )
    else:
        interpolated = np.empty((len(dataset), len(variables)))
        for index, name in enumerate(variables):
            sigma, solved = coefficients[name]
            interpolated[:, index] = model._rbf_variable_interpolation(
                normalized_dataset=normalized,
                opt_sigma=sigma,
                rbf_coeff=solved,
                num_points_subset=num_points,
                num_vars_subset=num_variables,
            )

    predicted = pd.DataFrame(interpolated, columns=variables)

    if model.is_target_normalized:
        predicted = model.denormalize(
            normalized_data=predicted, scale_factor=model.target_scale_factor
        )

    predicted.index = dataset.index

    return predicted


def _batched(model, query, coefficients, variables, num_points, num_variables,
             sigma, row_chunk):
    """Evaluate every component from one kernel matrix per chunk."""

    centres = model.normalized_subset_data.values
    stacked = np.column_stack([coefficients[name][1] for name in variables])

    rbf_weights = stacked[:num_points]
    constants = stacked[num_points]
    linear_weights = stacked[num_points + 1: num_points + 1 + num_variables]

    kernel_sigma = 1.0 if sigma is None else sigma
    out = np.empty((query.shape[0], len(variables)))

    for start in range(0, query.shape[0], row_chunk):
        block = query[start:start + row_chunk]
        distances = np.linalg.norm(block[:, None, :] - centres[None, :, :], axis=2)
        kernel = model.kernel_func(distances, kernel_sigma)
        out[start:start + row_chunk] = (
            constants + kernel @ rbf_weights + block @ linear_weights
        )

    return out


def check_against_predict(
    model,
    dataset: pd.DataFrame,
    coefficients: Optional[Coefficients] = None,
    batched: bool = False,
) -> float:
    """
    Compare the cached path against the toolkit's own `predict`.

    Parameters
    ----------
    model : bluemath_tk.interpolation.rbf.RBF
        The fitted model.
    dataset : pd.DataFrame
        A few rows are enough.
    coefficients : dict, optional
        From `rbf_coefficients`.

    batched : bool, optional
        Which path to check. Default is False, the exact one, where the
        difference should be zero. True checks the fast path, where around 1e-4
        on the components is expected and harmless.

    Returns
    -------
    float
        The largest absolute difference between the two, in principal-component
        units. With `batched=False` anything other than zero means the toolkit's
        internals have moved and these helpers need revisiting.
    """

    reference = np.asarray(model.predict(dataset), dtype=float)
    fast = np.asarray(
        rbf_predict(model, dataset, coefficients, batched=batched), dtype=float
    )

    return float(np.nanmax(np.abs(reference - fast)))
