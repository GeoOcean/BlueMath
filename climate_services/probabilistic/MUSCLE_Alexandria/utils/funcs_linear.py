from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import xarray as xr
from bluemath_tk.datamining.pca import PCA
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split


def plot_pcs(pca: PCA, n_pcs: int) -> None:
    """
    Plot the principal components (PCs) from a PCA analysis.

    Parameters
    ----------
    pca : PCA
        The PCA object containing the principal components.
    n_pcs : int
        The number of principal components to plot.
    """

    fig, axs = plt.subplots(n_pcs, 1, figsize=(15, 5 * n_pcs))

    # Handle case where n_pcs = 1 (axs becomes a single axis, not an array)
    if n_pcs == 1:
        axs = [axs]

    for i in range(n_pcs):
        pca.pcs.PCs.isel(n_component=i).plot(ax=axs[i], color="darkmagenta")
        axs[i].axhline(0, color="black", lw=1)
        axs[i].grid(":", lw=0.5)
        axs[i].set_title(f"PC {i + 1}")
        axs[i].set_xlim([pca.pcs.time.min(), pca.pcs.time.max()])
        axs[i].set_ylim(
            [
                -np.nanmax(np.abs(pca.pcs.PCs.isel(n_component=range(n_pcs)))),
                np.nanmax(np.abs(pca.pcs.PCs.isel(n_component=range(n_pcs)))),
            ]
        )


def plot_waves(waves: xr.Dataset) -> None:
    """
    Plot the wave data.

    This function creates a subplots figure and plots the wave data for each variable.

    Parameters
    ----------
    waves : xarray.Dataset
        Dataset containing wave data variables.
    """

    n_vars = len(waves.data_vars.keys())
    fig, axs = plt.subplots(n_vars, 1, figsize=(20, 4 * n_vars))

    # Handle case where n_vars = 1 (axs becomes a single axis, not an array)
    if n_vars == 1:
        axs = [axs]

    for iv, var in enumerate(list(waves.data_vars.keys())):
        ax = axs[iv]

        if var == "mwd":
            waves[var].plot(ax=ax, color="crimson", marker=".", lw=0, ms=1)
        else:
            waves[var].plot(ax=ax, color="crimson")

        ax.grid(":", lw=0.5)
        ax.set_title(var)
        ax.set_xlim([waves.time.min(), waves.time.max()])
        ax.set_ylim([waves[var].min(), waves[var].max()])


def fit_plot_linear_model(
    X: np.ndarray,
    Ys: np.ndarray,
    keys: Optional[List[str]] = None,
    perc_train: float = 0.8,
) -> List[Any]:
    """
    Fit and plot linear regression models for multiple dependent variables.

    Parameters
    ----------
    X : np.ndarray
        The independent variable data.
    Ys : np.ndarray
        The dependent variable data.
    keys : list of str, optional
        List of keys/names for the dependent variables. If None, default names
        'Y0', 'Y1', etc. will be used.
    perc_train : float, optional
        The percentage of data to use for training. Default is 0.8.

    Returns
    -------
    list
        A list of fitted linear regression models (statsmodels OLS results objects).

    Notes
    -----
    This function creates scatter plots showing actual vs predicted values for
    the validation set, with density coloring using Gaussian KDE.
    """

    fig, axs = plt.subplots(1, Ys.shape[1], figsize=(7 * Ys.shape[1], 5))

    # Handle case where Ys.shape[1] = 1 (axs becomes a single axis, not an array)
    if Ys.shape[1] == 1:
        axs = [axs]

    models = []

    if keys is None:
        keys = [f"Y{i}" for i in range(Ys.shape[1])]

    for i in range(Ys.shape[1]):
        ax = axs[i]

        # Add constant column for intercept term
        X_with_const = sm.add_constant(X)
        y = Ys[:, i]

        X_train, X_val, y_train, y_val = train_test_split(
            X_with_const, y, test_size=1 - perc_train, random_state=42
        )

        # Create and fit the OLS model with the training data
        model = sm.OLS(y_train, X_train)
        results = model.fit()

        models.append(results)

        # Get predictions for the validation set
        y_val_pred = results.predict(X_val)

        # Compute density using Gaussian KDE
        xy = np.vstack([y_val, y_val_pred])
        density = gaussian_kde(xy)(xy)

        # Create the scatter plot for the validation set with density coloring
        sc = ax.scatter(y_val, y_val_pred, c=density, cmap="viridis", alpha=0.8, s=2)
        ax.plot(
            [min(y_val), max(y_val)],
            [min(y_val), max(y_val)],
            color="red",
            linestyle="--",
        )  # 45° reference line

        ax.set_xlabel(f"Actual Values ({keys[i]}) - Validation")
        ax.set_ylabel(f"Predicted Values ({keys[i]}) - Validation")
        ax.set_aspect("equal", "box")
        ax.set_title(f"{keys[i]}")

        # Add colorbar
        fig.colorbar(sc, ax=ax, label="Density", shrink=0.6)

    return models
