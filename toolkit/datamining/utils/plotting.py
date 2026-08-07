import itertools
from typing import List, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Improve readability in every figure generated with these plotting utilities.
plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
)


def plot_variable_combinations(
    data: pd.DataFrame,
    vars: List[str],
    sel_datasets: Union[list, pd.DataFrame] = None,
    data_color: str = None,
    sel_color: str = None,
    labels: List[str] = None,
    size_point: int = 5,
) -> None:
    """
    Plots scatter plots of variable combinations.

    Parameters
    ----------
    data: DataFrame
        The data containing the variables.
    vars: list
        The names of the variables to plot.
    sel_datasets: list or DataFrame, optional
        The selected datasets to highlight on the scatter plots.
    data_color: str, optional
        The color of the data points.
    sel_color: str, optional
        The color of the selected datasets.
    labels: list, optional
        The labels for the selected datasets.
    size_point: int, optional
        The size of the points in the scatter plots.
    """

    num_vars = len(vars)
    combinations = list(itertools.combinations(range(num_vars), 2))

    fig = plt.figure(figsize=[12, 10], tight_layout=True)
    gs = gridspec.GridSpec(num_vars - 1, num_vars - 1)

    axes = {}

    for idx, (i, j) in enumerate(combinations):
        row = idx // (num_vars - 1)
        col = idx % (num_vars - 1)
        ax = fig.add_subplot(gs[row, col])
        axes[(i, j)] = ax

        v1, v1_label = data[vars[i]].values, vars[i]
        v2, v2_label = data[vars[j]].values, vars[j]

        if data_color is None:
            data_color = "k"

        ax.scatter(v1, v2, c=data_color, s=size_point, cmap="rainbow", alpha=0.2)
        ax.set_xlabel(v1_label, fontsize=plt.rcParams["axes.labelsize"])
        ax.set_ylabel(v2_label, fontsize=plt.rcParams["axes.labelsize"])
        ax.tick_params(
            axis="both",
            labelsize=plt.rcParams["xtick.labelsize"],
        )
        ax.grid(":", color="plum", linewidth=0.3)

        # Selected points
        if sel_datasets is not None:
            if isinstance(sel_datasets, list):
                color_list = [
                    "crimson",
                    "royalblue",
                    "lime",
                    "gold",
                    "purple",
                    "teal",
                    "orange",
                    "indigo",
                    "maroon",
                    "aqua",
                ]
                for ic, sel in enumerate(sel_datasets):
                    if labels is None:
                        label = "List " + str(ic)
                    else:
                        label = labels[ic]

                    if sel_color is None:
                        color_dataset = color_list[ic]
                    else:
                        color_dataset = sel_color
                    im = ax.scatter(
                        sel[vars[i]],
                        sel[vars[j]],
                        s=70,
                        c=color_dataset,
                        alpha=1,
                        zorder=2,
                        label=label,
                    )
            else:
                if labels is None:
                    labels = "List"
                if sel_color is None:
                    sel_color = range(len(sel_datasets))

                im = ax.scatter(
                    sel_datasets[vars[i]],
                    sel_datasets[vars[j]],
                    s=70,
                    c=sel_color,
                    ec="white",
                    alpha=1,
                    zorder=2,
                    cmap="rainbow",
                    label=labels,
                )
                plt.colorbar(im, ax=ax)

        ax.legend(loc="upper right", fontsize=plt.rcParams["legend.fontsize"])

    plt.show()


def plot_rbf_result(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    exact: np.ndarray,
    sample: pd.DataFrame,
    predicted: np.ndarray,
    title: str,
) -> None:
    """Compare an exact surface, its RBF reconstruction, and their error.

    Parameters
    ----------
    x_grid, y_grid : np.ndarray
        Two-dimensional coordinate grids shared by the exact and predicted surfaces.
    exact : np.ndarray
        Exact theoretical values with the same shape as the coordinate grids.
    sample : pd.DataFrame
        Training samples with ``x``, ``y``, and ``z`` columns.
    predicted : np.ndarray
        Flattened RBF predictions in the same order as the coordinate grids.
    title : str
        Name used in the exact-surface panel title.
    """

    predicted_grid = predicted.reshape(x_grid.shape)
    error = predicted_grid - exact

    fig = plt.figure(figsize=(16, 6))
    ax_exact = fig.add_subplot(131, projection="3d")
    ax_rbf = fig.add_subplot(132, projection="3d")
    ax_error = fig.add_subplot(133)

    ax_exact.plot_surface(x_grid, y_grid, exact, cmap="rainbow", alpha=0.75)
    ax_exact.scatter(
        sample["x"],
        sample["y"],
        sample["z"],
        c=sample["z"],
        cmap="rainbow",
        edgecolor="black",
        s=55,
    )
    ax_exact.set_title(f"{title}: exact surface")

    ax_rbf.plot_surface(
        x_grid,
        y_grid,
        predicted_grid,
        cmap="rainbow",
        alpha=0.85,
    )
    ax_rbf.set_title("RBF reconstruction")

    image = ax_error.pcolormesh(
        x_grid,
        y_grid,
        error,
        shading="auto",
        cmap="RdBu_r",
    )
    fig.colorbar(image, ax=ax_error, label="RBF − exact")
    ax_error.set_title("Interpolation error")

    for axis in (ax_exact, ax_rbf):
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    ax_error.set_xlabel("x")
    ax_error.set_ylabel("y")

    plt.tight_layout()
    plt.show()
