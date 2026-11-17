"""
Utilities of the AWT analysis. Importing the package sets the common figure style:
larger fonts and a purple / green / orange palette (no red-blue).
"""

import matplotlib as mpl
from cycler import cycler

# main colours
PURPLE = "#6A3D9A"
GREEN = "#1B9E77"
ORANGE = "#E66101"
GOLD = "#D9A441"
BROWN = "#8C510A"
LIGHT_PURPLE = "#B2ABD2"

# approaches
APPROACH_COLORS = {"PCA": PURPLE, "RPCA": GREEN, "RPCA sel": ORANGE}

# colormaps
CMAP_DIV = "PuOr_r"      # diverging (correlations, SST patterns): purple (-) / orange (+)
CMAP_DIV2 = "PRGn"       # diverging (MSL patterns): purple (-) / green (+)
CMAP_SEQ = "magma"       # sequential (skill, maps of means)

mpl.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "figure.titlesize": 19,
    "axes.prop_cycle": cycler(color=[PURPLE, GREEN, ORANGE, GOLD, BROWN, LIGHT_PURPLE,
                                     "#66A61E", "#5E3C99", "#FDB863", "#35978F"]),
})
