"""Shared figure style so every Task 1 figure reads as one set."""
import matplotlib

matplotlib.use("Agg")                  # figures are written to disk by main.py, never shown
import matplotlib.pyplot as plt

# Fixed categorical order: the same entity always gets the same colour.
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "grey": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e",
}
# Model/condition colours used across all B2 figures.
MODEL_COLORS = {
    "EEGNet": C["blue"], "TF-time": C["orange"], "TF-chan-id": C["aqua"],
    "TF-chan-noid": C["violet"], "TF-chan-id-mirror": C["magenta"],
}
# Diverging map for power change: blue = desynchronisation, red = synchronisation,
# white midpoint = no change.
DIVERGING = "RdBu_r"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
        "lines.linewidth": 2, "legend.frameon": False,
    })
