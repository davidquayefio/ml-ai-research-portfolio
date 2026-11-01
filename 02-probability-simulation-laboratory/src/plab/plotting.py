"""Consistent plotting style and figure saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def set_style() -> None:
    """Apply the laboratory-wide plotting style."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def save_fig(fig, directory, name: str) -> Path:
    """Save ``fig`` as ``directory/name`` (creating the directory) and return the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path)
    return path
