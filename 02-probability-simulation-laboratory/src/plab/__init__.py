"""plab: shared utilities for the Probability Simulation Laboratory.

Modules
-------
rng        reproducible random number generators
stats      small statistical helpers (standard errors, intervals, ECDF)
plotting   consistent plotting style and figure saving
processes  Brownian motion and geometric Brownian motion simulators
"""

from .rng import DEFAULT_SEED, make_rng, spawn_rngs
from .stats import ecdf, mean_ci, standard_error, wilson_interval
from .plotting import save_fig, set_style

__all__ = [
    "DEFAULT_SEED",
    "make_rng",
    "spawn_rngs",
    "ecdf",
    "mean_ci",
    "standard_error",
    "wilson_interval",
    "save_fig",
    "set_style",
]
