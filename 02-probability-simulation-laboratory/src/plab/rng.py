"""Reproducible random number generation.

We use NumPy's modern ``Generator`` API (PCG64 bit generator) instead of the
legacy ``np.random.seed`` global state. A ``Generator`` object is passed
explicitly to every simulation function, so results never depend on hidden
global state and every experiment can be reproduced from its seed.
"""

from __future__ import annotations

import numpy as np

#: Default seed used across the laboratory (the date the project started).
DEFAULT_SEED: int = 20261031


def make_rng(seed: int | None = DEFAULT_SEED) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` seeded with ``seed``.

    Parameters
    ----------
    seed : int or None
        Seed for the PCG64 bit generator. ``None`` gives a non-reproducible
        generator seeded from OS entropy.
    """
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` statistically independent generators derived from ``seed``.

    Uses ``SeedSequence.spawn`` which guarantees non-overlapping streams. This
    is the recommended way to give independent randomness to independent
    experiments (or parallel workers).
    """
    children = np.random.SeedSequence(seed).spawn(n)
    return [np.random.default_rng(child) for child in children]
