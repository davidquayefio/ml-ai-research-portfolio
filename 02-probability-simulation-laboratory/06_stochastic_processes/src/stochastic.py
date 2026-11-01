"""Module 6: random walks, Markov chains and the Poisson process."""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Random walks
# ----------------------------------------------------------------------------
def simple_random_walk(n_steps: int, n_paths: int, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Paths of ``S_n = X_1 + ... + X_n`` with ``X_i = +1`` w.p. ``p`` else ``-1``.

    Returns an ``(n_paths, n_steps + 1)`` integer array with ``S_0 = 0``.
    """
    steps = np.where(rng.random((n_paths, n_steps)) < p, 1, -1)
    paths = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    np.cumsum(steps, axis=1, out=paths[:, 1:])
    return paths


def lattice_walk_2d(n_steps: int, n_paths: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric nearest-neighbour walk on Z^2 (E/W/N/S each with probability 1/4)."""
    d = rng.integers(0, 4, size=(n_paths, n_steps))
    dx = np.array([1, -1, 0, 0])[d]
    dy = np.array([0, 0, 1, -1])[d]
    x = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    y = np.zeros((n_paths, n_steps + 1), dtype=np.int64)
    np.cumsum(dx, axis=1, out=x[:, 1:])
    np.cumsum(dy, axis=1, out=y[:, 1:])
    return x, y


# ----------------------------------------------------------------------------
# Gambler's ruin: absorbing random walk on {0, ..., N}
# ----------------------------------------------------------------------------
def gamblers_ruin_simulate(
    start: int, target: int, p: float, n_paths: int, rng: np.random.Generator, max_steps: int = 10_000_000
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate until absorption at 0 or ``target``.

    Returns ``(reached_target, duration)`` arrays of length ``n_paths``.
    """
    pos = np.full(n_paths, start, dtype=np.int64)
    duration = np.zeros(n_paths, dtype=np.int64)
    active = np.ones(n_paths, dtype=bool)
    t = 0
    while active.any():
        t += 1
        if t > max_steps:
            raise RuntimeError("max_steps exceeded")
        idx = np.flatnonzero(active)
        pos[idx] += np.where(rng.random(idx.size) < p, 1, -1)
        duration[idx] += 1
        done = (pos[idx] == 0) | (pos[idx] == target)
        active[idx[done]] = False
    return pos == target, duration


def gamblers_ruin_win_probability(i: int, N: int, p: float) -> float:
    """``P_i(hit N before 0)``: ``i/N`` if fair, else ``(1 - r^i)/(1 - r^N)`` with ``r = q/p``."""
    q = 1.0 - p
    if abs(p - 0.5) < 1e-12:
        return i / N
    r = q / p
    return (1.0 - r**i) / (1.0 - r**N)


def gamblers_ruin_expected_duration(i: int, N: int, p: float) -> float:
    """``E_i[T]``: ``i (N - i)`` if fair, else ``i/(q-p) - (N/(q-p)) (1 - r^i)/(1 - r^N)``."""
    q = 1.0 - p
    if abs(p - 0.5) < 1e-12:
        return float(i * (N - i))
    r = q / p
    return i / (q - p) - (N / (q - p)) * (1.0 - r**i) / (1.0 - r**N)


def gamblers_ruin_matrix(N: int, p: float) -> np.ndarray:
    """Transition matrix of the gambler's-ruin chain on ``{0, ..., N}`` (0 and N absorbing)."""
    P = np.zeros((N + 1, N + 1))
    P[0, 0] = P[N, N] = 1.0
    for i in range(1, N):
        P[i, i + 1] = p
        P[i, i - 1] = 1.0 - p
    return P


def absorbing_chain_analysis(P: np.ndarray, absorbing) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Fundamental-matrix analysis of an absorbing Markov chain.

    With transient block ``Q`` and transient-to-absorbing block ``R``,
    ``F = (I - Q)^{-1}`` gives expected visits; ``F 1`` is the expected time to
    absorption and ``F R`` the absorption probabilities.

    Returns ``(transient_states, expected_steps, absorption_probabilities)``.
    """
    absorbing = list(absorbing)
    transient = [s for s in range(P.shape[0]) if s not in absorbing]
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]
    F = np.linalg.inv(np.eye(len(transient)) - Q)
    return transient, F.sum(axis=1), F @ R


# ----------------------------------------------------------------------------
# Finite Markov chains
# ----------------------------------------------------------------------------
def simulate_chain(P: np.ndarray, x0: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """One path ``X_0, ..., X_{n_steps}`` of a finite Markov chain with transition matrix ``P``.

    Inverse-transform sampling from each row: next state is the first index whose
    cumulative probability exceeds a uniform draw.
    """
    P = np.asarray(P, dtype=float)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0  # guard against floating-point round-off
    u = rng.random(n_steps)
    path = np.empty(n_steps + 1, dtype=np.int64)
    path[0] = x0
    x = x0
    for t in range(n_steps):
        x = int(np.searchsorted(cum[x], u[t], side="right"))
        path[t + 1] = x
    return path


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Solve ``pi P = pi``, ``sum(pi) = 1`` (least squares on the stacked linear system)."""
    k = P.shape[0]
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def n_step_distribution(P: np.ndarray, x0: int, n: int) -> np.ndarray:
    """Distribution of ``X_n`` given ``X_0 = x0``: row ``x0`` of ``P^n``."""
    return np.linalg.matrix_power(P, n)[x0]


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """``0.5 * sum |p_i - q_i|``."""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def empirical_distribution(path: np.ndarray, k: int) -> np.ndarray:
    """Fraction of time a path spends in each of the ``k`` states."""
    return np.bincount(path, minlength=k) / path.size


def return_times(path: np.ndarray, state: int) -> np.ndarray:
    """Gaps between successive visits to ``state``."""
    return np.diff(np.flatnonzero(path == state))


# ----------------------------------------------------------------------------
# Poisson process
# ----------------------------------------------------------------------------
def poisson_process_arrivals(rate: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """Arrival times in ``[0, T]`` of a Poisson process: cumulative Exp(rate) inter-arrival times."""
    arrivals: list[np.ndarray] = []
    t = 0.0
    batch = int(rate * T + 6 * np.sqrt(rate * T) + 10)
    while t <= T:
        times = t + np.cumsum(rng.exponential(1.0 / rate, size=batch))
        arrivals.append(times)
        t = times[-1]
    all_times = np.concatenate(arrivals)
    return all_times[all_times <= T]
