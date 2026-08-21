"""Stable predictive scoring and structured eclipse folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp


@dataclass(frozen=True)
class CVComparison:
    """Pointwise comparison of two predictive models."""

    delta_elpd: float
    standard_error: float
    z_score: float
    pointwise_delta: NDArray[np.float64]


def gaussian_pointwise_elpd(
    observed: ArrayLike,
    predicted_samples: ArrayLike,
    sigma: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate pointwise expected log predictive density.

    ``predicted_samples`` must have shape ``(draw, observation)``. The
    log-mean-exp calculation avoids the underflow in the legacy implementation.
    """

    y = np.asarray(observed, dtype=float)
    mu = np.asarray(predicted_samples, dtype=float)
    error = np.asarray(sigma, dtype=float)
    if y.ndim != 1:
        raise ValueError("observed must be one-dimensional")
    if mu.ndim != 2 or mu.shape[1] != y.size:
        raise ValueError("predicted_samples must have shape (draw, observation)")
    if error.ndim == 0:
        error = np.full_like(y, float(error))
    if error.shape != y.shape:
        raise ValueError("sigma must be scalar or have the same shape as observed")
    if np.any(~np.isfinite(error)) or np.any(error <= 0.0):
        raise ValueError("sigma must contain finite positive values")

    log_likelihood = (
        -0.5 * ((y[None, :] - mu) / error[None, :]) ** 2
        - np.log(error[None, :])
        - 0.5 * np.log(2.0 * np.pi)
    )
    return logsumexp(log_likelihood, axis=0) - np.log(mu.shape[0])


def compare_pointwise_elpd(
    candidate: ArrayLike, reference: ArrayLike
) -> CVComparison:
    """Compare candidate and reference pointwise ELPD values."""

    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if candidate_values.shape != reference_values.shape or candidate_values.ndim != 1:
        raise ValueError("candidate and reference must be one-dimensional and have equal shape")
    delta = candidate_values - reference_values
    total = float(np.sum(delta))
    if delta.size < 2:
        standard_error = 0.0
    else:
        standard_error = float(np.sqrt(delta.size * np.var(delta, ddof=1)))
    if standard_error == 0.0:
        z_score = float(np.sign(total) * np.inf) if total != 0.0 else 0.0
    else:
        z_score = total / standard_error
    return CVComparison(total, standard_error, z_score, delta)


def make_eclipse_folds(
    time: ArrayLike,
    intervals: ArrayLike,
    *,
    blocks_per_interval: int = 10,
) -> tuple[NDArray[np.int64], ...]:
    """Split eclipse intervals into contiguous validation blocks.

    Parameters
    ----------
    intervals
        Array with shape ``(event, 2)``. Each row gives the start and end time
        of an ingress or egress interval. Intervals can cover many eclipses.
    """

    t = np.asarray(time, dtype=float)
    bounds = np.asarray(intervals, dtype=float)
    if t.ndim != 1 or np.any(np.diff(t) < 0.0):
        raise ValueError("time must be a sorted one-dimensional array")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("intervals must have shape (event, 2)")
    if blocks_per_interval < 1:
        raise ValueError("blocks_per_interval must be at least one")

    folds: list[NDArray[np.int64]] = []
    for start, stop in bounds:
        if not np.isfinite(start + stop) or stop <= start:
            raise ValueError("each interval must have finite start < stop")
        indices = np.flatnonzero((t >= start) & (t <= stop))
        if indices.size == 0:
            continue
        for block in np.array_split(indices, min(blocks_per_interval, indices.size)):
            if block.size:
                folds.append(block.astype(np.int64, copy=False))
    if not folds:
        raise ValueError("no data points fall inside the supplied eclipse intervals")
    return tuple(folds)
