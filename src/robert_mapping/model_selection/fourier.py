"""Fourier null model used in the Hammond eclipse-map test."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def fourier_design_matrix(
    time: ArrayLike,
    *,
    period: float,
    t0: float,
    visibility: ArrayLike | None = None,
    degree: int = 2,
    include_intercept: bool = True,
) -> NDArray[np.float64]:
    """Build a uniform-disc Fourier phase-curve design matrix.

    The planet terms are multiplied by ``visibility``. This gives the uniform
    secondary-eclipse shape while the sinusoidal terms describe the phase
    variation outside eclipse.
    """

    t = np.asarray(time, dtype=float)
    if t.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be greater than zero")
    if degree < 0:
        raise ValueError("degree must be greater than or equal to zero")
    if visibility is None:
        visible = np.ones_like(t)
    else:
        visible = np.asarray(visibility, dtype=float)
        if visible.shape != t.shape:
            raise ValueError("visibility must have the same shape as time")
        if np.any((visible < 0.0) | (visible > 1.0)):
            raise ValueError("visibility must be between zero and one")

    phase = 2.0 * np.pi * (t - t0) / period
    columns: list[NDArray[np.float64]] = []
    if include_intercept:
        columns.append(np.ones_like(t))
    columns.append(visible)
    for order in range(1, degree + 1):
        columns.append(visible * np.sin(order * phase))
        columns.append(visible * np.cos(order * phase))
    return np.column_stack(columns)
