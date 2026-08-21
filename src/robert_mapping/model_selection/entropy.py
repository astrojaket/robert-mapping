"""Spatial entropy used by Hammond et al. (2024)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def spatial_entropy(pixels: ArrayLike, *, axis: int = -1) -> NDArray[np.float64]:
    """Return the Hammond map entropy relative to a uniform map.

    The returned value is zero for a uniform positive map and negative for a
    structured map. This matches the sign used by the legacy PyMC3 potential.
    """

    values = np.asarray(pixels, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("pixels must contain only finite values")
    if np.any(values <= 0.0):
        raise ValueError("pixels must be strictly positive")
    mean = np.mean(values, axis=axis, keepdims=True)
    return -np.sum(values * np.log(values / mean), axis=axis)


def entropy_log_weight(
    pixels: ArrayLike, alpha: float, *, axis: int = -1
) -> NDArray[np.float64]:
    """Return the log-density term ``2 * alpha * entropy``."""

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a finite value greater than or equal to zero")
    return 2.0 * alpha * spatial_entropy(pixels, axis=axis)
