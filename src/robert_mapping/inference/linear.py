"""Small exact Gaussian solver used for tests and fast comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LinearGaussianPosterior:
    """Posterior for a Gaussian linear model."""

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]

    def predict(self, design_matrix: ArrayLike) -> NDArray[np.float64]:
        design = np.asarray(design_matrix, dtype=float)
        return design @ self.mean

    def draw(self, draws: int, *, seed: int = 0) -> NDArray[np.float64]:
        if draws < 1:
            raise ValueError("draws must be at least one")
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.mean, self.covariance, size=draws)


def fit_linear_gaussian(
    design_matrix: ArrayLike,
    observed: ArrayLike,
    sigma: ArrayLike,
    *,
    prior_mean: ArrayLike | None = None,
    prior_scale: ArrayLike | float = 1.0e6,
) -> LinearGaussianPosterior:
    """Fit a Gaussian linear model with an independent normal prior."""

    design = np.asarray(design_matrix, dtype=float)
    y = np.asarray(observed, dtype=float)
    error = np.asarray(sigma, dtype=float)
    if design.ndim != 2 or y.ndim != 1 or design.shape[0] != y.size:
        raise ValueError("design_matrix must have shape (observation, parameter)")
    if error.ndim == 0:
        error = np.full_like(y, float(error))
    if error.shape != y.shape or np.any(error <= 0.0):
        raise ValueError("sigma must be scalar or a positive value per observation")

    parameter_count = design.shape[1]
    mean0 = (
        np.zeros(parameter_count)
        if prior_mean is None
        else np.asarray(prior_mean, dtype=float)
    )
    scale0 = np.broadcast_to(np.asarray(prior_scale, dtype=float), (parameter_count,))
    if mean0.shape != (parameter_count,) or np.any(scale0 <= 0.0):
        raise ValueError("prior parameters must match the design matrix columns")

    weighted_design = design / error[:, None]
    weighted_y = y / error
    prior_precision = 1.0 / scale0**2
    precision = weighted_design.T @ weighted_design + np.diag(prior_precision)
    right_hand_side = weighted_design.T @ weighted_y + prior_precision * mean0
    covariance = np.linalg.inv(precision)
    mean = np.linalg.solve(precision, right_hand_side)
    return LinearGaussianPosterior(mean=mean, covariance=covariance)
