"""Fast linear recovery-grid fits.

This module contains the small, deterministic part of a recovery analysis.  A
candidate grid usually represents different longitudes (or other fixed model
choices).  For each candidate we profile a Gaussian or generalized least
squares model with unbounded nuisance terms and up to two non-negative map
amplitudes.  The constrained solve is exact for this small problem: every
active set of map amplitudes is enumerated and solved with least squares.

The module is deliberately independent of the eclipse-mapping physics layer.
It only needs NumPy and SciPy, so it is also useful in a minimal Conda
environment or in a fast CPU benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, solve_triangular
from scipy.special import logsumexp


FloatArray = NDArray[np.float64]
IndexTuple = tuple[int, ...]


@dataclass(frozen=True)
class ProfileFit:
    """Profile solution for one candidate design matrix.

    ``coefficients`` has one value for every input design-matrix column.  A
    map column that is not in ``active_map_columns`` is fixed to zero.  The
    covariance is the local covariance of the free coefficients; rows and
    columns for fixed map amplitudes are zero.
    """

    coefficients: FloatArray
    fitted: FloatArray
    residuals: FloatArray
    covariance: FloatArray
    log_likelihood: float
    bic: float
    chi2: float
    active_map_columns: IndexTuple
    nuisance_columns: IndexTuple
    parameter_count: int

    @property
    def active_set(self) -> IndexTuple:
        """Alias for the globally indexed active map columns."""

        return self.active_map_columns

    @property
    def map_coefficients(self) -> FloatArray:
        """Return the fitted map amplitudes, including inactive zeros."""

        map_columns = tuple(
            index
            for index in range(self.coefficients.size)
            if index not in self.nuisance_columns
        )
        return self.coefficients[list(map_columns)]

    @property
    def profile_log_likelihood(self) -> float:
        """Long-form alias for :attr:`log_likelihood`."""

        return self.log_likelihood


@dataclass(frozen=True)
class CandidateGridResult:
    """Results from profiling all candidates in a recovery grid.

    Posterior weights are BIC weights, ``exp(-0.5 * BIC)``, normalised with a
    log-sum-exp calculation.  This gives a transparent, sampler-free measure
    of relative support for each candidate.
    """

    longitudes: FloatArray
    fits: tuple[ProfileFit, ...]
    log_likelihood: FloatArray
    bic: FloatArray
    posterior_weights: FloatArray
    longitude_q16: float
    longitude_median: float
    longitude_q84: float
    best_index: int
    longitude_period: float | None = None

    @property
    def weights(self) -> FloatArray:
        """Short alias for :attr:`posterior_weights`."""

        return self.posterior_weights

    @property
    def normalized_posterior_weights(self) -> FloatArray:
        """Long-form alias for :attr:`posterior_weights`."""

        return self.posterior_weights

    @property
    def profile_log_likelihood(self) -> FloatArray:
        """Long-form alias for the per-candidate log likelihood."""

        return self.log_likelihood

    @property
    def bic_values(self) -> FloatArray:
        """Long-form alias for the per-candidate BIC values."""

        return self.bic

    @property
    def q16(self) -> float:
        """Short alias for the weighted 16th-percentile longitude."""

        return self.longitude_q16

    @property
    def median(self) -> float:
        """Short alias for the weighted median longitude."""

        return self.longitude_median

    @property
    def q84(self) -> float:
        """Short alias for the weighted 84th-percentile longitude."""

        return self.longitude_q84

    @property
    def longitude_quantiles(self) -> tuple[float, float, float]:
        """Return the 16th, 50th, and 84th percentile longitude."""

        return (self.longitude_q16, self.longitude_median, self.longitude_q84)

    @property
    def best_fit(self) -> ProfileFit:
        """Return the candidate with the smallest BIC."""

        return self.fits[self.best_index]


@dataclass(frozen=True)
class BICComparison:
    """Comparison of a flexible model against a uniform model.

    The convention is ``delta_bic = flexible_bic - uniform_bic``.  Therefore,
    a negative value favours the flexible model.
    """

    uniform_bic: float
    flexible_bic: float
    delta_bic: float

    @property
    def flexible_preferred(self) -> bool:
        """Whether the flexible model has the smaller BIC."""

        return self.delta_bic < 0.0

    @property
    def bic_delta(self) -> float:
        """Alias for :attr:`delta_bic`."""

        return self.delta_bic


@dataclass(frozen=True)
class _Whitening:
    """Internal whitening operator and the Gaussian normalisation term."""

    design: FloatArray
    observed: FloatArray
    log_determinant: float


def _as_float_vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _validate_design_and_observed(
    design_matrix: ArrayLike, observed: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    design = np.asarray(design_matrix, dtype=float)
    y = _as_float_vector(observed, "observed")
    if design.ndim != 2:
        raise ValueError("design_matrix must be two-dimensional")
    if design.shape[0] != y.size:
        raise ValueError("design_matrix rows must match observed")
    if design.shape[1] == 0:
        raise ValueError("design_matrix must contain at least one column")
    if not np.all(np.isfinite(design)):
        raise ValueError("design_matrix must contain only finite values")
    return np.asarray(design, dtype=float), y


def _validate_columns(
    n_columns: int,
    map_columns: Sequence[int] | None,
    nuisance_columns: Sequence[int] | None,
    n_nuisance: int | None,
) -> tuple[IndexTuple, IndexTuple]:
    def normalise(
        columns: Sequence[int] | None, name: str
    ) -> IndexTuple | None:
        if columns is None:
            return None
        try:
            values = tuple(columns)
        except TypeError as error:
            raise ValueError(f"{name} must be a sequence of integer indexes") from error
        normalised: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must contain only integer indexes")
            index = int(value)
            if index < 0 or index >= n_columns:
                raise ValueError(f"{name} indexes must refer to design_matrix columns")
            normalised.append(index)
        return tuple(normalised)

    if n_nuisance is not None:
        if nuisance_columns is not None:
            raise ValueError("use either n_nuisance or nuisance_columns, not both")
        if isinstance(n_nuisance, bool) or not isinstance(n_nuisance, Integral):
            raise ValueError("n_nuisance must be an integer")
        if n_nuisance < 0 or n_nuisance > n_columns:
            raise ValueError("n_nuisance must be between zero and the column count")
        nuisance_columns = tuple(range(int(n_nuisance)))

    map_columns = normalise(map_columns, "map_columns")
    nuisance_columns = normalise(nuisance_columns, "nuisance_columns")

    if map_columns is None and nuisance_columns is None:
        # A common recovery design is [nuisance..., map].  Keep this useful
        # default while still allowing an explicit empty map set below.
        map_columns = (n_columns - 1,)
        nuisance_columns = tuple(range(n_columns - 1))
    elif map_columns is None:
        nuisance_columns = tuple(nuisance_columns or ())
        map_columns = tuple(index for index in range(n_columns) if index not in nuisance_columns)
    elif nuisance_columns is None:
        map_columns = tuple(map_columns)
        nuisance_columns = tuple(index for index in range(n_columns) if index not in map_columns)
    else:
        map_columns = tuple(map_columns)
        nuisance_columns = tuple(nuisance_columns)

    if len(map_columns) > 2:
        raise ValueError("at most two non-negative map columns are supported")
    if len(set(map_columns)) != len(map_columns):
        raise ValueError("map_columns must not contain duplicates")
    if len(set(nuisance_columns)) != len(nuisance_columns):
        raise ValueError("nuisance_columns must not contain duplicates")
    all_columns = set(range(n_columns))
    map_set = set(map_columns)
    nuisance_set = set(nuisance_columns)
    if not map_set.issubset(all_columns) or not nuisance_set.issubset(all_columns):
        raise ValueError("column indexes must refer to design_matrix columns")
    if map_set & nuisance_set:
        raise ValueError("map_columns and nuisance_columns must not overlap")
    if map_set | nuisance_set != all_columns:
        raise ValueError("map_columns and nuisance_columns must cover every design column")
    return tuple(map_columns), tuple(nuisance_columns)


def _whiten(
    design: FloatArray,
    observed: FloatArray,
    sigma: ArrayLike | None,
    covariance: ArrayLike | None,
) -> _Whitening:
    if sigma is not None and covariance is not None:
        raise ValueError("provide either sigma or covariance, not both")
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim != 2 or cov.shape != (observed.size, observed.size):
            raise ValueError("covariance must have shape (observation, observation)")
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariance must contain only finite values")
        if not np.allclose(cov, cov.T, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("covariance must be symmetric")
        try:
            factor, lower = cho_factor(cov, lower=True, check_finite=True)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error
        whitened_design = solve_triangular(
            factor, design, lower=lower, check_finite=True
        )
        whitened_observed = solve_triangular(
            factor, observed, lower=lower, check_finite=True
        )
        log_determinant = float(2.0 * np.sum(np.log(np.diag(factor))))
        return _Whitening(whitened_design, whitened_observed, log_determinant)

    if sigma is None:
        error = np.ones(observed.size, dtype=float)
    else:
        error = np.asarray(sigma, dtype=float)
        if error.ndim == 0:
            error = np.full(observed.size, float(error))
        elif error.ndim != 1 or error.shape != observed.shape:
            raise ValueError("sigma must be scalar or have shape (observation,)")
    if not np.all(np.isfinite(error)) or np.any(error <= 0.0):
        raise ValueError("sigma must contain finite positive values")
    return _Whitening(
        design / error[:, None],
        observed / error,
        float(2.0 * np.sum(np.log(error))),
    )


def _active_sets(map_columns: IndexTuple) -> Iterable[IndexTuple]:
    """Yield every active map subset, starting with the empty subset."""

    for size in range(len(map_columns) + 1):
        yield from combinations(map_columns, size)


def _profile_active_set(
    whitening: _Whitening,
    n_columns: int,
    nuisance_columns: IndexTuple,
    active_map_columns: IndexTuple,
    *,
    constraint_tolerance: float,
) -> ProfileFit | None:
    free_columns = nuisance_columns + active_map_columns
    if free_columns:
        free_design = whitening.design[:, free_columns]
        free_solution, _, _, _ = np.linalg.lstsq(
            free_design, whitening.observed, rcond=None
        )
    else:
        free_solution = np.empty(0, dtype=float)

    active_start = len(nuisance_columns)
    active_solution = free_solution[active_start:]
    scale = max(1.0, float(np.max(np.abs(active_solution), initial=0.0)))
    if np.any(active_solution < -constraint_tolerance * scale):
        return None
    active_solution = np.maximum(active_solution, 0.0)

    coefficients = np.zeros(n_columns, dtype=float)
    if nuisance_columns:
        coefficients[list(nuisance_columns)] = free_solution[:active_start]
    if active_map_columns:
        coefficients[list(active_map_columns)] = active_solution

    fitted = whitening.design @ coefficients
    whitened_residuals = whitening.observed - fitted
    chi2 = float(np.dot(whitened_residuals, whitened_residuals))
    n_observations = whitening.observed.size
    log_likelihood = float(
        -0.5
        * (chi2 + n_observations * np.log(2.0 * np.pi) + whitening.log_determinant)
    )

    covariance = np.zeros((n_columns, n_columns), dtype=float)
    if free_columns:
        normal_matrix = whitening.design[:, free_columns].T @ whitening.design[:, free_columns]
        free_covariance = np.linalg.pinv(normal_matrix, hermitian=True)
        covariance[np.ix_(free_columns, free_columns)] = free_covariance

    # The caller unwhitens the fitted and residual values.  Keeping these
    # temporary values out of this internal function avoids duplicating the
    # covariance handling in every active-set branch.
    bic = float(len(free_columns) * np.log(n_observations) - 2.0 * log_likelihood)
    return ProfileFit(
        coefficients=coefficients,
        fitted=fitted,
        residuals=whitened_residuals,
        covariance=covariance,
        log_likelihood=log_likelihood,
        bic=bic,
        chi2=chi2,
        active_map_columns=active_map_columns,
        nuisance_columns=nuisance_columns,
        parameter_count=len(free_columns),
    )


def fit_profile(
    design_matrix: ArrayLike,
    observed: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    covariance: ArrayLike | None = None,
    map_columns: Sequence[int] | None = None,
    nuisance_columns: Sequence[int] | None = None,
    n_nuisance: int | None = None,
    constraint_tolerance: float = 1.0e-12,
) -> ProfileFit:
    """Fit one Gaussian/GLS profile with exact non-negative active sets.

    Parameters
    ----------
    design_matrix
        Matrix with shape ``(observation, parameter)``.
    observed
        Data vector with shape ``(observation,)``.
    sigma
        Scalar or diagonal standard deviation.  If omitted, unit errors are
        used.  Do not provide this together with ``covariance``.
    covariance
        Optional dense covariance matrix.  It is whitened with a SciPy
        Cholesky factor before solving.
    map_columns
        Zero, one, or two columns whose fitted amplitudes must be non-negative.
        If omitted, the last column is treated as the map column and all prior
        columns are nuisance columns.
    nuisance_columns
        Columns with unbounded coefficients.  The two column sets must cover
        the design matrix without overlap.
    n_nuisance
        Convenience form for selecting the first ``n_nuisance`` columns as
        nuisance columns.  It cannot be combined with ``nuisance_columns``.
    constraint_tolerance
        Relative tolerance for accepting a small negative active coefficient
        from floating-point least squares.
    """

    if not np.isfinite(constraint_tolerance) or constraint_tolerance < 0.0:
        raise ValueError("constraint_tolerance must be finite and non-negative")
    design, y = _validate_design_and_observed(design_matrix, observed)
    map_set, nuisance_set = _validate_columns(
        design.shape[1], map_columns, nuisance_columns, n_nuisance
    )
    whitening = _whiten(design, y, sigma, covariance)

    best: ProfileFit | None = None
    for active in _active_sets(map_set):
        candidate = _profile_active_set(
            whitening,
            design.shape[1],
            nuisance_set,
            tuple(active),
            constraint_tolerance=constraint_tolerance,
        )
        if candidate is None:
            continue
        if best is None:
            best = candidate
            continue
        score_difference = candidate.log_likelihood - best.log_likelihood
        if score_difference > 1.0e-12 or (
            abs(score_difference) <= 1.0e-12
            and len(candidate.active_map_columns) < len(best.active_map_columns)
        ):
            best = candidate
    if best is None:  # The empty active set should always be feasible.
        raise RuntimeError("no feasible active set was found")

    # The fit and residuals are returned in data units.  Coefficients already
    # live in data units because whitening changes rows, not columns.
    unwhitened_fitted = design @ best.coefficients
    unwhitened_residuals = y - unwhitened_fitted
    return ProfileFit(
        coefficients=best.coefficients.copy(),
        fitted=unwhitened_fitted,
        residuals=unwhitened_residuals,
        covariance=best.covariance.copy(),
        log_likelihood=best.log_likelihood,
        bic=best.bic,
        chi2=best.chi2,
        active_map_columns=best.active_map_columns,
        nuisance_columns=best.nuisance_columns,
        parameter_count=best.parameter_count,
    )


def _weighted_quantile(values: FloatArray, weights: FloatArray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    centre_cdf = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    return float(np.interp(quantile, centre_cdf, ordered_values))


def _periodic_values_for_quantiles(
    values: FloatArray, weights: FloatArray, period: float | None
) -> FloatArray:
    if period is None:
        return values
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("longitude_period must be finite and positive")
    angle = 2.0 * np.pi * values / period
    resultant = np.sum(weights * np.exp(1j * angle))
    if abs(resultant) < 1.0e-12:
        anchor = float(values[np.argmax(weights)])
    else:
        anchor = float(np.angle(resultant) * period / (2.0 * np.pi))
    return anchor + (values - anchor + 0.5 * period) % period - 0.5 * period


def fit_candidate_grid(
    longitudes: ArrayLike,
    design_matrices: ArrayLike | Sequence[ArrayLike],
    observed: ArrayLike,
    sigma: ArrayLike | None = None,
    *,
    covariance: ArrayLike | None = None,
    map_columns: Sequence[int] | None = None,
    nuisance_columns: Sequence[int] | None = None,
    n_nuisance: int | None = None,
    constraint_tolerance: float = 1.0e-12,
    longitude_period: float | None = None,
) -> CandidateGridResult:
    """Profile every candidate in a longitude recovery grid.

    ``design_matrices`` may be a single matrix with shape ``(observation,
    parameter)`` or a stack with shape ``(candidate, observation, parameter)``.
    A sequence of two-dimensional matrices is also accepted.  Posterior
    weights are normalised BIC weights, and longitude quantiles use those
    weights.  Set ``longitude_period`` to obtain circular quantiles; leave it
    unset for ordinary linear longitude coordinates.
    """

    grid = _as_float_vector(longitudes, "longitudes")
    try:
        matrices = np.asarray(design_matrices, dtype=float)
    except (TypeError, ValueError):
        try:
            matrices = np.asarray(
                [np.asarray(matrix, dtype=float) for matrix in design_matrices],
                dtype=float,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "design_matrices must be a matrix, a 3D stack, or a sequence of matrices"
            ) from error
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    elif matrices.ndim != 3:
        try:
            matrices = np.asarray(
                [np.asarray(matrix, dtype=float) for matrix in design_matrices],
                dtype=float,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "design_matrices must be a matrix, a 3D stack, or a sequence of matrices"
            ) from error
    if matrices.ndim != 3:
        raise ValueError("design_matrices must have shape (candidate, observation, parameter)")
    if matrices.shape[0] != grid.size:
        raise ValueError("longitudes must match the number of candidate design matrices")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("design_matrices must contain only finite values")

    fits = tuple(
        fit_profile(
            matrix,
            observed,
            sigma,
            covariance=covariance,
            map_columns=map_columns,
            nuisance_columns=nuisance_columns,
            n_nuisance=n_nuisance,
            constraint_tolerance=constraint_tolerance,
        )
        for matrix in matrices
    )
    log_likelihood = np.asarray([fit.log_likelihood for fit in fits], dtype=float)
    bic = np.asarray([fit.bic for fit in fits], dtype=float)
    log_weights = -0.5 * bic
    posterior_weights = np.exp(log_weights - logsumexp(log_weights))
    quantile_values = _periodic_values_for_quantiles(
        grid, posterior_weights, longitude_period
    )
    q16 = _weighted_quantile(quantile_values, posterior_weights, 0.16)
    median = _weighted_quantile(quantile_values, posterior_weights, 0.50)
    q84 = _weighted_quantile(quantile_values, posterior_weights, 0.84)
    return CandidateGridResult(
        longitudes=grid.copy(),
        fits=fits,
        log_likelihood=log_likelihood,
        bic=bic,
        posterior_weights=posterior_weights,
        longitude_q16=q16,
        longitude_median=median,
        longitude_q84=q84,
        best_index=int(np.argmin(bic)),
        longitude_period=longitude_period,
    )


def compare_uniform_flexible_bic(
    uniform: ProfileFit | float,
    flexible: ProfileFit | float,
) -> BICComparison:
    """Compare uniform and flexible BIC values.

    ``delta_bic`` is always ``flexible - uniform``.  Negative values favour
    the flexible model.
    """

    uniform_bic = float(uniform.bic if isinstance(uniform, ProfileFit) else uniform)
    flexible_bic = float(flexible.bic if isinstance(flexible, ProfileFit) else flexible)
    if not np.isfinite(uniform_bic) or not np.isfinite(flexible_bic):
        raise ValueError("BIC values must be finite")
    return BICComparison(
        uniform_bic=uniform_bic,
        flexible_bic=flexible_bic,
        delta_bic=flexible_bic - uniform_bic,
    )


def cyclic_residual_shift(residuals: ArrayLike, shift: int, *, axis: int = -1) -> FloatArray:
    """Cyclically shift residuals along one axis.

    A positive shift moves the final values to the beginning, matching
    :func:`numpy.roll`.  The helper returns a new floating-point array and
    preserves all non-shifted axes.
    """

    if isinstance(shift, bool) or not isinstance(shift, Integral):
        raise ValueError("shift must be an integer")
    values = np.asarray(residuals, dtype=float)
    if values.ndim == 0:
        raise ValueError("residuals must be an array")
    if not np.all(np.isfinite(values)):
        raise ValueError("residuals must contain only finite values")
    if not isinstance(axis, Integral) or isinstance(axis, bool):
        raise ValueError("axis must be an integer")
    if axis < -values.ndim or axis >= values.ndim:
        raise ValueError("axis is outside the residual array")
    return np.roll(values, int(shift), axis=int(axis))


# Descriptive aliases for callers that prefer explicit function names.
fit_linear_profile = fit_profile
fit_recovery_grid = fit_candidate_grid
compare_bic = compare_uniform_flexible_bic
shift_residuals_cyclic = cyclic_residual_shift


__all__ = [
    "BICComparison",
    "CandidateGridResult",
    "ProfileFit",
    "compare_bic",
    "compare_uniform_flexible_bic",
    "cyclic_residual_shift",
    "fit_candidate_grid",
    "fit_linear_profile",
    "fit_profile",
    "fit_recovery_grid",
    "shift_residuals_cyclic",
]
