"""Fast tests for the sampler-free recovery-grid core."""

from __future__ import annotations

import numpy as np
import pytest

from robert_mapping.recovery import (
    compare_uniform_flexible_bic,
    cyclic_residual_shift,
    fit_candidate_grid,
    fit_profile,
)


def test_one_nonnegative_map_amplitude_and_unbounded_nuisance() -> None:
    x = np.linspace(-1.0, 1.0, 21)
    design = np.column_stack((np.ones(x.size), x))
    observed = 0.75 + 0.4 * x
    fit = fit_profile(design, observed, 0.01, map_columns=(1,), nuisance_columns=(0,))

    assert fit.active_map_columns == (1,)
    assert np.allclose(fit.coefficients, (0.75, 0.4), atol=1.0e-10)
    assert np.isclose(fit.chi2, 0.0, atol=1.0e-10)
    assert np.allclose(fit.fitted, observed)


def test_active_set_enumeration_rejects_negative_map_column() -> None:
    x = np.linspace(-1.0, 1.0, 31)
    design = np.column_stack((np.ones(x.size), x, x**2))
    observed = 1.0 - 0.3 * x + 0.5 * x**2
    fit = fit_profile(
        design,
        observed,
        0.02,
        map_columns=(1, 2),
        nuisance_columns=(0,),
    )

    # x is a non-negative map amplitude column and must be fixed to zero;
    # x**2 remains active.  The constant remains unbounded.
    assert fit.active_map_columns == (2,)
    assert np.isclose(fit.coefficients[1], 0.0)
    assert np.isclose(fit.coefficients[2], 0.5, atol=1.0e-10)
    assert np.isclose(fit.coefficients[0], 1.0, atol=1.0e-10)


def test_dense_covariance_matches_explicit_cholesky_whitening() -> None:
    design = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    truth = np.array([0.5, 0.25])
    covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.05, 0.01], [0.0, 0.01, 0.03]])
    observed = design @ truth
    fit = fit_profile(design, observed, covariance=covariance, map_columns=(1,))

    chol = np.linalg.cholesky(covariance)
    whitened_design = np.linalg.solve(chol, design)
    whitened_observed = np.linalg.solve(chol, observed)
    expected = np.linalg.lstsq(whitened_design, whitened_observed, rcond=None)[0]
    assert np.allclose(fit.coefficients, expected)
    assert np.isclose(fit.chi2, 0.0, atol=1.0e-10)


def test_candidate_grid_returns_bic_weights_and_weighted_quantiles() -> None:
    x = np.linspace(-1.0, 1.0, 41)
    longitudes = np.array([-20.0, 0.0, 20.0])
    designs = np.stack(
        [np.column_stack((np.ones(x.size), x - offset / 20.0)) for offset in longitudes]
    )
    observed = 0.8 + 0.5 * x
    result = fit_candidate_grid(
        longitudes,
        designs,
        observed,
        0.03,
        map_columns=(1,),
        nuisance_columns=(0,),
    )

    assert result.log_likelihood.shape == longitudes.shape
    assert result.bic.shape == longitudes.shape
    assert result.posterior_weights.shape == longitudes.shape
    assert np.isclose(np.sum(result.posterior_weights), 1.0)
    assert result.best_index == int(np.argmax(result.posterior_weights))
    assert result.longitude_q16 <= result.longitude_median <= result.longitude_q84


def test_bic_sign_convention_and_cyclic_shift() -> None:
    comparison = compare_uniform_flexible_bic(100.0, 92.0)
    assert comparison.delta_bic == -8.0
    assert comparison.flexible_preferred
    assert np.array_equal(cyclic_residual_shift(np.arange(5.0), 2), np.array([3.0, 4.0, 0.0, 1.0, 2.0]))


def test_shape_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="rows"):
        fit_profile(np.ones((3, 2)), np.ones(4), 1.0)
    with pytest.raises(ValueError, match="at most two"):
        fit_profile(np.ones((4, 3)), np.ones(4), 1.0, map_columns=(0, 1, 2))
    with pytest.raises(ValueError, match="positive definite"):
        fit_profile(np.ones((2, 1)), np.ones(2), covariance=np.zeros((2, 2)))
