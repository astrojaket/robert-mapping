import numpy as np

from robert_mapping.model_selection import (
    compare_information_criteria,
    compare_pointwise_elpd,
    entropy_log_weight,
    fourier_design_matrix,
    gaussian_pointwise_elpd,
    information_criteria,
    make_eclipse_folds,
    spatial_entropy,
)


def test_information_criteria_use_smaller_is_better_convention():
    mapped = information_criteria(-90.0, 9, 100)
    fourier = information_criteria(-100.0, 6, 100)
    difference = compare_information_criteria(mapped, fourier)
    assert difference.aic > 0.0
    assert difference.bic > 0.0


def test_uniform_map_has_zero_entropy():
    assert np.isclose(spatial_entropy(np.ones(16)), 0.0)
    assert spatial_entropy(np.array([1.0, 2.0, 1.0])) < 0.0
    assert entropy_log_weight(np.array([1.0, 2.0]), 10.0) < 0.0


def test_fourier_design_matrix_shape_and_eclipse():
    time = np.linspace(0.0, 1.0, 11)
    visibility = np.ones_like(time)
    visibility[5] = 0.0
    matrix = fourier_design_matrix(
        time, period=1.0, t0=0.0, visibility=visibility, degree=2
    )
    assert matrix.shape == (11, 6)
    assert np.allclose(matrix[5, 1:], 0.0)
    assert matrix[5, 0] == 1.0


def test_gaussian_elpd_prefers_correct_predictions():
    observed = np.array([0.0, 1.0, -1.0])
    correct = np.tile(observed, (20, 1))
    wrong = correct + 4.0
    sigma = np.ones(3)
    elpd_correct = gaussian_pointwise_elpd(observed, correct, sigma)
    elpd_wrong = gaussian_pointwise_elpd(observed, wrong, sigma)
    comparison = compare_pointwise_elpd(elpd_correct, elpd_wrong)
    assert comparison.delta_elpd > 0.0
    assert comparison.pointwise_delta.shape == observed.shape


def test_eclipse_folds_are_contiguous_and_cover_intervals():
    time = np.linspace(0.0, 1.0, 101)
    folds = make_eclipse_folds(
        time, [[0.20, 0.30], [0.70, 0.80]], blocks_per_interval=3
    )
    assert len(folds) == 6
    for fold in folds:
        assert np.all(np.diff(fold) == 1)
