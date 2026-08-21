import numpy as np
from dataclasses import replace

from robert_mapping.benchmark import (
    quick_hammond_comparison,
    run_benchmark,
    select_entropy_alpha,
)
from robert_mapping.config import default_config


def test_quick_comparison_detects_extra_map_signal():
    rng = np.random.default_rng(7)
    count = 120
    phase = np.linspace(-1.0, 1.0, count)
    fourier = np.column_stack([np.ones(count), phase, phase**2])
    eclipse_feature = np.exp(-0.5 * ((phase - 0.35) / 0.09) ** 2)
    mapping = np.column_stack([fourier, eclipse_feature])
    truth = mapping @ np.array([1.0, 0.01, -0.02, 0.08])
    sigma = np.full(count, 0.005)
    observed = truth + rng.normal(0.0, sigma)
    folds = tuple(np.array(block) for block in np.array_split(np.arange(70, 105), 5))
    result = quick_hammond_comparison(
        observed, sigma, mapping, fourier, folds, map_prior_scale=10.0
    )
    assert result.comparison.delta_elpd > 0.0


def test_entropy_selection_returns_grid_member():
    rng = np.random.default_rng(11)
    design = np.abs(rng.normal(size=(50, 4))) * 0.1
    pixels = np.array([1.0, 1.1, 0.9, 1.0])
    sigma = np.full(50, 0.02)
    observed = design @ pixels + rng.normal(0.0, sigma)
    folds = tuple(np.array(block) for block in np.array_split(np.arange(50), 2))
    selection = select_entropy_alpha(
        observed,
        sigma,
        design,
        folds,
        [0.0, 0.1],
        prior_mean=1.0,
        prior_log_sigma=1.0,
    )
    assert selection.selected_alpha in {0.0, 0.1}
    assert selection.score.shape == (2,)


def test_quick_hammond_workflow_has_broad_paper_consistency(tmp_path):
    config = default_config()
    config = replace(
        config,
        output=replace(config.output, directory=tmp_path, overwrite=True),
    )
    report = run_benchmark(config)
    assert report.status == "passed"
    assert all(case.passed for case in report.cases)
    assert report.injection_correlation > 0.5
    assert (tmp_path / "hammond2024_benchmark.json").is_file()
    assert (tmp_path / "hammond2024_injection.npz").is_file()
    assert (tmp_path / "hammond2024_benchmark.png").is_file()
