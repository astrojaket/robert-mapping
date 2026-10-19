import os
import numpy as np
from dataclasses import replace

from robert_mapping.benchmark import (
    quick_hammond_comparison,
    run_benchmark,
    select_entropy_alpha,
)
from robert_mapping.config import default_config
from robert_mapping.compute import resolve_plan


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
    serial = quick_hammond_comparison(
        observed, sigma, mapping, fourier, folds, map_prior_scale=10.0
    )
    plan = resolve_plan(max_cpus=2, threads=1, workers=2)
    parallel = quick_hammond_comparison(
        observed,
        sigma,
        mapping,
        fourier,
        folds,
        map_prior_scale=10.0,
        compute_plan=plan,
    )
    assert serial.comparison.delta_elpd > 0.0
    assert parallel.comparison.delta_elpd == serial.comparison.delta_elpd
    assert parallel.comparison.standard_error == serial.comparison.standard_error
    assert parallel.comparison.z_score == serial.comparison.z_score
    np.testing.assert_array_equal(
        parallel.comparison.pointwise_delta, serial.comparison.pointwise_delta
    )
    np.testing.assert_array_equal(
        parallel.map_pointwise_elpd, serial.map_pointwise_elpd
    )
    np.testing.assert_array_equal(
        parallel.fourier_pointwise_elpd, serial.fourier_pointwise_elpd
    )
    assert parallel.worker_process_ids
    if plan.workers > 1:
        assert os.getpid() not in parallel.worker_process_ids
    else:
        assert parallel.worker_process_ids == (os.getpid(),)


def test_entropy_selection_returns_grid_member_and_matches_parallel_scores():
    rng = np.random.default_rng(11)
    design = np.abs(rng.normal(size=(50, 4))) * 0.1
    pixels = np.array([1.0, 1.1, 0.9, 1.0])
    sigma = np.full(50, 0.02)
    observed = design @ pixels + rng.normal(0.0, sigma)
    folds = tuple(np.array(block) for block in np.array_split(np.arange(50), 2))
    serial = select_entropy_alpha(
        observed,
        sigma,
        design,
        folds,
        [0.0, 0.1],
        prior_mean=1.0,
        prior_log_sigma=1.0,
    )
    plan = resolve_plan(max_cpus=2, threads=1, workers=2)
    parallel = select_entropy_alpha(
        observed,
        sigma,
        design,
        folds,
        [0.0, 0.1],
        prior_mean=1.0,
        prior_log_sigma=1.0,
        compute_plan=plan,
    )
    assert serial.selected_alpha in {0.0, 0.1}
    np.testing.assert_array_equal(parallel.score, serial.score)
    assert parallel.selected_alpha == serial.selected_alpha
    assert parallel.score.shape == (2,)
    assert parallel.worker_process_ids
    if plan.workers > 1:
        assert os.getpid() not in parallel.worker_process_ids
    else:
        assert parallel.worker_process_ids == (os.getpid(),)


def test_quick_hammond_workflow_has_broad_paper_consistency(tmp_path):
    config = default_config()
    plan = resolve_plan(max_cpus=2, threads=1, workers=2)
    config = replace(
        config,
        output=replace(config.output, directory=tmp_path, overwrite=True),
        compute=replace(
            config.compute,
            max_cpus=plan.requested_max_cpus,
            threads=plan.requested_threads,
            workers=plan.requested_workers,
        ),
    )
    report = run_benchmark(config)
    assert report.status == "passed"
    assert all(case.passed for case in report.cases)
    assert report.injection_correlation > 0.5
    assert 1 <= report.execution_plan["actual_workers"] <= plan.workers
    if plan.workers > 1:
        assert os.getpid() not in report.execution_plan["worker_process_ids"]
    else:
        assert report.execution_plan["worker_process_ids"] == [os.getpid()]
    assert (tmp_path / "hammond2024_benchmark.json").is_file()
    assert (tmp_path / "hammond2024_injection.npz").is_file()
    assert (tmp_path / "hammond2024_benchmark.png").is_file()
