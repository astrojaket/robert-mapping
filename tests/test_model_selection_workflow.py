"""Tests for held-out predictive checks on supplied light curves."""

from dataclasses import replace
import json

import numpy as np
import pytest

from robert_mapping.compute import resolve_plan
from robert_mapping.config import mapping_config_from_dict
from robert_mapping.data import load_light_curve
from robert_mapping.inference.run import _secondary_design, _time_days
from robert_mapping.model_selection.fourier import fourier_design_matrix
from robert_mapping.model_selection import workflow as workflow_module
from robert_mapping.model_selection.workflow import (
    _eclipse_transition_intervals,
    _null_design,
    _unocculted_uniform_visibility,
    _uniform_visibility_at_samples,
    build_cv_folds,
    run_cross_validation,
    validate_cross_validation,
)


def _config(tmp_path, *, workers=1, strategy="eclipse_blocks", representation="direct_harmonics", n_folds=5, exposure_seconds=10.0):
    time = np.linspace(0.45, 0.55, 401)
    flux = 1.0 + 0.005 * np.cos(2.0 * np.pi * (time - 0.5))
    flux += 2.0e-5 * np.sin(np.arange(time.size) * 0.31)
    flux_err = np.full(time.size, 8.0e-5)
    data_path = tmp_path / "light_curve.npz"
    np.savez(data_path, time=time, flux=flux, flux_err=flux_err)
    output_path = tmp_path / f"results-{workers}-{strategy}-{representation}-{n_folds}"
    map_config = {
        "representation": representation,
        "harmonic_degree": 1,
        "positive": representation != "longitude",
        "regularization": "none",
    }
    if representation == "direct_harmonics":
        map_config["positive"] = False
        map_config["active_harmonic_indices"] = [3, 0, 2]
    if representation == "longitude":
        map_config["positive"] = False
    return mapping_config_from_dict(
        {
            "project": {"name": "workflow-test", "seed": 11},
            "data": {
                "kind": "files",
                "file": str(data_path),
                "format": "npz",
                "time_unit": "day",
                "exposure_seconds": exposure_seconds,
                "normalize": "none",
            },
            "system": {
                "period_days": 1.0,
                "transit_time": 0.0,
                "a_over_rstar": 5.0,
                "radius_ratio": 0.1,
                "inclination_degrees": 90.0,
                "planet_flux_ratio": 0.005,
            },
            "map": map_config,
            "model": {"fourier_degree": 1, "integrate_exposure": True},
            "inference": {"sampler": "map"},
            "cross_validation": {
                "n_folds": n_folds,
                "strategy": strategy,
                "null_model": "fourier",
                "prior_scale": 0.05,
            },
            "compute": {
                "max_cpus": max(2, workers),
                "workers": workers,
                "threads": 1,
                "quadrature_radial": 4,
                "quadrature_azimuth": 8,
            },
            "output": {
                "directory": str(output_path),
                "save_resolved_config": False,
            },
        }
    )


def test_eclipse_blocks_hold_out_only_ingress_and_egress(tmp_path):
    config = _config(tmp_path)
    curve = load_light_curve(config)
    folds = build_cv_folds(config, curve)
    intervals = _eclipse_transition_intervals(
        config, _time_days(curve), exposure_seconds=curve.exposure_seconds
    )
    assigned = np.concatenate(folds)
    assert len(folds) == config.cross_validation.n_folds
    assert all(fold.size > 0 for fold in folds)
    assert np.unique(assigned).size == assigned.size
    inside_transition = np.zeros(curve.n_observations, dtype=bool)
    time_days = _time_days(curve)
    for start, stop in intervals:
        inside_transition |= (time_days >= start) & (time_days <= stop)
    assert np.array_equal(np.sort(assigned), np.flatnonzero(inside_transition))
    assert np.any(~inside_transition)


def test_contiguous_folds_cover_each_observation_once(tmp_path):
    config = _config(tmp_path, strategy="contiguous")
    curve = load_light_curve(config)
    folds = build_cv_folds(config, curve)
    assigned = np.concatenate(folds)
    assert sorted(assigned.tolist()) == list(range(curve.n_observations))
    assert all(fold.size > 0 for fold in folds)


def test_cv_rejects_full_data_normalization_and_eccentric_geometry(tmp_path):
    config = _config(tmp_path)
    normalized = replace(config, data=replace(config.data, normalize="median"))
    with pytest.raises(ValueError, match="data.normalize: none"):
        validate_cross_validation(normalized)
    eccentric = replace(config, system=replace(config.system, eccentricity=0.01))
    with pytest.raises(ValueError, match="circular-orbit"):
        validate_cross_validation(eccentric)
    no_warmup = replace(config, inference=replace(config.inference, sampler="nuts", warmup=0))
    with pytest.raises(ValueError, match="positive warmup"):
        validate_cross_validation(no_warmup)


def test_fourier_null_averages_each_exposure_sample(tmp_path):
    config = _config(tmp_path, exposure_seconds=1800.0)
    curve = load_light_curve(config)
    bundle = _secondary_design(config, curve)
    actual = _null_design(config, curve, bundle)
    time_days = _time_days(curve)
    offsets = ((np.arange(4) + 0.5) / 4.0 - 0.5) * curve.exposure_seconds / 86400.0
    sampled_time = time_days[:, None] + offsets[None, :]
    sampled_visibility = _uniform_visibility_at_samples(
        config, sampled_time.reshape(-1)
    ).reshape(time_days.size, 4)
    expected = fourier_design_matrix(
        sampled_time.reshape(-1),
        period=config.system.period_days,
        t0=config.system.transit_time,
        visibility=sampled_visibility.reshape(-1),
        degree=config.model.fourier_degree,
        include_intercept=False,
    ).reshape(time_days.size, 4, -1).mean(axis=1)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)

    reference_visibility = _unocculted_uniform_visibility(config)
    midpoint_visibility = bundle.map_design[:, 0] / reference_visibility
    midpoint_design = fourier_design_matrix(
        bundle.time_days,
        period=config.system.period_days,
        t0=config.system.transit_time,
        visibility=midpoint_visibility,
        degree=config.model.fourier_degree,
        include_intercept=False,
    )
    assert np.max(np.abs(actual - midpoint_design)) > 1.0e-8


def test_exact_cv_outputs_match_serial_and_parallel_runs(tmp_path):
    serial_config = _config(tmp_path, workers=1)
    parallel_config = _config(tmp_path, workers=2)
    serial = run_cross_validation(serial_config)
    parallel = run_cross_validation(parallel_config)
    assert serial.status == parallel.status == "complete"
    assert serial.evaluated_observations == parallel.evaluated_observations
    assert serial.map_elpd == pytest.approx(parallel.map_elpd, abs=1.0e-12)
    assert serial.null_elpd == pytest.approx(parallel.null_elpd, abs=1.0e-12)
    assert serial.delta_elpd == pytest.approx(parallel.delta_elpd, abs=1.0e-12)

    with np.load(serial.pointwise_path) as serial_data, np.load(parallel.pointwise_path) as parallel_data:
        for name in (
            "fold_membership",
            "map_log_predictive_density",
            "null_log_predictive_density",
            "pointwise_delta",
        ):
            np.testing.assert_allclose(
                serial_data[name], parallel_data[name], rtol=0.0, atol=1.0e-12, equal_nan=True
            )
    with parallel.summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    assert summary["execution_plan"]["parent_process_id"] != summary["execution_plan"]["worker_process_ids"][0] if summary["execution_plan"]["workers"] > 1 else True
    assert summary["execution_plan"]["active_workers"] >= 1
    assert summary["evaluated_observations"] == serial.evaluated_observations
    assert summary["settings"]["sampler"] == "map"
    assert summary["settings"]["master_seed"] == serial_config.project.seed
    assert summary["priors"]["baseline"] == {
        "distribution": "normal",
        "mean": 1.0,
        "standard_deviation": 0.05,
    }
    assert summary["priors"]["map_coefficients"]["mean"] == [
        0.0,
        serial_config.system.planet_flux_ratio,
        0.0,
    ]
    assert summary["priors"]["map_coefficients"]["active_harmonic_indices"] == [
        3,
        0,
        2,
    ]
    assert summary["priors"]["null_coefficients"]["mean"][0] == pytest.approx(
        serial_config.system.planet_flux_ratio
    )
    assert not (serial.summary_path.parent / "cross_validation_config.yml").exists()


def test_exact_longitude_cv_is_supported(tmp_path):
    config = _config(tmp_path, representation="longitude", n_folds=2)
    plan = validate_cross_validation(config)
    assert plan["map_model"] == "longitude_analytic_gaussian"
    report = run_cross_validation(config)
    assert report.status == "complete"
    assert report.n_folds == 2


def test_held_out_flux_does_not_enter_training_fits(tmp_path, monkeypatch):
    config = _config(tmp_path, n_folds=2)
    curve = load_light_curve(config)
    bundle = _secondary_design(config, curve)
    null_design = _null_design(config, curve, bundle)
    held_out = build_cv_folds(config, curve)[0]
    worker_plan = resolve_plan(max_cpus=1, workers=1, threads=1)
    job = {
        "config": config,
        "curve": curve,
        "bundle": bundle,
        "null_design": null_design,
        "fold_index": 0,
        "held_out": held_out,
        "worker_plan": worker_plan,
        "map_seed": 1,
        "null_seed": 2,
    }
    original_fit = workflow_module.fit_linear_gaussian
    fitted_observations = []

    def record_training_observations(design, observed, sigma, **kwargs):
        fitted_observations.append(np.asarray(observed).copy())
        return original_fit(design, observed, sigma, **kwargs)

    monkeypatch.setattr(
        workflow_module, "fit_linear_gaussian", record_training_observations
    )
    first = workflow_module._score_fold(job)
    changed_flux = np.asarray(curve.flux).copy()
    changed_flux[held_out] += 0.01
    changed_curve = replace(curve, flux=changed_flux)
    second = workflow_module._score_fold({**job, "curve": changed_curve})

    assert len(fitted_observations) == 4
    np.testing.assert_array_equal(fitted_observations[0], fitted_observations[2])
    np.testing.assert_array_equal(fitted_observations[1], fitted_observations[3])
    assert not np.allclose(
        first["map_log_predictive_density"],
        second["map_log_predictive_density"],
    )
    assert not np.allclose(
        first["null_log_predictive_density"],
        second["null_log_predictive_density"],
    )
