"""Focused tests for the combined WASP-121b study report helpers."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np


_TOOL = Path(__file__).parents[1] / "tools" / "report_wasp121b_study.py"
_SPEC = importlib.util.spec_from_file_location("report_wasp121b_study", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_acf_is_normalized_at_zero_lag() -> None:
    values = np.array([1.0, -1.0, 1.0, -1.0])
    acf = _MODULE._acf(values, maximum_lag=2)

    assert acf[0] == 1.0
    np.testing.assert_allclose(acf[1:], np.array([-0.75, 0.5]))


def test_ou_innovations_reduce_to_independent_residuals_at_zero_amplitude() -> None:
    residual = np.array([0.5, -1.0, 2.0])
    sigma = np.ones(3)
    time = np.array([0.0, 1.0, 2.0]) / 86400.0

    innovations = _MODULE._ou_standardized_innovations(
        residual,
        sigma,
        time,
        amplitude=0.0,
        timescale=100.0,
        jitter=0.0,
    )

    np.testing.assert_allclose(innovations, residual)


def test_ou_innovations_are_finite_for_irregular_cadence() -> None:
    innovations = _MODULE._ou_standardized_innovations(
        np.array([1.0, 0.3, -0.2, 0.1]) * 1.0e-4,
        np.full(4, 5.0e-5),
        np.array([0.0, 20.0, 75.0, 140.0]) / 86400.0,
        amplitude=8.0e-5,
        timescale=900.0,
        jitter=3.0e-5,
    )

    assert innovations.shape == (4,)
    assert np.all(np.isfinite(innovations))


def test_map_physicality_requires_a_valid_positive_draw_diagnostic() -> None:
    zero_draws = {
        "status": "no_accepted_draws",
        "accepted_draws": 0,
        "total_draws": 2000,
        "accepted_fraction": 0.0,
    }
    unknown = {}
    null_diagnostic = None

    for positivity in (zero_draws, unknown, null_diagnostic):
        is_physical, warning = _MODULE._map_physicality(positivity)

        assert is_physical is False
        assert warning is not None

    available = {
        "status": "available",
        "accepted_draws": 1994,
        "total_draws": 2000,
        "accepted_fraction": 0.997,
    }
    is_physical, warning = _MODULE._map_physicality(available)

    assert is_physical is True
    assert warning is not None
    assert "1994 of 2000" in warning
    assert "99.7%" in warning


def test_load_run_does_not_trust_miri_label_when_positivity_has_zero_draws(tmp_path: Path) -> None:
    mapped = tmp_path / "mapped"
    uniform = tmp_path / "uniform"
    mapped.mkdir()
    uniform.mkdir()
    data_path = tmp_path / "data.npz"
    np.savez(data_path, time=np.array([0.0, 1.0]), flux_err=np.ones(2))
    np.save(mapped / "residuals.npy", np.zeros(2))
    np.save(uniform / "residuals.npy", np.zeros(2))

    fit_summary = {
        "residual_rms": 1.0,
        "divergences": 0,
        "maximum_rhat": 1.0,
        "minimum_effective_sample_size": 100.0,
        "n_samples": 100,
    }
    (mapped / "fit_summary.json").write_text(
        json.dumps(fit_summary), encoding="utf-8"
    )
    (uniform / "fit_summary.json").write_text(
        json.dumps(fit_summary), encoding="utf-8"
    )
    (mapped / "production_report.json").write_text(
        json.dumps(
            {
                "hotspot_longitude_degrees_east": {
                    "q16": 1.0,
                    "median": 2.0,
                    "q84": 3.0,
                },
                "positivity_conditioned": {
                    "status": "no_accepted_draws",
                    "accepted_draws": 0,
                    "total_draws": 100,
                    "accepted_fraction": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )

    settings = dict(_MODULE.RUNS["MIRI"])
    settings.update(mapped=mapped, uniform=uniform, data=data_path)
    run = _MODULE._load_run("MIRI", settings)

    assert run["map_is_physical"] is False
    assert run["positive_draw_fraction"] == 0.0
    assert "No globally non-negative posterior draws" in run["map_warning"]
