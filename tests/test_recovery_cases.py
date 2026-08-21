"""Focused tests for the physical recovery-case layer."""

from __future__ import annotations

import numpy as np

from robert_mapping.benchmark.recovery_cases import _hotspot_coefficients, run_recovery
from robert_mapping.config import mapping_config_from_dict
from robert_mapping.physics import evaluate_map


def test_positive_longitude_hotspot_peaks_east() -> None:
    longitude = np.arange(-90.0, 91.0)
    latitude = np.zeros(longitude.size)
    coefficients = _hotspot_coefficients(27.0, 40.0, 2)
    profile = np.asarray(
        evaluate_map(coefficients, np.deg2rad(longitude), latitude), dtype=float
    )
    assert longitude[int(np.argmax(profile))] == 27.0


def test_latitude_hotspot_template_peaks_north() -> None:
    longitude = np.zeros(181)
    latitude = np.arange(-90.0, 91.0)
    coefficients = _hotspot_coefficients(0.0, 30.0, 2, latitude_degrees=30.0)
    profile = np.asarray(
        evaluate_map(
            coefficients,
            np.deg2rad(longitude),
            np.deg2rad(latitude),
        ),
        dtype=float,
    )
    assert latitude[int(np.argmax(profile))] > 0.0


def test_small_noiseless_hatp32_recovery(tmp_path) -> None:
    config = mapping_config_from_dict(
        {
            "project": {"seed": 3},
            "data": {"kind": "synthetic"},
            "system": {
                "period_days": 2.150009,
                "transit_time": 2457740.0443955003,
                "a_over_rstar": 6.05,
                "radius_ratio": 0.1508,
                "inclination_degrees": 88.9,
                "planet_flux_ratio": 0.000445,
            },
            "map": {"representation": "harmonics", "harmonic_degree": 2},
            "recovery": {
                "enabled": True,
                "case": "hatp32",
                "injected_longitudes_degrees": [10.0],
                "noise_ppm": 0.01,
                "longitude_grid_min_degrees": -30.0,
                "longitude_grid_max_degrees": 30.0,
                "longitude_grid_step_degrees": 10.0,
                "width_grid_degrees": [40.0],
                "timing_grid_seconds": [0.0],
                "trials_per_case": 1,
            },
            "output": {
                "directory": str(tmp_path),
                "save_report": False,
                "overwrite": True,
            },
        }
    )
    report = run_recovery(config)
    trial = report.trials[0]
    assert abs(trial.recovered_longitude_degrees - 10.0) <= 5.0
    assert trial.interval_contains_injection
    assert report.longitude_sign.startswith("positive is east")


def test_small_synthetic_matrix_covers_requested_strata(tmp_path) -> None:
    config = mapping_config_from_dict(
        {
            "project": {"seed": 17},
            "data": {"kind": "synthetic"},
            "system": {
                "period_days": 2.150009,
                "transit_time": 2457740.0443955003,
                "a_over_rstar": 6.05,
                "radius_ratio": 0.1508,
                "inclination_degrees": 88.9,
                "planet_flux_ratio": 0.000445,
            },
            "map": {"representation": "harmonics", "harmonic_degree": 2},
            "recovery": {
                "enabled": True,
                "case": "synthetic_matrix",
                "injected_longitudes_degrees": [-30.0, 30.0],
                "injected_latitudes_degrees": [-30.0, 30.0],
                "noise_levels_ppm": [10.0, 20.0],
                "eclipse_counts": [1, 2],
                "points_per_eclipse": 21,
                "trials_per_case": 1,
                "longitude_grid_min_degrees": -60.0,
                "longitude_grid_max_degrees": 60.0,
                "longitude_grid_step_degrees": 30.0,
                "latitude_grid_degrees": [-30.0, 0.0, 30.0],
                "width_grid_degrees": [40.0],
                "timing_grid_seconds": [0.0],
            },
            "output": {
                "directory": str(tmp_path),
                "save_report": False,
                "overwrite": True,
            },
        }
    )
    report = run_recovery(config)
    assert report.case == "synthetic_matrix"
    assert report.null_trial_count == 4
    assert report.injection_trial_count == 16
    assert {trial.noise_ppm for trial in report.trials} == {10.0, 20.0}
    assert {trial.eclipse_count for trial in report.trials} == {1, 2}
    assert all(
        trial.injected_latitude_degrees is None
        or trial.latitude_q16_degrees is not None
        for trial in report.trials
    )
    assert "mapping_evidence" in report.comparison
    assert "conditional_location" in report.comparison
