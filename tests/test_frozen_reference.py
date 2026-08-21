"""Tests for the starry-free frozen reference comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from robert_mapping.benchmark.frozen_reference import run_hatp32_frozen_reference
from robert_mapping.physics import evaluate_map, secondary_eclipse_design_matrix


REFERENCE = Path(
    os.environ.get("ROBERT_MAPPING_HATP32_REFERENCE", "reference_data/hatp32_60ppm")
)


def _write_fixture(directory: Path) -> None:
    system = {
        "period_days": 2.150009,
        "derived_transit_epoch_bjd_tdb": 2457740.0443955003,
        "eclipse_mid_bjd_tdb": 2457741.1194,
        "a_over_rstar": 6.05,
        "rp_over_rstar": 0.1508,
        "inclination_deg": 88.9,
        "stellar_mass_msun": 1.16,
        "stellar_radius_rsun": 1.22,
        "planet_mass_mjup": 0.86,
    }
    coefficients = np.array(
        [1.0, 0.0, 0.198943737807463, 0.035079148618446486, 0.0, 0.0, 0.02933156657967967, 0.009099531943521704, 0.0008022464967307069]
    )
    injection = {
        "map_degree": 2,
        "longitude_deg": 10.0,
        "latitude_deg": 0.0,
        "starry_map_amplitude": 0.0003558903196804383,
        "starry_coefficients": coefficients.tolist(),
    }
    config = {"target": "HAT-P-32b", "system": system, "injection": injection}
    (directory / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    longitude = np.linspace(-90.0, 90.0, 19)
    latitude = np.linspace(-90.0, 90.0, 5)
    grid_lon, grid_lat = np.meshgrid(np.deg2rad(longitude), np.deg2rad(latitude), indexing="xy")
    map_percent = (
        np.asarray(evaluate_map(coefficients, grid_lon, grid_lat), dtype=float)
        * injection["starry_map_amplitude"]
        * np.pi
        * 100.0
    )
    np.savez(
        directory / "map_data.npz",
        longitude_deg=longitude,
        latitude_deg=latitude,
        injected_specific_intensity_percent=map_percent,
    )

    time = system["eclipse_mid_bjd_tdb"] + np.linspace(-0.1, 0.1, 7)
    design = np.asarray(
        secondary_eclipse_design_matrix(
            time,
            system["period_days"],
            system["a_over_rstar"],
            system["inclination_deg"],
            system["rp_over_rstar"],
            2,
            system["derived_transit_epoch_bjd_tdb"],
            theta0=np.pi,
            angle_unit="deg",
        ),
        dtype=float,
    )
    planet = injection["starry_map_amplitude"] * design.dot(coefficients)
    rows = np.column_stack(
        (
            time,
            (time - system["eclipse_mid_bjd_tdb"]) * 24.0,
            1.0 + planet,
            planet,
            np.full(time.size, 60.0e-6),
        )
    )
    np.savetxt(
        directory / "synthetic_observation.csv",
        rows,
        delimiter=",",
        header="time_bjd_tdb,time_from_eclipse_hours,flux_true,planet_flux_true,flux_uncertainty",
        comments="",
    )


def test_frozen_reference_fixture_is_exact(tmp_path) -> None:
    reference = tmp_path / "reference"
    output = tmp_path / "output"
    reference.mkdir()
    _write_fixture(reference)
    report = run_hatp32_frozen_reference(reference, output, save_plots=False)
    assert report.status == "passed"
    assert report.map_comparison.map_maximum_absolute_difference_percent < 1.0e-12
    assert report.map_comparison.robert_peak_longitude_degrees == 10.0
    assert report.map_comparison.robert_peak_latitude_degrees == 0.0
    assert report.literal_curve_comparison.rmse_ppm < 1.0e-8
    assert report.geometry["saved_a_over_rstar"] == 6.05
    assert (output / "frozen_hatp32_report.json").is_file()
    assert (output / "frozen_hatp32_comparison.npz").is_file()


def test_frozen_reference_matches_saved_hatp32_map(tmp_path) -> None:
    if not REFERENCE.is_dir():
        import pytest

        pytest.skip("The external frozen HAT-P-32b reference is not mounted")
    report = run_hatp32_frozen_reference(REFERENCE, tmp_path / "output", save_plots=False)
    assert report.status == "passed"
    assert report.map_comparison.map_correlation > 1.0 - 1.0e-12
    assert report.map_comparison.legacy_peak_longitude_degrees == 10.0
    assert report.map_comparison.robert_peak_longitude_degrees == 10.0
    assert report.map_comparison.legacy_peak_latitude_degrees == 0.0
    assert report.map_comparison.robert_peak_latitude_degrees == 0.0
    assert report.geometry["saved_a_over_rstar"] == 6.05
    assert abs(report.geometry["kepler_derived_a_over_rstar"] - 6.03899) < 1.0e-3
    assert report.reference_aligned_curve_comparison.rmse_ppm < 5.0
    assert 7.2 < report.geometry["reference_aligned_a_over_rstar"] < 7.5
