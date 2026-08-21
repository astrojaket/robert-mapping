"""Tests for the frozen, starry-free WASP-43b forward comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robert_mapping.benchmark.frozen_wasp43b import run_frozen_wasp43b


REFERENCE = Path(__file__).resolve().parents[1] / "wasp43b_simulation"


def test_frozen_wasp43b_matches_saved_reference_with_exact_windows(tmp_path: Path) -> None:
    report = run_frozen_wasp43b(REFERENCE, tmp_path / "output", save_plots=False)

    assert report.status == "passed"
    assert report.n_observations == 1561
    assert report.numerical_settings["quadrature_n_radial"] == 32
    assert report.numerical_settings["quadrature_n_azimuth"] == 128
    assert report.numerical_settings["eclipse_window_half_phase"] == 0.12
    assert report.pass_criterion["scope"] == ["eclipse_0", "eclipse_1"]
    assert report.event_windows["counts"]["eclipse_0"] == 187
    assert report.event_windows["counts"]["eclipse_1"] == 187

    matrix = report.reference_matrix
    assert matrix["status"] == "passed"
    assert matrix["assets_used"] == (
        "w43b_time.npy",
        "sim_flux_clean.npy",
        "sim_ylm_coeffs.npy",
    )
    matrix_cases = {item["name"]: item for item in matrix["cases"]}
    assert set(matrix_cases) == {
        "full_phase_curve",
        "orbit_0",
        "orbit_1",
        "both_secondary_eclipses",
        "eclipse_0_ingress",
        "eclipse_0_egress",
        "eclipse_1_ingress",
        "eclipse_1_egress",
        "transit_0_contact",
        "transit_1_contact",
        "out_of_event",
    }
    assert all(item["status"] == "passed" for item in matrix_cases.values())
    assert matrix_cases["full_phase_curve"]["n_observations"] == 1561
    assert matrix_cases["orbit_0"]["n_observations"] == 780
    assert matrix_cases["orbit_1"]["n_observations"] == 781
    assert matrix_cases["both_secondary_eclipses"]["n_observations"] == 374
    assert len(matrix["blocked_cases"]) == 6

    comparisons = {item.window: item for item in report.comparisons}
    all_points = comparisons["all"]
    assert all_points.rmse_ppm == pytest.approx(0.7466871433706161, abs=1.0e-9)
    assert all_points.maximum_absolute_error_ppm == pytest.approx(8.232330020563339, abs=1.0e-9)
    assert all_points.correlation > 0.99999999

    for name in ("eclipse_0", "eclipse_1"):
        eclipse = comparisons[name]
        assert eclipse.rmse_ppm == pytest.approx(1.51711617485, abs=1.0e-9)
        assert eclipse.maximum_absolute_error_ppm < 8.3
        assert eclipse.correlation > 0.999999
        assert eclipse.reference_peak_excess_ppm == pytest.approx(6600.0, abs=1.0e-9)

    assert report.map_summary.coefficient_count == 9
    assert report.map_summary.peak_longitude_degrees == pytest.approx(30.0, abs=1.0e-10)
    assert report.map_summary.peak_latitude_degrees == pytest.approx(16.0, abs=1.0e-10)
    assert report.map_summary.map_maximum_intensity_percent == pytest.approx(
        0.7497995232147557, abs=1.0e-12
    )

    report_path = tmp_path / "output" / "frozen_wasp43b_report.json"
    comparison_path = tmp_path / "output" / "frozen_wasp43b_comparison.npz"
    assert report_path.is_file()
    assert comparison_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "passed"
    assert saved["comparisons"][2]["window"] == "eclipse_0"
    assert saved["reference_matrix"]["status"] == "passed"
    assert len(saved["reference_matrix"]["cases"]) == 11
