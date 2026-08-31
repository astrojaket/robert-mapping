"""Tests for the sampler-free WASP-18b 25-bin validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from robert_mapping.benchmark.wasp18b import (
    load_wasp18b_25bin,
    run_wasp18b_benchmark,
)
from robert_mapping.physics import secondary_eclipse_design_matrix


@pytest.fixture()
def input_path(tmp_path: Path) -> Path:
    """Write a small ThERESA-format input with no external data dependency."""

    period = 0.941452382
    transit_time = 2459802.4078798564
    time = transit_time + np.linspace(0.05, 0.95, 180) * period
    wavelength = np.linspace(0.8937745957740919, 2.7875921576025764, 25)
    uniform_design = secondary_eclipse_design_matrix(
        time,
        period,
        3.48023,
        84.35320,
        0.09783,
        0,
        transit_time,
        theta0=np.pi,
        rotation_period=period,
        subobserver_lat=np.deg2rad(5.64680),
        angle_unit="deg",
    )[:, 0]
    scale = np.linspace(900.0, 1700.0, wavelength.size)
    signal_ppm = scale[:, None] * uniform_design[None, :]
    error_ppm = np.full_like(signal_ppm, 100.0)
    path = tmp_path / "spec_lambin_25.npz"
    np.savez(path, arr_0=time, arr_1=wavelength, arr_2=signal_ppm, arr_3=error_ppm)
    return path


def test_load_theresa_wasp18b_25bin_input(input_path: Path) -> None:
    data = load_wasp18b_25bin(input_path)

    assert data.n_observations == 180
    assert data.n_bins == 25
    assert data.flux.shape == (180, 25)
    assert data.flux_err.shape == (180, 25)
    assert np.all(np.isfinite(data.flux))
    assert np.all(data.flux_err > 0.0)
    assert data.wavelength_um[0] == np.float64(0.8937745957740919)
    assert data.wavelength_um[-1] == np.float64(2.7875921576025764)


def test_wasp18b_benchmark_writes_summary_profiles_and_plot(
    tmp_path: Path, input_path: Path
) -> None:
    report = run_wasp18b_benchmark(
        input_path,
        tmp_path / "results",
        bin_indices=(0, 12, 24),
        quadrature_radial=4,
        quadrature_azimuth=16,
        profile_nlon=61,
        profile_nlat=31,
    )

    assert report.status == "complete"
    assert report.n_observations == 180
    assert report.n_bins == 25
    assert len(report.bins) == 3
    assert all(np.isfinite(item.delta_bic_mapped_preference) for item in report.bins)
    assert all(0.0 <= item.mapped_negative_fraction <= 1.0 for item in report.bins)
    assert report.numerical_settings["parallel"] is False
    assert report.numerical_settings["cpu_count"] == 1

    for key in (
        "summary_json",
        "results_csv",
        "longitude_profiles_csv",
        "mapped_predictions_npz",
        "overview_png",
        "overview_pdf",
    ):
        assert Path(report.files[key]).is_file()

    summary = json.loads(Path(report.files["summary_json"]).read_text(encoding="utf-8"))
    assert summary["input"]["selected_bins"] == [0, 12, 24]
    assert len(summary["bins"]) == 3

    with Path(report.files["results_csv"]).open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    with Path(report.files["longitude_profiles_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        profile_rows = list(csv.DictReader(handle))
        assert len(profile_rows) == 3 * 61
        assert {int(row["bin_index"]) for row in profile_rows} == {0, 12, 24}

    saved = np.load(report.files["mapped_predictions_npz"], allow_pickle=False)
    assert saved["mapped_prediction_ppm"].shape == (3, 180)
