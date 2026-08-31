"""Tests for the sampler-free WASP-18b 25-bin validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from robert_mapping.benchmark.wasp18b import (
    load_wasp18b_25bin,
    run_wasp18b_benchmark,
)


INPUT = (
    Path(__file__).resolve().parents[1]
    / "literature_data"
    / "WASP-18b"
    / "JWST-NIRISS-SOSS"
    / "source"
    / "WASP-18b 3D Mapping Archive"
    / "theresa"
    / "inputs"
    / "spec_lambin_25.npz"
)


def test_load_published_wasp18b_25bin_input() -> None:
    data = load_wasp18b_25bin(INPUT)

    assert data.n_observations == 2719
    assert data.n_bins == 25
    assert data.flux.shape == (2719, 25)
    assert data.flux_err.shape == (2719, 25)
    assert np.all(np.isfinite(data.flux))
    assert np.all(data.flux_err > 0.0)
    assert data.wavelength_um[0] == np.float64(0.8937745957740919)
    assert data.wavelength_um[-1] == np.float64(2.7875921576025764)


def test_wasp18b_benchmark_writes_summary_profiles_and_plot(tmp_path: Path) -> None:
    report = run_wasp18b_benchmark(
        INPUT,
        tmp_path,
        bin_indices=(0, 12, 24),
        quadrature_radial=4,
        quadrature_azimuth=16,
        profile_nlon=61,
        profile_nlat=31,
    )

    assert report.status == "complete"
    assert report.n_observations == 2719
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
    assert saved["mapped_prediction_ppm"].shape == (3, 2719)
