from __future__ import annotations

import csv
import json

import numpy as np

from robert_mapping.benchmark.production_inputs import (
    prepare_wasp43b_eclipse_windows,
    prepare_wasp43b_full_phase_curve,
    prepare_wasp178b_fixed_detrending,
)


def test_prepare_wasp178b_fixed_detrending(tmp_path) -> None:
    source = tmp_path / "input.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "time_mjd_tdb",
                "relative_flux",
                "relative_flux_err",
                "systematics_model",
            )
        )
        writer.writerow((1.0, 0.998, 0.001, -0.002))
        writer.writerow((2.0, 0.999, 0.001, -0.001))

    result = prepare_wasp178b_fixed_detrending(source, tmp_path / "out")
    table = np.genfromtxt(result, delimiter=",", names=True)
    assert np.allclose(table["relative_flux"], 1.0)
    provenance = json.loads((result.parent / "input_provenance.json").read_text())
    assert provenance["n_observations"] == 2
    assert provenance["operation"] == "relative_flux - systematics_model"


def test_prepare_wasp43b_eclipse_windows(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    time = np.linspace(0.0, 1.0, 21)
    clean = np.ones(21)
    total = clean + 0.001
    observed = total + np.linspace(-1.0e-4, 1.0e-4, 21)
    error = np.full(21, 1.0e-4)
    for name, values in {
        "w43b_time.npy": time,
        "w43b_flux.npy": observed,
        "w43b_error.npy": error,
        "sim_flux_clean.npy": clean,
        "sim_flux_total.npy": total,
    }.items():
        np.save(source / name, values)
    result = prepare_wasp43b_eclipse_windows(
        source,
        tmp_path / "out",
        transit_time=0.0,
        period_days=1.0,
        half_window_phase=0.1,
    )
    table = np.genfromtxt(result, delimiter=",", names=True)
    assert table.size == 4
    assert np.allclose(table["flux"], observed[8:12] - 0.001)


def test_prepare_wasp43b_full_phase_curve(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    time = np.linspace(0.0, 2.0, 21)
    clean = 1.0 + 0.001 * np.cos(np.pi * time)
    total = clean + 0.002
    observed = total + np.linspace(-1.0e-4, 1.0e-4, 21)
    error = np.full(21, 1.0e-4)
    for name, values in {
        "w43b_time.npy": time,
        "w43b_flux.npy": observed,
        "w43b_error.npy": error,
        "sim_flux_clean.npy": clean,
        "sim_flux_total.npy": total,
    }.items():
        np.save(source / name, values)

    result = prepare_wasp43b_full_phase_curve(
        source,
        tmp_path / "out",
        transit_time=0.0,
        period_days=1.0,
    )
    table = np.genfromtxt(result, delimiter=",", names=True)
    assert table.size == 21
    assert np.allclose(table["phase"], time)
    assert np.allclose(table["flux"], observed - 0.002)
    provenance = json.loads(
        (result.parent / "full_phase_input_provenance.json").read_text()
    )
    assert provenance["selection"] == "all saved samples; no phase selection"
    assert provenance["n_observations"] == 21
