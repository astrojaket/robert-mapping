"""Checks for saved-result case selection and map reconstruction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPO_ROOT / "UNITY_Theory"
CODE_ROOT = STUDY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import build_case_folders as cases  # noqa: E402


def test_case_noise_filters_use_saved_input_noise() -> None:
    run_inputs = json.loads(
        (STUDY_ROOT / "reference/expected_noise_ensemble/run_inputs.json").read_text()
    )
    snapshot = run_inputs["input_snapshot"]
    target_names = [target["name"] for target in snapshot["targets"]]
    noises = cases._noise_map(
        "JWST_Noise", target_names, snapshot, run_inputs["run_settings"]
    )
    source_rows = list(
        csv.DictReader(
            (STUDY_ROOT / "reference/expected_noise_ensemble/recovery_trials.csv").open()
        )
    )

    selected = cases._selected_rows(source_rows, target_names, noises)

    assert len(selected) == 1200
    assert len({row["noise_input_ppm"] for row in selected if row["target"] == "WASP-12 b"}) == 1
    assert cases._selected_rows(
        [
            {"target": "x", "noise_input_ppm": "50", "trial": "0"},
            {"target": "x", "noise_input_ppm": "100", "trial": "1"},
        ],
        ["x"],
        {"x": 100.0},
    ) == [{"target": "x", "noise_input_ppm": "100", "trial": "1"}]


def test_case_inputs_store_one_noise_level_and_no_null_global_noise() -> None:
    target = {"name": "Target b", "noise_ppm": 50.0}
    snapshot = {
        "targets": [target],
        "noise_levels_ppm": [50.0, 100.0],
        "assumptions": {"single_realization": "historical text"},
    }

    baseline = cases._prepare_case_input(
        snapshot, ["Target b"], {"Target b": 50.0}, ["temperature_scaled"], "50ppm", expected_trials=7
    )
    jwst = cases._prepare_case_input(
        snapshot, ["Target b"], {"Target b": 43.2}, ["temperature_scaled"], "JWST_Noise", expected_trials=7
    )

    assert baseline["noise_levels_ppm"] == [50.0]
    assert baseline["targets"][0]["noise_levels_ppm"] == [50.0]
    assert "noise_levels_ppm" not in jwst
    assert jwst["targets"][0]["noise_levels_ppm"] == [43.2]
    assert "single_realization" not in jwst["assumptions"]
    assert "7 saved draws" in jwst["assumptions"]["realizations"]


def test_reconstructed_map_matches_saved_trial_zero_metrics() -> None:
    target_name = "HAT-P-30 b"
    run_inputs = json.loads((STUDY_ROOT / "reference/study/run_inputs.json").read_text())
    target = next(target for target in run_inputs["input_snapshot"]["targets"] if target["name"] == target_name)
    source_rows = csv.DictReader((STUDY_ROOT / "reference/study/recovery_trials.csv").open())
    row = next(
        row
        for row in source_rows
        if row["target"] == target_name
        and row["noise_input_ppm"] == "50.0"
        and row["scenario"] == "temperature_scaled"
        and row["trial"] == "0"
        and row["fit_scope"] == "full_window"
    )
    row["depth_ppm"] = str(target["depth_ppm"])

    coefficients, details = cases._recovered_map(row, target, 2)
    metrics = cases._validate_map_metrics(row, coefficients, truth=False, ydeg=2)

    assert coefficients.shape == (9,)
    assert np.all(np.isfinite(coefficients))
    assert np.isclose(
        details["full_phase_flux_scale"],
        float(row["best_uniform_map_amplitude_scale"]) + float(row["best_spot_amplitude_scale"]),
    )
    assert abs(metrics["minimum_intensity"] - float(row["best_map_minimum_intensity"])) <= 2.0e-6


def test_finegrid_source_bundle_is_packaged() -> None:
    source_root = STUDY_ROOT / "reference"
    run_inputs, summary, trials, examples = cases._bundle_paths(source_root, "wasp12_finegrid")

    assert run_inputs.is_file()
    assert summary.is_file()
    assert trials.is_file()
    assert examples.is_dir()


def test_reference_profile_pointer_uses_tracked_archive_parts() -> None:
    case_root = STUDY_ROOT / "50ppm"
    pointer = cases._profile_source(
        source_label="reference",
        source_root=STUDY_ROOT / "reference",
        bundle_name="study",
        output_dir=case_root,
        case_name="50ppm",
        selected_rows=[],
        noise_by_target={},
    )

    assert pointer["relative_to"] == "case root"
    assert pointer["restore_command"] == "python UNITY_Theory/archive_profiles.py --restore --study study"
    assert pointer["profiles_files"]
    assert all((case_root / path).is_file() for path in pointer["profiles_files"])
