"""Serial and process-parallel checks for the UNITY runner."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNITY_CODE = ROOT / "UNITY_Theory" / "code"
if str(UNITY_CODE) not in sys.path:
    sys.path.insert(0, str(UNITY_CODE))

from unity_injection_recovery import run_study  # noqa: E402
from robert_mapping.compute import resolve_plan  # noqa: E402


def _small_two_target_input(path: Path) -> Path:
    source = ROOT / "UNITY_Theory" / "data" / "inputs" / "unity_expected_noise.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["targets"] = payload["targets"][:2]
    payload["scenarios"] = ["temperature_scaled", "uniform_null"]
    analysis = payload.setdefault("analysis", {})
    analysis.update(
        longitude_grid_step_degrees=30.0,
        width_grid_degrees=[40.0],
        latitude_grid_degrees=[0.0],
        timing_grid_seconds=[0.0],
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_two_target_parallel_run_matches_serial_outputs(tmp_path: Path) -> None:
    inputs = _small_two_target_input(tmp_path / "inputs.json")
    serial_output = tmp_path / "serial"
    parallel_output = tmp_path / "parallel"

    serial = run_study(
        inputs,
        serial_output,
        trials=2,
        exposure_subsamples=2,
        disk_radial=8,
        disk_azimuth=16,
        chunk_size=64,
        max_cpus=1,
        workers=1,
        threads=1,
    )
    plan = resolve_plan(max_cpus=2, workers=2, threads=1)
    parallel = run_study(
        inputs,
        parallel_output,
        trials=2,
        exposure_subsamples=2,
        disk_radial=8,
        disk_azimuth=16,
        chunk_size=64,
        max_cpus=2,
        workers=2,
        threads=1,
    )

    serial_rows = list(csv.DictReader((serial_output / "recovery_trials.csv").open()))
    parallel_rows = list(csv.DictReader((parallel_output / "recovery_trials.csv").open()))
    assert parallel_rows == serial_rows
    assert [row["target"] for row in parallel_rows[:2]] == [
        "HAT-P-30 b",
        "HAT-P-30 b",
    ]
    assert [row["random_seed"] for row in parallel_rows] == [
        row["random_seed"] for row in serial_rows
    ]

    with np.load(serial_output / "profiles.npz", allow_pickle=False) as serial_profiles:
        with np.load(parallel_output / "profiles.npz", allow_pickle=False) as parallel_profiles:
            assert parallel_profiles.files == serial_profiles.files
            for key in serial_profiles.files:
                np.testing.assert_array_equal(parallel_profiles[key], serial_profiles[key])

    assert serial["trial_summaries"] == parallel["trial_summaries"]
    assert [item["target"] for item in parallel["targets"]] == [
        "HAT-P-30 b",
        "KELT-7 b",
    ]
    assert serial["execution_plan"]["worker_process_ids"] == [os.getpid()]
    parallel_pids = parallel["execution_plan"]["worker_process_ids"]
    assert parallel["execution_plan"]["actual_workers"] == len(parallel_pids)
    assert parallel_pids
    if plan.workers > 1:
        assert os.getpid() not in parallel_pids
    else:
        assert parallel_pids == [os.getpid()]
