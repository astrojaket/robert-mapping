"""Small tests for the top-level UNITY reproduction commands."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


RUN_PATH = Path(__file__).resolve().parents[1] / "run.py"
SPEC = importlib.util.spec_from_file_location("unity_run", RUN_PATH)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


def test_ensemble_simulation_builds_runner_and_plot_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, tuple[object, ...]]] = []
    monkeypatch.setattr(run, "child", lambda script, *args: calls.append((script, args)))

    run.simulate("ensemble", trials=2)

    assert len(calls) == 2
    runner, runner_args = calls[0]
    assert runner == run.CODE / "unity_injection_recovery.py"
    assert runner_args == (
        "--inputs",
        run.DATA / "inputs" / "unity_expected_noise.json",
        "--output",
        run.GENERATED / "expected_noise_ensemble",
        "--trials",
        2,
        "--exposure-subsamples",
        8,
        "--disk-radial",
        32,
        "--disk-azimuth",
        128,
        "--chunk-size",
        128,
    )

    plotter, plot_args = calls[1]
    assert plotter == run.CODE / "plot_unity_expected_ensemble.py"
    assert plot_args == (
        "--trials",
        run.GENERATED / "expected_noise_ensemble" / "recovery_trials.csv",
        "--inputs",
        run.DATA / "inputs" / "unity_expected_noise.json",
        "--output-dir",
        run.GENERATED / "expected_noise_ensemble_deliverables",
        "--profiles",
        run.GENERATED / "expected_noise_ensemble" / "profiles.npz",
        "--expected-draws",
        2,
    )


def test_verify_reports_a_missing_package_checksum_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "HERE", tmp_path)

    with pytest.raises(FileNotFoundError, match="SHA256SUMS.json"):
        run.verify()


def test_cli_rejects_nonpositive_trial_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(RUN_PATH), "simulate", "--trials", "0"])

    with pytest.raises(SystemExit) as error:
        run.main()

    assert error.value.code == 2


@pytest.mark.parametrize(
    "arguments",
    [["all", "--trials", "1"], ["all", "--case", "baseline"]],
)
def test_all_rejects_options_it_cannot_apply(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", [str(RUN_PATH), *arguments])

    with pytest.raises(SystemExit) as error:
        run.main()

    assert error.value.code == 2
