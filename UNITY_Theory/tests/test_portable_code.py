"""Portability checks for the copied UNITY analysis scripts."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = REPO_ROOT / "UNITY_Theory"
CODE_ROOT = STUDY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import compute_unity_expected_noise as compute_noise  # noqa: E402
import plot_unity_expected_ensemble as plot_ensemble  # noqa: E402
import plot_unity_expected_noise as plot_expected_noise  # noqa: E402
import plot_unity_recovery as plot_recovery  # noqa: E402
import unity_injection_recovery as injection_recovery  # noqa: E402


SCRIPTS = (
    "unity_injection_recovery.py",
    "compute_unity_expected_noise.py",
    "plot_unity_expected_noise.py",
    "plot_unity_expected_ensemble.py",
    "plot_unity_recovery.py",
)


def _targets() -> list[dict[str, object]]:
    return [
        {
            "target": f"Target {index + 1}",
            "teq_k": float(1200 + 100 * index),
            "configured_noise_ppm": float(30 + index),
            "analytic_longitude_degrees": 20.0 + index,
        }
        for index in range(6)
    ]


def test_all_cli_help_runs_from_an_unrelated_working_directory(tmp_path: Path) -> None:
    for name in SCRIPTS:
        script = CODE_ROOT / name
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_code_uses_the_repository_root_and_bundled_style() -> None:
    assert injection_recovery.PROJECT_ROOT == REPO_ROOT
    assert injection_recovery.STUDY_ROOT == STUDY_ROOT
    assert compute_noise.ROOT == REPO_ROOT
    assert compute_noise.STUDY_ROOT == STUDY_ROOT
    assert compute_noise.DEFAULT_MANIFEST == (
        STUDY_ROOT / "data" / "expected_noise_calculations" / "online" / "manifest.json"
    )
    assert compute_noise.DEFAULT_EVIDENCE == (
        STUDY_ROOT
        / "data"
        / "expected_noise_inputs"
        / "pid10786_detector_magnitudes_evidence.json"
    )
    assert compute_noise.DEFAULT_JSON == (
        STUDY_ROOT / "generated" / "expected_noise_calculations" / "expected_noise.json"
    )
    assert plot_ensemble.DEFAULT_INPUTS == STUDY_ROOT / "data" / "inputs" / "unity_expected_noise.json"
    assert plot_ensemble.REFERENCE_RUN_DIR == STUDY_ROOT / "reference" / "expected_noise_ensemble"
    assert plot_ensemble._parser().parse_args([]).profiles == (
        STUDY_ROOT / "reference" / "expected_noise_ensemble" / "profiles.npz"
    )
    assert plot_ensemble._parser().parse_args([]).output_dir == (
        STUDY_ROOT / "generated" / "expected_noise_ensemble"
    )
    assert plot_recovery._parser().parse_args([]).output_dir == (
        STUDY_ROOT / "generated" / "injection_recovery"
    )
    assert plot_recovery._parser().parse_args([]).inputs == (
        STUDY_ROOT / "data" / "inputs" / "unity_inputs.json"
    )
    assert plot_recovery._parser().parse_args([]).trials == (
        STUDY_ROOT / "reference" / "study" / "recovery_trials.csv"
    )
    assert plot_recovery._parser().parse_args([]).expected_target_count == 6

    for name in SCRIPTS:
        assert "/Users/jaketaylor/" not in (CODE_ROOT / name).read_text(encoding="utf-8")
    assert plot_ensemble.STYLE_HELPER == CODE_ROOT / "wasp15b_plot_style.py"
    assert plot_expected_noise.STYLE_HELPER == CODE_ROOT / "wasp15b_plot_style.py"
    assert plot_recovery.STYLE_HELPER == CODE_ROOT / "wasp15b_plot_style.py"
    assert plot_ensemble.STYLE_HELPER.is_file()


def test_one_target_finegrid_cli_succeeds_without_population_slopes(tmp_path: Path) -> None:
    finegrid = STUDY_ROOT / "reference" / "wasp12_finegrid"
    inputs = STUDY_ROOT / "data" / "inputs" / "wasp12_finegrid.json"
    output = tmp_path / "finegrid_deliverables"
    result = subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "plot_unity_recovery.py"),
            "--trials",
            str(finegrid / "recovery_trials.csv"),
            "--inputs",
            str(inputs),
            "--run-inputs",
            str(finegrid / "run_inputs.json"),
            "--summary",
            str(finegrid / "summary.json"),
            "--profiles",
            str(finegrid / "profiles.npz"),
            "--output-dir",
            str(output),
            "--expected-target-count",
            "1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "methodology.txt").is_file()
    methodology = (output / "methodology.txt").read_text(encoding="utf-8")
    assert "This run has 1, so no population slope was calculated." in methodology
    assert not list(output.glob("population_slope_recovery_*.png"))
    assert not list(output.glob("population_slope_recovery_*.pdf"))
    assert not list(output.glob("population_slope_recovery_*.svg"))


def test_manifest_pickle_path_prefers_the_relative_path(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "online"
    nested = manifest_dir / "saved" / "target.p"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")
    (manifest_dir / "target.p").write_bytes(b"legacy")

    assert compute_noise._resolve_pickle_path("saved/target.p", manifest_dir) == nested


def test_restricted_pickle_allows_numpy_and_blocks_untrusted_globals(tmp_path: Path) -> None:
    safe_path = tmp_path / "safe.p"
    with safe_path.open("wb") as handle:
        pickle.dump({"counts": np.array([1.0, 2.0])}, handle)
    assert np.array_equal(compute_noise._load_pickle(safe_path)["counts"], [1.0, 2.0])

    unsafe_path = tmp_path / "unsafe.p"
    unsafe_path.write_bytes(pickle.dumps(os.system))
    with pytest.raises(pickle.UnpicklingError, match="Blocked pickle global"):
        compute_noise._load_pickle(unsafe_path)


def test_single_realization_note_is_adjusted_only_in_run_metadata() -> None:
    original = {
        "assumptions": {
            "single_realization": "One predeclared seed per target; no sample selection."
        }
    }

    unchanged = injection_recovery._effective_input_assumptions(original, 1)
    adjusted = injection_recovery._effective_input_assumptions(original, 100)

    assert original["assumptions"]["single_realization"] == (
        "One predeclared seed per target; no sample selection."
    )
    assert unchanged == original["assumptions"]
    assert "100 independent trials" in adjusted["single_realization"]
    assert "does not describe these results" in adjusted["single_realization"]
    assert "noise-index component stays at 0" in injection_recovery.SEED_PROVENANCE
    assert "paired standardized Gaussian draws across noise levels" in injection_recovery.SEED_PROVENANCE


def test_expected_noise_csv_normalises_six_targets() -> None:
    targets = _targets()
    rows = []
    for target in targets:
        for scope in ("full_window", "eclipse_only"):
            rows.append(
                {
                    "scenario": "temperature_scaled",
                    "target": str(target["target"]),
                    "fit_scope": scope,
                    "trial": "0",
                    "teq_k": str(target["teq_k"]),
                    "noise_ppm": str(target["configured_noise_ppm"]),
                    "injected_longitude_degrees": "20",
                    "recovered_longitude_q16_degrees": "18",
                    "recovered_longitude_median_degrees": "20",
                    "recovered_longitude_q84_degrees": "22",
                }
            )

    result = plot_expected_noise._normalise_results(rows, targets)

    assert len(result) == 12
    assert {row["target"] for row in result} == {target["target"] for target in targets}
    assert {row["fit_scope"] for row in result} == {"full_window", "eclipse_only"}


def test_reference_ensemble_csv_normalises_six_targets_and_100_draws() -> None:
    targets = _targets()
    rows = []
    for target in targets:
        for scope in plot_ensemble.SCOPES:
            for trial in range(100):
                rows.append(
                    {
                        "scenario": "temperature_scaled",
                        "target": str(target["target"]),
                        "fit_scope": scope,
                        "trial": str(trial),
                        "teq_k": str(target["teq_k"]),
                        "injected_longitude_degrees": "20",
                        "recovered_longitude_q16_degrees": "18",
                        "recovered_longitude_median_degrees": "20",
                        "recovered_longitude_q84_degrees": "22",
                    }
                )

    result = plot_ensemble._normalise_draws(rows, targets, expected_draws=100)

    assert len(result) == 6 * 2 * 100
    assert {row["target"] for row in result} == {target["target"] for target in targets}
    assert {row["trial"] for row in result} == set(range(100))
    assert {row["fit_scope"] for row in result} == {"full_window", "eclipse_only"}


def test_reference_profile_mixture_accepts_all_six_targets_and_100_draws(
    tmp_path: Path,
) -> None:
    targets = _targets()
    names = np.repeat([target["target"] for target in targets], 100)
    trials = np.tile(np.arange(100, dtype=int), len(targets))
    rows = len(names)
    profile_path = tmp_path / "profiles.npz"
    np.savez_compressed(
        profile_path,
        target=names,
        scenario=np.full(rows, "temperature_scaled"),
        trial=trials,
        fit_scope=np.full(rows, "full_window"),
        candidate_peak_longitude_degrees=np.tile(np.array([20.0, 30.0]), (rows, 1)),
        posterior_weights=np.tile(np.array([0.75, 0.25]), (rows, 1)),
    )

    mixtures, metadata = plot_ensemble._load_profile_mixtures(
        profile_path, targets, expected_draws=100
    )

    assert metadata["available"] is True
    assert len(mixtures) == 6
    assert {row["target"] for row in mixtures} == {target["target"] for target in targets}
    assert {row["n_noise_draws"] for row in mixtures} == {100}
