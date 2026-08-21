"""Frozen starry 1.0.0 forward-matrix checks."""

from pathlib import Path

from robert_mapping.benchmark import run_starry_v1_matrix


REFERENCE = Path(__file__).parents[1] / "reference_data" / "starry_v1"


def test_starry_v1_matrix_has_no_unexpected_failures(tmp_path: Path) -> None:
    report = run_starry_v1_matrix(REFERENCE, tmp_path)
    assert report["status"] == "pass"
    assert report["runtime_imports_starry"] is False
    assert report["failed_cases"] == 0
    assert report["passed_cases"] == 7
    assert report["blocked_cases"] == 1
    assert {case["status"] for case in report["cases"]} == {"pass", "blocked"}
    assert (tmp_path / "starry_v1_matrix_report.json").is_file()
