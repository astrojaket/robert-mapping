"""Tests for the compact WASP-121b validation comparison report."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


_TOOL_PATH = Path(__file__).parents[1] / "tools" / "report_wasp121b_validation.py"
_SPEC = importlib.util.spec_from_file_location("report_wasp121b_validation", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_wasp121b_validation_comparison = _MODULE.make_wasp121b_validation_comparison


def _write_products(directory: Path, channel: str, median: float, jitter: float) -> None:
    directory.mkdir()
    (directory / "production_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "target": f"wasp121b-nirspec-{channel.lower()}-validation",
                "posterior_draws": 2000,
                "divergences": 0,
                "maximum_rhat": 1.01,
                "residual_rms_ppm": 130.0 if channel == "NRS1" else 155.0,
                "hotspot_longitude_degrees_east": {
                    "q16": median - 0.1,
                    "median": median,
                    "q84": median + 0.1,
                },
                "longitude_comparison": {
                    "published_reference": {
                        "label": f"Mikal-Evans {channel}",
                        "median": 3.36 if channel == "NRS1" else 2.66,
                        "sigma": 0.11 if channel == "NRS1" else 0.12,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (directory / "fit_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "project": f"wasp121b-nirspec-{channel.lower()}-validation",
                "residual_rms": (130.0 if channel == "NRS1" else 155.0) / 1.0e6,
                "white_jitter_mean": jitter / 1.0e6,
                "white_jitter_standard_deviation": 2.0 / 1.0e6,
            }
        ),
        encoding="utf-8",
    )


def test_wasp121b_validation_comparison_writes_summary_and_figures(tmp_path: Path) -> None:
    nrs1 = tmp_path / "nrs1"
    nrs2 = tmp_path / "nrs2"
    _write_products(nrs1, "NRS1", 3.0, 91.0)
    _write_products(nrs2, "NRS2", 2.4, 79.0)

    output = tmp_path / "comparison"
    summary = make_wasp121b_validation_comparison(nrs1, nrs2, output)

    assert summary["status"] == "complete"
    assert [entry["channel"] for entry in summary["channels"]] == ["NRS1", "NRS2"]
    assert summary["channels"][0]["robert_mapping"]["white_jitter_ppm"] == 91.0
    assert (
        abs(
            summary["channels"][0]["differences"]["hotspot_median_minus_published_deg"]
            + 0.36
        )
        < 1.0e-12
    )
    for name in (
        "wasp121b_validation_comparison.png",
        "wasp121b_validation_comparison.pdf",
        "wasp121b_validation_comparison.json",
    ):
        assert (output / name).is_file()

    saved = json.loads(
        (output / "wasp121b_validation_comparison.json").read_text(encoding="utf-8")
    )
    assert saved["channels"][1]["mikal_evans_2023"]["hotspot_offset_deg_east"] == 2.66
