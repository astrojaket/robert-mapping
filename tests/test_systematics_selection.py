"""Focused tests for deterministic raw-systematics candidate selection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robert_mapping.benchmark.systematics_selection import (
    compare_systematics_candidates,
    run_systematics_selection,
)
from robert_mapping.config import mapping_config_from_dict
from robert_mapping.data import LightCurve


def _selection_config(metric: str = "bic"):
    return mapping_config_from_dict(
        {
            "systematics_selection": {
                "enabled": True,
                "metric": metric,
                "validation_fraction": 0.2,
                "min_training_points": 20,
                "candidates": [
                    {"name": "corrected", "mode": "corrected"},
                    {
                        "name": "additive_ramp",
                        "mode": "additive",
                        "exponential_ramp": True,
                        "ramp_timescale_hours": 4.0,
                    },
                    {
                        "name": "multiplicative_ramp",
                        "mode": "multiplicative",
                        "exponential_ramp": True,
                        "ramp_timescale_hours": 4.0,
                    },
                ],
            }
        }
    )


def _ramped_curve() -> LightCurve:
    time = np.linspace(0.0, 1.0, 100)
    ramp = 0.002 * np.exp(-time / (4.0 / 24.0))
    flux = 1.0 + ramp
    error = np.full(time.size, 2.0e-5)
    return LightCurve(time=time, flux=flux, flux_err=error, time_unit="day")


def test_bic_selects_the_ramp_candidate_without_sampling() -> None:
    report = compare_systematics_candidates(_selection_config(), _ramped_curve())

    assert report.status == "complete"
    assert report.chosen_candidate == "additive_ramp"
    assert report.n_training == 80
    assert report.n_held_out == 20
    assert report.to_dict()["map_detection_evidence"] is None
    assert report.to_dict()["conditional_hotspot_location"] is None
    assert all(score.parameter_count >= 1 for score in report.scores)


def test_held_out_selection_is_deterministic_and_keeps_candidate_order() -> None:
    config = _selection_config(metric="held_out_elpd")
    first = compare_systematics_candidates(config, _ramped_curve())
    second = compare_systematics_candidates(config, _ramped_curve())

    assert first.to_dict() == second.to_dict()
    assert [score.name for score in first.scores] == [
        "corrected",
        "additive_ramp",
        "multiplicative_ramp",
    ]


def test_default_candidate_set_is_small_and_bounded() -> None:
    config = mapping_config_from_dict({})
    assert len(config.systematics_selection.candidates) == 3
    assert config.systematics_selection.metric == "bic"


def test_corrected_candidate_rejects_nuisance_terms() -> None:
    with pytest.raises(ValueError, match="corrected candidates"):
        mapping_config_from_dict(
            {
                "systematics_selection": {
                    "candidates": [
                        {
                            "name": "bad_corrected",
                            "mode": "corrected",
                            "polynomial_order": 1,
                        }
                    ]
                }
            }
        )


def test_run_writes_machine_readable_selection_products(tmp_path: Path) -> None:
    config = _selection_config()
    config = replace(
        config,
        output=replace(
            config.output,
            directory=tmp_path / "selection",
            save_resolved_config=False,
        ),
    )
    report = run_systematics_selection(config, curve=_ramped_curve())

    output = tmp_path / "selection"
    assert report.output_directory == str(output.resolve())
    assert (output / "systematics_selection.json").exists()
    assert (output / "systematics_selection_candidates.csv").exists()
    assert (output / "systematics_selection.json").read_text(encoding="utf-8").find(
        '"map_detection_evidence": null'
    ) >= 0


def test_segmented_ramp_design_resets_for_each_segment() -> None:
    base = _ramped_curve()
    segments = np.repeat(("A", "B"), 50)
    curve = LightCurve(
        time=base.time,
        flux=base.flux,
        flux_err=base.flux_err,
        time_unit="day",
        segments=segments,
    )
    config = mapping_config_from_dict(
        {
            "systematics_selection": {
                "enabled": True,
                "candidates": [
                    {
                        "name": "segmented_ramp",
                        "mode": "additive",
                        "fit_offset": True,
                        "exponential_ramp": True,
                        "ramp_timescale_hours": 4.0,
                        "segment_column": "visit",
                    }
                ],
            }
        }
    )
    report = compare_systematics_candidates(config, curve)
    assert report.scores[0].design_columns == (
        "baseline",
        "offset[segment=B]",
        "ramp[segment=A]",
        "ramp[segment=B]",
    )
