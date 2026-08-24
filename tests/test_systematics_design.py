"""Focused tests for the modular systematics design layer."""

from __future__ import annotations

import numpy as np
import pytest

from robert_mapping.systematics import build_systematics_design


def test_two_segments_have_offsets_and_segment_reset_ramps() -> None:
    time = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    segments = np.array(["A", "A", "A", "B", "B", "B"])

    result = build_systematics_design(
        time,
        segment_ids=segments,
        polynomial_order=2,
        ramp_timescale=2.0,
    )

    assert result.names == (
        "offset[segment=A]",
        "offset[segment=B]",
        "time",
        "time^2",
        "ramp[segment=A]",
        "ramp[segment=B]",
    )
    assert result.matrix.shape == (6, 6)
    assert np.allclose(result.matrix[:3, 0], 1.0)
    assert np.allclose(result.matrix[:3, 1], 0.0)
    assert np.allclose(result.matrix[3:, 0], 0.0)
    assert np.allclose(result.matrix[3:, 1], 1.0)
    assert np.allclose(
        result.matrix[[0, 1, 2], 4], np.exp(-np.array([0.0, 1.0, 2.0]) / 2.0)
    )
    assert np.allclose(
        result.matrix[[3, 4, 5], 5], np.exp(-np.array([0.0, 1.0, 2.0]) / 2.0)
    )
    assert np.allclose(result.matrix[:, 2], (time - 6.0) / 6.0)


def test_named_and_array_auxiliary_regressors_are_preserved() -> None:
    time = np.array([1.0, 2.0, 3.0, 4.0])
    named = {
        "airmass": np.array([1.1, 1.2, 1.3, 1.4]),
        "detector_x": np.array([-1.0, -0.5, 0.5, 1.0]),
    }
    named_result = build_systematics_design(time, auxiliary_regressors=named)
    assert named_result.names == ("offset", "auxiliary[airmass]", "auxiliary[detector_x]")
    assert np.allclose(named_result.matrix[:, 1:], np.column_stack(tuple(named.values())))

    array_result = build_systematics_design(
        time,
        auxiliary_regressors=np.column_stack((time, time**2)),
        auxiliary_names=("elapsed", "elapsed_squared"),
    )
    assert array_result.names == ("offset", "auxiliary[elapsed]", "auxiliary[elapsed_squared]")
    assert np.allclose(array_result.matrix[:, 1], time)
    assert np.allclose(array_result.matrix[:, 2], time**2)


def test_offsets_can_be_disabled_without_dropping_other_terms() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    result = build_systematics_design(
        time,
        include_offsets=False,
        polynomial_order=1,
        ramp_timescale=2.0,
        auxiliary_regressors=np.arange(4.0),
        auxiliary_names=("detector",),
    )

    assert result.names == ("time", "ramp", "auxiliary[detector]")
    assert result.matrix.shape == (4, 3)
    assert np.allclose(result.matrix[:, 0], (time - 1.5) / 1.5)
    assert np.allclose(result.matrix[:, 2], time)


def test_time_polynomial_can_use_days_from_midpoint() -> None:
    time = np.array([10.0, 10.5, 11.0])
    result = build_systematics_design(
        time,
        polynomial_order=2,
        standardize_time=False,
    )

    centred = time - np.mean(time)
    assert np.allclose(result.matrix[:, 1], centred)
    assert np.allclose(result.matrix[:, 2], centred**2)


def test_disabling_offsets_requires_another_nuisance_term() -> None:
    with pytest.raises(ValueError, match="at least one nuisance column"):
        build_systematics_design(np.arange(3.0), include_offsets=False)


def test_include_offsets_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="include_offsets"):
        build_systematics_design(np.arange(3.0), include_offsets=1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"polynomial_order": -1}, "polynomial_order"),
        ({"ramp_timescale": 0.0}, "ramp_timescale"),
        ({"segment_ids": [0, 1]}, "segment_ids"),
        ({"auxiliary_regressors": [[1.0], [2.0], [np.nan]]}, "finite"),
        ({"standardize_time": 1}, "standardize_time"),
    ],
)
def test_systematics_design_validates_inputs(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_systematics_design(np.arange(3.0), **kwargs)
