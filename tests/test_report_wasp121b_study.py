"""Focused tests for the combined WASP-121b study report helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_TOOL = Path(__file__).parents[1] / "tools" / "report_wasp121b_study.py"
_SPEC = importlib.util.spec_from_file_location("report_wasp121b_study", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_acf_is_normalized_at_zero_lag() -> None:
    values = np.array([1.0, -1.0, 1.0, -1.0])
    acf = _MODULE._acf(values, maximum_lag=2)

    assert acf[0] == 1.0
    np.testing.assert_allclose(acf[1:], np.array([-0.75, 0.5]))


def test_ou_innovations_reduce_to_independent_residuals_at_zero_amplitude() -> None:
    residual = np.array([0.5, -1.0, 2.0])
    sigma = np.ones(3)
    time = np.array([0.0, 1.0, 2.0]) / 86400.0

    innovations = _MODULE._ou_standardized_innovations(
        residual,
        sigma,
        time,
        amplitude=0.0,
        timescale=100.0,
        jitter=0.0,
    )

    np.testing.assert_allclose(innovations, residual)


def test_ou_innovations_are_finite_for_irregular_cadence() -> None:
    innovations = _MODULE._ou_standardized_innovations(
        np.array([1.0, 0.3, -0.2, 0.1]) * 1.0e-4,
        np.full(4, 5.0e-5),
        np.array([0.0, 20.0, 75.0, 140.0]) / 86400.0,
        amplitude=8.0e-5,
        timescale=900.0,
        jitter=3.0e-5,
    )

    assert innovations.shape == (4,)
    assert np.all(np.isfinite(innovations))
