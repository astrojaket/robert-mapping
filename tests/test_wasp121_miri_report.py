"""WASP-121b MIRI report checks."""

from __future__ import annotations

import numpy as np

from robert_mapping.benchmark.production_report import (
    _latitude_profile_weights,
    _longitude_comparison_references,
    _published_longitude_reference,
)


def test_wasp121_miri_uses_cosine_squared_profile_weighting() -> None:
    latitude = np.array([-60.0, 0.0, 60.0])
    weights = _latitude_profile_weights("wasp121b-miri-lrs", latitude)

    np.testing.assert_allclose(weights, np.array([0.25, 1.0, 0.25]))


def test_wasp121_miri_reference_keeps_asymmetric_errors() -> None:
    published = _published_longitude_reference("wasp121b-miri-lrs")
    assert published is not None

    references, summary = _longitude_comparison_references(
        np.array([3.0, 5.0, 8.0]),
        include_hammond=False,
        published_reference=published,
    )

    literature = next(item for item in references if item["kind"] == "literature")
    assert literature["longitude"] == 4.8
    assert literature["lower"] == 2.0
    assert literature["upper"] == 7.5
    assert summary["published_reference"]["lower"] == 2.0
    assert summary["published_reference"]["upper"] == 7.5
