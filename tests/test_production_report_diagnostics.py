"""Focused tests for posterior map-information diagnostics."""

from __future__ import annotations

import numpy as np

from robert_mapping.benchmark.production_report import (
    _condition_positive_rendered_maps,
    _latitude_profile_weights,
    _map_information_diagnostics,
    _longitude_comparison_references,
    _profile_peak_longitudes,
    _published_longitude_reference,
)


def test_profile_peak_longitudes_interpolates_below_grid_spacing() -> None:
    longitude = np.arange(-180.0, 181.0, 3.0)
    dayside = (longitude >= -90.0) & (longitude <= 90.0)
    centres = np.array([7.75, -12.4])
    profiles = np.stack([-(longitude - centre) ** 2 for centre in centres])

    peaks = _profile_peak_longitudes(profiles, longitude, dayside)

    np.testing.assert_allclose(peaks, centres, atol=1.0e-10)


def test_longitude_comparison_includes_hammond_and_asymmetric_robert_errors() -> None:
    offsets = np.array([6.0, 7.0, 7.5, 8.0, 9.0])

    references, summary = _longitude_comparison_references(
        offsets,
        include_hammond=True,
    )

    assert [reference["kind"] for reference in references] == [
        "substellar",
        "robert",
        "hammond",
    ]
    assert references[0]["longitude"] == 0.0
    assert references[2]["longitude"] == 7.75
    assert references[2]["lower"] == 7.39
    assert references[2]["upper"] == 8.11
    assert summary["hammond_et_al_2024"]["sigma"] == 0.36


def test_positive_rendered_map_conditioning_checks_the_full_grid() -> None:
    maps = np.ones((3, 2, 4), dtype=float)
    maps[0, 0, 0] = -5.0e-13  # accepted within the numerical tolerance
    maps[1, 1, 2] = -2.0e-12  # rejected as genuinely negative

    selected, accepted, minimum = _condition_positive_rendered_maps(maps)

    np.testing.assert_array_equal(accepted, np.array([True, False, True]))
    np.testing.assert_allclose(minimum, np.array([-5.0e-13, -2.0e-12, 1.0]))
    assert selected.shape == (2, 2, 4)
    np.testing.assert_allclose(selected[0, 0, 0], -5.0e-13)


def test_positive_rendered_map_conditioning_rejects_invalid_inputs() -> None:
    with np.testing.assert_raises(ValueError):
        _condition_positive_rendered_maps(np.ones((2, 3)))
    with np.testing.assert_raises(ValueError):
        _condition_positive_rendered_maps(np.ones((2, 3, 4)), tolerance=-1.0)
    with np.testing.assert_raises(ValueError):
        _condition_positive_rendered_maps(
            np.array([[[1.0, np.nan]]]), tolerance=1.0e-12
        )


def test_map_information_diagnostics_separates_peak_coordinates_and_latitude() -> None:
    longitude = np.linspace(-180.0, 180.0, 13)
    latitude = np.array([-60.0, 0.0, 60.0])
    maps = np.ones((4, latitude.size, longitude.size), dtype=float)
    east = int(np.flatnonzero(longitude == 30.0)[0])
    west = int(np.flatnonzero(longitude == -30.0)[0])
    maps[0, 2, east] += 2.0
    maps[1, 0, west] += 2.0
    maps[2, 2, east] += 2.0
    maps[3, 1, int(np.flatnonzero(longitude == 0.0)[0])] += 2.0

    result = _map_information_diagnostics(maps, longitude, latitude)

    np.testing.assert_array_equal(
        result["peak_longitude_degrees_east"], np.array([30.0, -30.0, 30.0, 0.0])
    )
    np.testing.assert_array_equal(
        result["peak_latitude_degrees"], np.array([60.0, -60.0, 60.0, 0.0])
    )
    assert result["north_peak_fraction"] == 0.5
    assert result["south_peak_fraction"] == 0.25
    assert result["pole_peak_fraction"] == 0.75
    assert result["latitude_information_status"] == "prior_dominated"
    assert "Warning:" in result["latitude_information_warning"]
    assert result["asymmetry_interval_contains_zero"]


def test_map_information_diagnostics_rejects_non_map_shape() -> None:
    with np.testing.assert_raises(ValueError):
        _map_information_diagnostics(
            np.ones((3, 4)),
            np.arange(4.0),
            np.arange(3.0),
        )


def test_boundary_pinned_peaks_are_prior_dominated_even_with_narrow_interval() -> None:
    longitude = np.linspace(-180.0, 180.0, 13)
    latitude = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    maps = np.ones((10, latitude.size, longitude.size), dtype=float)
    east = int(np.flatnonzero(longitude == 30.0)[0])
    maps[:, -1, east] += 2.0

    result = _map_information_diagnostics(maps, longitude, latitude)

    assert result["pole_peak_fraction"] == 1.0
    assert result["latitude_information_status"] == "prior_dominated"
    assert "boundary-pinned" in result["latitude_information_warning"]
    assert "not evidence for latitude" in result["latitude_information_warning"]


def test_wasp18_profile_uses_published_cosine_squared_weighting() -> None:
    latitude = np.array([-60.0, 0.0, 60.0])

    wasp18 = _latitude_profile_weights("wasp18b-niriss", latitude)
    wasp43 = _latitude_profile_weights("hammond-wasp43b", latitude)

    np.testing.assert_allclose(wasp18, np.array([0.25, 1.0, 0.25]))
    np.testing.assert_allclose(wasp43, np.array([0.5, 1.0, 0.5]))


def test_wasp121_published_longitude_references_are_channel_specific() -> None:
    nrs1 = _published_longitude_reference("wasp121b-nirspec-nrs1")
    nrs2 = _published_longitude_reference("wasp121b-nirspec-nrs2")

    assert nrs1 is not None and nrs1["longitude"] == 3.36
    assert nrs1["sigma"] == 0.11
    assert nrs2 is not None and nrs2["longitude"] == 2.66
    assert nrs2["sigma"] == 0.12
