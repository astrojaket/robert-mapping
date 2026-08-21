"""Focused tests for the band-integrated temperature conversion."""

from __future__ import annotations

import numpy as np
import pytest

from robert_mapping.benchmark.temperature import (
    BandpassTemperatureConverter,
    band_integrated_contrast,
    blackbody_stellar_radiance,
    brightness_temperature_from_contrast,
)


def _spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength = np.linspace(2.0, 5.0, 41)
    stellar = blackbody_stellar_radiance(
        wavelength, 6_000.0, wavelength_unit="micron"
    )
    weights = 1.0 + 0.2 * np.cos(wavelength)
    return wavelength, stellar, weights


def test_blackbody_helper_has_expected_units_and_shape() -> None:
    wavelength = np.array([2.0, 4.0, 5.0])
    per_m = blackbody_stellar_radiance(
        wavelength, 6_000.0, wavelength_unit="micron"
    )
    per_micron = blackbody_stellar_radiance(
        wavelength, 6_000.0, wavelength_unit="micron", radiance_unit="per_micron"
    )
    assert per_m.shape == wavelength.shape
    assert np.all(per_m > 0.0)
    np.testing.assert_allclose(per_micron, per_m / 1.0e6, rtol=1.0e-14)


def test_band_conversion_recovers_map_shaped_temperatures() -> None:
    wavelength, stellar, weights = _spectrum()
    converter = BandpassTemperatureConverter(
        wavelength,
        stellar,
        weights,
        0.1125,
        wavelength_unit="micron",
        temperature_grid_k=np.linspace(500.0, 8_000.0, 7_501),
    )
    true_temperatures = np.array([[1_800.0, 2_700.0], [3_600.0, 5_100.0]])
    contrast = converter.contrast_from_temperature(true_temperatures)
    recovered = converter.temperature_from_contrast(contrast)
    np.testing.assert_allclose(recovered, true_temperatures, rtol=2.0e-6, atol=0.02)
    np.testing.assert_allclose(
        brightness_temperature_from_contrast(
            contrast,
            wavelength,
            stellar,
            weights,
            0.1125,
            wavelength_unit="micron",
            temperature_grid_k=np.linspace(500.0, 8_000.0, 7_501),
        ),
        true_temperatures,
        rtol=2.0e-6,
        atol=0.02,
    )


def test_contrast_includes_radius_ratio_and_is_monotonic() -> None:
    wavelength, stellar, weights = _spectrum()
    temperatures = np.array([1_000.0, 2_000.0, 3_000.0])
    contrast_small = band_integrated_contrast(
        temperatures, wavelength, stellar, weights, 0.1, wavelength_unit="micron"
    )
    contrast_large = band_integrated_contrast(
        temperatures, wavelength, stellar, weights, 0.2, wavelength_unit="micron"
    )
    np.testing.assert_allclose(contrast_large, contrast_small * 4.0, rtol=1.0e-14)
    assert np.all(np.diff(contrast_small) > 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stellar_radiance": np.ones(3), "weights": np.ones(4)}, "same shape"),
        ({"stellar_radiance": np.zeros(3), "weights": np.ones(3)}, "strictly positive"),
        ({"stellar_radiance": np.ones(3), "weights": np.zeros(3)}, "positive sum"),
        ({"stellar_radiance": np.ones(3), "weights": np.ones(3), "radius_ratio": 0.0}, "radius_ratio"),
    ],
)
def test_converter_rejects_invalid_spectrum(kwargs: dict[str, object], message: str) -> None:
    wavelength = np.array([2.0, 3.0, 4.0])
    values: dict[str, object] = {
        "stellar_radiance": np.ones(3),
        "weights": np.ones(3),
        "radius_ratio": 0.1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        BandpassTemperatureConverter(wavelength, wavelength_unit="micron", **values)


def test_converter_rejects_non_monotonic_grid_and_out_of_range_map() -> None:
    wavelength, stellar, weights = _spectrum()
    with pytest.raises(ValueError, match="strictly increasing"):
        BandpassTemperatureConverter(
            wavelength,
            stellar,
            weights,
            0.1,
            wavelength_unit="micron",
            temperature_grid_k=[500.0, 1_000.0, 900.0],
        )
    converter = BandpassTemperatureConverter(
        wavelength,
        stellar,
        weights,
        0.1,
        wavelength_unit="micron",
        temperature_grid_k=[500.0, 1_000.0, 2_000.0],
    )
    with pytest.raises(ValueError, match="outside"):
        converter.temperature_from_contrast([converter.contrast_grid[-1] * 1.01])
    with pytest.raises(ValueError, match="strictly positive"):
        converter.temperature_from_contrast([0.0])
