import numpy as np

from robert_mapping.physics.pixels import (
    equal_area_pixels,
    pixels_for_ydeg,
    harmonics_to_pixels,
    mollweide_pixels_for_ydeg,
    pixel_design_matrix,
    pixels_to_harmonics,
    starry_pixel_transforms,
)


def test_pixel_areas_are_equal_and_cover_sphere():
    pixels = equal_area_pixels(12, 24)
    assert pixels.npix == 288
    assert np.allclose(pixels.area, pixels.area[0])
    assert np.isclose(pixels.area.sum(), 4 * np.pi)


def test_pixel_harmonic_round_trip_for_low_degree():
    pixels = equal_area_pixels(20, 40)
    coefficients = np.array([1.0, -0.2, 0.5, 0.1, 0.03, -0.4, 0.2, 0.07, -0.1])
    values = harmonics_to_pixels(coefficients, pixels)
    recovered = pixels_to_harmonics(values, pixels, 2)
    assert np.allclose(recovered, coefficients, atol=2e-4)


def test_design_matrix_shape():
    pixels = equal_area_pixels(5, 10)
    assert pixel_design_matrix(pixels, 3).shape == (50, 16)


def test_hammond_quick_pixel_counts():
    assert pixels_for_ydeg(2).npix == 16
    assert pixels_for_ydeg(4).npix == 62
    assert np.isclose(pixels_for_ydeg(4).area.sum(), 4 * np.pi)


def test_starry_mollweide_transform_counts_and_inverse():
    assert mollweide_pixels_for_ydeg(2, oversample=3).npix == 16
    assert mollweide_pixels_for_ydeg(4, oversample=3).npix == 62
    _, y2p, p2y = starry_pixel_transforms(2, oversample=3)
    assert y2p.shape == (16, 9)
    assert p2y.shape == (9, 16)
    assert np.max(np.abs(p2y @ y2p - np.eye(9))) < 1.0e-5
