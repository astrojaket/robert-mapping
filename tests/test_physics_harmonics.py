import numpy as np

from robert_mapping.physics.harmonics import (
    evaluate_map,
    index_from_lm,
    lm_from_index,
    ncoeff,
    real_sph_harm,
    real_sph_harm_all,
)
from robert_mapping.physics.pixels import equal_area_pixels


def test_starry_flat_index_round_trip():
    for ydeg in range(7):
        for index in range(ncoeff(ydeg)):
            l, m = lm_from_index(index)
            assert index_from_lm(l, m) == index
            assert l <= ydeg


def test_real_harmonic_normalization_and_low_order_values():
    assert np.isclose(real_sph_harm(0, 0, 0.0, 0.0), 1.0 / np.pi)
    # Starry's polar axis is physical longitude=0, latitude=0. Its +1 axis is
    # east and its -1 axis is north.
    assert np.isclose(real_sph_harm(1, 0, 0.0, 0.0), np.sqrt(3) / np.pi)
    assert np.isclose(real_sph_harm(1, 1, np.pi / 2, 0.0), np.sqrt(3) / np.pi)
    assert np.isclose(real_sph_harm(1, -1, 0.0, np.pi / 2), np.sqrt(3) / np.pi)


def test_frozen_starry_hatp32_map_coefficients():
    """A frozen starry v1 coefficient vector must peak at +10 degrees east."""

    coefficients = np.array(
        [
            1.0,
            0.0,
            0.198943737807463,
            0.035079148618446486,
            0.0,
            0.0,
            0.02933156657967967,
            0.009099531943521704,
            0.0008022464967307069,
        ]
    )
    longitude = np.deg2rad(np.arange(-90.0, 91.0))
    profile = evaluate_map(coefficients, longitude, np.zeros(longitude.size))
    assert np.rad2deg(longitude[int(np.argmax(profile))]) == 10.0


def test_equal_area_grid_integrates_harmonics():
    pixels = equal_area_pixels(40, 80)
    basis = real_sph_harm_all(4, pixels.lon, pixels.lat)
    integral = np.einsum("p,pc->c", pixels.area, basis)
    assert np.isclose(integral[0], 4.0, rtol=2e-5)
    # Midpoint cells are exactly equal in area but are not Gauss--Legendre
    # cells.  The degree-four midpoint quadrature error is therefore small,
    # not machine precision.
    assert np.max(np.abs(integral[1:])) < 2e-2


def test_map_evaluation_supports_batches():
    coeff = np.zeros((2, 4))
    coeff[0, 0] = 1.0
    coeff[1, 3] = 1.0
    values = evaluate_map(coeff, np.array([0.0, np.pi / 2]), 0.0)
    assert values.shape == (2, 2)
    assert np.isclose(values[0, 0], 1 / np.pi)
