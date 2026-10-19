from __future__ import annotations

import numpy as np
import pytest

from robert_mapping.inference.linear import fit_linear_gaussian
from robert_mapping.physics.harmonics import evaluate_map, real_sph_harm_all
from robert_mapping.physics.longitude import (
    evaluate_longitude_map,
    longitude_basis,
    longitude_eclipse_design_matrix,
)
from robert_mapping.physics.occultation import (
    disk_quadrature,
    secondary_eclipse_design_matrix,
)


def test_longitude_basis_is_fourier_ordered_and_normalized_like_monopole() -> None:
    longitude = np.array([0.0, np.pi / 2.0])

    basis = longitude_basis(longitude, degree=1)

    assert basis.shape == (2, 3)
    assert np.allclose(basis[:, 0], 1.0 / np.pi)
    assert np.allclose(
        basis[0], np.array([1.0, 1.0, 0.0]) / np.pi, atol=1e-15
    )


def test_evaluated_longitude_map_is_flat_in_latitude_and_broadcasts() -> None:
    longitude = np.array([[0.1], [1.2]])
    latitude = np.array([[-1.0, -0.3, 0.2, 1.1]])
    coefficients = np.array([0.9, 0.2, -0.1, 0.05, 0.03])

    values = evaluate_longitude_map(coefficients, longitude, latitude)

    assert values.shape == (2, 4)
    assert np.allclose(values, values[:, :1])


def test_latitude_flat_basis_is_not_a_latitude_dependent_harmonic_subset() -> None:
    longitude = np.full(3, 0.6)
    latitude = np.array([-0.8, -0.2, 0.8])
    coefficients = np.array([0.0, 1.0, 0.0])
    harmonics = np.zeros(9)
    harmonics[8] = 1.0  # one l=2, |m|=2 spherical harmonic

    longitude_values = evaluate_longitude_map(
        coefficients, longitude, latitude
    )
    harmonic_values = evaluate_map(harmonics, longitude, latitude)

    assert np.allclose(longitude_values, longitude_values[0])
    assert not np.allclose(harmonic_values, harmonic_values[0])


@pytest.mark.parametrize("area_weighted", [False, True])
def test_latitude_averaged_harmonics_have_the_fourier_column_span(
    area_weighted: bool,
) -> None:
    degree = 2
    longitude = np.linspace(-np.pi, np.pi, 121, endpoint=False)
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, 401)
    spacing = latitude[1] - latitude[0]
    latitude_weights = np.ones(latitude.size)
    latitude_weights[[0, -1]] *= 0.5
    latitude_weights *= spacing
    if area_weighted:
        latitude_weights *= np.cos(latitude)
    latitude_weights /= np.sum(latitude_weights)

    harmonic_basis = real_sph_harm_all(
        degree, longitude[:, None], latitude[None, :]
    )
    averaged_harmonics = np.einsum(
        "ljc,j->lc", harmonic_basis, latitude_weights
    )
    fourier = np.stack(
        [
            np.ones_like(longitude),
            *[
                function(order * longitude)
                for order in range(1, degree + 1)
                for function in (np.cos, np.sin)
            ],
        ],
        axis=-1,
    )

    assert np.linalg.matrix_rank(averaged_harmonics) == 1 + 2 * degree
    assert np.linalg.matrix_rank(fourier) == 1 + 2 * degree
    assert np.linalg.matrix_rank(
        np.column_stack((averaged_harmonics, fourier))
    ) == 1 + 2 * degree


def test_uniform_longitude_map_matches_spherical_harmonic_monopole() -> None:
    time = np.array([0.0, 0.48, 0.5, 0.52, 0.9])
    quadrature = disk_quadrature(24, 96)

    longitude_design = longitude_eclipse_design_matrix(
        time,
        1.0,
        5.0,
        89.0,
        0.1,
        degree=2,
        t0=0.0,
        angle_unit="deg",
        quadrature=quadrature,
    )
    harmonic_design = secondary_eclipse_design_matrix(
        time,
        1.0,
        5.0,
        89.0,
        0.1,
        ydeg=0,
        t0=0.0,
        angle_unit="deg",
        quadrature=quadrature,
    )

    assert longitude_design.shape == (time.size, 5)
    assert np.allclose(longitude_design[:, 0], harmonic_design[:, 0], atol=1e-13)


def test_longitude_offset_is_recovered_by_exact_gaussian_linear_fit() -> None:
    time = np.linspace(0.0, 1.0, 101, endpoint=False)
    design = longitude_eclipse_design_matrix(
        time,
        1.0,
        5.0,
        89.0,
        0.1,
        degree=1,
        t0=0.0,
        angle_unit="deg",
        quadrature=disk_quadrature(30, 120),
    )
    offset = np.deg2rad(35.0)
    expected = np.array([1.0, 0.2 * np.cos(offset), 0.2 * np.sin(offset)])
    observed = design @ expected

    posterior = fit_linear_gaussian(
        design,
        observed,
        1.0e-6,
        prior_mean=np.zeros(expected.size),
        prior_scale=1.0e8,
    )

    assert np.allclose(posterior.mean, expected, atol=1.0e-10)
    recovered_offset = np.arctan2(posterior.mean[2], posterior.mean[1])
    assert recovered_offset == pytest.approx(offset, abs=1.0e-10)


def test_eclipse_matrix_supports_inclination_rotation_and_light_delay() -> None:
    time = np.array([0.46, 0.49, 0.5, 0.51, 0.54])

    design = longitude_eclipse_design_matrix(
        time,
        1.0,
        5.0,
        87.0,
        0.1,
        degree=2,
        t0=0.0,
        theta0=0.2,
        rotation_period=1.02,
        subobserver_lat=0.1,
        angle_unit="rad",
        quadrature=disk_quadrature(20, 80),
        light_delay=True,
        rstar_meters=7.0e8,
    )

    assert design.shape == (time.size, 5)
    assert np.all(np.isfinite(design))


def test_scalar_time_returns_vector_and_light_delay_needs_stellar_radius() -> None:
    args = (0.5, 1.0, 5.0, 89.0, 0.1)

    design = longitude_eclipse_design_matrix(
        *args, degree=1, angle_unit="deg", quadrature=disk_quadrature(12, 48)
    )
    assert design.shape == (3,)

    with pytest.raises(ValueError, match="rstar_meters is required"):
        longitude_eclipse_design_matrix(
            *args, degree=1, light_delay=True, angle_unit="deg"
        )


@pytest.mark.parametrize("degree", [-1, 11, 1.5])
def test_longitude_degree_has_a_clear_supported_range(degree: float) -> None:
    with pytest.raises(ValueError, match="degree must be an integer"):
        longitude_basis(0.0, degree)
