"""Accuracy checks for the starry-free radial-ring transit operator."""

from __future__ import annotations

import numpy as np

from robert_mapping.physics import disk_quadrature, stellar_transit_flux


def _circle_overlap_area(distance: float, planet_radius: float) -> float:
    """Area shared by a unit star and a planet disc."""

    if distance >= 1.0 + planet_radius:
        return 0.0
    if distance <= abs(1.0 - planet_radius):
        return np.pi * min(1.0, planet_radius) ** 2
    star_angle = np.arccos(
        np.clip((distance**2 + 1.0 - planet_radius**2) / (2.0 * distance), -1.0, 1.0)
    )
    planet_angle = np.arccos(
        np.clip(
            (distance**2 + planet_radius**2 - 1.0) / (2.0 * distance * planet_radius),
            -1.0,
            1.0,
        )
    )
    radical = np.sqrt(
        max(
            0.0,
            (-distance + 1.0 + planet_radius)
            * (distance + 1.0 - planet_radius)
            * (distance - 1.0 + planet_radius)
            * (distance + 1.0 + planet_radius),
        )
    )
    return star_angle + planet_radius**2 * planet_angle - 0.5 * radical


def test_out_of_transit_is_unity_for_arbitrary_time_array() -> None:
    times = np.array([[-0.4, -0.25], [0.25, 0.4]])
    flux = stellar_transit_flux(times, 1.0, 5.0, np.pi / 2.0, 0.1)
    assert flux.shape == times.shape
    np.testing.assert_allclose(flux, 1.0, rtol=0.0, atol=1.0e-14)


def test_transit_repeats_periodically_for_three_cycles() -> None:
    offsets = np.array([-0.012, -0.003, 0.0, 0.004, 0.015])
    times = np.concatenate([offsets - 1.0, offsets, offsets + 1.0])
    flux = stellar_transit_flux(times, 1.0, 5.0, np.deg2rad(88.5), 0.12)
    np.testing.assert_allclose(flux[: offsets.size], flux[offsets.size : 2 * offsets.size], atol=1.0e-13)
    np.testing.assert_allclose(flux[offsets.size : 2 * offsets.size], flux[2 * offsets.size :], atol=1.0e-13)


def test_transit_is_symmetric_about_midpoint() -> None:
    offsets = np.linspace(0.0, 0.045, 25)
    positive = stellar_transit_flux(offsets, 1.0, 5.0, np.deg2rad(87.0), 0.1, u1=0.2, u2=0.1)
    negative = stellar_transit_flux(-offsets, 1.0, 5.0, np.deg2rad(87.0), 0.1, u1=0.2, u2=0.1)
    np.testing.assert_allclose(positive, negative, rtol=0.0, atol=1.0e-13)


def test_uniform_star_central_depth_matches_area() -> None:
    radius_ratio = 0.1
    flux = stellar_transit_flux(0.0, 1.0, 5.0, np.pi / 2.0, radius_ratio, u1=0.0, u2=0.0)
    assert np.isclose(flux, 1.0 - radius_ratio**2, rtol=0.0, atol=1.0e-13)


def test_uniform_star_off_centre_flux_matches_circle_area() -> None:
    radius_ratio = 0.1
    distance = 0.5
    time = np.arcsin(distance / 5.0) / (2.0 * np.pi)
    flux = stellar_transit_flux(
        time,
        1.0,
        5.0,
        np.pi / 2.0,
        radius_ratio,
        u1=0.0,
        u2=0.0,
    )
    expected = 1.0 - _circle_overlap_area(distance, radius_ratio) / np.pi
    assert np.isclose(flux, expected, rtol=0.0, atol=4.0e-9)


def test_radial_ring_convergence_is_monotonic() -> None:
    radius_ratio = 0.1
    time = np.arcsin(0.5 / 5.0) / (2.0 * np.pi)
    reference = stellar_transit_flux(
        time,
        1.0,
        5.0,
        np.pi / 2.0,
        radius_ratio,
        u1=0.2,
        u2=0.1,
        quadrature=disk_quadrature(256, 8),
    )
    errors = []
    for n_radial in (16, 32, 64, 128):
        estimate = stellar_transit_flux(
            time,
            1.0,
            5.0,
            np.pi / 2.0,
            radius_ratio,
            u1=0.2,
            u2=0.1,
            quadrature=disk_quadrature(n_radial, 8),
        )
        errors.append(abs(float(estimate - reference)))
    assert all(left > right for left, right in zip(errors, errors[1:]))
    assert errors[-1] < 4.0e-9
