"""Regression tests for phase curves with multiple secondary eclipses."""

from __future__ import annotations

import numpy as np

from robert_mapping.physics import disk_quadrature, secondary_eclipse_flux


def test_secondary_eclipse_flux_handles_three_separated_eclipses() -> None:
    """A vector phase curve can contain separated eclipse windows over many cycles."""

    period = 1.0
    t0 = 0.0
    coefficients = np.array([1.0, 0.05, 0.08, 0.03])
    quadrature = disk_quadrature(12, 48)

    # Keep the three windows separate. The star is large enough in planet
    # radii to fully occult the planet at each central-eclipse sample.
    eclipse_centres = np.array([0.5, 2.5, 4.5])
    offsets = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])
    eclipse_times = np.concatenate(
        [centre + offsets for centre in eclipse_centres]
    )
    full_phase_curve = np.linspace(0.0, period, 41, endpoint=False)
    periodic_times = np.array([0.11, 0.27, 0.43, 0.57, 0.73, 0.89])
    times = np.concatenate(
        (full_phase_curve, eclipse_times, periodic_times, periodic_times + 2.0 * period)
    )

    flux = secondary_eclipse_flux(
        coefficients,
        times,
        period,
        5.0,
        np.pi / 2.0,
        0.1,
        t0,
        theta0=np.pi,
        rotation_period=period,
        quadrature=quadrature,
    )
    repeated_flux = secondary_eclipse_flux(
        coefficients,
        times,
        period,
        5.0,
        np.pi / 2.0,
        0.1,
        t0,
        theta0=np.pi,
        rotation_period=period,
        quadrature=quadrature,
    )

    assert flux.shape == times.shape
    assert np.allclose(flux, repeated_flux, rtol=0.0, atol=1.0e-14)

    first_window = flux[full_phase_curve.size : full_phase_curve.size + offsets.size]
    for index in range(1, eclipse_centres.size):
        start = full_phase_curve.size + index * offsets.size
        window = flux[start : start + offsets.size]
        assert np.allclose(window, first_window, rtol=0.0, atol=1.0e-12)
        assert window[offsets.size // 2] == 0.0
        assert window[offsets.size // 2] < window[0]
        assert window[offsets.size // 2] < window[-1]

    first_period = flux[-2 * periodic_times.size : -periodic_times.size]
    second_period = flux[-periodic_times.size :]
    assert np.allclose(first_period, second_period, rtol=0.0, atol=1.0e-12)
    assert np.all(flux[: full_phase_curve.size] >= 0.0)
    assert np.any(flux[: full_phase_curve.size] > 0.0)
