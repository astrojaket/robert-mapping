import numpy as np

from robert_mapping.physics.orbit import (
    light_travel_time_days,
    orbital_phase,
    projected_separation,
    sky_position,
    subobserver_longitude,
)
from robert_mapping.physics.occultation import disk_quadrature, map_flux
from robert_mapping.physics.occultation import secondary_eclipse_flux


def test_transit_and_eclipse_geometry():
    pos = sky_position(0.0, 2.0, 5.0, np.pi / 2)
    assert np.allclose(pos, [0.0, 0.0, 5.0], atol=1e-12)
    pos = sky_position(1.0, 2.0, 5.0, np.pi / 2)
    assert np.allclose(pos, [0.0, 0.0, -5.0], atol=1e-12)
    assert np.isclose(projected_separation(0.0, 2.0, 5.0, np.pi / 2), 0.0)


def test_inclination_sets_impact_parameter():
    inc = np.deg2rad(82.0)
    a = 5.0
    assert np.isclose(projected_separation(0.0, 1.0, a, inc), a * np.cos(inc))


def test_rotation_and_light_delay():
    assert np.isclose(orbital_phase(0.25, 1.0), np.pi / 2)
    assert np.isclose(subobserver_longitude(0.25, 1.0, theta0=0.2), 0.2 - np.pi / 2)
    delay = light_travel_time_days(0.0, 1.0, 5.0, np.pi / 2, 7.0e8)
    assert delay > 0
    assert np.isclose(delay, 5 * 7.0e8 / 299792458.0 / 86400.0)


def test_theta0_pi_at_transit_places_dayside_at_secondary():
    # At transit the sub-observer longitude is pi (the nightside).  Half a
    # period later it is zero, the dayside. A Y_1,+1 map changes sign.
    q = disk_quadrature(12, 48)
    coeff = np.array([0.0, 0.0, 0.0, 1.0])
    transit_lon = subobserver_longitude(0.0, 1.0, theta0=np.pi)
    secondary_lon = subobserver_longitude(0.5, 1.0, theta0=np.pi)
    transit = map_flux(coeff, transit_lon, 0.0, quadrature=q)
    secondary = map_flux(coeff, secondary_lon, 0.0, quadrature=q)
    assert np.isclose(secondary_lon, 0.0)
    assert np.isclose(secondary, -transit, rtol=1e-6, atol=1e-10)


def test_eastward_hotspot_is_brighter_before_eclipse():
    coefficients = np.array([1.0, 0.0, 0.0, 0.2])
    flux = secondary_eclipse_flux(
        coefficients,
        np.array([0.35, 0.65]),
        1.0,
        5.0,
        np.pi / 2,
        0.1,
        theta0=np.pi,
        quadrature=disk_quadrature(12, 48),
    )
    assert flux[0] > flux[1]
