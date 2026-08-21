"""Deterministic projected-disc integration for eclipse maps.

This module provides the independent numerical occultation operator used as a
reference for the fast inference layer.  It integrates a fixed polar rule on
the visible planetary disc.  The rule is smooth and JAX-compatible away from
the occultor boundary; a binary mask handles the stellar limb.  Increasing
``n_radial`` and ``n_azimuth`` gives a direct convergence check near contact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._backend import xp_for
from .harmonics import evaluate_map, ncoeff, real_sph_harm_all
from .orbit import light_travel_time_days, sky_position, subobserver_longitude


@dataclass(frozen=True)
class DiskQuadrature:
    """Fixed quadrature nodes on the unit projected planetary disc."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    weight: np.ndarray
    n_radial: int
    n_azimuth: int

    @property
    def size(self) -> int:
        """Number of quadrature nodes."""

        return int(self.x.size)


def disk_quadrature(n_radial: int = 32, n_azimuth: int = 128) -> DiskQuadrature:
    """Create a Gauss--Legendre radial and uniform-azimuth disc rule.

    The weights include the projected-area Jacobian ``r dr dphi`` and sum to
    ``pi`` to floating-point precision.
    """

    n_radial = int(n_radial)
    n_azimuth = int(n_azimuth)
    if n_radial < 2:
        raise ValueError("n_radial must be at least 2")
    if n_azimuth < 8:
        raise ValueError("n_azimuth must be at least 8")
    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    radius = 0.5 * (nodes + 1.0)
    radial_weight = 0.5 * weights * radius
    phi = 2.0 * np.pi * (np.arange(n_azimuth, dtype=float) + 0.5) / n_azimuth
    rr, pp = np.meshgrid(radius, phi, indexing="ij")
    ww = radial_weight[:, None] * (2.0 * np.pi / n_azimuth)
    x = (rr * np.cos(pp)).ravel()
    y = (rr * np.sin(pp)).ravel()
    z = np.sqrt(np.maximum(0.0, 1.0 - x * x - y * y))
    weight = np.broadcast_to(ww, rr.shape).ravel()
    return DiskQuadrature(x, y, z, weight, n_radial, n_azimuth)


def _map_coordinates(
    quadrature: DiskQuadrature,
    subobserver_lon: Any,
    subobserver_lat: Any,
):
    """Map projected-disc nodes into the planet's longitude/latitude frame."""

    xp = xp_for(subobserver_lon, subobserver_lat)
    lon0 = xp.asarray(subobserver_lon)[..., None]
    lat0 = xp.asarray(subobserver_lat)[..., None]
    x = xp.asarray(quadrature.x)[None, :]
    y = xp.asarray(quadrature.y)[None, :]
    z = xp.asarray(quadrature.z)[None, :]

    # ``n`` points to the sub-observer longitude/latitude.  ``e_lon`` is the
    # direction of increasing longitude, and ``e_lat`` points north.  The
    # resulting position is a unit vector in map coordinates.
    cos_lon = xp.cos(lon0)
    sin_lon = xp.sin(lon0)
    cos_lat = xp.cos(lat0)
    sin_lat = xp.sin(lat0)
    rx = x * (-sin_lon) + y * (-sin_lat * cos_lon) + z * (cos_lat * cos_lon)
    ry = x * cos_lon + y * (-sin_lat * sin_lon) + z * (cos_lat * sin_lon)
    rz = y * cos_lat + z * sin_lat
    lon = xp.arctan2(ry, rx)
    lat = xp.arcsin(xp.clip(rz, -1.0, 1.0))
    # A scalar sub-observer latitude must broadcast over a vector of epochs.
    lon, lat = xp.broadcast_arrays(lon, lat)
    return lon, lat


def _occultation_mask(
    quadrature: DiskQuadrature,
    occultor_center: Any | None,
    occultor_radius: Any | None,
):
    """Return a visible (one) / blocked (zero) mask."""

    xp = xp_for(occultor_center, occultor_radius)
    x = xp.asarray(quadrature.x)[None, :]
    y = xp.asarray(quadrature.y)[None, :]
    if occultor_center is None or occultor_radius is None:
        return xp.ones_like(x)
    center = xp.asarray(occultor_center)
    if center.shape[-1] != 2:
        raise ValueError("occultor_center must end in an (x, y) pair")
    cx = center[..., 0, None]
    cy = center[..., 1, None]
    radius = xp.asarray(occultor_radius)[..., None]
    return (xp.square(x - cx) + xp.square(y - cy) >= xp.square(radius)).astype(xp.asarray(1.0).dtype)


def map_design_matrix(
    subobserver_lon: Any,
    subobserver_lat: Any = 0.0,
    ydeg: int = 4,
    *,
    occultor_center: Any | None = None,
    occultor_radius: Any | None = None,
    quadrature: DiskQuadrature | None = None,
):
    """Return the linear map-to-flux design vector or matrix.

    A scalar geometry returns shape ``(ncoeff,)``.  A vector of geometries
    returns shape ``(ntime, ncoeff)``.  Each row gives the projected flux for
    one unit spherical-harmonic coefficient.
    """

    quadrature = disk_quadrature() if quadrature is None else quadrature
    xp = xp_for(subobserver_lon, subobserver_lat, occultor_center, occultor_radius)
    lon, lat = _map_coordinates(quadrature, subobserver_lon, subobserver_lat)
    basis = real_sph_harm_all(int(ydeg), lon, lat)
    mask = _occultation_mask(quadrature, occultor_center, occultor_radius)
    weight = xp.asarray(quadrature.weight)[None, :, None]
    design = xp.sum(basis * mask[..., None] * weight, axis=-2)
    # The coordinate helper introduces a leading length-one axis for scalar
    # inputs.  Remove it for the ergonomic scalar return shape.
    if np.ndim(subobserver_lon) == 0 and np.ndim(subobserver_lat) == 0:
        design = design[0]
    return design


def map_flux(
    coefficients: Any,
    subobserver_lon: Any = 0.0,
    subobserver_lat: Any = 0.0,
    *,
    occultor_center: Any | None = None,
    occultor_radius: Any | None = None,
    quadrature: DiskQuadrature | None = None,
):
    """Integrate a spherical-harmonic map over the visible, unblocked disc."""

    coeff = coefficients
    design = map_design_matrix(
        subobserver_lon,
        subobserver_lat,
        int(np.sqrt(np.shape(coeff)[-1])) - 1,
        occultor_center=occultor_center,
        occultor_radius=occultor_radius,
        quadrature=quadrature,
    )
    xp = xp_for(coeff, design)
    coeff = xp.asarray(coeff)
    if coeff.shape[-1] != design.shape[-1]:
        raise ValueError("Coefficient axis does not match the requested map degree")
    return xp.tensordot(coeff, design, axes=([-1], [-1]))


def _secondary_geometry(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    rplanet_over_rstar: float,
    t0: float,
    *,
    angle_unit: str,
):
    """Return planet position, star centre in planet radii, and behind mask."""

    xp = xp_for(time)
    position = sky_position(
        time,
        period,
        a_over_rstar,
        inclination,
        t0,
        angle_unit=angle_unit,
    )
    rprs = xp.asarray(rplanet_over_rstar)
    center = -position[..., :2] / rprs[..., None]
    radius = xp.ones_like(position[..., 0]) / rprs
    # A star can only occult a planet when the planet is behind the star.  The
    # huge radius is retained for all phases but the radius is set to zero in
    # front, giving an all-visible mask without Python conditionals.
    radius = xp.where(position[..., 2] < 0.0, radius, 0.0)
    return position, center, radius


def secondary_eclipse_design_matrix(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    rplanet_over_rstar: float,
    ydeg: int = 4,
    t0: float = 0.0,
    *,
    theta0: float = 0.0,
    rotation_period: float | None = None,
    subobserver_lat: float = 0.0,
    angle_unit: str = "rad",
    quadrature: DiskQuadrature | None = None,
    light_delay: bool = False,
    rstar_meters: float | None = None,
):
    """Return the map design matrix for a circular secondary eclipse.

    Coordinates use stellar radii for the orbit and planetary radii for the
    occultor.  ``light_delay`` applies the first-order planet--star delay; a
    stellar radius is required when it is enabled.
    """

    xp = xp_for(time)
    eval_time = xp.asarray(time)
    if light_delay:
        if rstar_meters is None:
            raise ValueError("rstar_meters is required when light_delay=True")
        delay = light_travel_time_days(
            eval_time,
            period,
            a_over_rstar,
            inclination,
            rstar_meters,
            t0,
            angle_unit=angle_unit,
        )
        # Match starry's observer-time convention: a body on the near side
        # is evaluated at the later orbital state that produces the same
        # arrival time relative to the stellar reference signal.
        eval_time = eval_time + delay
    position, center, radius = _secondary_geometry(
        eval_time,
        period,
        a_over_rstar,
        inclination,
        rplanet_over_rstar,
        t0,
        angle_unit=angle_unit,
    )
    lon = subobserver_longitude(
        eval_time,
        period,
        t0,
        theta0=theta0,
        rotation_period=rotation_period,
    )
    return map_design_matrix(
        lon,
        subobserver_lat,
        ydeg,
        occultor_center=center,
        occultor_radius=radius,
        quadrature=quadrature,
    )


def secondary_eclipse_flux(
    coefficients: Any,
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    rplanet_over_rstar: float,
    t0: float = 0.0,
    *,
    theta0: float = 0.0,
    rotation_period: float | None = None,
    subobserver_lat: float = 0.0,
    angle_unit: str = "rad",
    quadrature: DiskQuadrature | None = None,
    light_delay: bool = False,
    rstar_meters: float | None = None,
):
    """Evaluate a map's thermal phase curve and secondary eclipse."""

    design = secondary_eclipse_design_matrix(
        time,
        period,
        a_over_rstar,
        inclination,
        rplanet_over_rstar,
        int(np.sqrt(np.shape(coefficients)[-1])) - 1,
        t0,
        theta0=theta0,
        rotation_period=rotation_period,
        subobserver_lat=subobserver_lat,
        angle_unit=angle_unit,
        quadrature=quadrature,
        light_delay=light_delay,
        rstar_meters=rstar_meters,
    )
    xp = xp_for(coefficients, design)
    return xp.tensordot(xp.asarray(coefficients), design, axes=([-1], [-1]))


# Clear aliases used by callers that prefer “planet” terminology.
planet_flux = map_flux
planet_design_matrix = map_design_matrix
