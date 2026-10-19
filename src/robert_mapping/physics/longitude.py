"""Latitude-flat longitude maps for secondary-eclipse mapping.

Hammond et al. (2024, arXiv:2404.16488, pp. 6–7, Fig. 4) describe a
longitude-only map formed by averaging each fitting map basis over latitude
and removing repeated or uniform results. Through degree ``d``, the resulting
span is the real Fourier span: a constant plus sine and cosine terms for
orders ``1`` through ``d``. This module evaluates that span directly, so no
latitude-dependent spherical-harmonic structure remains in the fitted map.

The columns use the intensity normalization ``1 / pi``. Thus coefficient
``[1, 0, ...]`` is the same uniform surface as the starry-normalized
spherical-harmonic monopole ``Y00 = 1 / pi`` and has unit unocculted
projected-disc flux. Averaging with uniform latitude or with the spherical
area weight ``cos(latitude)`` can change the scale of each non-constant
Fourier column. It does not change their span when coefficients are free;
the common intensity normalization here keeps the monopole convention clear.

Coefficient order is ``[constant, cos(1 lon), sin(1 lon), ...]``. The basis
is differentiable and uses NumPy or JAX based on the input arrays.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._backend import xp_for
from .occultation import (
    DiskQuadrature,
    _map_coordinates,
    _occultation_mask,
    _secondary_geometry,
    disk_quadrature,
)
from .orbit import light_travel_time_days, subobserver_longitude


_MAX_DEGREE = 10
_INTENSITY_SCALE = 1.0 / math.pi


def _degree_value(degree: int) -> int:
    if isinstance(degree, bool):
        raise ValueError("degree must be an integer from 0 to 10")
    try:
        value = int(degree)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("degree must be an integer from 0 to 10") from exc
    if value != degree or value < 0 or value > _MAX_DEGREE:
        raise ValueError(f"degree must be an integer from 0 to {_MAX_DEGREE}")
    return value


def longitude_basis(longitude: Any, degree: int):
    """Return the normalized real Fourier basis through ``degree``.

    The final axis has ``1 + 2 * degree`` columns in constant, cosine,
    sine order. The columns are latitude-flat because they depend only on
    longitude. The leading shape follows ``longitude``.
    """

    degree = _degree_value(degree)
    xp = xp_for(longitude)
    lon = xp.asarray(longitude)
    values = [xp.ones_like(lon) * _INTENSITY_SCALE]
    for order in range(1, degree + 1):
        values.append(xp.cos(order * lon) * _INTENSITY_SCALE)
        values.append(xp.sin(order * lon) * _INTENSITY_SCALE)
    return xp.stack(values, axis=-1)


def evaluate_longitude_map(
    coefficients: Any,
    longitude: Any,
    latitude: Any | None = None,
):
    """Evaluate a latitude-independent map at longitude and optional latitude.

    ``latitude`` is accepted to make coordinate-array broadcasting explicit;
    its values do not enter the map. Coefficients may have leading batch
    dimensions, as with :func:`~robert_mapping.physics.harmonics.evaluate_map`.
    """

    xp = xp_for(coefficients, longitude, latitude)
    coeff = xp.asarray(coefficients)
    if coeff.ndim < 1 or coeff.shape[-1] < 1 or coeff.shape[-1] % 2 != 1:
        raise ValueError("coefficients must end in 1 + 2 * degree values")
    degree = (int(coeff.shape[-1]) - 1) // 2
    _degree_value(degree)

    lon = xp.asarray(longitude)
    if latitude is not None:
        lon, _ = xp.broadcast_arrays(lon, xp.asarray(latitude))
    basis = longitude_basis(lon, degree)
    return xp.tensordot(coeff, basis, axes=([-1], [-1]))


def longitude_eclipse_design_matrix(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    rplanet_over_rstar: float,
    degree: int = 4,
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
    """Integrate longitude-only basis columns over a secondary eclipse.

    Geometry, angle units, optional rotation, and light-delay controls match
    :func:`~robert_mapping.physics.occultation.secondary_eclipse_design_matrix`.
    Each returned row is projected flux for one unit Fourier coefficient.
    A scalar time returns shape ``(1 + 2 * degree,)``; an array of times
    returns ``(ntime, 1 + 2 * degree)``.
    """

    degree = _degree_value(degree)
    quadrature = disk_quadrature() if quadrature is None else quadrature
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
        eval_time = eval_time + delay

    _, center, radius = _secondary_geometry(
        eval_time,
        period,
        a_over_rstar,
        inclination,
        rplanet_over_rstar,
        t0,
        angle_unit=angle_unit,
    )
    observer_lon = subobserver_longitude(
        eval_time,
        period,
        t0,
        theta0=theta0,
        rotation_period=rotation_period,
    )
    map_lon, _ = _map_coordinates(quadrature, observer_lon, subobserver_lat)
    basis = longitude_basis(map_lon, degree)
    visible = _occultation_mask(quadrature, center, radius)
    weight = xp.asarray(quadrature.weight)[None, :, None]
    design = xp.sum(basis * visible[..., None] * weight, axis=-2)

    if np.ndim(observer_lon) == 0 and np.ndim(subobserver_lat) == 0:
        design = design[0]
    return design


__all__ = [
    "longitude_basis",
    "evaluate_longitude_map",
    "longitude_eclipse_design_matrix",
]
