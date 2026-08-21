"""Circular-orbit geometry and light-travel-time helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._backend import xp_for


def orbital_phase(time: Any, period: float, t0: float = 0.0):
    """Return orbital phase angle in radians.

    ``phase = 0`` is the centre of transit and ``phase = pi`` is the centre
    of secondary eclipse.  The output is not wrapped, which preserves smooth
    derivatives for inference.
    """

    xp = xp_for(time)
    return 2.0 * math.pi * (xp.asarray(time) - t0) / period


def sky_position(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    t0: float = 0.0,
    *,
    angle_unit: str = "rad",
):
    """Return circular-orbit position ``(x, y, z)`` in stellar-radius units.

    The observer is on the positive ``z`` axis.  At transit (phase zero) the
    planet is in front of the star and ``z`` is positive.  ``x`` and ``y`` are
    sky-plane coordinates.  ``inclination`` is accepted in radians by
    default; pass ``angle_unit='deg'`` for degrees.
    """

    xp = xp_for(time)
    phase = orbital_phase(time, period, t0)
    inc = xp.asarray(inclination)
    if angle_unit.lower().startswith("deg"):
        inc = inc * math.pi / 180.0
    a = xp.asarray(a_over_rstar)
    x = a * xp.sin(phase)
    y = -a * xp.cos(inc) * xp.cos(phase)
    z = a * xp.sin(inc) * xp.cos(phase)
    return xp.stack((x, y, z), axis=-1)


def projected_separation(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    t0: float = 0.0,
    *,
    angle_unit: str = "rad",
):
    """Return sky-projected separation in stellar-radius units."""

    xp = xp_for(time)
    pos = sky_position(time, period, a_over_rstar, inclination, t0, angle_unit=angle_unit)
    return xp.sqrt(xp.sum(pos[..., :2] ** 2, axis=-1))


def subobserver_longitude(
    time: Any,
    period: float,
    t0: float = 0.0,
    *,
    theta0: float = 0.0,
    rotation_period: float | None = None,
):
    """Return the map longitude at the centre of the visible disc.

    ``theta0`` is the sub-observer longitude at ``t0``.  A synchronously
    rotating planet uses ``rotation_period=period``.  For a non-synchronous
    map, pass its rotation period explicitly.  The returned angle is unwrapped.
    """

    xp = xp_for(time)
    prot = period if rotation_period is None else rotation_period
    # East-positive longitudes rotate out of view as orbital phase increases.
    # The minus sign matches starry's prograde, synchronous convention and
    # makes an eastward hotspot brighten before secondary eclipse.
    return theta0 - 2.0 * math.pi * (xp.asarray(time) - t0) / prot


def light_travel_time_days(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    rstar_meters: float,
    t0: float = 0.0,
    *,
    angle_unit: str = "rad",
    speed_of_light_m_per_s: float = 299_792_458.0,
):
    """Return the planet--star light-travel delay in days.

    The line-of-sight coordinate is measured toward the observer.  A positive
    delay is positive on the observer-facing side. To reproduce starry's
    observer-time convention, evaluate the apparent relative orbit at
    ``time + delay``.
    ``period`` and ``t0`` are in days, while ``rstar_meters`` is in metres.
    """

    xp = xp_for(time)
    z = sky_position(
        time,
        period,
        a_over_rstar,
        inclination,
        t0,
        angle_unit=angle_unit,
    )[..., 2]
    delay_seconds = z * rstar_meters / speed_of_light_m_per_s
    return delay_seconds / 86_400.0
