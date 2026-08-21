"""Stellar transit and exposure-integration helpers."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from ._backend import xp_for
from .orbit import sky_position


_DEFAULT_TRANSIT_RADIAL_ORDER = 128


@lru_cache(maxsize=16)
def _transit_gauss_legendre_rule(n_radial: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached Gauss--Legendre nodes and weights on ``[-1, 1]``."""

    nodes, weights = np.polynomial.legendre.leggauss(int(n_radial))
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def _transit_radial_order(quadrature: Any | None) -> int:
    """Read the radial order from the legacy quadrature argument.

    ``stellar_transit_flux`` historically accepted a :class:`DiskQuadrature`
    object.  The new ring rule does not use its azimuth nodes, but retaining
    ``n_radial`` keeps existing configurations and notebooks compatible.
    """

    if quadrature is None:
        return _DEFAULT_TRANSIT_RADIAL_ORDER
    if not hasattr(quadrature, "n_radial"):
        raise TypeError("quadrature must provide a positive n_radial attribute")
    n_radial = int(quadrature.n_radial)
    if n_radial < 4:
        raise ValueError("quadrature.n_radial must be at least 4")
    return n_radial


def _occulted_quadratic_flux(
    projected_distance: Any,
    planet_radius: Any,
    front: Any,
    u1: float,
    u2: float,
    xp: Any,
    n_radial: int,
):
    """Integrate occulted flux with exact angular fractions on radial rings.

    At a stellar radius ``r``, the planet blocks either a full ring, no ring,
    or an angular interval ``2 acos(cos(phi))``.  Splitting the radial domain at
    ``|d-rp|`` and ``d+rp`` keeps the Gauss--Legendre rule away from the two
    contact discontinuities.  The calculation is vectorised over any leading
    time dimensions and uses only NumPy/JAX operations after the static rule
    is created.
    """

    distance, radius = xp.broadcast_arrays(
        xp.asarray(projected_distance), xp.asarray(planet_radius)
    )
    nodes, node_weights = _transit_gauss_legendre_rule(n_radial)
    node_shape = (1,) * distance.ndim + (1, n_radial)
    nodes = xp.asarray(nodes).reshape(node_shape)
    node_weights = xp.asarray(node_weights).reshape(node_shape)

    # The only radii at which the occulted angular fraction changes branch are
    # the internal and external circle contacts.  Clipping to the stellar disc
    # preserves sorted boundaries even when the planet is far from the star.
    lower_contact = xp.clip(xp.abs(distance - radius), 0.0, 1.0)
    upper_contact = xp.clip(distance + radius, 0.0, 1.0)
    boundaries = xp.stack(
        (
            xp.zeros_like(distance),
            lower_contact,
            upper_contact,
            xp.ones_like(distance),
        ),
        axis=-1,
    )
    left = boundaries[..., :-1, None]
    right = boundaries[..., 1:, None]
    half_width = 0.5 * (right - left)
    radius_midpoint = 0.5 * (right + left)
    stellar_radius = radius_midpoint + half_width * nodes

    distance_expanded = distance[..., None, None]
    planet_radius_expanded = radius[..., None, None]
    inside_planet = stellar_radius + distance_expanded <= planet_radius_expanded
    outside_planet = (distance_expanded >= stellar_radius + planet_radius_expanded) | (
        stellar_radius >= distance_expanded + planet_radius_expanded
    )
    denominator = 2.0 * stellar_radius * distance_expanded
    safe_denominator = xp.where(
        denominator > 0.0, denominator, xp.ones_like(denominator)
    )
    cosine = (
        stellar_radius**2 + distance_expanded**2 - planet_radius_expanded**2
    ) / safe_denominator
    # Keep the inactive ``arccos`` branch finite for automatic
    # differentiation.  A hard clip at exactly +/-1 gives an infinite
    # derivative; those values are selected by the full/no-overlap branches
    # above, so moving the clip by one machine epsilon does not change the
    # returned flux but avoids ``0 * NaN`` in JAX reverse mode.
    cosine_epsilon = xp.finfo(cosine.dtype).eps
    partial_angle = 2.0 * xp.arccos(
        xp.clip(cosine, -1.0 + cosine_epsilon, 1.0 - cosine_epsilon)
    )
    occulted_angle = xp.where(
        inside_planet,
        2.0 * math.pi,
        xp.where(outside_planet, 0.0, partial_angle),
    )

    # Zero-width intervals occur for out-of-transit points after contact
    # clipping.  Keeping the inactive limb value above zero prevents an
    # otherwise harmless ``d sqrt(mu) / d mu`` infinity from producing NaN in
    # JAX gradients; the interval carries exactly zero integration weight.
    mu_epsilon = xp.finfo(stellar_radius.dtype).eps
    mu = xp.sqrt(xp.clip(1.0 - stellar_radius**2, mu_epsilon, 1.0))
    intensity = quadratic_limb_darkening(mu, u1, u2)
    ring_integrand = stellar_radius * occulted_angle * intensity
    blocked = xp.sum(half_width * node_weights * ring_integrand, axis=(-2, -1))
    # If the planet fully contains the stellar disc, use the analytic stellar
    # flux.  This avoids a tiny radial-rule error at the square-root limb and
    # guarantees exactly zero residual flux for the (rare) oversized-planet
    # case.
    total = math.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    geometry_epsilon = 8.0 * xp.finfo(distance.dtype).eps
    contains_star = distance + 1.0 <= radius + geometry_epsilon
    blocked = xp.where(contains_star, total, blocked)
    return xp.where(front, blocked, xp.zeros_like(blocked))


def quadratic_limb_darkening(mu: Any, u1: float = 0.0, u2: float = 0.0):
    """Return quadratic limb-darkened intensity for ``mu = cos(theta)``."""

    xp = xp_for(mu)
    mu = xp.asarray(mu)
    one_minus_mu = 1.0 - mu
    return 1.0 - u1 * one_minus_mu - u2 * one_minus_mu * one_minus_mu


def stellar_flux(u1: float = 0.0, u2: float = 0.0):
    """Return the analytic total flux of a unit-radius quadratic star."""

    return math.pi * (1.0 - u1 / 3.0 - u2 / 6.0)


def stellar_transit_flux(
    time: Any,
    period: float,
    a_over_rstar: float,
    inclination: float,
    rplanet_over_rstar: float,
    t0: float = 0.0,
    *,
    u1: float = 0.0,
    u2: float = 0.0,
    angle_unit: str = "rad",
    quadrature=None,
):
    """Return stellar flux during a planet transit, normalised out of transit.

    The stellar disc is integrated with Gauss--Legendre radial rings.  Each
    ring uses the exact occulted angular fraction of two circles, rather than
    a binary mask on a Cartesian or polar point grid.  This removes the
    azimuth-resolution dependence of the former disk quadrature and gives a
    smooth, high-accuracy calculation through ingress and egress.  The planet
    blocks stellar points only when it is in front of the star (positive
    line-of-sight ``z`` in :func:`sky_position`).

    The legacy ``quadrature`` argument remains accepted.  Its ``n_radial``
    value controls the radial order; its azimuth order is no longer used.
    """

    n_radial = _transit_radial_order(quadrature)
    xp = xp_for(time, rplanet_over_rstar)
    time = xp.asarray(time)
    position = sky_position(
        time,
        period,
        a_over_rstar,
        inclination,
        t0,
        angle_unit=angle_unit,
    )
    # Planet centre in stellar-radius units.  Positive z means it is in front
    # of the star and only those samples can remove stellar flux.
    projected_distance = xp.sqrt(xp.sum(xp.square(position[..., :2]), axis=-1))
    front = position[..., 2] > 0.0
    blocked = _occulted_quadratic_flux(
        projected_distance,
        rplanet_over_rstar,
        front,
        u1,
        u2,
        xp,
        n_radial,
    )
    return 1.0 - blocked / stellar_flux(u1, u2)


def exposure_integrate(
    function: Callable[[Any], Any],
    time: Any,
    exposure_duration: float,
    *,
    n_subsamples: int = 8,
):
    """Average a time-dependent function over a finite exposure.

    ``time`` and ``exposure_duration`` must use the same units.  Midpoint
    samples make the rule symmetric and avoid evaluating exactly at an
    occultation contact.  The callback must support a final sample axis; all
    physics functions in this package do.
    """

    n_subsamples = int(n_subsamples)
    if n_subsamples < 1:
        raise ValueError("n_subsamples must be at least 1")
    xp = xp_for(time)
    time = xp.asarray(time)
    offsets = ((xp.arange(n_subsamples) + 0.5) / n_subsamples - 0.5) * exposure_duration
    samples = function(time[..., None] + offsets)
    return xp.mean(xp.asarray(samples), axis=-1)


def integrated_stellar_transit_flux(
    time: Any,
    exposure_duration: float,
    *args,
    n_subsamples: int = 8,
    **kwargs,
):
    """Convenience wrapper for exposure-integrated stellar transit flux."""

    return exposure_integrate(
        lambda sample_time: stellar_transit_flux(sample_time, *args, **kwargs),
        time,
        exposure_duration,
        n_subsamples=n_subsamples,
    )
