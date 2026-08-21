"""Real spherical harmonics used by the eclipse maps.

The coefficient ordering follows the convention used by ``starry`` and by
the original Hammond et al. code: for each degree ``l`` in increasing order,
the orders are ``m=-l, ..., +l``.  Thus

``index = l*l + l + m``.

The underlying basis is the real, orthonormal basis used by ``starry``. The
public values include ``starry``'s historical ``2 / sqrt(pi)`` scale, so a
coefficient vector with ``y[0] = 1`` has uniform intensity ``1 / pi`` and
unit disk-integrated flux. ``starry`` uses its polar axis at physical
longitude zero and latitude zero. For physical longitude ``lon`` and latitude
``lat``, its internal Cartesian coordinates are

``x = cos(lat) sin(lon), y = sin(lat), z = cos(lat) cos(lon)``.

This axis choice is important. It makes ``Y_1,0`` peak at the substellar
point, ``Y_1,1`` point east, and ``Y_1,-1`` point north. The public functions
accept physical longitude and latitude and apply this transform internally.
With ``P_l^|m|`` the associated Legendre polynomial without the
Condon--Shortley sign,

* ``m = 0`` uses ``N P_l^0(z)``;
* ``m > 0`` uses the cosine component around the internal polar axis;
* ``m < 0`` uses the sine component around the internal polar axis.

The resulting ``Y00`` is ``1 / pi``.  Pass ``normalization='orthonormal'`` to
obtain the unit-sphere convention (``Y00 = 1 / sqrt(4 pi)``).  The
implementation uses a short recurrence instead of a SciPy call, so it is
differentiable and JIT-compatible with JAX.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._backend import xp_for


def ncoeff(ydeg: int) -> int:
    """Return the number of coefficients through spherical-harmonic degree."""

    ydeg = int(ydeg)
    if ydeg < 0:
        raise ValueError("ydeg must be non-negative")
    return (ydeg + 1) ** 2


def index_from_lm(l: int, m: int) -> int:
    """Return the flat ``starry``-compatible index for ``(l, m)``."""

    l = int(l)
    m = int(m)
    if l < 0 or abs(m) > l:
        raise ValueError(f"Require l >= 0 and -l <= m <= l; got l={l}, m={m}")
    return l * l + l + m


def lm_from_index(index: int) -> tuple[int, int]:
    """Return ``(l, m)`` for a flat coefficient index."""

    index = int(index)
    if index < 0:
        raise ValueError("index must be non-negative")
    l = math.isqrt(index)
    # The first index at degree l is l**2.  ``isqrt`` is therefore exact for
    # all indices; round up in the rare case of a floating caller converted to
    # an integer just below the boundary.
    while (l + 1) ** 2 <= index:
        l += 1
    while l * l > index:
        l -= 1
    m = index - l * l - l
    if abs(m) > l:  # defensive check for malformed integer input
        raise ValueError(f"Invalid harmonic index {index}")
    return l, m


def _associated_legendre(l: int, m: int, x: Any, xp):
    """Evaluate ``P_l^m(x)`` without the Condon--Shortley sign."""

    # Numerical clipping keeps round-off from producing NaNs in sqrt(1-x*x)
    # at the poles.  ``xp.clip`` is JAX-safe.
    x = xp.clip(x, -1.0, 1.0)
    m = int(m)
    l = int(l)
    if m < 0 or m > l:
        raise ValueError("Require 0 <= m <= l")

    # Starry's real polynomial basis omits the Condon--Shortley phase:
    # P_m^m(x) = (2m-1)!! (1-x^2)^(m/2).
    p_mm = xp.ones_like(x)
    if m:
        p_mm = math.prod(range(1, 2 * m, 2)) * xp.power(
            xp.maximum(0.0, 1.0 - x * x), 0.5 * m
        )
    if l == m:
        return p_mm

    p_m1m = x * (2 * m + 1) * p_mm
    if l == m + 1:
        return p_m1m

    p_prev = p_mm
    p_curr = p_m1m
    for ell in range(m + 2, l + 1):
        p_next = ((2 * ell - 1) * x * p_curr - (ell + m - 1) * p_prev) / (ell - m)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def real_sph_harm(
    l: int,
    m: int,
    lon: Any,
    lat: Any,
    *,
    normalization: str = "starry",
):
    """Evaluate one normalized real spherical harmonic.

    Inputs can be scalars or arrays.  The return shape is the broadcast shape
    of ``lon`` and ``lat`` and the backend follows the inputs.
    """

    xp = xp_for(lon, lat)
    physical_lon = xp.asarray(lon)
    physical_lat = xp.asarray(lat)
    physical_lon, physical_lat = xp.broadcast_arrays(physical_lon, physical_lat)
    l = int(l)
    m = int(m)
    if l < 0 or abs(m) > l:
        raise ValueError(f"Require l >= 0 and -l <= m <= l; got l={l}, m={m}")
    normalization = str(normalization).lower()
    if normalization not in {"starry", "orthonormal", "unit"}:
        raise ValueError("normalization must be 'starry' or 'orthonormal'")

    # Match starry's native map axes exactly. A direct use of physical
    # latitude as the Legendre coordinate rotates every non-uniform map and
    # does not reproduce frozen starry coefficient vectors.
    internal_x = xp.cos(physical_lat) * xp.sin(physical_lon)
    internal_y = xp.sin(physical_lat)
    internal_z = xp.cos(physical_lat) * xp.cos(physical_lon)
    internal_lon = xp.arctan2(internal_y, internal_x)

    abs_m = abs(m)
    p = _associated_legendre(l, abs_m, internal_z, xp)
    norm = math.sqrt(
        (2 * l + 1)
        / (4 * math.pi)
        * math.factorial(l - abs_m)
        / math.factorial(l + abs_m)
    )
    if m == 0:
        value = norm * p
    else:
        trig = (
            xp.cos(abs_m * internal_lon)
            if m > 0
            else xp.sin(abs_m * internal_lon)
        )
        value = math.sqrt(2.0) * norm * p * trig
    if normalization == "starry":
        value = value * (2.0 / math.sqrt(math.pi))
    return value


def real_sph_harm_all(
    ydeg: int,
    lon: Any,
    lat: Any,
    *,
    normalization: str = "starry",
):
    """Evaluate all coefficients through degree ``ydeg``.

    The final axis follows the flat ordering described in this module's
    docstring.  ``lon`` and ``lat`` may be arrays of any common broadcast
    shape.
    """

    xp = xp_for(lon, lat)
    lon = xp.asarray(lon)
    lat = xp.asarray(lat)
    values = [
        real_sph_harm(l, m, lon, lat, normalization=normalization)
        for l in range(int(ydeg) + 1)
        for m in range(-l, l + 1)
    ]
    return xp.stack(values, axis=-1)


def evaluate_map(
    coefficients: Any,
    lon: Any,
    lat: Any,
    *,
    normalization: str = "starry",
):
    """Evaluate a real spherical-harmonic map at ``(lon, lat)``."""

    xp = xp_for(coefficients, lon, lat)
    coefficients = xp.asarray(coefficients)
    expected = int(math.isqrt(int(coefficients.shape[-1])))
    if expected * expected != coefficients.shape[-1]:
        raise ValueError("The coefficient axis must contain (ydeg + 1)^2 values")
    basis = real_sph_harm_all(expected - 1, lon, lat, normalization=normalization)
    # Keep coefficient batch axes separate from coordinate axes.  This is
    # useful for posterior map samples: ``(draw, ncoeff)`` evaluated on a
    # ``(time,)`` grid returns ``(draw, time)`` rather than pairing draws with
    # epochs by NumPy's ordinary broadcasting rules.
    return xp.tensordot(coefficients, basis, axes=([-1], [-1]))


def harmonic_index_table(ydeg: int) -> np.ndarray:
    """Return an ``(ncoeff, 2)`` integer table of ``(l, m)`` pairs."""

    return np.asarray(
        [(l, m) for l in range(int(ydeg) + 1) for m in range(-l, l + 1)], dtype=int
    )
