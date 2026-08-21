"""Map rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._backend import xp_for
from .harmonics import evaluate_map


def render_map(
    coefficients: Any,
    *,
    nlon: int = 360,
    nlat: int = 180,
    longitude_range: tuple[float, float] = (-np.pi, np.pi),
    latitude_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
):
    """Render a harmonic map on a regular longitude--latitude grid.

    The returned tuple is ``(longitude, latitude, values)``.  ``longitude``
    and ``latitude`` are one-dimensional arrays, while ``values`` has shape
    ``(..., nlat, nlon)`` for any leading coefficient batch dimensions.
    """

    nlon = int(nlon)
    nlat = int(nlat)
    if nlon < 2 or nlat < 2:
        raise ValueError("nlon and nlat must be at least 2")
    xp = xp_for(coefficients)
    lon = xp.linspace(longitude_range[0], longitude_range[1], nlon)
    lat = xp.linspace(latitude_range[0], latitude_range[1], nlat)
    grid_lon, grid_lat = xp.meshgrid(lon, lat, indexing="xy")
    values = evaluate_map(coefficients, grid_lon, grid_lat)
    return lon, lat, values


def render_visible_map(
    coefficients: Any,
    subobserver_lon: Any = 0.0,
    subobserver_lat: Any = 0.0,
    *,
    nlon: int = 360,
    nlat: int = 180,
):
    """Render a map in observer-centred coordinates.

    ``subobserver_lon`` and ``subobserver_lat`` specify the map location at
    the centre of the projected disc.  The returned grid spans a full sphere;
    callers can mask the far hemisphere with ``cos(latitude) <= 0`` after
    shifting coordinates if they need a plotting surface.
    """

    xp = xp_for(coefficients, subobserver_lon, subobserver_lat)
    rel_lon = xp.linspace(-np.pi, np.pi, int(nlon))
    rel_lat = xp.linspace(-np.pi / 2, np.pi / 2, int(nlat))
    lon_grid, lat_grid = xp.meshgrid(rel_lon, rel_lat, indexing="xy")

    # Build the same observer-centred tangent basis used by the projected-disc
    # integrator.  ``lon_grid`` and ``lat_grid`` describe a full spherical
    # display; the visible hemisphere is the region with cos(lat_grid) > 0
    # around the central meridian.
    lon0 = xp.asarray(subobserver_lon)
    lat0 = xp.asarray(subobserver_lat)
    x = xp.cos(lat_grid) * xp.cos(lon_grid)
    y = xp.cos(lat_grid) * xp.sin(lon_grid)
    z = xp.sin(lat_grid)
    cos_lon = xp.cos(lon0)
    sin_lon = xp.sin(lon0)
    cos_lat = xp.cos(lat0)
    sin_lat = xp.sin(lat0)
    rx = x * (-sin_lon) + y * (-sin_lat * cos_lon) + z * (cos_lat * cos_lon)
    ry = x * cos_lon + y * (-sin_lat * sin_lon) + z * (cos_lat * sin_lon)
    rz = y * cos_lat + z * sin_lat
    map_lon = xp.arctan2(ry, rx)
    map_lat = xp.arcsin(xp.clip(rz, -1.0, 1.0))
    values = evaluate_map(coefficients, map_lon, map_lat)
    return rel_lon, rel_lat, values
