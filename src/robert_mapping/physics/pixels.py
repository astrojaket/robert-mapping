"""Equal-area spherical pixels and harmonic conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import qr

from ._backend import xp_for
from .harmonics import ncoeff, real_sph_harm_all


@dataclass(frozen=True)
class EqualAreaPixels:
    """Centres and areas of an equal-area latitude--longitude grid.

    The grid is uniform in ``mu = sin(latitude)`` and longitude.  Every cell
    has exactly the same area, ``4 pi / (nlat * nlon)``.  Cell centres are
    returned, so this object is useful both for map visualisation and for the
    pixel prior used by the inference layer.
    """

    lon: np.ndarray
    lat: np.ndarray
    area: np.ndarray
    nlat: int
    nlon: int

    @property
    def npix(self) -> int:
        """Number of pixels."""

        return int(self.lon.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Underlying ``(nlat, nlon)`` grid shape."""

        return self.nlat, self.nlon


def equal_area_pixels(
    nlat: int | None = None,
    nlon: int | None = None,
    *,
    npix: int | None = None,
) -> EqualAreaPixels:
    """Construct an equal-area pixel grid.

    Parameters
    ----------
    nlat
        Number of bands in uniform ``sin(latitude)``.
    nlon
        Number of longitude cells per band.  The default is ``2 * nlat``.
    npix
        Optional exact number of pixels.  When supplied, a Fibonacci
        equal-area sphere is returned.  This is useful for the small Hammond
        quick cases, whose pixel counts are 16 (degree 2) and 62 (degree 4).
    """

    if npix is not None:
        if nlat is not None or nlon is not None:
            raise ValueError("Pass either nlat/nlon or npix, not both")
        return fibonacci_pixels(npix)
    if nlat is None:
        raise ValueError("nlat is required unless npix is supplied")
    nlat = int(nlat)
    if nlat < 1:
        raise ValueError("nlat must be at least 1")
    nlon = 2 * nlat if nlon is None else int(nlon)
    if nlon < 1:
        raise ValueError("nlon must be at least 1")

    mu = -1.0 + (np.arange(nlat, dtype=float) + 0.5) * (2.0 / nlat)
    lon_1d = -np.pi + (np.arange(nlon, dtype=float) + 0.5) * (2.0 * np.pi / nlon)
    lat_1d = np.arcsin(mu)
    lon, lat = np.meshgrid(lon_1d, lat_1d, indexing="xy")
    npix = nlat * nlon
    area = np.full(npix, 4.0 * np.pi / npix, dtype=float)
    return EqualAreaPixels(lon.ravel(), lat.ravel(), area, nlat, nlon)


def fibonacci_pixels(npix: int) -> EqualAreaPixels:
    """Construct an exact-size, approximately equal-area Fibonacci sphere.

    The latitude bands have equal solid angle and the longitude is advanced
    by the golden angle.  Unlike a rectangular latitude--longitude grid this
    permits any requested pixel count.  Pixel centres are returned, with a
    common area of ``4 pi / npix``.
    """

    npix = int(npix)
    if npix < 1:
        raise ValueError("npix must be at least 1")
    index = np.arange(npix, dtype=float)
    mu = 1.0 - 2.0 * (index + 0.5) / npix
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    lon = (index * golden_angle + np.pi) % (2.0 * np.pi) - np.pi
    lat = np.arcsin(mu)
    area = np.full(npix, 4.0 * np.pi / npix, dtype=float)
    # nlat=npix, nlon=1 records that the grid is unstructured while retaining
    # the shape metadata required by the dataclass.
    return EqualAreaPixels(lon, lat, area, npix, 1)


def pixels_for_ydeg(ydeg: int, *, oversample: int = 3) -> EqualAreaPixels:
    """Return a small exact-size grid for a Hammond-style harmonic map.

    With the default oversampling, this follows the pixel counts used by the
    original ``starry`` workflow: 16 pixels for ``ydeg=2`` and 62 pixels for
    ``ydeg=4``.  Other degrees use a conservative ``oversample * ncoeff``
    count.  The returned Fibonacci sphere is equal-area to the precision of
    its point-centre quadrature.
    """

    ydeg = int(ydeg)
    oversample = int(oversample)
    if ydeg < 0 or oversample < 1:
        raise ValueError("ydeg must be non-negative and oversample >= 1")
    known = {2: 16, 4: 62}
    count = known.get(ydeg, oversample * ncoeff(ydeg))
    return fibonacci_pixels(count)


def mollweide_pixels_for_ydeg(
    ydeg: int, *, oversample: int = 3
) -> EqualAreaPixels:
    """Return the regular Mollweide grid used by ``starry`` 1.1.x."""

    ydeg = int(ydeg)
    oversample = int(oversample)
    if ydeg < 0 or oversample < 1:
        raise ValueError("ydeg must be non-negative and oversample >= 1")
    if ydeg <= 1:
        oversample = max(oversample, 3)
    target = oversample * ncoeff(ydeg)
    ny = int(np.sqrt(target * np.pi / 4.0))
    nx = 2 * ny
    projected_y, projected_x = np.meshgrid(
        np.sqrt(2.0) * np.linspace(-1.0, 1.0, ny),
        2.0 * np.sqrt(2.0) * np.linspace(-1.0, 1.0, nx),
    )
    projected_x = projected_x.ravel()
    projected_y = projected_y.ravel()
    inside = (projected_y / np.sqrt(2.0)) ** 2 + (
        projected_x / (2.0 * np.sqrt(2.0))
    ) ** 2 <= 1.0
    projected_x = projected_x[inside]
    projected_y = projected_y[inside]
    theta = np.arcsin(projected_y / np.sqrt(2.0))
    latitude = np.arcsin((2.0 * theta + np.sin(2.0 * theta)) / np.pi)
    longitude = 1.5 * np.pi + np.pi * projected_x / (
        2.0 * np.sqrt(2.0) * np.cos(theta)
    )
    latitude = np.append(latitude, (-np.pi / 2.0, 0.0, 0.0, np.pi / 2.0))
    longitude = np.append(
        longitude, (1.5 * np.pi, 1.5 * np.pi, 2.5 * np.pi, 1.5 * np.pi)
    )
    longitude = longitude - 1.5 * np.pi
    order = np.lexsort((longitude, latitude))
    longitude = longitude[order]
    latitude = latitude[order]
    count = latitude.size
    return EqualAreaPixels(
        longitude,
        latitude,
        np.full(count, 4.0 * np.pi / count, dtype=float),
        count,
        1,
    )


def starry_pixel_transforms(
    ydeg: int,
    *,
    oversample: int = 3,
    regularization: float = 1.0e-6,
) -> tuple[EqualAreaPixels, np.ndarray, np.ndarray]:
    """Return frozen-compatible Mollweide ``Y2P`` and ``P2Y`` matrices."""

    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    pixels = mollweide_pixels_for_ydeg(ydeg, oversample=oversample)
    y2p = np.asarray(pixel_design_matrix(pixels, ydeg), dtype=float)
    p2y = np.linalg.solve(
        y2p.T @ y2p + float(regularization) * np.eye(ncoeff(ydeg)),
        y2p.T,
    )
    return pixels, y2p, p2y


@dataclass(frozen=True)
class HarmonicAnchorTransform:
    """Rank-revealing square transform from map anchors to harmonics.

    The anchors are selected from the existing starry-compatible Mollweide
    evaluation grid. If ``A`` is the evaluation matrix at the selected
    anchors, then ``anchor_to_harmonics`` is ``A^-1`` and maps positive anchor
    values to the degree-``ydeg`` harmonic coefficients.
    """

    pixels: EqualAreaPixels
    evaluation_matrix: np.ndarray
    anchor_indices: np.ndarray
    anchor_to_harmonics: np.ndarray
    rank: int
    condition_number: float

    @property
    def anchor_longitude_degrees(self) -> np.ndarray:
        """Return selected anchor longitudes in degrees."""

        return np.rad2deg(np.asarray(self.pixels.lon)[self.anchor_indices])

    @property
    def anchor_latitude_degrees(self) -> np.ndarray:
        """Return selected anchor latitudes in degrees."""

        return np.rad2deg(np.asarray(self.pixels.lat)[self.anchor_indices])


def rank_revealing_anchor_transform(
    ydeg: int,
    *,
    oversample: int = 3,
    rank_tolerance: float = 1.0e-12,
) -> HarmonicAnchorTransform:
    """Select a deterministic full-rank anchor set for a harmonic map.

    Pivoted QR is applied to the transpose of the existing starry-compatible
    Mollweide evaluation matrix. Its first ``(ydeg + 1)**2`` pivot rows are
    the anchors. The returned square inverse lets the inference layer sample
    one positive value per anchor while retaining the usual harmonic output.
    """

    ydeg = int(ydeg)
    oversample = int(oversample)
    if ydeg < 0 or oversample < 1:
        raise ValueError("ydeg must be non-negative and oversample >= 1")
    if not np.isfinite(rank_tolerance) or rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be a positive finite number")

    pixels = mollweide_pixels_for_ydeg(ydeg, oversample=oversample)
    evaluation_matrix = np.asarray(pixel_design_matrix(pixels, ydeg), dtype=float)
    n_harmonics = ncoeff(ydeg)
    if pixels.npix < n_harmonics:
        raise ValueError("The evaluation grid must contain at least ncoeff anchors")

    _, triangular, pivot = qr(evaluation_matrix.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    scale = max(float(diagonal[0]), np.finfo(float).eps)
    rank = int(np.count_nonzero(diagonal > float(rank_tolerance) * scale))
    if rank < n_harmonics:
        raise ValueError(
            f"The Mollweide evaluation grid has rank {rank}; "
            f"{n_harmonics} anchors are required"
        )
    anchor_indices = np.asarray(pivot[:n_harmonics], dtype=int)
    anchor_matrix = evaluation_matrix[anchor_indices]
    anchor_to_harmonics = np.linalg.solve(
        anchor_matrix, np.eye(n_harmonics, dtype=float)
    )
    condition_number = float(np.linalg.cond(anchor_matrix))
    return HarmonicAnchorTransform(
        pixels=pixels,
        evaluation_matrix=evaluation_matrix,
        anchor_indices=anchor_indices,
        anchor_to_harmonics=anchor_to_harmonics,
        rank=rank,
        condition_number=condition_number,
    )


def pixel_design_matrix(pixels: EqualAreaPixels, ydeg: int, *, backend=None):
    """Return harmonic values at pixel centres.

    The matrix has shape ``(pixels.npix, (ydeg + 1)**2)`` and maps harmonic
    coefficients to pixel values.
    """

    xp = backend or xp_for(pixels.lon, pixels.lat)
    lon = xp.asarray(pixels.lon)
    lat = xp.asarray(pixels.lat)
    return real_sph_harm_all(int(ydeg), lon, lat)


def _weighted_pseudoinverse(design, area, xp):
    """Build the area-weighted least-squares inverse of a design matrix."""

    # Solve the weighted normal equations directly.  This avoids an expensive
    # SVD for the small fixed transforms and keeps the operation available in
    # JAX's CPU and accelerator backends.  The intended grids are
    # overdetermined (more pixels than harmonic coefficients), so the Gram
    # matrix is well conditioned for the supported degrees.
    area = xp.asarray(area)
    gram = xp.einsum("pc,p,pk->ck", design, area, design)
    rhs = xp.einsum("pc,p->cp", design, area)
    return xp.linalg.solve(gram, rhs)


def pixels_to_harmonics(
    pixel_values: Any,
    pixels: EqualAreaPixels,
    ydeg: int,
    *,
    area_weighted: bool = True,
):
    """Fit harmonic coefficients to values on an equal-area pixel grid.

    ``pixel_values`` may have leading batch dimensions.  The final axis must
    have length ``pixels.npix``.  The transform is differentiable when the
    values are JAX arrays; the grid and pseudoinverse are fixed numerically.
    """

    xp = xp_for(pixel_values)
    values = xp.asarray(pixel_values)
    if values.shape[-1] != pixels.npix:
        raise ValueError(
            f"Expected {pixels.npix} pixel values on the final axis; got {values.shape[-1]}"
        )
    design = pixel_design_matrix(pixels, ydeg, backend=xp)
    if area_weighted:
        inverse = _weighted_pseudoinverse(design, pixels.area, xp)
    else:
        inverse = xp.linalg.pinv(design)
    return xp.einsum("...p,cp->...c", values, inverse)


def harmonics_to_pixels(
    coefficients: Any,
    pixels: EqualAreaPixels,
    ydeg: int | None = None,
):
    """Evaluate harmonic coefficients at pixel centres."""

    xp = xp_for(coefficients)
    coeff = xp.asarray(coefficients)
    if ydeg is None:
        n = int(coeff.shape[-1])
        root = int(np.sqrt(n))
        if root * root != n:
            raise ValueError("The coefficient axis must contain a square number of values")
        ydeg = root - 1
    expected = ncoeff(ydeg)
    if coeff.shape[-1] != expected:
        raise ValueError(f"Expected {expected} coefficients for ydeg={ydeg}")
    design = pixel_design_matrix(pixels, ydeg, backend=xp)
    return xp.einsum("...c,pc->...p", coeff, design)


def pixel_area_map_to_flux(pixel_values: Any, pixels: EqualAreaPixels):
    """Integrate a full-sphere pixel map over solid angle."""

    xp = xp_for(pixel_values)
    values = xp.asarray(pixel_values)
    if values.shape[-1] != pixels.npix:
        raise ValueError("Final axis does not match the number of pixels")
    return xp.sum(values * xp.asarray(pixels.area), axis=-1)
