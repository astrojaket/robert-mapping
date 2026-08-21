r"""Band-integrated brightness-temperature conversion.

The eclipse maps in :mod:`robert_mapping` are dimensionless planet--star
contrasts.  This module converts those contrasts to an equivalent blackbody
brightness temperature for a selected wavelength band.  The convention is

.. math::

    C(T) = (R_p/R_*)^2 \left\langle
        B_\lambda(T) / I_{\lambda,*}
    \right\rangle_w,

where ``I_lambda,*`` is the stellar spectral radiance and the angle brackets
are a weighted mean over wavelength.  The stellar radiance must be in SI
``W m^-2 sr^-1 m^-1``.  A Phoenix spectrum can therefore be passed directly
when its wavelength grid has been converted to metres.

The inverse is evaluated from a monotonic, precomputed temperature grid.  A
single :class:`BandpassTemperatureConverter` can be reused for every pixel in
a posterior map, so a map conversion does not repeat the spectral setup.
"""

from __future__ import annotations

from numbers import Real
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


# CODATA 2018 exact SI values.
PLANCK_CONSTANT: Final[float] = 6.62607015e-34
SPEED_OF_LIGHT: Final[float] = 299_792_458.0
BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23

# A broad grid covers cool planets and the hot daysides used by the benchmark
# systems.  The grid is deliberately a module constant: constructing a
# converter still evaluates the band spectrum once, but never has to choose a
# temperature grid implicitly in a way that changes between calls.
DEFAULT_TEMPERATURE_GRID_K: Final[NDArray[np.float64]] = np.linspace(
    100.0, 100_000.0, 8_193, dtype=float
)


def _wavelength_to_metres(wavelength: ArrayLike, unit: str) -> NDArray[np.float64]:
    """Convert a wavelength array to metres and validate its values."""

    values = np.asarray(wavelength, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("wavelength must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("wavelength must contain finite, strictly positive values")

    normalised = str(unit).lower().replace("µ", "u").replace("μ", "u").strip()
    if normalised in {"m", "metre", "meter", "metres", "meters"}:
        scale = 1.0
    elif normalised in {"um", "micron", "microns", "micrometre", "micrometer"}:
        scale = 1.0e-6
    elif normalised in {"nm", "nanometre", "nanometer"}:
        scale = 1.0e-9
    elif normalised in {"angstrom", "angstroms", "a"}:
        scale = 1.0e-10
    else:
        raise ValueError("wavelength_unit must be m, micron, nm, or angstrom")
    return np.asarray(values * scale, dtype=float)


def _radiance_unit_scale(unit: str) -> float:
    """Return the factor that converts per-wavelength radiance to per metre."""

    normalised = str(unit).lower().replace("µ", "u").replace("μ", "u")
    normalised = normalised.replace(" ", "").replace("-", "_")
    if normalised in {"si", "per_m", "per_meter", "per_metre", "w_m2_sr_m"}:
        return 1.0
    if normalised in {"per_um", "per_micron", "w_m2_sr_um"}:
        # I_lambda d lambda is invariant.  One micron is 1e-6 metre.
        return 1.0e6
    raise ValueError("radiance_unit must be SI per metre or per micron")


def planck_radiance(wavelength_m: ArrayLike, temperature_k: ArrayLike) -> NDArray[np.float64] | np.float64:
    """Return blackbody spectral radiance ``B_lambda`` in SI units.

    Parameters
    ----------
    wavelength_m
        Wavelength in metres.  It may be broadcast with ``temperature_k``.
    temperature_k
        Blackbody temperature in kelvin.  Values must be finite and positive.

    Returns
    -------
    numpy.ndarray or numpy.float64
        Radiance in ``W m^-2 sr^-1 m^-1`` with the broadcast input shape.
    """

    wavelength = np.asarray(wavelength_m, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    try:
        wavelength, temperature = np.broadcast_arrays(wavelength, temperature)
    except ValueError as error:
        raise ValueError("wavelength_m and temperature_k cannot be broadcast") from error
    if wavelength.size == 0:
        raise ValueError("wavelength_m and temperature_k must not be empty")
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength_m must contain finite, strictly positive values")
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_k must contain finite, strictly positive values")

    exponent = PLANCK_CONSTANT * SPEED_OF_LIGHT / (
        wavelength * BOLTZMANN_CONSTANT * temperature
    )
    # exp(-x) / (1 - exp(-x)) is stable for both very large and small x.  It
    # avoids overflow for the cool end of a user-supplied temperature grid.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        denominator = -np.expm1(-exponent)
        radiance = (
            2.0
            * PLANCK_CONSTANT
            * SPEED_OF_LIGHT**2
            / wavelength**5
            * np.exp(-exponent)
            / denominator
        )
    if not np.all(np.isfinite(radiance)) or np.any(radiance <= 0.0):
        raise ValueError("Planck radiance is not finite and strictly positive")
    if radiance.ndim == 0:
        return np.float64(radiance)
    return np.asarray(radiance, dtype=float)


def blackbody_stellar_radiance(
    wavelength: ArrayLike,
    effective_temperature_k: float,
    *,
    wavelength_unit: str = "m",
    radiance_unit: str = "per_m",
) -> NDArray[np.float64]:
    """Build a blackbody stellar spectrum on a requested wavelength grid.

    The returned spectrum is spectral radiance per metre by default.  Set
    ``radiance_unit="per_micron"`` to return radiance per micrometre.  This
    helper is a transparent fallback when a Phoenix stellar spectrum is not
    available; it does not include a stellar radius or a distance factor.
    """

    if isinstance(effective_temperature_k, bool) or not isinstance(
        effective_temperature_k, Real
    ):
        raise ValueError("effective_temperature_k must be a finite positive number")
    temperature = float(effective_temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("effective_temperature_k must be a finite positive number")
    wavelength_m = _wavelength_to_metres(wavelength, wavelength_unit)
    radiance = np.asarray(planck_radiance(wavelength_m, temperature), dtype=float)
    scale = _radiance_unit_scale(radiance_unit)
    return np.asarray(radiance / scale, dtype=float)


def _validate_spectrum(
    wavelength: ArrayLike,
    stellar_radiance: ArrayLike,
    weights: ArrayLike,
    *,
    wavelength_unit: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    wavelength_m = _wavelength_to_metres(wavelength, wavelength_unit)
    stellar = np.asarray(stellar_radiance, dtype=float)
    band_weights = np.asarray(weights, dtype=float)
    if stellar.ndim != 1 or band_weights.ndim != 1:
        raise ValueError("stellar_radiance and weights must be one-dimensional arrays")
    if stellar.shape != wavelength_m.shape or band_weights.shape != wavelength_m.shape:
        raise ValueError("wavelength, stellar_radiance, and weights must have the same shape")
    if not np.all(np.isfinite(stellar)) or np.any(stellar <= 0.0):
        raise ValueError("stellar_radiance must contain finite, strictly positive values")
    if not np.all(np.isfinite(band_weights)) or np.any(band_weights < 0.0):
        raise ValueError("weights must contain finite, non-negative values")
    if float(np.sum(band_weights)) <= 0.0:
        raise ValueError("weights must have a positive sum")
    return wavelength_m, stellar, band_weights


def _validate_radius_ratio(radius_ratio: float) -> float:
    if isinstance(radius_ratio, bool) or not isinstance(radius_ratio, Real):
        raise ValueError("radius_ratio must be a finite positive number")
    value = float(radius_ratio)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("radius_ratio must be a finite positive number")
    return value


def _validate_temperature_grid(temperature_grid_k: ArrayLike) -> NDArray[np.float64]:
    grid = np.asarray(temperature_grid_k, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("temperature_grid_k must contain at least two temperatures")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("temperature_grid_k must contain finite, strictly positive values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("temperature_grid_k must be strictly increasing")
    return np.asarray(grid, dtype=float)


def _validate_contrast(contrast: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(contrast, dtype=float)
    if values.size == 0:
        raise ValueError("map_contrast must not be empty")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("map_contrast must contain finite, strictly positive values")
    return values


def band_integrated_contrast(
    temperature_k: ArrayLike,
    wavelength: ArrayLike,
    stellar_radiance: ArrayLike,
    weights: ArrayLike,
    radius_ratio: float,
    *,
    wavelength_unit: str = "m",
) -> NDArray[np.float64] | np.float64:
    """Calculate the planet--star contrast for one or more temperatures.

    ``radius_ratio`` is ``R_p/R_*``.  ``temperature_k`` can be a scalar or an
    array.  The output has the same shape as ``temperature_k`` and is suitable
    as the ``map_contrast`` input to :func:`brightness_temperature_from_contrast`.
    """

    wavelength_m, stellar, band_weights = _validate_spectrum(
        wavelength, stellar_radiance, weights, wavelength_unit=wavelength_unit
    )
    ratio = _validate_radius_ratio(radius_ratio)
    temperatures = np.asarray(temperature_k, dtype=float)
    if temperatures.size == 0:
        raise ValueError("temperature_k must not be empty")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("temperature_k must contain finite, strictly positive values")
    wavelength_shape = (1,) * temperatures.ndim + (wavelength_m.size,)
    radiance = np.asarray(
        planck_radiance(wavelength_m.reshape(wavelength_shape), np.expand_dims(temperatures, -1))
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        spectral_ratio = radiance / stellar
        contrasts = ratio**2 * np.average(spectral_ratio, axis=-1, weights=band_weights)
    contrasts = np.asarray(contrasts, dtype=float).reshape(temperatures.shape)
    if not np.all(np.isfinite(contrasts)) or np.any(contrasts <= 0.0):
        raise ValueError("band-integrated contrast is not finite and strictly positive")
    if contrasts.ndim == 0:
        return np.float64(contrasts)
    return np.asarray(contrasts, dtype=float)


class BandpassTemperatureConverter:
    """Convert a bandpass map contrast to brightness temperature.

    The stellar spectrum, radius ratio, and wavelength weights are validated
    once.  The constructor then precomputes a strictly monotonic contrast grid
    and uses linear interpolation for every conversion.  This is efficient for
    posterior arrays and ensures that every map pixel uses the same inversion.
    """

    def __init__(
        self,
        wavelength: ArrayLike,
        stellar_radiance: ArrayLike,
        weights: ArrayLike,
        radius_ratio: float,
        *,
        wavelength_unit: str = "m",
        temperature_grid_k: ArrayLike | None = None,
    ) -> None:
        self.wavelength_m, self.stellar_radiance, self.weights = _validate_spectrum(
            wavelength, stellar_radiance, weights, wavelength_unit=wavelength_unit
        )
        self.radius_ratio = _validate_radius_ratio(radius_ratio)
        selected_grid = (
            DEFAULT_TEMPERATURE_GRID_K if temperature_grid_k is None else temperature_grid_k
        )
        self.temperature_grid_k = _validate_temperature_grid(selected_grid).copy()
        self.contrast_grid = np.asarray(
            band_integrated_contrast(
                self.temperature_grid_k,
                self.wavelength_m,
                self.stellar_radiance,
                self.weights,
                self.radius_ratio,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(self.contrast_grid)) or np.any(self.contrast_grid <= 0.0):
            raise ValueError("precomputed contrast grid must be finite and strictly positive")
        if np.any(np.diff(self.contrast_grid) <= 0.0):
            raise ValueError("the precomputed band contrast grid must be strictly increasing")

    def contrast_from_temperature(
        self, temperature_k: ArrayLike
    ) -> NDArray[np.float64] | np.float64:
        """Return exact band contrast for one or more temperatures."""

        return band_integrated_contrast(
            temperature_k,
            self.wavelength_m,
            self.stellar_radiance,
            self.weights,
            self.radius_ratio,
        )

    def temperature_from_contrast(
        self, map_contrast: ArrayLike
    ) -> NDArray[np.float64] | np.float64:
        """Return interpolated brightness temperature for a map contrast.

        The input must lie within the precomputed grid range.  Raising for an
        out-of-range value prevents silent clipping of a posterior map.
        """

        values = _validate_contrast(map_contrast)
        minimum = float(self.contrast_grid[0])
        maximum = float(self.contrast_grid[-1])
        if np.any(values < minimum) or np.any(values > maximum):
            raise ValueError(
                "map_contrast is outside the precomputed temperature range "
                f"({minimum:.6g} to {maximum:.6g})"
            )
        temperatures = np.interp(values.ravel(), self.contrast_grid, self.temperature_grid_k)
        temperatures = temperatures.reshape(values.shape)
        if temperatures.ndim == 0:
            return np.float64(temperatures)
        return np.asarray(temperatures, dtype=float)

    # Short aliases are useful in map-processing notebooks.
    contrast = contrast_from_temperature
    temperature = temperature_from_contrast


def brightness_temperature_from_contrast(
    map_contrast: ArrayLike,
    wavelength: ArrayLike,
    stellar_radiance: ArrayLike,
    weights: ArrayLike,
    radius_ratio: float,
    *,
    wavelength_unit: str = "m",
    temperature_grid_k: ArrayLike | None = None,
) -> NDArray[np.float64] | np.float64:
    """Convert a scalar or map-shaped contrast to band brightness temperature.

    ``map_contrast`` is dimensionless and must be the planet--star contrast
    represented by each map value.  The returned array has exactly the same
    shape.  The conversion is equivalent to constructing a
    :class:`BandpassTemperatureConverter` and calling its
    :meth:`~BandpassTemperatureConverter.temperature_from_contrast` method.
    """

    converter = BandpassTemperatureConverter(
        wavelength,
        stellar_radiance,
        weights,
        radius_ratio,
        wavelength_unit=wavelength_unit,
        temperature_grid_k=temperature_grid_k,
    )
    return converter.temperature_from_contrast(map_contrast)


# Names used by earlier analysis notebooks and by the Hammond post-processing
# scripts.  Keep them as simple aliases so there is one implementation.
band_brightness_temperature = brightness_temperature_from_contrast
temperature_from_contrast = brightness_temperature_from_contrast
blackbody_radiance = blackbody_stellar_radiance


__all__ = [
    "BandpassTemperatureConverter",
    "DEFAULT_TEMPERATURE_GRID_K",
    "PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
    "BOLTZMANN_CONSTANT",
    "band_brightness_temperature",
    "band_integrated_contrast",
    "blackbody_radiance",
    "blackbody_stellar_radiance",
    "brightness_temperature_from_contrast",
    "planck_radiance",
    "temperature_from_contrast",
]
