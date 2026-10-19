"""Gaussian-process noise kernels with optional NumPy and JAX backends.

The kernel registry and :func:`available_kernels` use only the standard
library. This lets configuration loading validate kernel names before it
imports a numerical backend.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any


KernelFunction = Callable[[Any, Any], Any]


def _exponential(distance: Any, xp: Any) -> Any:
    return xp.exp(-distance)


def _matern32(distance: Any, xp: Any) -> Any:
    scaled = math.sqrt(3.0) * distance
    return (1.0 + scaled) * xp.exp(-scaled)


def _squared_exponential(distance: Any, xp: Any) -> Any:
    return xp.exp(-0.5 * distance**2)


_KERNELS: dict[str, KernelFunction] = {
    "exponential": _exponential,
    "matern32": _matern32,
    "squared_exponential": _squared_exponential,
}


def available_kernels() -> tuple[str, ...]:
    """Return registered kernel names without importing numerical packages."""

    return tuple(sorted(_KERNELS))


def register_kernel(
    name: str, function: KernelFunction, *, replace: bool = False
) -> None:
    """Register a correlation function of scaled distance and array module.

    ``function(distance, xp)`` must return correlation values with a final
    dimension matching ``distance``. It must support both NumPy and JAX NumPy
    operations through ``xp``. To make a custom kernel available to fresh CLI
    runs and spawned workers, call this function from an importable module that
    the CLI and worker processes load; a registration made at runtime exists
    only in the current process.
    """

    normalized = str(name).strip().lower()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError("kernel name must contain only letters, numbers, and underscores")
    if not callable(function):
        raise TypeError("kernel function must be callable")
    if normalized in _KERNELS and not replace:
        raise ValueError(
            f"GP kernel {normalized!r} is already registered; "
            "pass replace=True to replace it"
        )
    _KERNELS[normalized] = function


def gp_covariance(
    time_seconds: Any,
    amplitude: Any,
    timescale: Any,
    kernel: str = "matern32",
    xp: Any | None = None,
) -> Any:
    """Build a dense stationary GP covariance in the requested array module."""

    if xp is None:
        import numpy as xp  # type: ignore[no-redef]
    check_values = getattr(xp, "__name__", "") == "numpy"

    name = str(kernel).strip().lower()
    try:
        correlation = _KERNELS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown GP kernel {kernel!r}; available kernels: {', '.join(available_kernels())}"
        ) from exc

    times = xp.asarray(time_seconds)
    if times.ndim != 1:
        raise ValueError("time_seconds must be a one-dimensional array")
    if check_values:
        import numpy as np

        amplitude_value = np.asarray(amplitude, dtype=float)
        timescale_value = np.asarray(timescale, dtype=float)
        if (
            amplitude_value.ndim != 0
            or not np.isfinite(amplitude_value)
            or amplitude_value < 0.0
        ):
            raise ValueError("amplitude must be a finite non-negative scalar")
        if (
            timescale_value.ndim != 0
            or not np.isfinite(timescale_value)
            or timescale_value <= 0.0
        ):
            raise ValueError("timescale must be a positive finite scalar")
        if not np.all(np.isfinite(times)):
            raise ValueError("time_seconds must contain only finite values")
    # Numeric validation is done before model construction in the inference
    # API. Avoid converting tracers to Python scalars here, so this function
    # remains differentiable when amplitude and timescale are JAX samples.
    distance = xp.abs(times[:, None] - times[None, :]) / timescale
    return amplitude**2 * correlation(distance, xp)


def gp_log_likelihood(
    residual: Any,
    sigma: Any,
    time_seconds: Any,
    amplitude: Any,
    timescale: Any,
    jitter: Any,
    kernel: str = "matern32",
    xp: Any | None = None,
) -> Any:
    """Return a dense Gaussian-process log likelihood.

    The covariance is the GP covariance plus diagonal measurement variance
    ``sigma**2 + jitter**2``. This calculation has cubic time and quadratic
    memory cost in the number of observations.
    """

    if xp is None:
        import numpy as xp  # type: ignore[no-redef]
    check_values = getattr(xp, "__name__", "") == "numpy"

    values = xp.asarray(residual)
    errors = xp.asarray(sigma)
    times = xp.asarray(time_seconds)
    if (
        values.ndim != 1
        or errors.shape != values.shape
        or times.shape != values.shape
    ):
        raise ValueError("residual, sigma, and time_seconds must be matching vectors")
    if values.size < 1:
        raise ValueError("GP likelihood requires at least one observation")
    if check_values:
        import numpy as np

        jitter_value = np.asarray(jitter, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("residual must contain only finite values")
        if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
            raise ValueError("sigma must contain only positive finite values")
        if (
            jitter_value.ndim != 0
            or not np.isfinite(jitter_value)
            or jitter_value < 0.0
        ):
            raise ValueError("jitter must be a finite non-negative scalar")
    covariance = gp_covariance(times, amplitude, timescale, kernel=kernel, xp=xp)
    diagonal = errors**2 + jitter**2
    covariance = covariance + xp.diag(diagonal)
    factor = xp.linalg.cholesky(covariance)
    if check_values:
        from scipy.linalg import solve_triangular
    else:
        from jax.scipy.linalg import solve_triangular
    whitened = solve_triangular(factor, values, lower=True)
    return -0.5 * (
        values.size * math.log(2.0 * math.pi)
        + 2.0 * xp.sum(xp.log(xp.diag(factor)))
        + xp.sum(whitened**2)
    )


__all__ = [
    "available_kernels",
    "gp_covariance",
    "gp_log_likelihood",
    "register_kernel",
]
