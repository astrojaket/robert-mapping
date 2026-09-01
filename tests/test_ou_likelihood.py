"""Focused tests for the irregular-cadence OU innovations likelihood."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from robert_mapping.config import ConfigError, mapping_config_from_dict
from robert_mapping.inference import numpyro_backend as backend


def _dense_gaussian_log_likelihood(
    residual: np.ndarray,
    sigma: np.ndarray,
    times: np.ndarray,
    amplitude: float,
    timescale: float,
    jitter: float,
) -> float:
    covariance = amplitude**2 * np.exp(
        -np.abs(times[:, None] - times[None, :]) / timescale
    )
    covariance.flat[:: times.size + 1] += sigma**2 + jitter**2
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0.0
    return float(
        -0.5
        * (
            residual.size * np.log(2.0 * np.pi)
            + logdet
            + residual @ np.linalg.solve(covariance, residual)
        )
    )


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX is required for the OU recursion test",
)
def test_ou_innovations_matches_dense_covariance_on_irregular_times() -> None:
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist

    residual = np.array([0.0003, -0.0001, 0.0002, -0.0004, 0.00005])
    sigma = np.array([0.00007, 0.00008, 0.00006, 0.00009, 0.00007])
    times = np.array([0.0, 3.0, 11.0, 12.5, 30.0])
    amplitude = 0.00025
    timescale = 8.0
    jitter = 0.00004

    innovations = backend._ou_kalman_log_likelihood(
        residual,
        sigma,
        times,
        amplitude,
        timescale,
        jitter,
        likelihood="gaussian",
        student_t_nu=4.0,
        jax=jax,
        jnp=jnp,
        dist=dist,
    )
    dense = _dense_gaussian_log_likelihood(
        residual, sigma, times, amplitude, timescale, jitter
    )
    assert np.isclose(float(innovations), dense, rtol=2.0e-6, atol=2.0e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="JAX is required for the OU recursion test",
)
def test_ou_innovations_are_differentiable_and_student_t_is_finite() -> None:
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist

    residual = jnp.array([0.1, -0.04, 0.02])
    sigma = jnp.array([0.03, 0.03, 0.04])
    times = jnp.array([0.0, 0.5, 2.0])

    def objective(amplitude):
        return backend._ou_kalman_log_likelihood(
            residual,
            sigma,
            times,
            amplitude,
            1.2,
            0.01,
            likelihood="student_t",
            student_t_nu=5.0,
            jax=jax,
            jnp=jnp,
            dist=dist,
        )

    value = objective(0.08)
    gradient = jax.grad(objective)(0.08)
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))


def test_ou_inputs_require_sorted_times_and_positive_priors() -> None:
    with pytest.raises(ValueError, match="time_seconds is required"):
        backend.sample_harmonic_map(
            np.ones((3, 1)),
            np.ones(3),
            0.1,
            noise_model="ou",
            progress_bar=False,
        )
    with pytest.raises(ValueError, match="early to late"):
        backend.sample_harmonic_map(
            np.ones((3, 1)),
            np.ones(3),
            0.1,
            time_seconds=[0.0, 2.0, 1.0],
            noise_model="ou",
            progress_bar=False,
        )
    with pytest.raises(ValueError, match="ou_amplitude_prior_scale"):
        backend.sample_harmonic_map(
            np.ones((3, 1)),
            np.ones(3),
            0.1,
            time_seconds=[0.0, 1.0, 2.0],
            noise_model="ou",
            ou_amplitude_prior_scale=0.0,
            progress_bar=False,
        )


def test_ou_configuration_is_explicit_and_unit_friendly() -> None:
    config = mapping_config_from_dict(
        {
            "model": {
                "noise_model": "ou",
                "ou_amplitude_prior_scale_ppm": 62.0,
                "ou_timescale_prior_median_seconds": 950.0,
                "ou_timescale_prior_sigma_ln": 0.7,
                "jitter_prior_scale_ppm": 66.0,
            }
        }
    )
    assert config.model.noise_model == "ou"
    assert config.model.ou_amplitude_prior_scale_ppm == 62.0
    assert config.model.ou_timescale_prior_median_seconds == 950.0
    assert config.model.ou_timescale_prior_sigma_ln == 0.7
    assert config.model.jitter_prior_scale_ppm == 66.0

    with pytest.raises(ConfigError, match="ou_timescale_prior_median_seconds"):
        mapping_config_from_dict(
            {"model": {"ou_timescale_prior_median_seconds": 0.0}}
        )
