"""Tests for Gaussian-process covariance and inference backends."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from robert_mapping import noise
from robert_mapping.compute import resolve_plan
from robert_mapping.inference import numpyro_backend


def test_registry_import_does_not_import_numpy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from robert_mapping.noise import available_kernels; "
            "assert available_kernels(); assert 'numpy' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "kernel", ["exponential", "matern32", "squared_exponential"]
)
def test_builtin_covariances_are_symmetric_and_positive_definite(kernel: str) -> None:
    times = np.array([0.0, 19.0, 83.0, 141.0])
    covariance = noise.gp_covariance(times, 0.4, 55.0, kernel=kernel)

    assert covariance.shape == (times.size, times.size)
    assert np.allclose(covariance, covariance.T)
    assert np.allclose(np.diag(covariance), 0.4**2)
    assert np.linalg.eigvalsh(covariance).min() > 0.0


def test_exponential_gp_matches_dense_ou_covariance_likelihood() -> None:
    times = np.array([0.0, 35.0, 120.0, 240.0])
    residual = np.array([0.02, -0.01, 0.005, 0.008])
    sigma = np.array([0.01, 0.02, 0.015, 0.012])
    amplitude = 0.025
    timescale = 90.0
    jitter = 0.003
    covariance = amplitude**2 * np.exp(
        -np.abs(times[:, None] - times[None, :]) / timescale
    ) + np.diag(sigma**2 + jitter**2)
    sign, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (
        residual.size * np.log(2.0 * np.pi)
        + logdet
        + residual @ np.linalg.solve(covariance, residual)
    )

    actual = noise.gp_log_likelihood(
        residual,
        sigma,
        times,
        amplitude,
        timescale,
        jitter,
        kernel="exponential",
    )

    assert sign == 1.0
    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    ou_actual = numpyro_backend._ou_kalman_log_likelihood(
        jnp.asarray(residual),
        jnp.asarray(sigma),
        jnp.asarray(times),
        amplitude,
        timescale,
        jitter,
        likelihood="gaussian",
        student_t_nu=4.0,
        jax=jax,
        jnp=jnp,
        dist=None,
    )
    assert float(ou_actual) == pytest.approx(expected, rel=1.0e-5, abs=1.0e-5)


def test_kernel_registry_is_explicit_and_can_add_a_student_kernel() -> None:
    name = "test_linear_decay"
    noise.register_kernel(name, lambda distance, xp: xp.exp(-2.0 * distance))
    try:
        covariance = noise.gp_covariance([0.0, 1.0], 2.0, 1.0, kernel=name)
        assert name in noise.available_kernels()
        assert covariance[0, 1] == pytest.approx(4.0 * np.exp(-2.0))
    finally:
        noise._KERNELS.pop(name, None)
    with pytest.raises(ValueError, match="already registered"):
        noise.register_kernel("matern32", lambda distance, xp: xp.ones_like(distance))


def test_covariance_changes_when_kernel_parameters_change() -> None:
    times = np.array([0.0, 10.0, 30.0])
    short = noise.gp_covariance(times, 0.2, 8.0, kernel="matern32")
    long = noise.gp_covariance(times, 0.2, 80.0, kernel="matern32")
    high_amplitude = noise.gp_covariance(times, 0.4, 8.0, kernel="matern32")

    assert not np.allclose(short, long)
    assert np.allclose(high_amplitude, 4.0 * short)


def test_repeated_times_keep_a_valid_jittered_covariance() -> None:
    times = np.array([0.0, 20.0, 20.0, 95.0])
    covariance = noise.gp_covariance(times, 0.3, 40.0, kernel="exponential")
    total_covariance = covariance + np.diag(np.full(times.size, 0.01**2))

    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > -1.0e-12
    assert np.linalg.eigvalsh(total_covariance).min() > 0.0
    assert np.isfinite(
        noise.gp_log_likelihood(
            np.zeros(times.size),
            np.full(times.size, 0.01),
            times,
            0.3,
            40.0,
            0.0,
            kernel="exponential",
        )
    )


def test_zero_gp_amplitude_matches_independent_gaussian_limit() -> None:
    residual = np.array([0.01, -0.03, 0.015])
    sigma = np.array([0.02, 0.015, 0.025])
    jitter = 0.004
    variance = sigma**2 + jitter**2
    expected = -0.5 * np.sum(
        np.log(2.0 * np.pi * variance) + residual**2 / variance
    )

    actual = noise.gp_log_likelihood(
        residual,
        sigma,
        np.array([0.0, 60.0, 130.0]),
        0.0,
        40.0,
        jitter,
        kernel="matern32",
    )

    assert actual == pytest.approx(expected, rel=1.0e-12, abs=1.0e-12)


def test_gp_likelihood_changes_with_kernel_choice() -> None:
    times = np.array([0.0, 12.0, 68.0, 141.0])
    residual = np.array([0.02, -0.01, 0.03, 0.005])
    sigma = np.full(times.size, 0.01)
    values = [
        noise.gp_log_likelihood(
            residual,
            sigma,
            times,
            0.025,
            55.0,
            0.002,
            kernel=kernel,
        )
        for kernel in noise.available_kernels()
    ]

    assert len(np.unique(np.round(values, decimals=8))) == len(values)


@pytest.mark.parametrize("kernel", ["exponential", "matern32", "squared_exponential"])
def test_gp_log_likelihood_has_finite_parameter_gradients(kernel: str) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    residual = jnp.array([0.01, -0.02, 0.005])
    sigma = jnp.full(3, 0.01)
    times = jnp.array([0.0, 30.0, 75.0])

    def log_likelihood(amplitude, log_timescale, jitter):
        return noise.gp_log_likelihood(
            residual,
            sigma,
            times,
            amplitude,
            jnp.exp(log_timescale),
            jitter,
            kernel=kernel,
            xp=jnp,
        )

    gradients = jax.grad(log_likelihood, argnums=(0, 1, 2))(
        0.03, jnp.log(40.0), 0.002
    )

    assert all(np.isfinite(float(value)) for value in gradients)
    assert all(float(value) != 0.0 for value in gradients)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"likelihood": "student_t"}, "likelihood='gaussian' only"),
        ({"fit_white_jitter": True}, "already samples diagonal jitter"),
        ({"gp_kernel": "not_registered"}, "unknown GP kernel"),
        ({"gp_max_points": 3}, "above gp_max_points=3"),
    ],
)
def test_gp_backend_rejects_unsupported_controls_before_sampling(
    kwargs, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        numpyro_backend.sample_harmonic_map(
            np.ones((4, 1)),
            np.ones(4),
            0.01,
            time_seconds=np.arange(4, dtype=float),
            noise_model="gp",
            warmup=2,
            draws=2,
            chains=1,
            progress_bar=False,
            **kwargs,
        )


@pytest.mark.parametrize("sampler", ["positive", "harmonic"])
def test_gp_runs_a_short_real_nuts_chain_for_each_sampler(sampler: str) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("numpyro")
    n_observations = 24
    times = np.arange(n_observations, dtype=float) * 45.0
    sigma = np.full(n_observations, 0.015)
    noise_residual = 0.004 * np.sin(np.arange(n_observations) * 0.8)
    common = dict(
        time_seconds=times,
        noise_model="gp",
        gp_kernel="matern32",
        gp_amplitude_prior_scale=0.01,
        gp_timescale_prior_median=120.0,
        gp_timescale_prior_sigma_ln=0.6,
        gp_max_points=30,
        jitter_prior_scale=0.005,
        warmup=5,
        draws=4,
        chains=1,
        seed=17,
        progress_bar=False,
        compute_plan=resolve_plan(max_cpus=1, threads=1, workers=1),
    )

    if sampler == "positive":
        design = np.ones((n_observations, 1))
        observed = 1.0 + 0.0015 + noise_residual
        result = numpyro_backend.sample_positive_map(
            design,
            observed,
            sigma,
            stellar_flux=1.0,
            pixel_prior_mean=0.0015,
            pixel_log_sigma=0.5,
            **common,
        )
    else:
        design = np.ones((n_observations, 1))
        observed = 1.0 + noise_residual
        result = numpyro_backend.sample_harmonic_map(
            design,
            observed,
            sigma,
            stellar_flux=0.0,
            coefficient_prior_mean=1.0,
            coefficient_prior_sigma=0.2,
            **common,
        )

    assert result.samples["gp_amplitude"].shape == (4,)
    assert result.samples["gp_timescale"].shape == (4,)
    assert result.samples["jitter"].shape == (4,)
    assert result.grouped_samples is not None
    assert result.grouped_samples["gp_amplitude"].shape == (1, 4)
    assert result.chain_method == "sequential"
