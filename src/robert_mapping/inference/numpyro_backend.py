"""NumPyro models with no dependency on the physics implementation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class NumpyroRun:
    """Portable sampling output."""

    samples: dict[str, NDArray[np.float64]]
    extra_fields: dict[str, NDArray[np.float64]]
    sampler: Any
    grouped_samples: dict[str, NDArray[np.float64]] | None = None


def _imports():
    try:
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - exercised in a minimal install
        raise RuntimeError(
            "NumPyro inference is not installed. Activate the eclipse-mapping "
            "Conda environment and install the locked dependencies."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax, jnp, numpyro, dist, MCMC, NUTS


def _validated_inputs(design_matrix: ArrayLike, observed: ArrayLike, sigma: ArrayLike):
    design = np.asarray(design_matrix, dtype=float)
    y = np.asarray(observed, dtype=float)
    error = np.asarray(sigma, dtype=float)
    if design.ndim != 2 or y.ndim != 1 or design.shape[0] != y.size:
        raise ValueError("design_matrix must have shape (observation, parameter)")
    if error.ndim == 0:
        error = np.full_like(y, float(error))
    if error.shape != y.shape or np.any(error <= 0.0):
        raise ValueError("sigma must be scalar or a positive value per observation")
    return design, y, error


def _validated_noise_inputs(
    time_seconds: ArrayLike | None,
    n_observations: int,
    *,
    noise_model: str,
    ou_amplitude_prior_scale: float,
    ou_timescale_prior_median: float,
    ou_timescale_prior_sigma_ln: float,
    jitter_prior_scale: float,
) -> NDArray[np.float64] | None:
    """Validate the optional correlated-noise inputs.

    The OU recursion is causal, so observations must be supplied in
    increasing time order.  ``time_seconds`` is required only for the OU
    model.  Prior scales use the same flux units as ``observed`` and seconds
    for the OU timescale.
    """

    model = str(noise_model).lower()
    if model not in {"white", "independent", "ou"}:
        raise ValueError("noise_model must be white, independent, or ou")
    if not np.isfinite(ou_amplitude_prior_scale) or ou_amplitude_prior_scale <= 0.0:
        raise ValueError("ou_amplitude_prior_scale must be positive")
    if not np.isfinite(ou_timescale_prior_median) or ou_timescale_prior_median <= 0.0:
        raise ValueError("ou_timescale_prior_median must be positive")
    if not np.isfinite(ou_timescale_prior_sigma_ln) or ou_timescale_prior_sigma_ln <= 0.0:
        raise ValueError("ou_timescale_prior_sigma_ln must be positive")
    if not np.isfinite(jitter_prior_scale) or jitter_prior_scale <= 0.0:
        raise ValueError("jitter_prior_scale must be positive")
    if time_seconds is None:
        if model == "ou":
            raise ValueError("time_seconds is required when noise_model is ou")
        return None
    times = np.asarray(time_seconds, dtype=float)
    if times.ndim != 1 or times.size != n_observations:
        raise ValueError(
            "time_seconds must have one finite value per observation"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("time_seconds must contain only finite values")
    if model == "ou" and times.size > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError("time_seconds must be in non-decreasing order for OU noise")
    return times


def _ou_kalman_log_likelihood(
    residual: Any,
    sigma: Any,
    time_seconds: Any,
    amplitude: Any,
    timescale: Any,
    jitter: Any,
    *,
    likelihood: str,
    student_t_nu: float,
    jax: Any,
    jnp: Any,
    dist: Any,
) -> Any:
    """Return an OU marginal log likelihood with an O(n) innovations scan.

    The latent process is stationary at the first observation and follows
    ``x[i] = exp(-dt / timescale) * x[i-1] + eta[i]``.  The process variance
    is ``amplitude**2 * (1 - exp(-2*dt/timescale))``.  The scan therefore
    supports irregular cadence without building an n-by-n covariance matrix.

    For a Gaussian likelihood this is the exact OU marginal likelihood.  For
    ``student_t`` the same Kalman prediction variances are used with Student-t
    innovation densities.  This preserves the robust likelihood option while
    keeping the recursion linear in the number of observations.
    """

    residual = jnp.asarray(residual)
    sigma = jnp.asarray(sigma)
    times = jnp.asarray(time_seconds)
    # ``decay[i]`` advances the filtered state from observation ``i`` to
    # observation ``i + 1``.  The final element is a zero-length transition.
    # Keeping the transition with the preceding scan step is important for
    # irregular cadence: applying ``diff(times)`` at the start of a step would
    # advance the state after, rather than before, the next observation.
    dt = jnp.concatenate((jnp.diff(times), jnp.zeros((1,), dtype=times.dtype)))
    safe_timescale = jnp.maximum(timescale, jnp.asarray(1.0e-12, dtype=times.dtype))
    decay = jnp.exp(-dt / safe_timescale)
    # ``-expm1`` is stable for short gaps and gives zero process variance at
    # the first observation, where dt is exactly zero.
    process_variance = amplitude**2 * (-jnp.expm1(-2.0 * dt / safe_timescale))
    observation_variance = sigma**2 + jitter**2
    initial_variance = amplitude**2

    def step(carry: tuple[Any, Any], values: tuple[Any, Any, Any, Any]):
        state_mean, state_variance = carry
        residual_i, observation_variance_i, decay_i, process_variance_i = values
        predicted_variance = jnp.maximum(state_variance, jnp.asarray(1.0e-30))
        innovation_variance = jnp.maximum(
            predicted_variance + observation_variance_i,
            jnp.asarray(1.0e-30),
        )
        innovation = residual_i - state_mean
        if likelihood == "student_t":
            log_likelihood = dist.StudentT(
                student_t_nu,
                0.0,
                jnp.sqrt(innovation_variance),
            ).log_prob(innovation)
        else:
            log_likelihood = -0.5 * (
                jnp.log(2.0 * jnp.pi)
                + jnp.log(innovation_variance)
                + innovation**2 / innovation_variance
            )
        gain = predicted_variance / innovation_variance
        updated_mean = state_mean + gain * innovation
        updated_variance = jnp.maximum(
            (1.0 - gain) * predicted_variance,
            jnp.asarray(0.0),
        )
        next_mean = decay_i * updated_mean
        next_variance = decay_i**2 * updated_variance + process_variance_i
        return (next_mean, next_variance), log_likelihood

    (_, _), log_likelihood = jax.lax.scan(
        step,
        (jnp.asarray(0.0, dtype=residual.dtype), initial_variance),
        (residual, observation_variance, decay, process_variance),
    )
    return jnp.sum(log_likelihood)


def sample_positive_map(
    design_matrix: ArrayLike,
    observed: ArrayLike,
    sigma: ArrayLike,
    *,
    stellar_flux: ArrayLike | float = 1.0,
    pixel_prior_mean: float = 5.0e-3 / np.pi,
    pixel_log_sigma: float = np.sqrt(10.0),
    alpha: float = 0.0,
    warmup: int = 200,
    draws: int = 200,
    chains: int = 2,
    seed: int = 0,
    target_accept: float = 0.9,
    progress_bar: bool = True,
    dense_mass: bool = False,
    init_strategy: str = "median",
    systematics_design: ArrayLike | None = None,
    systematics_mode: str = "additive",
    systematics_prior_sigma: ArrayLike | float = 0.01,
    likelihood: str = "gaussian",
    student_t_nu: float = 4.0,
    time_seconds: ArrayLike | None = None,
    noise_model: str = "white",
    ou_amplitude_prior_scale: float = 100.0e-6,
    ou_timescale_prior_median: float = 900.0,
    ou_timescale_prior_sigma_ln: float = 1.0,
    jitter_prior_scale: float = 100.0e-6,
    sample_ramp_rate: bool = False,
    ramp_rate_prior_mean_per_day: float = 3.7,
    ramp_rate_prior_sigma_per_day: float = 1.0,
    ramp_amplitude_prior_sigma: float = 0.1,
    fit_error_scale: bool = False,
    error_scale_log_sigma: float = 0.1,
    fit_white_jitter: bool = False,
    multiplicative_composition: str = "linearized",
    systematics_names: tuple[str, ...] | list[str] | None = None,
) -> NumpyroRun:
    """Sample positive map pixels for a precomputed light-curve operator.

    Set ``noise_model="ou"`` and pass ``time_seconds`` to sample a
    stationary OU residual process.  The three prior scales are in flux
    units, except for the OU timescale, which is in seconds.
    """

    design, y, error = _validated_inputs(design_matrix, observed, sigma)
    star = np.broadcast_to(np.asarray(stellar_flux, dtype=float), y.shape)
    if systematics_design is None:
        nuisance = np.empty((y.size, 0), dtype=float)
    else:
        nuisance = np.asarray(systematics_design, dtype=float)
        if nuisance.ndim != 2 or nuisance.shape[0] != y.size:
            raise ValueError(
                "systematics_design must have shape (observation, nuisance_parameter)"
            )
        if not np.all(np.isfinite(nuisance)):
            raise ValueError("systematics_design contains non-finite values")
    if systematics_mode not in {"additive", "multiplicative"}:
        raise ValueError("systematics_mode must be additive or multiplicative")
    if multiplicative_composition not in {"linearized", "product"}:
        raise ValueError(
            "multiplicative_composition must be linearized or product"
        )
    if likelihood not in {"gaussian", "student_t"}:
        raise ValueError("likelihood must be gaussian or student_t")
    if student_t_nu < 2.0:
        raise ValueError("student_t_nu must be at least 2")
    if pixel_prior_mean <= 0.0 or pixel_log_sigma <= 0.0:
        raise ValueError("pixel prior values must be positive")
    if alpha < 0.0:
        raise ValueError("alpha must be greater than or equal to zero")
    if ramp_rate_prior_mean_per_day <= 0.0 or ramp_rate_prior_sigma_per_day <= 0.0:
        raise ValueError("ramp-rate prior values must be positive")
    if ramp_amplitude_prior_sigma <= 0.0:
        raise ValueError("ramp_amplitude_prior_sigma must be positive")
    if error_scale_log_sigma <= 0.0:
        raise ValueError("error_scale_log_sigma must be positive")
    if min(warmup, draws, chains) < 1 or chains > 3:
        raise ValueError("warmup and draws must be positive; chains must be from 1 to 3")
    times = _validated_noise_inputs(
        time_seconds,
        y.size,
        noise_model=noise_model,
        ou_amplitude_prior_scale=float(ou_amplitude_prior_scale),
        ou_timescale_prior_median=float(ou_timescale_prior_median),
        ou_timescale_prior_sigma_ln=float(ou_timescale_prior_sigma_ln),
        jitter_prior_scale=float(jitter_prior_scale),
    )
    noise_model = "white" if str(noise_model).lower() == "independent" else str(noise_model).lower()
    if sample_ramp_rate and times is None:
        raise ValueError("time_seconds is required when sample_ramp_rate is true")
    if (
        multiplicative_composition == "product"
        and nuisance.shape[1]
        and systematics_names is None
    ):
        raise ValueError(
            "systematics_names is required for product-composed systematics"
        )
    if systematics_names is None:
        nuisance_names = tuple(f"term_{index}" for index in range(nuisance.shape[1]))
    else:
        nuisance_names = tuple(str(name) for name in systematics_names)
        if len(nuisance_names) != nuisance.shape[1]:
            raise ValueError("systematics_names must match systematics_design columns")
    systematics_prior_scales = _coefficient_prior_vector(
        systematics_prior_sigma,
        "systematics_prior_sigma",
        nuisance.shape[1],
        strictly_positive=True,
    )
    ramp_cv = float(ramp_rate_prior_sigma_per_day / ramp_rate_prior_mean_per_day)
    ramp_sigma_ln = float(np.sqrt(np.log1p(ramp_cv**2)))
    ramp_mu_ln = float(
        np.log(ramp_rate_prior_mean_per_day) - 0.5 * ramp_sigma_ln**2
    )

    # The first factor is the linear baseline. Each detector regressor and
    # fixed ramp is a separate multiplicative factor, as in Hammond et al.
    product_group_ids: list[int] = []
    next_group = 1
    for name in nuisance_names:
        if name.startswith("offset") or name.startswith("time"):
            product_group_ids.append(0)
        else:
            product_group_ids.append(next_group)
            next_group += 1

    # Match the legacy workflow: optimize in unconstrained log-pixel space,
    # then initialize all chains at the posterior mode. This avoids sending
    # broad lognormal chains into different weakly identified pixel modes.
    from scipy.optimize import minimize

    log_prior_mean = float(np.log(pixel_prior_mean))

    def negative_log_density(parameters):
        values = np.asarray(parameters, dtype=float)
        log_pixels = values[: design.shape[1]]
        beta = values[design.shape[1] :]
        pixels = np.exp(log_pixels)
        astrophysical = star + design @ pixels
        nuisance_model = nuisance @ beta
        model_flux = (
            astrophysical + nuisance_model
            if systematics_mode == "additive"
            else astrophysical * (1.0 + nuisance_model)
        )
        residual = (y - model_flux) / error
        if likelihood == "student_t":
            data_density = 0.5 * (student_t_nu + 1.0) * np.sum(
                np.log1p(residual**2 / student_t_nu)
            )
        else:
            data_density = 0.5 * np.sum(residual**2)
        prior = (log_pixels - log_prior_mean) / pixel_log_sigma
        nuisance_prior = beta / systematics_prior_scales
        mean_pixel = float(np.mean(pixels))
        entropy = -float(np.sum(pixels * np.log(pixels / mean_pixel)))
        return float(
            data_density
            + 0.5 * np.sum(prior**2)
            + 0.5 * np.sum(nuisance_prior**2)
            - 2.0 * alpha * entropy
        )

    initial_parameters = np.concatenate(
        (
            np.full(design.shape[1], log_prior_mean),
            np.zeros(nuisance.shape[1], dtype=float),
        )
    )
    optimized = minimize(
        negative_log_density,
        initial_parameters,
        method="L-BFGS-B",
        bounds=[(-25.0, 0.0)] * design.shape[1]
        + [(None, None)] * nuisance.shape[1],
        options={"maxiter": 5000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    initial_pixels = (
        np.exp(optimized.x[: design.shape[1]])
        if optimized.success and np.all(np.isfinite(optimized.x))
        else np.full(design.shape[1], pixel_prior_mean)
    )
    initial_systematics = (
        np.asarray(optimized.x[design.shape[1] :], dtype=float)
        if optimized.success and np.all(np.isfinite(optimized.x))
        else np.zeros(nuisance.shape[1], dtype=float)
    )

    # Sample a locally whitened latent vector.  The physical coordinates are
    # log-pixels plus nuisance coefficients, so every pixel remains strictly
    # positive after the affine map and exponentiation.
    map_parameters = np.concatenate(
        (np.log(initial_pixels), initial_systematics)
    )
    whitening, whitening_logdet, _whitening_precision = (
        _positive_map_posterior_whitener(
            design,
            y,
            error,
            np.asarray(star, dtype=float),
            map_parameters,
            float(pixel_log_sigma),
            nuisance,
            systematics_mode,
            systematics_prior_scales,
            likelihood,
            float(student_t_nu),
        )
    )

    jax, jnp, numpyro, dist, MCMC, NUTS = _imports()
    numpyro.set_host_device_count(chains)
    design_jax = jnp.asarray(design)
    y_jax = jnp.asarray(y)
    error_jax = jnp.asarray(error)
    star_jax = jnp.asarray(star)
    nuisance_jax = jnp.asarray(nuisance)
    systematics_prior_scales_jax = jnp.asarray(systematics_prior_scales)
    times_jax = None if times is None else jnp.asarray(times)
    map_parameters_jax = jnp.asarray(map_parameters)
    whitening_jax = jnp.asarray(whitening)
    whitening_logdet_jax = jnp.asarray(whitening_logdet)
    standard_normal = dist.Normal(
        jnp.zeros(map_parameters.size), jnp.ones(map_parameters.size)
    ).to_event(1)

    def map_init(site):
        """Start each chain at, or close to, the optimized posterior mode."""

        if site["type"] == "sample" and site["name"] == "whitened_parameters":
            if init_strategy != "jitter+adapt_diag":
                return jnp.zeros(map_parameters.size)
            rng_key = site["kwargs"].get("rng_key")
            jitter = 0.05 * jax.random.normal(
                rng_key, shape=np.shape(map_parameters)
            )
            return jitter
        return None

    def model():
        whitened_parameters = numpyro.sample(
            "whitened_parameters", standard_normal
        )
        parameters = map_parameters_jax + whitening_jax @ whitened_parameters
        log_pixels = parameters[: design.shape[1]]
        pixels = jnp.exp(log_pixels)
        if nuisance.shape[1]:
            systematics_coefficients = parameters[design.shape[1] :]
        else:
            systematics_coefficients = jnp.zeros(0, dtype=parameters.dtype)

        # The original LogNormal pixel prior is exactly a Normal prior in
        # log-pixel coordinates.  The latent base density is cancelled and
        # the affine Jacobian is restored, leaving the requested physical
        # prior unchanged under the new sampling coordinates.
        pixel_prior = dist.Normal(
            jnp.log(pixel_prior_mean), pixel_log_sigma
        ).expand((design.shape[1],)).to_event(1)
        physical_prior_log_prob = pixel_prior.log_prob(log_pixels)
        if nuisance.shape[1]:
            physical_prior_log_prob += dist.Normal(
                0.0, systematics_prior_scales_jax
            ).expand((nuisance.shape[1],)).to_event(1).log_prob(
                systematics_coefficients
            )
        numpyro.factor(
            "physical_prior_correction",
            physical_prior_log_prob
            + whitening_logdet_jax
            - standard_normal.log_prob(whitened_parameters),
        )

        mean_pixel = jnp.mean(pixels)
        entropy = -jnp.sum(pixels * jnp.log(pixels / mean_pixel))
        numpyro.deterministic("entropy", entropy)
        numpyro.factor("entropy_regularization", 2.0 * alpha * entropy)
        astrophysical_flux = star_jax + design_jax @ pixels
        has_systematics = bool(nuisance.shape[1]) or bool(sample_ramp_rate)
        if has_systematics:
            nuisance_model = nuisance_jax @ systematics_coefficients
            if sample_ramp_rate:
                ramp_amplitude = numpyro.sample(
                    "ramp_amplitude",
                    dist.Normal(0.0, float(ramp_amplitude_prior_sigma)),
                )
                ramp_rate_per_day = numpyro.sample(
                    "ramp_rate_per_day",
                    dist.LogNormal(ramp_mu_ln, ramp_sigma_ln),
                )
                elapsed_days = (times_jax - times_jax[0]) / 86_400.0
                ramp_model = ramp_amplitude * jnp.exp(
                    -ramp_rate_per_day * elapsed_days
                )
                nuisance_model = nuisance_model + ramp_model
            if systematics_mode == "additive":
                flux = astrophysical_flux + nuisance_model
            elif multiplicative_composition == "linearized":
                flux = astrophysical_flux * (1.0 + nuisance_model)
            else:
                systematics_factor = jnp.ones_like(astrophysical_flux)
                for group_id in sorted(set(product_group_ids)):
                    indices = [
                        index
                        for index, value in enumerate(product_group_ids)
                        if value == group_id
                    ]
                    group_model = nuisance_jax[:, indices] @ systematics_coefficients[
                        jnp.asarray(indices)
                    ]
                    systematics_factor = systematics_factor * (1.0 + group_model)
                if sample_ramp_rate:
                    systematics_factor = systematics_factor * (1.0 + ramp_model)
                flux = astrophysical_flux * systematics_factor
                nuisance_model = systematics_factor - 1.0
            numpyro.deterministic("systematics_model", nuisance_model)
        else:
            flux = astrophysical_flux
        numpyro.deterministic("pixels", pixels)
        if nuisance.shape[1]:
            numpyro.deterministic(
                "systematics_coefficients", systematics_coefficients
            )
        numpyro.deterministic("flux", flux)
        if fit_error_scale:
            error_scale = numpyro.sample(
                "error_scale",
                dist.LogNormal(0.0, float(error_scale_log_sigma)),
            )
        else:
            error_scale = jnp.asarray(1.0, dtype=error_jax.dtype)
        scaled_error = error_jax * error_scale
        if fit_white_jitter:
            white_jitter = numpyro.sample(
                "white_jitter",
                dist.HalfNormal(float(jitter_prior_scale)),
            )
            scaled_error = jnp.sqrt(scaled_error**2 + white_jitter**2)
        if noise_model == "ou":
            ou_amplitude = numpyro.sample(
                "ou_amplitude",
                dist.HalfNormal(float(ou_amplitude_prior_scale)),
            )
            ou_timescale = numpyro.sample(
                "ou_timescale",
                dist.LogNormal(
                    jnp.log(float(ou_timescale_prior_median)),
                    float(ou_timescale_prior_sigma_ln),
                ),
            )
            jitter = numpyro.sample(
                "jitter",
                dist.HalfNormal(float(jitter_prior_scale)),
            )
            numpyro.factor(
                "ou_log_likelihood",
                _ou_kalman_log_likelihood(
                    y_jax - flux,
                    scaled_error,
                    times_jax,
                    ou_amplitude,
                    ou_timescale,
                    jitter,
                    likelihood=likelihood,
                    student_t_nu=student_t_nu,
                    jax=jax,
                    jnp=jnp,
                    dist=dist,
                ),
            )
        else:
            observation_distribution = (
                dist.StudentT(student_t_nu, flux, scaled_error)
                if likelihood == "student_t"
                else dist.Normal(flux, scaled_error)
            )
            numpyro.sample("obs", observation_distribution, obs=y_jax)

    mcmc = MCMC(
        NUTS(
            model,
            target_accept_prob=target_accept,
            dense_mass=bool(dense_mass),
            init_strategy=partial(map_init),
        ),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="parallel" if chains > 1 else "sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.key(seed), extra_fields=("diverging", "accept_prob"))
    return NumpyroRun(
        samples={key: np.asarray(value) for key, value in mcmc.get_samples().items()},
        extra_fields={
            key: np.asarray(value) for key, value in mcmc.get_extra_fields().items()
        },
        sampler=mcmc,
        grouped_samples={
            key: np.asarray(value)
            for key, value in mcmc.get_samples(group_by_chain=True).items()
        },
    )


def _coefficient_prior_vector(
    value: ArrayLike | float,
    name: str,
    size: int,
    *,
    strictly_positive: bool = False,
) -> NDArray[np.float64]:
    """Broadcast and validate a direct-coefficient prior vector."""

    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(size, float(values), dtype=float)
    if values.shape != (size,):
        raise ValueError(f"{name} must be scalar or have one value per design column")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if strictly_positive and np.any(values <= 0.0):
        raise ValueError(f"{name} must contain strictly positive values")
    return np.asarray(values, dtype=float)


def _local_posterior_whitener(
    design: NDArray[np.float64],
    observed: NDArray[np.float64],
    error: NDArray[np.float64],
    stellar_flux: NDArray[np.float64],
    map_parameters: NDArray[np.float64],
    coefficient_prior_sigma: NDArray[np.float64],
    nuisance: NDArray[np.float64],
    systematics_mode: str,
    systematics_prior_sigma: ArrayLike | float,
    likelihood: str,
    student_t_nu: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Build a local posterior whitening map in prior-scaled coordinates.

    The returned affine matrix maps a standard normal latent vector ``z`` to
    physical parameters with ``parameters = map_parameters + affine @ z``.
    The matrix uses the inverse Cholesky factor of a Gauss--Newton posterior
    precision.  The returned log determinant is used by the model to apply an
    exact change-of-variables correction to the physical Normal priors.

    Scaling each parameter by its prior standard deviation before forming the
    precision avoids numerical problems when map coefficients and systematic
    coefficients have very different units or amplitudes.
    """

    n_coefficients = design.shape[1]
    if map_parameters.shape != (n_coefficients + nuisance.shape[1],):
        raise ValueError("map_parameters has an invalid shape")
    prior_scales = np.concatenate(
        (
            np.asarray(coefficient_prior_sigma, dtype=float),
            np.full(nuisance.shape[1], systematics_prior_sigma, dtype=float),
        )
    )
    if prior_scales.size == 0 or np.any(~np.isfinite(prior_scales)):
        raise ValueError("posterior whitening requires finite prior scales")
    if np.any(prior_scales <= 0.0):
        raise ValueError("posterior whitening requires positive prior scales")

    coefficients = map_parameters[:n_coefficients]
    beta = map_parameters[n_coefficients:]
    astrophysical = stellar_flux + design @ coefficients
    nuisance_model = nuisance @ beta
    if systematics_mode == "additive":
        model_flux = astrophysical + nuisance_model
        jacobian_coefficients = design
        jacobian_nuisance = nuisance
    else:
        model_flux = astrophysical * (1.0 + nuisance_model)
        jacobian_coefficients = design * (1.0 + nuisance_model)[:, None]
        jacobian_nuisance = astrophysical[:, None] * nuisance
    residual = (observed - model_flux) / error
    if likelihood == "student_t":
        likelihood_weights = (student_t_nu + 1.0) / (
            (student_t_nu + residual**2) * error**2
        )
    else:
        likelihood_weights = 1.0 / error**2
    jacobian = np.column_stack((jacobian_coefficients, jacobian_nuisance))
    scaled_jacobian = jacobian * prior_scales[None, :]
    precision = scaled_jacobian.T @ (likelihood_weights[:, None] * scaled_jacobian)
    precision += np.eye(prior_scales.size, dtype=float)
    precision = 0.5 * (precision + precision.T)

    # The prior precision makes this matrix positive definite.  A small
    # relative jitter is retained as a safety guard for extreme inputs and
    # changes only the proposal geometry, not the exact prior correction.
    diagonal_scale = max(1.0, float(np.max(np.diag(precision))))
    identity = np.eye(prior_scales.size, dtype=float)
    chol = None
    for exponent in range(8):
        jitter = 0.0 if exponent == 0 else 10.0 ** (exponent - 12) * diagonal_scale
        try:
            candidate = np.linalg.cholesky(precision + jitter * identity)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(candidate)):
            chol = candidate
            break
    if chol is None:
        raise ValueError("could not construct a finite posterior whitening transform")

    # If H = L L^T, then L^-T L^-T^T = H^-1.  The prior-scale matrix converts
    # this standardized covariance back to physical parameter units.
    inverse_upper_cholesky = np.linalg.solve(chol.T, identity)
    affine = prior_scales[:, None] * inverse_upper_cholesky
    log_determinant = float(
        np.sum(np.log(prior_scales)) - np.sum(np.log(np.diag(chol)))
    )
    return affine, log_determinant, precision


def _positive_map_posterior_whitener(
    design: NDArray[np.float64],
    observed: NDArray[np.float64],
    error: NDArray[np.float64],
    stellar_flux: NDArray[np.float64],
    map_parameters: NDArray[np.float64],
    pixel_log_sigma: float,
    nuisance: NDArray[np.float64],
    systematics_mode: str,
    systematics_prior_sigma: float,
    likelihood: str,
    student_t_nu: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Build a local whitening map for log-pixels and nuisance coefficients.

    The positive-map model is naturally expressed in ``x = log(pixels)``.
    This helper builds a Gauss--Newton precision in those coordinates, with
    the existing Normal prior scales used to standardise each parameter.  The
    returned affine map is ``[x, beta] = map_parameters + affine @ z`` for a
    standard-normal latent vector ``z``.  The likelihood and prior target are
    corrected exactly in :func:`sample_positive_map`, so this transform only
    changes the sampling geometry.
    """

    n_pixels = design.shape[1]
    if map_parameters.shape != (n_pixels + nuisance.shape[1],):
        raise ValueError("map_parameters has an invalid shape")
    if not np.isfinite(pixel_log_sigma) or pixel_log_sigma <= 0.0:
        raise ValueError("pixel_log_sigma must be positive")
    systematics_prior_scales = _coefficient_prior_vector(
        systematics_prior_sigma,
        "systematics_prior_sigma",
        nuisance.shape[1],
        strictly_positive=True,
    )

    log_pixels = np.asarray(map_parameters[:n_pixels], dtype=float)
    beta = np.asarray(map_parameters[n_pixels:], dtype=float)
    pixels = np.exp(log_pixels)
    astrophysical = stellar_flux + design @ pixels
    nuisance_model = nuisance @ beta
    if systematics_mode == "additive":
        model_flux = astrophysical + nuisance_model
        # d flux / d log(pixel) = d flux / d pixel * pixel.
        jacobian_pixels = design * pixels[None, :]
        jacobian_nuisance = nuisance
    elif systematics_mode == "multiplicative":
        model_flux = astrophysical * (1.0 + nuisance_model)
        jacobian_pixels = design * pixels[None, :] * (1.0 + nuisance_model)[:, None]
        jacobian_nuisance = astrophysical[:, None] * nuisance
    else:  # pragma: no cover - validated by the public sampler
        raise ValueError("systematics_mode must be additive or multiplicative")

    residual = (observed - model_flux) / error
    if likelihood == "student_t":
        likelihood_weights = (student_t_nu + 1.0) / (
            (student_t_nu + residual**2) * error**2
        )
    else:
        likelihood_weights = 1.0 / error**2
    jacobian = np.column_stack((jacobian_pixels, jacobian_nuisance))
    prior_scales = np.concatenate(
        (
            np.full(n_pixels, pixel_log_sigma, dtype=float),
            systematics_prior_scales,
        )
    )
    scaled_jacobian = jacobian * prior_scales[None, :]
    precision = scaled_jacobian.T @ (likelihood_weights[:, None] * scaled_jacobian)
    precision += np.eye(prior_scales.size, dtype=float)
    precision = 0.5 * (precision + precision.T)

    diagonal_scale = max(1.0, float(np.max(np.diag(precision))))
    identity = np.eye(prior_scales.size, dtype=float)
    chol = None
    for exponent in range(8):
        jitter = 0.0 if exponent == 0 else 10.0 ** (exponent - 12) * diagonal_scale
        try:
            candidate = np.linalg.cholesky(precision + jitter * identity)
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(candidate)):
            chol = candidate
            break
    if chol is None:
        raise ValueError("could not construct a finite posterior whitening transform")

    inverse_upper_cholesky = np.linalg.solve(chol.T, identity)
    affine = prior_scales[:, None] * inverse_upper_cholesky
    log_determinant = float(
        np.sum(np.log(prior_scales)) - np.sum(np.log(np.diag(chol)))
    )
    return affine, log_determinant, precision


def sample_harmonic_map(
    design_matrix: ArrayLike,
    observed: ArrayLike,
    sigma: ArrayLike,
    *,
    stellar_flux: ArrayLike | float = 1.0,
    coefficient_prior_mean: ArrayLike | float = 0.0,
    coefficient_prior_sigma: ArrayLike | float = 0.05,
    prior_mean: ArrayLike | float | None = None,
    prior_sigma: ArrayLike | float | None = None,
    warmup: int = 200,
    draws: int = 200,
    chains: int = 2,
    seed: int = 0,
    target_accept: float = 0.9,
    progress_bar: bool = True,
    dense_mass: bool = False,
    init_strategy: str = "median",
    systematics_design: ArrayLike | None = None,
    systematics_mode: str = "additive",
    systematics_prior_sigma: float = 0.01,
    likelihood: str = "gaussian",
    student_t_nu: float = 4.0,
    time_seconds: ArrayLike | None = None,
    noise_model: str = "white",
    ou_amplitude_prior_scale: float = 100.0e-6,
    ou_timescale_prior_median: float = 900.0,
    ou_timescale_prior_sigma_ln: float = 1.0,
    jitter_prior_scale: float = 100.0e-6,
    fit_white_jitter: bool = False,
) -> NumpyroRun:
    """Sample direct harmonic coefficients with a posterior-whitened state.

    One independent Normal prior is assigned to every design-matrix column.
    ``coefficient_prior_mean`` and ``coefficient_prior_sigma`` may be scalars
    or vectors with one value per column.  The shorter ``prior_mean`` and
    ``prior_sigma`` names are accepted as aliases.  No positivity transform is
    applied: this function is intended for the direct Hammond harmonic model.

    The model uses the same additive and multiplicative systematics forms as
    :func:`sample_positive_map`.  A standard-normal latent vector is mapped to
    physical coefficients with an inverse-Cholesky transform built from the
    MAP Gauss--Newton posterior precision.  A change-of-variables factor makes
    the independent physical Normal priors exact; whitening therefore changes
    sampling geometry without changing the target posterior.  All chains
    start at the finite MAP estimate, with optional small latent jitter.
    Deterministic physical ``coefficients``, ``flux``, and
    ``systematics_model`` values are returned in :class:`NumpyroRun` for
    downstream diagnostics.  Set ``noise_model="ou"`` and pass
    ``time_seconds`` to sample an irregular-cadence OU residual process.
    """

    design, y, error = _validated_inputs(design_matrix, observed, sigma)
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(y)):
        raise ValueError("design_matrix and observed must contain only finite values")
    if not np.all(np.isfinite(error)):
        raise ValueError("sigma must contain only finite values")
    try:
        star = np.broadcast_to(np.asarray(stellar_flux, dtype=float), y.shape)
    except ValueError as exc:
        raise ValueError("stellar_flux must be scalar or have one value per observation") from exc
    if not np.all(np.isfinite(star)):
        raise ValueError("stellar_flux must contain only finite values")

    if prior_mean is not None:
        coefficient_prior_mean = prior_mean
    if prior_sigma is not None:
        coefficient_prior_sigma = prior_sigma
    prior_mean_values = _coefficient_prior_vector(
        coefficient_prior_mean,
        "coefficient_prior_mean",
        design.shape[1],
    )
    prior_sigma_values = _coefficient_prior_vector(
        coefficient_prior_sigma,
        "coefficient_prior_sigma",
        design.shape[1],
        strictly_positive=True,
    )

    if systematics_design is None:
        nuisance = np.empty((y.size, 0), dtype=float)
    else:
        nuisance = np.asarray(systematics_design, dtype=float)
        if nuisance.ndim != 2 or nuisance.shape[0] != y.size:
            raise ValueError(
                "systematics_design must have shape (observation, nuisance_parameter)"
            )
        if not np.all(np.isfinite(nuisance)):
            raise ValueError("systematics_design contains non-finite values")
    if systematics_mode not in {"additive", "multiplicative"}:
        raise ValueError("systematics_mode must be additive or multiplicative")
    if not np.isfinite(systematics_prior_sigma) or systematics_prior_sigma <= 0.0:
        raise ValueError("systematics_prior_sigma must be positive")
    if likelihood not in {"gaussian", "student_t"}:
        raise ValueError("likelihood must be gaussian or student_t")
    if not np.isfinite(student_t_nu) or student_t_nu < 2.0:
        raise ValueError("student_t_nu must be at least 2")
    if not np.isfinite(target_accept) or not 0.0 < target_accept < 1.0:
        raise ValueError("target_accept must be between zero and one")
    if init_strategy not in {"median", "adapt_diag", "jitter+adapt_diag"}:
        raise ValueError(
            "init_strategy must be median, adapt_diag, or jitter+adapt_diag"
        )
    if min(warmup, draws, chains) < 1 or chains > 3:
        raise ValueError("warmup and draws must be positive; chains must be from 1 to 3")
    times = _validated_noise_inputs(
        time_seconds,
        y.size,
        noise_model=noise_model,
        ou_amplitude_prior_scale=float(ou_amplitude_prior_scale),
        ou_timescale_prior_median=float(ou_timescale_prior_median),
        ou_timescale_prior_sigma_ln=float(ou_timescale_prior_sigma_ln),
        jitter_prior_scale=float(jitter_prior_scale),
    )
    noise_model = "white" if str(noise_model).lower() == "independent" else str(noise_model).lower()

    # MAP optimization is done in the same direct coefficient coordinates used
    # by NUTS.  This removes the broad random initialisation that can send a
    # weakly constrained harmonic model into a poor geometry before adaptation.
    from scipy.optimize import minimize

    n_coefficients = design.shape[1]

    def negative_log_density(parameters: NDArray[np.float64]) -> float:
        values = np.asarray(parameters, dtype=float)
        coefficients = values[:n_coefficients]
        beta = values[n_coefficients:]
        astrophysical = star + design @ coefficients
        nuisance_model = nuisance @ beta
        model_flux = (
            astrophysical + nuisance_model
            if systematics_mode == "additive"
            else astrophysical * (1.0 + nuisance_model)
        )
        residual = (y - model_flux) / error
        if likelihood == "student_t":
            data_density = 0.5 * (student_t_nu + 1.0) * np.sum(
                np.log1p(residual**2 / student_t_nu)
            )
        else:
            data_density = 0.5 * np.sum(residual**2)
        coefficient_prior = (coefficients - prior_mean_values) / prior_sigma_values
        nuisance_prior = beta / systematics_prior_sigma
        return float(
            data_density
            + 0.5 * np.sum(coefficient_prior**2)
            + 0.5 * np.sum(nuisance_prior**2)
        )

    initial_parameters = np.concatenate(
        (prior_mean_values, np.zeros(nuisance.shape[1], dtype=float))
    )
    optimized = minimize(
        negative_log_density,
        initial_parameters,
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if optimized.success and np.all(np.isfinite(optimized.x)):
        initial_coefficients = np.asarray(optimized.x[:n_coefficients], dtype=float)
        initial_systematics = np.asarray(
            optimized.x[n_coefficients:], dtype=float
        )
    else:
        initial_coefficients = prior_mean_values.copy()
        initial_systematics = np.zeros(nuisance.shape[1], dtype=float)

    map_parameters = np.concatenate((initial_coefficients, initial_systematics))
    whitening, whitening_logdet, _whitening_precision = _local_posterior_whitener(
        design,
        y,
        error,
        np.asarray(star, dtype=float),
        map_parameters,
        prior_sigma_values,
        nuisance,
        systematics_mode,
        float(systematics_prior_sigma),
        likelihood,
        float(student_t_nu),
    )

    jax, jnp, numpyro, dist, MCMC, NUTS = _imports()
    numpyro.set_host_device_count(chains)
    design_jax = jnp.asarray(design)
    y_jax = jnp.asarray(y)
    error_jax = jnp.asarray(error)
    star_jax = jnp.asarray(star)
    nuisance_jax = jnp.asarray(nuisance)
    times_jax = None if times is None else jnp.asarray(times)
    prior_mean_jax = jnp.asarray(prior_mean_values)
    prior_sigma_jax = jnp.asarray(prior_sigma_values)
    map_parameters_jax = jnp.asarray(map_parameters)
    whitening_jax = jnp.asarray(whitening)
    whitening_logdet_jax = jnp.asarray(whitening_logdet)
    standard_normal = dist.Normal(
        jnp.zeros(map_parameters.size), jnp.ones(map_parameters.size)
    ).to_event(1)

    def map_init(site):
        """Start the latent state at the MAP estimate (zero)."""

        if site["type"] == "sample" and site["name"] == "whitened_parameters":
            if init_strategy != "jitter+adapt_diag":
                return jnp.zeros(map_parameters.size)
            rng_key = site["kwargs"].get("rng_key")
            jitter = 0.05 * jax.random.normal(
                rng_key, shape=np.shape(map_parameters)
            )
            return jitter
        return None

    def model():
        whitened_parameters = numpyro.sample("whitened_parameters", standard_normal)
        parameters = map_parameters_jax + whitening_jax @ whitened_parameters
        coefficients = parameters[:n_coefficients]
        if nuisance.shape[1]:
            systematics_coefficients = parameters[n_coefficients:]
        else:
            systematics_coefficients = jnp.zeros(0, dtype=parameters.dtype)

        # The base latent site is N(0, I).  Add log p(physical parameters) and
        # the affine Jacobian, then remove log p(latent), so the induced latent
        # density is exactly the requested physical Normal prior.
        physical_prior = dist.Normal(prior_mean_jax, prior_sigma_jax).to_event(1)
        if nuisance.shape[1]:
            physical_prior_log_prob = physical_prior.log_prob(coefficients)
            physical_prior_log_prob += dist.Normal(
                0.0, systematics_prior_sigma
            ).expand((nuisance.shape[1],)).to_event(1).log_prob(
                systematics_coefficients
            )
        else:
            physical_prior_log_prob = physical_prior.log_prob(coefficients)
        numpyro.factor(
            "physical_prior_correction",
            physical_prior_log_prob
            + whitening_logdet_jax
            - standard_normal.log_prob(whitened_parameters),
        )
        astrophysical_flux = star_jax + design_jax @ coefficients
        if nuisance.shape[1]:
            nuisance_model = nuisance_jax @ systematics_coefficients
        else:
            nuisance_model = jnp.zeros_like(y_jax)
        flux = (
            astrophysical_flux + nuisance_model
            if systematics_mode == "additive"
            else astrophysical_flux * (1.0 + nuisance_model)
        )
        numpyro.deterministic("coefficients", coefficients)
        if nuisance.shape[1]:
            numpyro.deterministic(
                "systematics_coefficients", systematics_coefficients
            )
        numpyro.deterministic("systematics_model", nuisance_model)
        numpyro.deterministic("flux", flux)
        if fit_white_jitter:
            white_jitter = numpyro.sample(
                "white_jitter",
                dist.HalfNormal(float(jitter_prior_scale)),
            )
            effective_error = jnp.sqrt(error_jax**2 + white_jitter**2)
        else:
            effective_error = error_jax
        if noise_model == "ou":
            ou_amplitude = numpyro.sample(
                "ou_amplitude",
                dist.HalfNormal(float(ou_amplitude_prior_scale)),
            )
            ou_timescale = numpyro.sample(
                "ou_timescale",
                dist.LogNormal(
                    jnp.log(float(ou_timescale_prior_median)),
                    float(ou_timescale_prior_sigma_ln),
                ),
            )
            jitter = numpyro.sample(
                "jitter",
                dist.HalfNormal(float(jitter_prior_scale)),
            )
            numpyro.factor(
                "ou_log_likelihood",
                _ou_kalman_log_likelihood(
                    y_jax - flux,
                    effective_error,
                    times_jax,
                    ou_amplitude,
                    ou_timescale,
                    jitter,
                    likelihood=likelihood,
                    student_t_nu=student_t_nu,
                    jax=jax,
                    jnp=jnp,
                    dist=dist,
                ),
            )
        else:
            observation_distribution = (
                dist.StudentT(student_t_nu, flux, effective_error)
                if likelihood == "student_t"
                else dist.Normal(flux, effective_error)
            )
            numpyro.sample("obs", observation_distribution, obs=y_jax)

    mcmc = MCMC(
        NUTS(
            model,
            target_accept_prob=target_accept,
            dense_mass=bool(dense_mass),
            init_strategy=partial(map_init),
        ),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="parallel" if chains > 1 else "sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.key(seed), extra_fields=("diverging", "accept_prob"))
    return NumpyroRun(
        samples={key: np.asarray(value) for key, value in mcmc.get_samples().items()},
        extra_fields={
            key: np.asarray(value) for key, value in mcmc.get_extra_fields().items()
        },
        sampler=mcmc,
        grouped_samples={
            key: np.asarray(value)
            for key, value in mcmc.get_samples(group_by_chain=True).items()
        },
    )


def sample_fourier_model(
    design_matrix: ArrayLike,
    observed: ArrayLike,
    sigma: ArrayLike,
    *,
    coefficient_scale: float = 0.1,
    warmup: int = 200,
    draws: int = 200,
    chains: int = 2,
    seed: int = 0,
    target_accept: float = 0.9,
    progress_bar: bool = True,
) -> NumpyroRun:
    """Sample a Gaussian Fourier model from its design matrix."""

    design, y, error = _validated_inputs(design_matrix, observed, sigma)
    if coefficient_scale <= 0.0:
        raise ValueError("coefficient_scale must be positive")
    if min(warmup, draws, chains) < 1 or chains > 3:
        raise ValueError("warmup and draws must be positive; chains must be from 1 to 3")
    jax, jnp, numpyro, dist, MCMC, NUTS = _imports()
    numpyro.set_host_device_count(chains)
    design_jax = jnp.asarray(design)
    y_jax = jnp.asarray(y)
    error_jax = jnp.asarray(error)

    def model():
        coefficients = numpyro.sample(
            "coefficients",
            dist.Normal(0.0, coefficient_scale).expand((design.shape[1],)),
        )
        flux = design_jax @ coefficients
        numpyro.deterministic("flux", flux)
        numpyro.sample("obs", dist.Normal(flux, error_jax), obs=y_jax)

    mcmc = MCMC(
        NUTS(model, target_accept_prob=target_accept),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="parallel" if chains > 1 else "sequential",
        progress_bar=progress_bar,
    )
    mcmc.run(jax.random.key(seed), extra_fields=("diverging", "accept_prob"))
    return NumpyroRun(
        samples={key: np.asarray(value) for key, value in mcmc.get_samples().items()},
        extra_fields={
            key: np.asarray(value) for key, value in mcmc.get_extra_fields().items()
        },
        sampler=mcmc,
        grouped_samples={
            key: np.asarray(value)
            for key, value in mcmc.get_samples(group_by_chain=True).items()
        },
    )
