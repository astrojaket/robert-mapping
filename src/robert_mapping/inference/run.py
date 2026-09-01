"""Small, configuration-driven fit engine.

The command line interface only needs one stable entry point: :func:`run_fit`.
This module keeps that entry point deliberately thin.  It loads a light curve,
builds the numerical secondary-eclipse operator, and dispatches either to the
exact Gaussian harmonic solver or to the small NumPyro positive-pixel model.

The total light curve contains a quadratic-limb-darkened stellar transit plus
the thermal planet map and secondary eclipse.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..config import MappingConfig, write_resolved_config
from ..data import LightCurve, load_light_curve
from ..physics import (
    disk_quadrature,
    fibonacci_pixels,
    ncoeff,
    pixels_to_harmonics,
    rank_revealing_anchor_transform,
    starry_pixel_transforms,
    secondary_eclipse_design_matrix,
    stellar_transit_flux,
)
from ..systematics import build_systematics_design
from .linear import LinearGaussianPosterior, fit_linear_gaussian
from . import numpyro_backend
from .numpyro_backend import NumpyroRun, sample_positive_map

try:
    from .numpyro_backend import sample_harmonic_map
except ImportError:  # The direct-harmonic backend is added in a later module step.
    sample_harmonic_map = None


@dataclass(frozen=True)
class FitResult:
    """Machine-readable description of a completed fit.

    The numerical arrays are written to ``output_directory``.  Keeping the
    result object small makes it useful from both Python and the CLI without
    duplicating posterior arrays in memory after a run.
    """

    status: str
    sampler: str
    n_observations: int
    n_parameters: int
    output_directory: Path
    summary_path: Path
    coefficients_path: Path
    samples_path: Path | None
    summary: dict[str, Any]


@dataclass(frozen=True)
class _DesignBundle:
    """The fixed operator and metadata used by both inference branches."""

    time_days: NDArray[np.float64]
    map_design: NDArray[np.float64]
    stellar_flux: NDArray[np.float64]
    quadrature_radial: int
    quadrature_azimuth: int
    exposure_subsamples: int
    harmonic_degree: int
    systematics_design: NDArray[np.float64]
    systematics_names: tuple[str, ...]
    systematics_mode: str


def _set_thread_limits(config: MappingConfig) -> int:
    """Apply the hard three-CPU limit before importing numerical backends."""

    compute = config.compute
    requested = min(3, int(compute.max_cpus), int(compute.threads))
    threads = max(1, requested)
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    # Do not append duplicate XLA flags.  The CLI sets this variable before
    # loading the fit engine; a default is enough for direct Python use.
    os.environ.setdefault(
        "XLA_FLAGS",
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={threads}",
    )
    return threads


def _time_days(curve: LightCurve) -> NDArray[np.float64]:
    """Convert the configured input time unit to days."""

    unit = str(curve.time_unit).lower()
    scale = {
        "day": 1.0,
        "days": 1.0,
        "hour": 1.0 / 24.0,
        "hours": 1.0 / 24.0,
        "second": 1.0 / 86_400.0,
        "seconds": 1.0 / 86_400.0,
    }.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported time unit {curve.time_unit!r}.")
    return np.asarray(curve.time, dtype=float) * scale


def _noise_diagnostics(
    run: NumpyroRun,
    config: MappingConfig,
) -> dict[str, Any]:
    """Summarize sampled time-correlated-noise values and chain checks."""

    noise_model = str(config.model.noise_model).lower()
    if noise_model == "independent":
        noise_model = "white"
    summary: dict[str, Any] = {
        "noise_model": noise_model,
        "ou_amplitude_prior_scale_ppm": float(
            config.model.ou_amplitude_prior_scale_ppm
        ),
        "ou_timescale_prior_median_seconds": float(
            config.model.ou_timescale_prior_median_seconds
        ),
        "ou_timescale_prior_sigma_ln": float(config.model.ou_timescale_prior_sigma_ln),
        "jitter_prior_scale_ppm": float(config.model.jitter_prior_scale_ppm),
        "maximum_noise_rhat": None,
        "minimum_noise_effective_sample_size": None,
    }
    finite_names: list[str] = []
    for name in ("ou_amplitude", "ou_timescale", "jitter", "white_jitter"):
        values = run.samples.get(name)
        if values is None:
            summary[f"{name}_mean"] = None
            summary[f"{name}_standard_deviation"] = None
            continue
        values = np.asarray(values, dtype=float)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_standard_deviation"] = float(np.std(values))
        finite_names.append(name)

    if run.grouped_samples is not None and run.grouped_samples:
        grouped = {
            name: np.asarray(run.grouped_samples[name], dtype=float)
            for name in finite_names
            if name in run.grouped_samples
        }
        if grouped and next(iter(grouped.values())).shape[0] > 1:
            from numpyro.diagnostics import summary as diagnostic_summary

            diagnostics = diagnostic_summary(
                grouped,
                prob=0.68,
                group_by_chain=True,
            )
            rhat_values = [float(np.nanmax(diagnostics[name]["r_hat"])) for name in grouped]
            ess_values = [float(np.nanmin(diagnostics[name]["n_eff"])) for name in grouped]
            summary["maximum_noise_rhat"] = float(np.nanmax(rhat_values))
            summary["minimum_noise_effective_sample_size"] = float(
                np.nanmin(ess_values)
            )
    return summary


def _add_noise_samples(
    payload: dict[str, Any],
    run: NumpyroRun,
) -> None:
    """Add sampled time-correlated-noise values to the samples archive."""

    for name in ("ou_amplitude", "ou_timescale", "jitter", "white_jitter"):
        if name in run.samples:
            payload[name] = np.asarray(run.samples[name])
        if run.grouped_samples is not None and name in run.grouped_samples:
            payload[f"{name}_by_chain"] = np.asarray(run.grouped_samples[name])


def _secondary_design(config: MappingConfig, curve: LightCurve) -> _DesignBundle:
    """Build fixed stellar-transit and secondary-eclipse operators.

    A modest 16-by-64 projected-disc quadrature is sufficient for the quick
    fit and keeps laptop and SLURM smoke runs short.  Four midpoint samples
    are used for a finite exposure when the configuration requests exposure
    integration.  The map longitude at transit is fixed to ``pi`` so the
    synchronous planet presents its dayside at secondary eclipse.
    """

    system = config.system
    period = float(system.period_days)
    t0 = float(system.transit_time)
    inclination = float(system.inclination_degrees)
    radius_ratio = float(system.radius_ratio)
    ydeg = int(config.map.harmonic_degree)
    time_days = _time_days(curve)
    quadrature = disk_quadrature(
        n_radial=int(config.compute.quadrature_radial),
        n_azimuth=int(config.compute.quadrature_azimuth),
    )

    def operator(evaluation_time: NDArray[np.float64]) -> NDArray[np.float64]:
        requested_time = np.asarray(evaluation_time, dtype=float)
        flat_time = requested_time.reshape(-1)
        chunks = []
        # The physics operator has one quadrature-node axis per time. Chunking
        # keeps a full 8000-point light curve safe on a laptop.
        for start in range(0, flat_time.size, 256):
            chunk_time = flat_time[start : start + 256]
            chunks.append(
                np.asarray(
                    secondary_eclipse_design_matrix(
                        chunk_time,
                        period,
                        float(system.a_over_rstar),
                        inclination,
                        radius_ratio,
                        ydeg,
                        t0,
                        # This is intentional: the secondary eclipse should show the
                        # dayside for a synchronously rotating planet.
                        theta0=np.pi,
                        # Match starry's viewing geometry. An orbit at 90 deg
                        # is equator-on; lower inclinations view the map from
                        # the corresponding positive sub-observer latitude.
                        subobserver_lat=np.deg2rad(90.0 - inclination),
                        angle_unit="deg",
                        quadrature=quadrature,
                        light_delay=bool(config.model.include_light_delay),
                        rstar_meters=(
                            None
                            if system.stellar_radius_rsun is None
                            else float(system.stellar_radius_rsun) * 6.957e8
                        ),
                    ),
                    dtype=float,
                )
            )
        flat_design = np.concatenate(chunks, axis=0)
        return flat_design.reshape(requested_time.shape + (flat_design.shape[-1],))

    def star_operator(evaluation_time: NDArray[np.float64]) -> NDArray[np.float64]:
        requested_time = np.asarray(evaluation_time, dtype=float)
        flat_time = requested_time.reshape(-1)
        if config.model.include_light_delay:
            from robert_mapping.physics import light_travel_time_days

            flat_time = flat_time + np.asarray(
                light_travel_time_days(
                    flat_time,
                    period,
                    float(system.a_over_rstar),
                    inclination,
                    float(system.stellar_radius_rsun) * 6.957e8,
                    t0,
                    angle_unit="deg",
                ),
                dtype=float,
            )
        chunks = []
        for start in range(0, flat_time.size, 256):
            chunk_time = flat_time[start : start + 256]
            chunks.append(
                np.asarray(
                    stellar_transit_flux(
                        chunk_time,
                        period,
                        float(system.a_over_rstar),
                        inclination,
                        radius_ratio,
                        t0,
                        u1=float(system.limb_darkening_u1),
                        u2=float(system.limb_darkening_u2),
                        angle_unit="deg",
                        quadrature=quadrature,
                    ),
                    dtype=float,
                )
            )
        return np.concatenate(chunks).reshape(requested_time.shape)

    if config.model.include_light_delay and system.stellar_radius_rsun is None:
        raise ValueError(
            "model.include_light_delay requires system.stellar_radius_rsun."
        )

    integrate = bool(config.model.integrate_exposure) and curve.exposure_seconds > 0.0
    n_subsamples = 4 if integrate else 1
    if integrate:
        exposure_days = float(curve.exposure_seconds) / 86_400.0
        offsets = ((np.arange(n_subsamples, dtype=float) + 0.5) / n_subsamples - 0.5)
        sampled_time = time_days[:, None] + offsets[None, :] * exposure_days
        sampled_design = operator(sampled_time)
        map_design = np.mean(sampled_design, axis=1)
        stellar_model = np.mean(star_operator(sampled_time), axis=1)
    else:
        map_design = operator(time_days)
        stellar_model = star_operator(time_days)

    if map_design.ndim != 2 or map_design.shape[0] != time_days.size:
        raise ValueError("The secondary-eclipse operator returned an invalid shape.")
    if not np.all(np.isfinite(map_design)):
        raise ValueError("The secondary-eclipse operator returned non-finite values.")
    if config.systematics.mode == "corrected":
        nuisance_design = np.empty((time_days.size, 0), dtype=float)
        nuisance_names: tuple[str, ...] = ()
    else:
        auxiliary = None if curve.regressors is None else np.asarray(curve.regressors, dtype=float)
        if auxiliary is not None and config.systematics.standardize_regressors:
            centre = np.mean(auxiliary, axis=0)
            scale = np.std(auxiliary, axis=0)
            scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
            auxiliary = (auxiliary - centre) / scale
        nuisance = build_systematics_design(
            time_days,
            segment_ids=curve.segments,
            include_offsets=bool(config.systematics.fit_offset),
            polynomial_order=int(config.systematics.polynomial_order),
            standardize_time=bool(config.systematics.standardize_time),
            ramp_timescale=(
                float(config.systematics.ramp_timescale_hours) / 24.0
                if config.systematics.exponential_ramp
                and not config.systematics.fit_ramp_rate
                else None
            ),
            auxiliary_regressors=auxiliary,
            auxiliary_names=(curve.regressor_names if auxiliary is not None else None),
        )
        nuisance_design = np.asarray(nuisance.matrix, dtype=float)
        nuisance_names = tuple(nuisance.names)
    return _DesignBundle(
        time_days=time_days,
        map_design=map_design,
        stellar_flux=stellar_model,
        quadrature_radial=quadrature.n_radial,
        quadrature_azimuth=quadrature.n_azimuth,
        exposure_subsamples=n_subsamples,
        harmonic_degree=ydeg,
        systematics_design=nuisance_design,
        systematics_names=nuisance_names,
        systematics_mode=str(config.systematics.mode),
    )


def _linear_prior(config: MappingConfig, ncoefficients: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a weak baseline prior and a broad thermal-map prior."""

    # The first column is the unit stellar baseline.  A 0.05 map scale is
    # broad relative to the expected ~0.005 planet-to-star flux and matches
    # the compact Hammond-style Gaussian comparison.
    planet_scale = max(0.05, 10.0 * float(config.system.planet_flux_ratio))
    mean = np.concatenate(([1.0], np.zeros(ncoefficients, dtype=float)))
    scale = np.concatenate(([0.05], np.full(ncoefficients, planet_scale)))
    return mean, scale


def _linear_fit(
    config: MappingConfig,
    curve: LightCurve,
    bundle: _DesignBundle,
) -> tuple[LinearGaussianPosterior, NDArray[np.float64], dict[str, Any]]:
    """Run the exact Gaussian harmonic fit."""

    if str(config.model.noise_model).lower() not in {"white", "independent"}:
        raise ValueError(
            "The exact map solver does not support correlated noise; use "
            "inference.sampler: nuts for time-correlated noise "
            "(noise_model: ou)."
        )
    if bundle.systematics_design.shape[1]:
        raise ValueError(
            "Joint systematics fitting currently requires inference.sampler: nuts."
        )

    design = np.column_stack((np.ones(curve.n_observations), bundle.map_design))
    prior_mean, prior_scale = _linear_prior(config, bundle.map_design.shape[1])
    adjusted_flux = np.asarray(curve.flux, dtype=float) - bundle.stellar_flux + 1.0
    posterior = fit_linear_gaussian(
        design,
        adjusted_flux,
        curve.flux_err,
        prior_mean=prior_mean,
        prior_scale=prior_scale,
    )
    model_flux = bundle.stellar_flux - 1.0 + posterior.predict(design)
    residual = np.asarray(curve.flux, dtype=float) - model_flux
    chi2 = float(np.sum((residual / curve.flux_err) ** 2))
    dof = max(1, curve.n_observations - design.shape[1])
    map_mean = np.asarray(posterior.mean[1:], dtype=float)
    map_covariance = np.asarray(posterior.covariance[1:, 1:], dtype=float)
    summary = {
        "baseline": float(posterior.mean[0]),
        "baseline_standard_deviation": float(np.sqrt(max(posterior.covariance[0, 0], 0.0))),
        "coefficient_mean": map_mean.tolist(),
        "coefficient_standard_deviation": np.sqrt(np.clip(np.diag(map_covariance), 0.0, None)).tolist(),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "chi_squared": chi2,
        "reduced_chi_squared": chi2 / dof,
        "degrees_of_freedom": dof,
        "design_shape": [int(value) for value in design.shape],
    }
    return posterior, model_flux, {
        **summary,
        "map_coefficients": map_mean,
        "map_covariance": map_covariance,
        "full_posterior_mean": np.asarray(posterior.mean, dtype=float),
        "full_posterior_covariance": np.asarray(posterior.covariance, dtype=float),
        "residuals": residual,
    }


def _harmonic_nuts_fit(
    config: MappingConfig,
    curve: LightCurve,
    bundle: _DesignBundle,
    *,
    threads: int,
) -> tuple[NumpyroRun, NDArray[np.float64], dict[str, Any]]:
    """Run the direct harmonic-coefficient NumPyro model.

    This branch deliberately keeps the positive-map policy separate from the
    parameterization. Harmonic coefficients may be negative; positivity is a
    diagnostic-only property for this future model.
    """

    ncoefficients = ncoeff(bundle.harmonic_degree)
    active_indices = np.asarray(
        config.map.active_harmonic_indices
        if config.map.active_harmonic_indices
        else tuple(range(ncoefficients)),
        dtype=int,
    )
    active_design = np.asarray(bundle.map_design, dtype=float)[:, active_indices]
    n_active = int(active_indices.size)
    planet_scale = max(float(config.system.planet_flux_ratio), 1.0e-8)
    prior_mean = np.zeros(n_active, dtype=float)
    prior_mean[0] = float(config.system.planet_flux_ratio)
    prior_sigma = np.full(
        n_active,
        float(config.map.pixel_log_sigma) * planet_scale,
        dtype=float,
    )
    prior_sigma[0] = 0.5 * planet_scale
    effective_chains = min(
        3, int(config.inference.chains), int(config.compute.max_cpus), threads
    )
    if effective_chains < 1:
        raise ValueError("At least one CPU is required for NumPyro sampling.")
    if int(config.inference.warmup) < 1:
        raise ValueError("inference.warmup must be at least one when sampler is nuts.")

    sampler = sample_harmonic_map
    if sampler is None:
        sampler = getattr(numpyro_backend, "sample_harmonic_map", None)
    if sampler is None:
        raise RuntimeError(
            "The direct-harmonic sampler is not available. "
            "Install or add numpyro_backend.sample_harmonic_map."
        )
    run = sampler(
        active_design,
        curve.flux,
        curve.flux_err,
        stellar_flux=bundle.stellar_flux,
        coefficient_prior_mean=prior_mean,
        coefficient_prior_sigma=prior_sigma,
        warmup=int(config.inference.warmup),
        draws=int(config.inference.draws),
        chains=effective_chains,
        seed=int(config.project.seed),
        target_accept=float(config.inference.target_accept),
        progress_bar=bool(config.inference.progress_bar),
        dense_mass=bool(config.inference.dense_mass),
        init_strategy=str(config.inference.init_strategy),
        systematics_design=(
            bundle.systematics_design
            if bundle.systematics_design.shape[1]
            else None
        ),
        systematics_mode=(
            bundle.systematics_mode
            if bundle.systematics_mode != "corrected"
            else "additive"
        ),
        systematics_prior_sigma=float(config.systematics.coefficient_prior_sigma),
        likelihood=str(config.model.likelihood),
        student_t_nu=float(config.model.student_t_nu),
        time_seconds=(
            np.asarray(bundle.time_days, dtype=float)
            - float(np.asarray(bundle.time_days, dtype=float)[0])
        )
        * 86_400.0,
        noise_model=str(config.model.noise_model),
        ou_amplitude_prior_scale=(
            float(config.model.ou_amplitude_prior_scale_ppm) * 1.0e-6
        ),
        ou_timescale_prior_median=float(
            config.model.ou_timescale_prior_median_seconds
        ),
        ou_timescale_prior_sigma_ln=float(config.model.ou_timescale_prior_sigma_ln),
        jitter_prior_scale=float(config.model.jitter_prior_scale_ppm) * 1.0e-6,
        fit_white_jitter=bool(config.model.fit_white_jitter),
    )
    raw_coefficients = run.samples.get("harmonic_coefficients")
    if raw_coefficients is None:
        raw_coefficients = run.samples.get("coefficients")
    if raw_coefficients is None:
        raise ValueError(
            "sample_harmonic_map must return harmonic_coefficients or coefficients"
        )
    active_samples = np.asarray(raw_coefficients, dtype=float)
    if active_samples.ndim != 2 or active_samples.shape[1] != n_active:
        raise ValueError(
            "Direct harmonic samples must have shape "
            f"(draw, {n_active})"
        )
    harmonic_samples = np.zeros(
        (active_samples.shape[0], ncoefficients), dtype=float
    )
    harmonic_samples[:, active_indices] = active_samples
    fallback_flux = (
        bundle.stellar_flux[None, :]
        + harmonic_samples @ np.asarray(bundle.map_design, dtype=float).T
    )
    flux_samples = np.asarray(
        run.samples.get("flux", fallback_flux), dtype=float
    )
    if flux_samples.shape != fallback_flux.shape:
        raise ValueError("Direct harmonic flux samples have an invalid shape")
    mean_flux = np.mean(flux_samples, axis=0)
    residual = np.asarray(curve.flux, dtype=float) - mean_flux
    error_scale_mean = (
        float(np.mean(np.asarray(run.samples["error_scale"], dtype=float)))
        if "error_scale" in run.samples
        else 1.0
    )
    white_jitter_mean = (
        float(np.mean(np.asarray(run.samples["white_jitter"], dtype=float)))
        if "white_jitter" in run.samples
        else 0.0
    )
    effective_error = np.sqrt(
        np.square(np.asarray(curve.flux_err, dtype=float) * error_scale_mean)
        + white_jitter_mean**2
    )
    divergences = np.asarray(
        run.extra_fields.get(
            "diverging", np.zeros(harmonic_samples.shape[0], dtype=bool)
        )
    )

    maximum_rhat = None
    minimum_effective_sample_size = None
    maximum_systematics_rhat = None
    minimum_systematics_effective_sample_size = None
    harmonic_by_chain = None
    if run.grouped_samples is not None:
        grouped = run.grouped_samples.get("harmonic_coefficients")
        if grouped is None:
            grouped = run.grouped_samples.get("coefficients")
        if grouped is not None:
            active_by_chain = np.asarray(grouped, dtype=float)
            if active_by_chain.ndim != 3 or active_by_chain.shape[-1] != n_active:
                raise ValueError("Grouped direct harmonic samples have an invalid shape")
            harmonic_by_chain = np.zeros(
                active_by_chain.shape[:-1] + (ncoefficients,), dtype=float
            )
            harmonic_by_chain[..., active_indices] = active_by_chain
            if active_by_chain.shape[0] > 1:
                from numpyro.diagnostics import summary as diagnostic_summary

                diagnostics = diagnostic_summary(
                    {"harmonics": active_by_chain},
                    prob=0.68,
                    group_by_chain=True,
                )["harmonics"]
                finite_rhat = np.asarray(diagnostics["r_hat"], dtype=float)
                finite_rhat = finite_rhat[np.isfinite(finite_rhat)]
                finite_ess = np.asarray(diagnostics["n_eff"], dtype=float)
                finite_ess = finite_ess[np.isfinite(finite_ess)]
                maximum_rhat = (
                    float(np.max(finite_rhat)) if finite_rhat.size else None
                )
                minimum_effective_sample_size = (
                    float(np.min(finite_ess)) if finite_ess.size else None
                )
                if "systematics_coefficients" in run.grouped_samples:
                    systematics_diagnostics = diagnostic_summary(
                        {
                            "systematics": np.asarray(
                                run.grouped_samples["systematics_coefficients"],
                                dtype=float,
                            )
                        },
                        prob=0.68,
                        group_by_chain=True,
                    )["systematics"]
                    maximum_systematics_rhat = float(
                        np.nanmax(systematics_diagnostics["r_hat"])
                    )
                    minimum_systematics_effective_sample_size = float(
                        np.nanmin(systematics_diagnostics["n_eff"])
                    )

    summary = {
        "baseline": 1.0,
        "coefficient_mean": np.mean(harmonic_samples, axis=0).tolist(),
        "coefficient_standard_deviation": np.std(harmonic_samples, axis=0).tolist(),
        "harmonic_coefficient_mean": np.mean(harmonic_samples, axis=0).tolist(),
        "harmonic_coefficient_standard_deviation": np.std(
            harmonic_samples, axis=0
        ).tolist(),
        "coefficient_prior_mean": prior_mean.tolist(),
        "coefficient_prior_sigma": prior_sigma.tolist(),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "chi_squared": float(np.sum((residual / effective_error) ** 2)),
        "n_parameters": int(n_active),
        "active_harmonic_indices": active_indices.tolist(),
        "n_samples": int(harmonic_samples.shape[0]),
        "chains": effective_chains,
        "warmup": int(config.inference.warmup),
        "draws": int(config.inference.draws),
        "dense_mass": bool(config.inference.dense_mass),
        "divergences": int(np.sum(divergences)),
        "maximum_rhat": maximum_rhat,
        "minimum_effective_sample_size": minimum_effective_sample_size,
        "maximum_pixel_rhat": None,
        "minimum_pixel_effective_sample_size": None,
        "maximum_systematics_rhat": maximum_systematics_rhat,
        "minimum_systematics_effective_sample_size": (
            minimum_systematics_effective_sample_size
        ),
        "mean_entropy": None,
        "design_shape": [int(value) for value in active_design.shape],
        "threads": int(threads),
        "pixel_transform_shape": [0, 0],
        "pixel_grid": "none",
        "parameterization": "direct_harmonics",
        "positivity_policy": "diagnostic_only",
        "anchor_indices": [],
        "anchor_longitude_degrees": [],
        "anchor_latitude_degrees": [],
        "anchor_coordinates_degrees": [],
        "anchor_rank": None,
        "anchor_condition_number": None,
        "systematics_mode": bundle.systematics_mode,
        "likelihood": str(config.model.likelihood),
        "student_t_nu": (
            float(config.model.student_t_nu)
            if config.model.likelihood == "student_t"
            else None
        ),
        "systematics_coefficient_names": list(bundle.systematics_names),
        "systematics_coefficient_mean": (
            np.mean(
                np.asarray(run.samples["systematics_coefficients"], dtype=float),
                axis=0,
            ).tolist()
            if "systematics_coefficients" in run.samples
            else []
        ),
        **_noise_diagnostics(run, config),
        "harmonic_samples": harmonic_samples,
        "flux_samples": flux_samples,
        "residuals": residual,
        "harmonic_by_chain": harmonic_by_chain,
    }
    return run, harmonic_samples, {**summary, "model_flux": mean_flux}


def _pixel_fit(
    config: MappingConfig,
    curve: LightCurve,
    bundle: _DesignBundle,
    *,
    threads: int,
) -> tuple[NumpyroRun, NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    """Run the small positive-pixel NumPyro model."""

    representation = str(config.map.representation).lower()
    anchor_transform = None
    if representation == "harmonics":
        anchor_transform = rank_revealing_anchor_transform(
            bundle.harmonic_degree, oversample=3
        )
        npixels = ncoeff(bundle.harmonic_degree)
        transform = np.asarray(anchor_transform.anchor_to_harmonics, dtype=float)
        pixel_grid = "starry_1.1_mollweide_harmonic_anchors"
    else:
        npixels = int(config.map.n_pixels)
        mollweide_pixels, _, mollweide_transform = starry_pixel_transforms(
            bundle.harmonic_degree, oversample=3
        )
        if npixels == mollweide_pixels.npix:
            transform = np.asarray(mollweide_transform, dtype=float)
            pixel_grid = "starry_1.1_mollweide"
        else:
            pixels = fibonacci_pixels(npixels)
            transform = np.asarray(
                pixels_to_harmonics(
                    np.eye(npixels, dtype=float), pixels, bundle.harmonic_degree
                ),
                dtype=float,
            ).T
            pixel_grid = "fibonacci_fallback"
    pixel_design = np.asarray(bundle.map_design @ transform, dtype=float)
    effective_chains = min(3, int(config.inference.chains), int(config.compute.max_cpus), threads)
    if effective_chains < 1:
        raise ValueError("At least one CPU is required for NumPyro sampling.")
    if int(config.inference.warmup) < 1:
        raise ValueError("inference.warmup must be at least one when sampler is nuts.")
    if config.map.pixel_prior_mean_ppm is not None:
        arithmetic_mean = float(config.map.pixel_prior_mean_ppm) * 1.0e-6 / np.pi
        arithmetic_sd = float(config.map.pixel_prior_sd_ppm) * 1.0e-6 / np.pi
        pixel_log_sigma = float(
            np.sqrt(np.log1p((arithmetic_sd / arithmetic_mean) ** 2))
        )
        pixel_prior_median = float(
            arithmetic_mean / np.sqrt(1.0 + (arithmetic_sd / arithmetic_mean) ** 2)
        )
    else:
        arithmetic_mean = None
        arithmetic_sd = None
        pixel_prior_median = max(
            float(config.system.planet_flux_ratio) / np.pi, 1.0e-8
        )
        pixel_log_sigma = float(config.map.pixel_log_sigma)

    systematics_prior_sigma: float | np.ndarray
    if config.systematics.coefficient_prior_sigmas:
        systematics_prior_sigma = np.asarray(
            config.systematics.coefficient_prior_sigmas, dtype=float
        )
        if systematics_prior_sigma.shape != (bundle.systematics_design.shape[1],):
            raise ValueError(
                "systematics.coefficient_prior_sigmas must contain one value "
                "for each fitted nuisance coefficient: "
                + ", ".join(bundle.systematics_names)
            )
    else:
        systematics_prior_sigma = float(
            config.systematics.coefficient_prior_sigma
        )

    run = sample_positive_map(
        pixel_design,
        curve.flux,
        curve.flux_err,
        stellar_flux=bundle.stellar_flux,
        pixel_prior_mean=pixel_prior_median,
        pixel_log_sigma=pixel_log_sigma,
        alpha=float(config.map.entropy_penalty),
        warmup=int(config.inference.warmup),
        draws=int(config.inference.draws),
        chains=effective_chains,
        seed=int(config.project.seed),
        target_accept=float(config.inference.target_accept),
        progress_bar=bool(config.inference.progress_bar),
        dense_mass=bool(config.inference.dense_mass),
        init_strategy=str(config.inference.init_strategy),
        systematics_design=(
            bundle.systematics_design
            if bundle.systematics_design.shape[1]
            else None
        ),
        systematics_mode=(
            bundle.systematics_mode
            if bundle.systematics_mode != "corrected"
            else "additive"
        ),
        systematics_prior_sigma=systematics_prior_sigma,
        likelihood=str(config.model.likelihood),
        student_t_nu=float(config.model.student_t_nu),
        time_seconds=(
            np.asarray(bundle.time_days, dtype=float)
            - float(np.asarray(bundle.time_days, dtype=float)[0])
        )
        * 86_400.0,
        noise_model=str(config.model.noise_model),
        ou_amplitude_prior_scale=(
            float(config.model.ou_amplitude_prior_scale_ppm) * 1.0e-6
        ),
        ou_timescale_prior_median=float(
            config.model.ou_timescale_prior_median_seconds
        ),
        ou_timescale_prior_sigma_ln=float(config.model.ou_timescale_prior_sigma_ln),
        jitter_prior_scale=float(config.model.jitter_prior_scale_ppm) * 1.0e-6,
        sample_ramp_rate=bool(config.systematics.fit_ramp_rate),
        ramp_rate_prior_mean_per_day=float(
            config.systematics.ramp_rate_prior_mean_per_day
        ),
        ramp_rate_prior_sigma_per_day=float(
            config.systematics.ramp_rate_prior_sigma_per_day
        ),
        ramp_amplitude_prior_sigma=float(
            config.systematics.ramp_amplitude_prior_sigma
        ),
        fit_error_scale=bool(config.model.fit_error_scale),
        error_scale_log_sigma=float(config.model.error_scale_log_sigma),
        fit_white_jitter=bool(config.model.fit_white_jitter),
        multiplicative_composition=str(
            config.systematics.multiplicative_composition
        ),
        systematics_names=bundle.systematics_names,
    )
    pixel_samples = np.asarray(run.samples["pixels"], dtype=float)
    harmonic_samples = pixel_samples @ transform.T
    flux_samples = np.asarray(run.samples.get("flux", 1.0 + pixel_samples @ pixel_design.T), dtype=float)
    mean_flux = np.mean(flux_samples, axis=0)
    residual = np.asarray(curve.flux, dtype=float) - mean_flux
    error_scale_mean = (
        float(np.mean(np.asarray(run.samples["error_scale"], dtype=float)))
        if "error_scale" in run.samples
        else 1.0
    )
    entropy_samples = np.asarray(run.samples.get("entropy", np.full(pixel_samples.shape[0], np.nan)), dtype=float)
    divergences = np.asarray(run.extra_fields.get("diverging", np.zeros(pixel_samples.shape[0], dtype=bool)))
    max_rhat = None
    minimum_ess = None
    maximum_harmonic_rhat = None
    minimum_harmonic_ess = None
    maximum_systematics_rhat = None
    minimum_systematics_ess = None
    if run.grouped_samples is not None:
        grouped_pixels = np.asarray(run.grouped_samples["pixels"], dtype=float)
        if grouped_pixels.shape[0] > 1:
            from numpyro.diagnostics import summary as diagnostic_summary

            diagnostics = diagnostic_summary(
                {"pixels": grouped_pixels}, prob=0.68, group_by_chain=True
            )
            pixel_diagnostics = diagnostics["pixels"]
            max_rhat = float(np.nanmax(pixel_diagnostics["r_hat"]))
            minimum_ess = float(np.nanmin(pixel_diagnostics["n_eff"]))
            grouped_harmonics = grouped_pixels @ transform.T
            harmonic_diagnostics = diagnostic_summary(
                {"harmonics": grouped_harmonics}, prob=0.68, group_by_chain=True
            )["harmonics"]
            maximum_harmonic_rhat = float(
                np.nanmax(harmonic_diagnostics["r_hat"])
            )
            minimum_harmonic_ess = float(
                np.nanmin(harmonic_diagnostics["n_eff"])
            )
            if "systematics_coefficients" in run.grouped_samples:
                systematics_diagnostics = diagnostic_summary(
                    {
                        "systematics": np.asarray(
                            run.grouped_samples["systematics_coefficients"], dtype=float
                        )
                    },
                    prob=0.68,
                    group_by_chain=True,
                )["systematics"]
                maximum_systematics_rhat = float(
                    np.nanmax(systematics_diagnostics["r_hat"])
                )
                minimum_systematics_ess = float(
                    np.nanmin(systematics_diagnostics["n_eff"])
                )
    summary = {
        "baseline": 1.0,
        "coefficient_mean": np.mean(pixel_samples, axis=0).tolist(),
        "coefficient_standard_deviation": np.std(pixel_samples, axis=0).tolist(),
        "harmonic_coefficient_mean": np.mean(harmonic_samples, axis=0).tolist(),
        "harmonic_coefficient_standard_deviation": np.std(harmonic_samples, axis=0).tolist(),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
        "chi_squared": float(
            np.sum((residual / (curve.flux_err * error_scale_mean)) ** 2)
        ),
        "n_pixels": npixels,
        "n_samples": int(pixel_samples.shape[0]),
        "chains": effective_chains,
        "warmup": int(config.inference.warmup),
        "draws": int(config.inference.draws),
        "dense_mass": bool(config.inference.dense_mass),
        "divergences": int(np.sum(divergences)),
        "maximum_rhat": maximum_harmonic_rhat,
        "minimum_effective_sample_size": minimum_harmonic_ess,
        "maximum_pixel_rhat": max_rhat,
        "minimum_pixel_effective_sample_size": minimum_ess,
        "maximum_systematics_rhat": maximum_systematics_rhat,
        "minimum_systematics_effective_sample_size": minimum_systematics_ess,
        "mean_entropy": float(np.nanmean(entropy_samples)) if np.any(np.isfinite(entropy_samples)) else None,
        "design_shape": [int(value) for value in pixel_design.shape],
        "threads": int(threads),
        "pixel_transform_shape": [int(value) for value in transform.shape],
        "pixel_grid": pixel_grid,
        "parameterization": (
            "harmonic_anchors" if anchor_transform is not None else "pixels"
        ),
        "pixel_prior_median_internal": float(pixel_prior_median),
        "pixel_prior_log_sigma": float(pixel_log_sigma),
        "pixel_prior_mean_ppm": (
            None if arithmetic_mean is None else float(config.map.pixel_prior_mean_ppm)
        ),
        "pixel_prior_sd_ppm": (
            None if arithmetic_sd is None else float(config.map.pixel_prior_sd_ppm)
        ),
        "anchor_indices": (
            anchor_transform.anchor_indices.tolist()
            if anchor_transform is not None
            else []
        ),
        "anchor_longitude_degrees": (
            anchor_transform.anchor_longitude_degrees.tolist()
            if anchor_transform is not None
            else []
        ),
        "anchor_latitude_degrees": (
            anchor_transform.anchor_latitude_degrees.tolist()
            if anchor_transform is not None
            else []
        ),
        "anchor_coordinates_degrees": (
            np.column_stack(
                (
                    anchor_transform.anchor_longitude_degrees,
                    anchor_transform.anchor_latitude_degrees,
                )
            ).tolist()
            if anchor_transform is not None
            else []
        ),
        "anchor_rank": (
            int(anchor_transform.rank) if anchor_transform is not None else None
        ),
        "anchor_condition_number": (
            float(anchor_transform.condition_number)
            if anchor_transform is not None
            else None
        ),
        "systematics_mode": bundle.systematics_mode,
        "likelihood": str(config.model.likelihood),
        "student_t_nu": (
            float(config.model.student_t_nu)
            if config.model.likelihood == "student_t"
            else None
        ),
        "systematics_coefficient_names": list(bundle.systematics_names),
        "systematics_coefficient_mean": (
            np.mean(
                np.asarray(run.samples["systematics_coefficients"], dtype=float),
                axis=0,
            ).tolist()
            if "systematics_coefficients" in run.samples
            else []
        ),
        "ramp_amplitude_mean": (
            float(np.mean(np.asarray(run.samples["ramp_amplitude"], dtype=float)))
            if "ramp_amplitude" in run.samples
            else None
        ),
        "ramp_rate_mean_per_day": (
            float(np.mean(np.asarray(run.samples["ramp_rate_per_day"], dtype=float)))
            if "ramp_rate_per_day" in run.samples
            else None
        ),
        "error_scale_mean": (
            error_scale_mean
            if "error_scale" in run.samples
            else None
        ),
        **_noise_diagnostics(run, config),
        "pixel_samples": pixel_samples,
        "harmonic_samples": harmonic_samples,
        "flux_samples": flux_samples,
        "transform": transform,
        "residuals": residual,
    }
    return run, pixel_samples, harmonic_samples, {**summary, "model_flux": mean_flux}


def _json_value(value: Any) -> Any:
    """Convert NumPy scalar/container values for the run manifest."""

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_common_outputs(
    output: Path,
    config: MappingConfig,
    bundle: _DesignBundle,
    curve: LightCurve,
    *,
    sampler: str,
    summary: dict[str, Any],
    model_flux: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    covariance: NDArray[np.float64] | None = None,
) -> tuple[Path, Path]:
    """Write arrays and the JSON summary shared by both branches."""

    output.mkdir(parents=True, exist_ok=True)
    coefficients_path = output / "coefficients.npy"
    np.save(coefficients_path, np.asarray(coefficients, dtype=float))
    # These named copies make the files self-explanatory in a results folder.
    np.save(output / "harmonic_coefficients.npy", np.asarray(coefficients, dtype=float))
    np.save(output / "model_flux.npy", np.asarray(model_flux, dtype=float))
    np.save(output / "stellar_flux.npy", np.asarray(bundle.stellar_flux, dtype=float))
    np.save(output / "residuals.npy", np.asarray(curve.flux, dtype=float) - np.asarray(model_flux, dtype=float))
    if covariance is not None:
        np.save(output / "coefficient_covariance.npy", np.asarray(covariance, dtype=float))

    payload = {
        "status": "complete",
        "sampler": sampler,
        "project": config.project.name,
        "n_observations": int(curve.n_observations),
        "harmonic_degree": int(bundle.harmonic_degree),
        "theta0_radians": float(np.pi),
        "quadrature": {
            "n_radial": int(bundle.quadrature_radial),
            "n_azimuth": int(bundle.quadrature_azimuth),
        },
        "exposure_subsamples": int(bundle.exposure_subsamples),
        "time_unit_input": curve.time_unit,
        "systematics": {
            "mode": bundle.systematics_mode,
            "coefficient_names": list(bundle.systematics_names),
            "design_shape": [int(value) for value in bundle.systematics_design.shape],
        },
        "files": {
            "coefficients": coefficients_path.name,
            "model_flux": "model_flux.npy",
            "stellar_flux": "stellar_flux.npy",
            "residuals": "residuals.npy",
        },
        **summary,
    }
    summary_path = output / "fit_summary.json"
    summary_path.write_text(json.dumps(_json_value(payload), indent=2) + "\n", encoding="utf-8")
    return summary_path, coefficients_path


def run_fit(config: MappingConfig) -> FitResult:
    """Run the configured quick eclipse-map fit and save its outputs.

    ``sampler: map`` and ``sampler: none`` use the same exact Gaussian
    harmonic posterior.  ``sampler: nuts`` samples positive pixel intensities
    with the configured small chain, warmup, and draw counts.  The returned
    :class:`FitResult` points to all machine-readable output files.
    """

    threads = _set_thread_limits(config)
    curve = load_light_curve(config)
    bundle = _secondary_design(config, curve)
    output = Path(config.output.directory).expanduser()
    if not output.is_absolute():
        output = (config.base_dir / output).resolve()

    if config.output.save_resolved_config:
        write_resolved_config(config, output / "resolved_config.yml")

    sampler = str(config.inference.sampler).lower()
    if sampler in {"map", "none"}:
        posterior, model_flux, details = _linear_fit(config, curve, bundle)
        summary_path, coefficients_path = _write_common_outputs(
            output,
            config,
            bundle,
            curve,
            sampler=sampler,
            summary={
                key: value
                for key, value in details.items()
                if key not in {"map_coefficients", "map_covariance", "full_posterior_mean", "full_posterior_covariance", "residuals"}
            },
            model_flux=model_flux,
            coefficients=details["map_coefficients"],
            covariance=details["map_covariance"],
        )
        np.save(output / "posterior_mean.npy", details["full_posterior_mean"])
        np.save(output / "posterior_covariance.npy", details["full_posterior_covariance"])
        return FitResult(
            status="complete",
            sampler=sampler,
            n_observations=curve.n_observations,
            n_parameters=int(bundle.map_design.shape[1]),
            output_directory=output,
            summary_path=summary_path,
            coefficients_path=coefficients_path,
            samples_path=None,
            summary={
                "sampler": sampler,
                "n_observations": curve.n_observations,
                **{key: value for key, value in details.items() if isinstance(value, (str, int, float, list, tuple))},
            },
        )

    if sampler != "nuts":
        raise ValueError(f"Unsupported sampler {sampler!r}; use nuts, map, or none.")

    if str(config.map.representation).lower() == "direct_harmonics":
        run, harmonic_samples, details = _harmonic_nuts_fit(
            config, curve, bundle, threads=threads
        )
        mean_harmonics = np.mean(harmonic_samples, axis=0)
        summary_path, coefficients_path = _write_common_outputs(
            output,
            config,
            bundle,
            curve,
            sampler=sampler,
            summary={
                key: value
                for key, value in details.items()
                if key
                not in {
                    "harmonic_samples",
                    "flux_samples",
                    "residuals",
                    "model_flux",
                    "harmonic_by_chain",
                }
            },
            model_flux=details["model_flux"],
            coefficients=mean_harmonics,
        )
        samples_path = output / "samples.npz"
        sample_payload: dict[str, Any] = {
            "harmonic_coefficients": harmonic_samples,
            "flux": details["flux_samples"],
        }
        _add_noise_samples(sample_payload, run)
        harmonic_by_chain = details.get("harmonic_by_chain")
        if harmonic_by_chain is not None:
            sample_payload["harmonic_coefficients_by_chain"] = np.asarray(
                harmonic_by_chain
            )
            if run.grouped_samples is not None and "flux" in run.grouped_samples:
                sample_payload["flux_by_chain"] = np.asarray(
                    run.grouped_samples["flux"]
                )
            else:
                sample_payload["flux_by_chain"] = (
                    np.asarray(bundle.stellar_flux)[None, None, :]
                    + np.asarray(harmonic_by_chain) @ np.asarray(bundle.map_design).T
                )
            if (
                run.grouped_samples is not None
                and "systematics_coefficients" in run.grouped_samples
            ):
                sample_payload["systematics_coefficients_by_chain"] = np.asarray(
                    run.grouped_samples["systematics_coefficients"]
                )
            if (
                run.grouped_samples is not None
                and "systematics_model" in run.grouped_samples
            ):
                sample_payload["systematics_model_by_chain"] = np.asarray(
                    run.grouped_samples["systematics_model"]
                )
        if "systematics_coefficients" in run.samples:
            sample_payload["systematics_coefficients"] = np.asarray(
                run.samples["systematics_coefficients"]
            )
        if "systematics_model" in run.samples:
            sample_payload["systematics_model"] = np.asarray(
                run.samples["systematics_model"]
            )
        sample_payload.update(
            {key: np.asarray(value) for key, value in run.extra_fields.items()}
        )
        np.savez_compressed(samples_path, **sample_payload)
        return FitResult(
            status="complete",
            sampler=sampler,
            n_observations=curve.n_observations,
            n_parameters=int(harmonic_samples.shape[-1]),
            output_directory=output,
            summary_path=summary_path,
            coefficients_path=coefficients_path,
            samples_path=samples_path,
            summary={
                "sampler": sampler,
                "n_samples": int(harmonic_samples.shape[0]),
                **details,
            },
        )

    run, pixel_samples, harmonic_samples, details = _pixel_fit(
        config, curve, bundle, threads=threads
    )
    mean_pixels = np.mean(pixel_samples, axis=0)
    mean_harmonics = np.mean(harmonic_samples, axis=0)
    summary_path, coefficients_path = _write_common_outputs(
        output,
        config,
        bundle,
        curve,
        sampler=sampler,
        summary={
            key: value
            for key, value in details.items()
            if key not in {"pixel_samples", "harmonic_samples", "flux_samples", "transform", "residuals", "model_flux"}
        },
        model_flux=details["model_flux"],
        coefficients=mean_pixels,
    )
    np.save(output / "pixel_coefficients.npy", mean_pixels)
    np.save(output / "harmonic_coefficients.npy", mean_harmonics)
    samples_path = output / "samples.npz"
    noise_payload: dict[str, Any] = {}
    _add_noise_samples(noise_payload, run)
    nonlinear_payload = {
        key: np.asarray(run.samples[key])
        for key in ("ramp_amplitude", "ramp_rate_per_day", "error_scale")
        if key in run.samples
    }
    grouped_nonlinear_payload = {
        f"{key}_by_chain": np.asarray(run.grouped_samples[key])
        for key in ("ramp_amplitude", "ramp_rate_per_day", "error_scale")
        if run.grouped_samples is not None and key in run.grouped_samples
    }
    np.savez_compressed(
        samples_path,
        pixels=pixel_samples,
        harmonic_coefficients=harmonic_samples,
        flux=details["flux_samples"],
        **(
            {
                "systematics_coefficients": np.asarray(
                    run.samples["systematics_coefficients"]
                ),
                "systematics_model": np.asarray(run.samples["systematics_model"]),
            }
            if "systematics_coefficients" in run.samples
            else {}
        ),
        **(
            {
                "pixels_by_chain": np.asarray(run.grouped_samples["pixels"]),
                "harmonic_coefficients_by_chain": (
                    np.asarray(run.grouped_samples["pixels"]) @ details["transform"].T
                ),
                "flux_by_chain": np.asarray(run.grouped_samples["flux"]),
                **(
                    {
                        "systematics_coefficients_by_chain": np.asarray(
                            run.grouped_samples["systematics_coefficients"]
                        ),
                        "systematics_model_by_chain": np.asarray(
                            run.grouped_samples["systematics_model"]
                        ),
                    }
                    if "systematics_coefficients" in run.grouped_samples
                    else {}
                ),
            }
            if run.grouped_samples is not None
            else {}
        ),
        **noise_payload,
        **nonlinear_payload,
        **grouped_nonlinear_payload,
        **{key: np.asarray(value) for key, value in run.extra_fields.items()},
    )
    return FitResult(
        status="complete",
        sampler=sampler,
        n_observations=curve.n_observations,
        n_parameters=int(pixel_samples.shape[-1]),
        output_directory=output,
        summary_path=summary_path,
        coefficients_path=coefficients_path,
        samples_path=samples_path,
        summary={"sampler": sampler, "n_samples": int(pixel_samples.shape[0]), **details},
    )


__all__ = ["FitResult", "run_fit"]
