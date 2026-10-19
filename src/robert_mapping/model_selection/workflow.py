"""User-data cross-validation for fixed-geometry eclipse maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from ..compute import ComputePlan, apply_environment, ordered_map, resolve_plan
from ..config import MappingConfig, write_resolved_config
from ..data import LightCurve, load_light_curve
from ..inference.linear import LinearGaussianPosterior, fit_linear_gaussian
from ..inference.run import (
    _DesignBundle,
    _harmonic_nuts_fit,
    _pixel_fit,
    _secondary_design,
    _time_days,
)
from ..inference.numpyro_backend import sample_harmonic_map
from ..physics import disk_quadrature, light_travel_time_days, secondary_eclipse_design_matrix
from .cross_validation import (
    CVComparison,
    compare_pointwise_elpd,
    gaussian_pointwise_elpd,
)
from .fourier import fourier_design_matrix


@dataclass(frozen=True)
class CrossValidationReport:
    """Saved predictive comparison for one user light curve."""

    status: str
    output_directory: Path
    summary_path: Path
    pointwise_path: Path
    n_observations: int
    n_folds: int
    evaluated_observations: int
    map_model: str
    null_model: str
    method: str
    map_elpd: float
    null_elpd: float
    delta_elpd: float
    standard_error: float
    z_score: float | None
    execution_plan: dict[str, Any]
    settings: dict[str, Any]
    priors: dict[str, Any]
    folds: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report summary."""

        return _json_safe(asdict(self))


def _supported_config(config: MappingConfig) -> str:
    """Check the likelihood and model options supported by this CV runner."""

    if str(config.data.kind).lower() != "files":
        raise ValueError(
            "cross-validation requires user data from files; synthetic data are not supported."
        )
    if str(config.data.normalize).lower() != "none":
        raise ValueError(
            "cross-validation requires data.normalize: none so each training fold does not use a full-data normalization."
        )
    if str(config.systematics.mode).lower() != "corrected":
        raise ValueError(
            "cross-validation currently requires corrected data. Joint systematics fitting is not supported."
        )
    if str(config.model.likelihood).lower() != "gaussian":
        raise ValueError(
            "cross-validation currently supports only independent Gaussian errors."
        )
    if str(config.model.noise_model).lower() not in {"white", "independent"}:
        raise ValueError(
            "cross-validation currently supports only independent Gaussian errors."
        )
    if config.model.fit_error_scale or config.model.fit_white_jitter:
        raise ValueError(
            "cross-validation requires fixed reported errors; fitted error scaling and jitter are not supported."
        )
    if config.model.fit_orbit or config.model.fit_limb_darkening:
        raise ValueError(
            "cross-validation requires fixed orbital geometry and limb darkening."
        )
    if not np.isclose(float(config.system.eccentricity), 0.0, atol=0.0, rtol=0.0):
        raise ValueError(
            "cross-validation currently supports fixed circular-orbit geometry only."
        )
    if config.model.fit_baseline or config.model.fit_ramp:
        raise ValueError(
            "cross-validation does not support a fitted baseline or ramp."
        )
    if config.model.include_light_delay and config.system.stellar_radius_rsun is None:
        raise ValueError(
            "model.include_light_delay requires system.stellar_radius_rsun."
        )

    sampler = str(config.inference.sampler).lower()
    representation = str(config.map.representation).lower()
    if sampler in {"map", "none"}:
        if representation not in {"harmonics", "direct_harmonics", "longitude"}:
            raise ValueError(
                "the exact Gaussian CV map supports harmonics, direct_harmonics, or longitude."
            )
    elif sampler == "nuts":
        if min(config.inference.warmup, config.inference.draws, config.inference.chains) < 1:
            raise ValueError("NUTS cross-validation requires positive warmup, draws, and chains.")
        if representation not in {"harmonics", "pixels", "direct_harmonics", "longitude"}:
            raise ValueError(
                "NUTS cross-validation supports harmonics, pixels, direct_harmonics, or longitude map representations."
            )
    else:
        raise ValueError("cross-validation supports inference.sampler map, none, or nuts.")

    if config.cross_validation.n_folds < 2:
        raise ValueError("cross_validation.n_folds must be at least 2.")
    if config.cross_validation.strategy not in {"eclipse_blocks", "contiguous"}:
        raise ValueError(
            "cross_validation.strategy must be eclipse_blocks or contiguous."
        )
    if config.cross_validation.null_model not in {"fourier", "uniform"}:
        raise ValueError("cross_validation.null_model must be fourier or uniform.")
    if (
        not np.isfinite(config.cross_validation.prior_scale)
        or config.cross_validation.prior_scale <= 0.0
    ):
        raise ValueError("cross_validation.prior_scale must be positive and finite.")
    return sampler


def _observer_time_for_orbit_time(
    config: MappingConfig,
    orbit_time_days: float,
) -> float:
    """Map a contact time to observer time when light delay is enabled."""

    if not config.model.include_light_delay:
        return float(orbit_time_days)
    system = config.system
    radius_meters = float(system.stellar_radius_rsun) * 6.957e8
    maximum_delay_days = (
        float(system.a_over_rstar)
        * radius_meters
        / 299_792_458.0
        / 86_400.0
    )

    def equation(observer_time: float) -> float:
        delay = float(
            light_travel_time_days(
                observer_time,
                float(system.period_days),
                float(system.a_over_rstar),
                float(system.inclination_degrees),
                radius_meters,
                float(system.transit_time),
                angle_unit="deg",
            )
        )
        return observer_time + delay - orbit_time_days

    padding = max(1.1 * maximum_delay_days, 1.0e-10)
    return float(
        brentq(
            equation,
            orbit_time_days - padding,
            orbit_time_days + padding,
            xtol=1.0e-13,
        )
    )


def _eclipse_transition_intervals(
    config: MappingConfig,
    time_days: np.ndarray,
    *,
    exposure_seconds: float,
) -> tuple[tuple[float, float], ...]:
    """Return ingress and egress intervals for every eclipse in the data."""

    system = config.system
    period = float(system.period_days)
    semimajor_axis = float(system.a_over_rstar)
    radius_ratio = float(system.radius_ratio)
    inclination = np.deg2rad(float(system.inclination_degrees))
    sin_inclination = float(np.sin(inclination))
    impact = semimajor_axis * abs(float(np.cos(inclination)))
    external_separation = 1.0 + radius_ratio
    if sin_inclination <= np.finfo(float).eps or impact >= external_separation:
        raise ValueError(
            "eclipse_blocks requires a secondary eclipse within the supplied geometry."
        )

    def half_duration_days(contact_separation: float) -> float | None:
        ratio = (1.0 - (contact_separation / semimajor_axis) ** 2) / (
            sin_inclination**2
        )
        if ratio <= 0.0 or ratio > 1.0:
            return None
        phase_offset = math.acos(math.sqrt(ratio))
        return period * phase_offset / (2.0 * math.pi)

    external_half = half_duration_days(external_separation)
    if external_half is None:
        raise ValueError(
            "eclipse_blocks could not find external eclipse contacts for the supplied geometry."
        )
    internal_half = half_duration_days(abs(1.0 - radius_ratio))
    transit_time = float(system.transit_time)
    lower_epoch = math.floor((float(np.min(time_days)) - transit_time) / period - 0.5) - 1
    upper_epoch = math.ceil((float(np.max(time_days)) - transit_time) / period - 0.5) + 1
    exposure_half_days = max(0.0, float(exposure_seconds)) / (2.0 * 86_400.0)
    intervals: list[tuple[float, float]] = []

    for epoch in range(lower_epoch, upper_epoch + 1):
        orbit_center = transit_time + (epoch + 0.5) * period
        outer_start = _observer_time_for_orbit_time(
            config, orbit_center - external_half
        )
        outer_end = _observer_time_for_orbit_time(
            config, orbit_center + external_half
        )
        if internal_half is None:
            center = _observer_time_for_orbit_time(config, orbit_center)
            ingress = (outer_start - exposure_half_days, center)
            egress = (center, outer_end + exposure_half_days)
        else:
            inner_start = _observer_time_for_orbit_time(
                config, orbit_center - internal_half
            )
            inner_end = _observer_time_for_orbit_time(
                config, orbit_center + internal_half
            )
            ingress = (outer_start - exposure_half_days, inner_start)
            egress = (inner_end, outer_end + exposure_half_days)
        if outer_end + exposure_half_days < float(np.min(time_days)):
            continue
        if outer_start - exposure_half_days > float(np.max(time_days)):
            continue
        intervals.extend((ingress, egress))
    return tuple(intervals)


def build_cv_folds(
    config: MappingConfig,
    curve: LightCurve,
) -> tuple[np.ndarray, ...]:
    """Build deterministic fold indices for one validated light curve.

    ``eclipse_blocks`` combines the same time-ordered block from every
    ingress and egress into one fold. Points outside these transitions remain
    in every training set. ``contiguous`` scores all observations in K time
    blocks.
    """

    _supported_config(config)
    count = int(config.cross_validation.n_folds)
    time_days = _time_days(curve)
    if count > curve.n_observations:
        raise ValueError("cross_validation.n_folds cannot exceed the observation count.")

    if config.cross_validation.strategy == "contiguous":
        sorted_indices = np.argsort(time_days, kind="stable")
        folds = tuple(
            block.astype(np.int64, copy=False)
            for block in np.array_split(sorted_indices, count)
        )
    else:
        intervals = _eclipse_transition_intervals(
            config,
            time_days,
            exposure_seconds=float(curve.exposure_seconds),
        )
        fold_parts: list[list[np.ndarray]] = [[] for _ in range(count)]
        assigned = np.zeros(curve.n_observations, dtype=bool)
        for interval_index, (start, stop) in enumerate(intervals):
            # A grazing event has a shared midpoint. Assign that point to the
            # egress interval only, so every row is scored at most once.
            system = config.system
            impact = float(system.a_over_rstar) * abs(
                math.cos(math.radians(float(system.inclination_degrees)))
            )
            grazing = impact >= abs(1.0 - float(system.radius_ratio))
            is_ingress = interval_index % 2 == 0
            is_grazing = grazing and not is_ingress
            if grazing and is_ingress:
                mask = (time_days >= start) & (time_days < stop)
            elif is_grazing:
                mask = (time_days >= start) & (time_days <= stop)
            else:
                mask = (time_days >= start) & (time_days <= stop)
            indices = np.flatnonzero(mask & ~assigned)
            indices = indices[np.argsort(time_days[indices], kind="stable")]
            if indices.size == 0:
                continue
            assigned[indices] = True
            for fold_index, block in enumerate(np.array_split(indices, count)):
                if block.size:
                    fold_parts[fold_index].append(block.astype(np.int64, copy=False))
        folds = tuple(
            np.sort(np.concatenate(parts)).astype(np.int64, copy=False)
            if parts
            else np.empty(0, dtype=np.int64)
            for parts in fold_parts
        )
    empty = [index for index, fold in enumerate(folds) if fold.size == 0]
    if empty:
        raise ValueError(
            "cross-validation produced empty held-out fold(s): "
            + ", ".join(str(index) for index in empty)
            + ". Increase the cadence or lower cross_validation.n_folds."
        )
    flattened = np.concatenate(folds)
    if np.unique(flattened).size != flattened.size:
        raise RuntimeError("cross-validation fold construction assigned rows more than once.")
    return folds


def _make_plan(config: MappingConfig) -> ComputePlan:
    return resolve_plan(
        max_cpus=int(config.compute.max_cpus),
        threads=int(config.compute.threads),
        workers=int(config.compute.workers),
    )


def validate_cross_validation(config: MappingConfig) -> dict[str, Any]:
    """Load the data and return the folds and CPU plan without fitting."""

    sampler = _supported_config(config)
    curve = load_light_curve(config)
    folds = build_cv_folds(config, curve)
    assigned = np.zeros(curve.n_observations, dtype=bool)
    for fold in folds:
        assigned[fold] = True
    plan = _make_plan(config)
    active_workers = min(plan.workers, len(folds))
    return {
        "n_observations": curve.n_observations,
        "n_folds": len(folds),
        "held_out_counts": [int(fold.size) for fold in folds],
        "evaluated_observations": int(np.sum(assigned)),
        "always_in_training_observations": int(curve.n_observations - np.sum(assigned)),
        "strategy": config.cross_validation.strategy,
        "map_model": (
            f"{config.map.representation}_analytic_gaussian"
            if sampler in {"map", "none"}
            else f"{config.map.representation}_nuts_map"
        ),
        "null_model": config.cross_validation.null_model,
        "method": (
            "analytic_gaussian_marginal"
            if sampler in {"map", "none"}
            else "nuts_posterior_draw_logmeanexp"
        ),
        "supported_limits": [
            "corrected data",
            "independent Gaussian errors with fixed reported uncertainties",
            "fixed circular-orbit geometry and fixed limb darkening",
            "signed Gaussian coefficient priors for analytic map fits",
        ],
        "execution_plan": {
            **plan.to_dict(),
            "active_workers": active_workers,
            "fold_threads": plan.threads,
        },
    }


def _linear_predictive_log_density(
    posterior: LinearGaussianPosterior,
    design: np.ndarray,
    observed: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Integrate a linear Gaussian posterior and observation noise exactly."""

    mean = design @ posterior.mean
    parameter_variance = np.einsum(
        "ij,jk,ik->i", design, posterior.covariance, design
    )
    predictive_sigma = np.sqrt(np.square(sigma) + np.maximum(parameter_variance, 0.0))
    standardized = (observed - mean) / predictive_sigma
    return -0.5 * np.square(standardized) - np.log(predictive_sigma) - 0.5 * np.log(2.0 * np.pi)


def _unocculted_uniform_visibility(config: MappingConfig) -> float:
    """Return the unocculted uniform harmonic column in this basis."""

    system = config.system
    quadrature = disk_quadrature(
        n_radial=int(config.compute.quadrature_radial),
        n_azimuth=int(config.compute.quadrature_azimuth),
    )
    row = np.asarray(
        secondary_eclipse_design_matrix(
            np.asarray([float(system.transit_time)]),
            float(system.period_days),
            float(system.a_over_rstar),
            float(system.inclination_degrees),
            float(system.radius_ratio),
            0,
            float(system.transit_time),
            theta0=np.pi,
            subobserver_lat=np.deg2rad(90.0 - float(system.inclination_degrees)),
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
    reference = float(row[0, 0])
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("could not normalize the uniform-disk null model.")
    return reference


def _uniform_visibility_at_samples(
    config: MappingConfig,
    time_days: np.ndarray,
) -> np.ndarray:
    """Return normalized uniform-map visibility at each requested time."""

    system = config.system
    quadrature = disk_quadrature(
        n_radial=int(config.compute.quadrature_radial),
        n_azimuth=int(config.compute.quadrature_azimuth),
    )
    reference = _unocculted_uniform_visibility(config)
    result = np.empty(time_days.size, dtype=float)
    for start in range(0, time_days.size, 256):
        stop = min(start + 256, time_days.size)
        result[start:stop] = np.asarray(
            secondary_eclipse_design_matrix(
                time_days[start:stop],
                float(system.period_days),
                float(system.a_over_rstar),
                float(system.inclination_degrees),
                float(system.radius_ratio),
                0,
                float(system.transit_time),
                theta0=np.pi,
                subobserver_lat=np.deg2rad(
                    90.0 - float(system.inclination_degrees)
                ),
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
        )[:, 0] / reference
    tolerance = 1.0e-8
    if np.any(result < -tolerance) or np.any(result > 1.0 + tolerance):
        raise ValueError(
            "uniform-map visibility is outside [0, 1]; check the fixed geometry and quadrature settings."
        )
    return np.clip(result, 0.0, 1.0)


def _null_design(
    config: MappingConfig,
    curve: LightCurve,
    bundle: _DesignBundle,
) -> np.ndarray:
    time_days = np.asarray(bundle.time_days, dtype=float)
    integrate = bool(config.model.integrate_exposure) and curve.exposure_seconds > 0.0
    if integrate:
        subsamples = 4
        exposure_days = float(curve.exposure_seconds) / 86_400.0
        offsets = (
            (np.arange(subsamples, dtype=float) + 0.5) / subsamples - 0.5
        ) * exposure_days
        sampled_time = time_days[:, None] + offsets[None, :]
    else:
        subsamples = 1
        sampled_time = time_days[:, None]
    sampled_visibility = _uniform_visibility_at_samples(
        config, sampled_time.reshape(-1)
    ).reshape(time_days.size, subsamples)
    if config.cross_validation.null_model == "uniform":
        return np.mean(sampled_visibility, axis=1)[:, None]
    sampled_design = fourier_design_matrix(
        sampled_time.reshape(-1),
        period=float(config.system.period_days),
        t0=float(config.system.transit_time),
        visibility=sampled_visibility.reshape(-1),
        degree=int(config.model.fourier_degree),
        include_intercept=False,
    )
    return np.mean(
        sampled_design.reshape(time_days.size, subsamples, -1), axis=1
    )


def _fold_seed(master_seed: int, fold_index: int, model_index: int) -> int:
    seed = np.random.SeedSequence(
        [int(master_seed), int(fold_index), int(model_index)]
    ).generate_state(1, dtype=np.uint32)
    return int(seed[0])


def _subset_curve(curve: LightCurve, indices: np.ndarray) -> LightCurve:
    regressors = None if curve.regressors is None else curve.regressors[indices]
    segments = None if curve.segments is None else curve.segments[indices]
    return LightCurve(
        time=curve.time[indices],
        flux=curve.flux[indices],
        flux_err=curve.flux_err[indices],
        source=curve.source,
        time_unit=curve.time_unit,
        exposure_seconds=curve.exposure_seconds,
        regressors=regressors,
        regressor_names=curve.regressor_names,
        segments=segments,
    )


def _subset_bundle(bundle: _DesignBundle, indices: np.ndarray) -> _DesignBundle:
    return replace(
        bundle,
        time_days=bundle.time_days[indices],
        map_design=bundle.map_design[indices],
        stellar_flux=bundle.stellar_flux[indices],
        systematics_design=bundle.systematics_design[indices],
    )


def _diagnostics(run, details: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_samples": int(
            next(iter(run.samples.values())).shape[0]
            if run.samples
            else 0
        ),
        "chain_method": str(run.chain_method or "unknown"),
        "divergences": int(
            np.sum(run.extra_fields.get("diverging", np.asarray([], dtype=bool)))
        ),
    }
    for name in ("maximum_rhat", "minimum_effective_sample_size"):
        if details is not None:
            result[name] = details.get(name)
    return result


def _json_safe(value: Any) -> Any:
    """Convert nested report values to strict JSON types."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _fit_nuts_map(
    config: MappingConfig,
    curve: LightCurve,
    bundle: _DesignBundle,
    train: np.ndarray,
    held_out: np.ndarray,
    plan: ComputePlan,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    map_config = config.map
    if (
        config.map.representation == "direct_harmonics"
        and config.map.active_harmonic_indices
    ):
        active_indices = tuple(config.map.active_harmonic_indices)
        if 0 in active_indices:
            active_indices = (0,) + tuple(
                index for index in active_indices if index != 0
            )
            map_config = replace(
                config.map, active_harmonic_indices=active_indices
            )
    fit_config = replace(
        config,
        project=replace(config.project, seed=seed),
        map=map_config,
    )
    fit_curve = _subset_curve(curve, train)
    fit_bundle = _subset_bundle(bundle, train)
    apply_environment(
        plan,
        jax_platform=str(config.compute.jax_platform),
        x64=bool(config.compute.x64),
        jit=bool(config.compute.jit),
        host_devices=1,
    )
    if config.map.representation in {"direct_harmonics", "longitude"}:
        run, harmonic_samples, details = _harmonic_nuts_fit(
            fit_config, fit_curve, fit_bundle, plan=plan
        )
    else:
        run, _pixel_samples, harmonic_samples, details = _pixel_fit(
            fit_config, fit_curve, fit_bundle, plan=plan
        )
    predictions = (
        bundle.stellar_flux[held_out][None, :]
        + np.asarray(harmonic_samples, dtype=float)
        @ np.asarray(bundle.map_design[held_out], dtype=float).T
    )
    log_density = gaussian_pointwise_elpd(
        curve.flux[held_out], predictions, curve.flux_err[held_out]
    )
    return log_density, _diagnostics(run, details)


def _fit_nuts_null(
    config: MappingConfig,
    curve: LightCurve,
    bundle: _DesignBundle,
    design: np.ndarray,
    train: np.ndarray,
    held_out: np.ndarray,
    plan: ComputePlan,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    coefficients = design.shape[1]
    prior_mean = np.zeros(coefficients, dtype=float)
    prior_mean[0] = float(config.system.planet_flux_ratio)
    run = sample_harmonic_map(
        design[train],
        curve.flux[train],
        curve.flux_err[train],
        stellar_flux=bundle.stellar_flux[train],
        coefficient_prior_mean=prior_mean,
        coefficient_prior_sigma=float(config.cross_validation.prior_scale),
        warmup=int(config.inference.warmup),
        draws=int(config.inference.draws),
        chains=int(config.inference.chains),
        seed=seed,
        target_accept=float(config.inference.target_accept),
        progress_bar=False,
        dense_mass=bool(config.inference.dense_mass),
        init_strategy=str(config.inference.init_strategy),
        compute_plan=plan,
        jax_platform=str(config.compute.jax_platform),
        x64=bool(config.compute.x64),
        jit=bool(config.compute.jit),
        host_devices=1,
    )
    samples = run.samples.get("coefficients")
    if samples is None:
        samples = run.samples.get("harmonic_coefficients")
    if samples is None:
        raise ValueError("the Fourier/uniform null sampler returned no coefficient samples.")
    predictions = (
        bundle.stellar_flux[held_out][None, :]
        + np.asarray(samples, dtype=float)
        @ np.asarray(design[held_out], dtype=float).T
    )
    log_density = gaussian_pointwise_elpd(
        curve.flux[held_out], predictions, curve.flux_err[held_out]
    )
    return log_density, _diagnostics(run)


def _score_fold(job: dict[str, Any]) -> dict[str, Any]:
    """Fit both models on one training fold and score held-out observations."""

    config: MappingConfig = job["config"]
    curve: LightCurve = job["curve"]
    bundle: _DesignBundle = job["bundle"]
    map_design = np.asarray(bundle.map_design, dtype=float)
    null_design = np.asarray(job["null_design"], dtype=float)
    held_out = np.asarray(job["held_out"], dtype=np.int64)
    train_mask = np.ones(curve.n_observations, dtype=bool)
    train_mask[held_out] = False
    train = np.flatnonzero(train_mask)
    if train.size == 0:
        raise ValueError("each cross-validation fold must leave training observations.")
    plan: ComputePlan = job["worker_plan"]
    sampler = str(config.inference.sampler).lower()
    map_seed = int(job["map_seed"])
    null_seed = int(job["null_seed"])
    diagnostics: dict[str, Any] = {}

    if sampler in {"map", "none"}:
        if (
            config.map.representation == "direct_harmonics"
            and config.map.active_harmonic_indices
        ):
            active_indices = np.asarray(config.map.active_harmonic_indices, dtype=int)
            map_design = map_design[:, active_indices]
        else:
            active_indices = np.arange(map_design.shape[1], dtype=int)
        adjusted = np.asarray(curve.flux, dtype=float) - bundle.stellar_flux + 1.0
        map_matrix = np.column_stack((np.ones(curve.n_observations), map_design))
        null_matrix = np.column_stack((np.ones(curve.n_observations), null_design))
        map_prior_mean = np.concatenate(([1.0], np.zeros(map_design.shape[1])))
        null_prior_mean = np.concatenate(([1.0], np.zeros(null_design.shape[1])))
        map_prior_mean[1 + int(np.flatnonzero(active_indices == 0)[0])] = float(
            config.system.planet_flux_ratio
        )
        null_prior_mean[1] = float(config.system.planet_flux_ratio)
        map_prior_scale = np.concatenate(
            ([0.05], np.full(map_design.shape[1], config.cross_validation.prior_scale))
        )
        null_prior_scale = np.concatenate(
            ([0.05], np.full(null_design.shape[1], config.cross_validation.prior_scale))
        )
        map_posterior = fit_linear_gaussian(
            map_matrix[train],
            adjusted[train],
            curve.flux_err[train],
            prior_mean=map_prior_mean,
            prior_scale=map_prior_scale,
        )
        null_posterior = fit_linear_gaussian(
            null_matrix[train],
            adjusted[train],
            curve.flux_err[train],
            prior_mean=null_prior_mean,
            prior_scale=null_prior_scale,
        )
        adjusted_test = adjusted[held_out]
        map_log_density = _linear_predictive_log_density(
            map_posterior,
            map_matrix[held_out],
            adjusted_test,
            curve.flux_err[held_out],
        )
        null_log_density = _linear_predictive_log_density(
            null_posterior,
            null_matrix[held_out],
            adjusted_test,
            curve.flux_err[held_out],
        )
        diagnostics["map"] = {"method": "analytic_gaussian_marginal"}
        diagnostics["null"] = {"method": "analytic_gaussian_marginal"}
    else:
        map_log_density, map_details = _fit_nuts_map(
            config,
            curve,
            bundle,
            train,
            held_out,
            plan,
            map_seed,
        )
        null_log_density, null_details = _fit_nuts_null(
            config,
            curve,
            bundle,
            null_design,
            train,
            held_out,
            plan,
            null_seed,
        )
        diagnostics["map"] = map_details
        diagnostics["null"] = null_details

    return {
        "fold_index": int(job["fold_index"]),
        "held_out": held_out,
        "map_log_predictive_density": map_log_density,
        "null_log_predictive_density": null_log_density,
        "map_seed": map_seed,
        "null_seed": null_seed,
        "process_id": os.getpid(),
        "diagnostics": diagnostics,
    }


def _run_settings_and_priors(
    config: MappingConfig,
    bundle: _DesignBundle,
    null_design: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Describe the model settings and priors used in every CV output."""

    sampler = str(config.inference.sampler).lower()
    representation = str(config.map.representation).lower()
    is_analytic = sampler in {"map", "none"}
    settings: dict[str, Any] = {
        "sampler": sampler,
        "map_representation": representation,
        "harmonic_degree": int(config.map.harmonic_degree),
        "active_harmonic_indices": list(config.map.active_harmonic_indices),
        "positive_map_requested": bool(config.map.positive),
        "null_model": config.cross_validation.null_model,
        "fourier_degree": int(config.model.fourier_degree),
        "fold_strategy": config.cross_validation.strategy,
        "n_folds": int(config.cross_validation.n_folds),
        "integrate_exposure": bool(config.model.integrate_exposure),
        "exposure_seconds": float(config.data.exposure_seconds),
        "likelihood": str(config.model.likelihood),
        "noise_model": str(config.model.noise_model),
        "fixed_geometry": {
            "period_days": float(config.system.period_days),
            "transit_time": float(config.system.transit_time),
            "a_over_rstar": float(config.system.a_over_rstar),
            "radius_ratio": float(config.system.radius_ratio),
            "inclination_degrees": float(config.system.inclination_degrees),
            "eccentricity": float(config.system.eccentricity),
            "include_light_delay": bool(config.model.include_light_delay),
        },
        "master_seed": int(config.project.seed),
        "seed_schedule": (
            "SeedSequence([master_seed, fold_index, model_index]); model_index 0 is map and 1 is null"
        ),
    }
    if sampler == "nuts":
        settings["inference"] = {
            "warmup": int(config.inference.warmup),
            "draws": int(config.inference.draws),
            "chains": int(config.inference.chains),
            "target_accept": float(config.inference.target_accept),
            "dense_mass": bool(config.inference.dense_mass),
            "init_strategy": str(config.inference.init_strategy),
            "progress_bar": bool(config.inference.progress_bar),
            "jit": bool(config.compute.jit),
            "x64": bool(config.compute.x64),
        }
    if representation in {"harmonics", "pixels"}:
        settings["positive_pixel_prior"] = {
            "n_pixels": int(config.map.n_pixels),
            "pixel_log_sigma": float(config.map.pixel_log_sigma),
            "pixel_prior_mean_ppm": config.map.pixel_prior_mean_ppm,
            "pixel_prior_sd_ppm": config.map.pixel_prior_sd_ppm,
            "entropy_penalty": float(config.map.entropy_penalty),
        }

    planet_flux = float(config.system.planet_flux_ratio)
    if is_analytic:
        if (
            representation == "direct_harmonics"
            and config.map.active_harmonic_indices
        ):
            active_indices = list(config.map.active_harmonic_indices)
        else:
            active_indices = list(range(bundle.map_design.shape[1]))
        map_mean = [0.0] * len(active_indices)
        map_mean[active_indices.index(0)] = planet_flux
        null_mean = [0.0] * null_design.shape[1]
        null_mean[0] = planet_flux
        priors = {
            "baseline": {
                "distribution": "normal",
                "mean": 1.0,
                "standard_deviation": 0.05,
            },
            "map_coefficients": {
                "distribution": "independent_normal",
                "basis": representation,
                "mean": map_mean,
                "standard_deviation": float(config.cross_validation.prior_scale),
                "active_harmonic_indices": active_indices,
            },
            "null_coefficients": {
                "distribution": "independent_normal",
                "basis": config.cross_validation.null_model,
                "mean": null_mean,
                "standard_deviation": float(config.cross_validation.prior_scale),
            },
        }
    else:
        if representation in {"direct_harmonics", "longitude"}:
            if representation == "longitude":
                active_indices = list(range(bundle.map_design.shape[1]))
            elif config.map.active_harmonic_indices:
                active_indices = list(config.map.active_harmonic_indices)
                if 0 in active_indices:
                    active_indices = [0] + [
                        index for index in active_indices if index != 0
                    ]
            else:
                active_indices = list(range(bundle.map_design.shape[1]))
            planet_scale = max(planet_flux, 1.0e-8)
            map_mean = [planet_flux if index == 0 else 0.0 for index in active_indices]
            map_sigma = [
                0.5 * planet_scale
                if index == 0
                else float(config.map.pixel_log_sigma) * planet_scale
                for index in active_indices
            ]
            map_prior: dict[str, Any] = {
                "distribution": "independent_normal",
                "basis": representation,
                "mean": map_mean,
                "standard_deviation": map_sigma,
                "active_harmonic_indices": active_indices,
            }
        else:
            if config.map.pixel_prior_mean_ppm is None:
                pixel_median = max(planet_flux / np.pi, 1.0e-8)
                pixel_log_sigma = float(config.map.pixel_log_sigma)
                source = "planet_flux_ratio and map.pixel_log_sigma"
            else:
                arithmetic_mean = float(config.map.pixel_prior_mean_ppm) * 1.0e-6 / np.pi
                arithmetic_sd = float(config.map.pixel_prior_sd_ppm) * 1.0e-6 / np.pi
                ratio = arithmetic_sd / arithmetic_mean
                pixel_log_sigma = float(np.sqrt(np.log1p(ratio**2)))
                pixel_median = float(
                    arithmetic_mean / np.sqrt(1.0 + ratio**2)
                )
                source = "map.pixel_prior_mean_ppm and map.pixel_prior_sd_ppm"
            pixel_count = (
                bundle.map_design.shape[1]
                if representation == "harmonics"
                else int(config.map.n_pixels)
            )
            map_prior = {
                "distribution": "independent_lognormal_positive_intensities",
                "basis": (
                    "rank_revealing_harmonic_anchors"
                    if representation == "harmonics"
                    else "surface_pixels"
                ),
                "intensity_count": int(pixel_count),
                "median_internal_intensity": float(pixel_median),
                "log_standard_deviation": float(pixel_log_sigma),
                "source": source,
                "entropy_penalty": float(config.map.entropy_penalty),
            }
        null_mean = [0.0] * null_design.shape[1]
        null_mean[0] = planet_flux
        priors = {
            "baseline": {
                "type": "fixed",
                "description": "stellar transit design with no fitted baseline coefficient",
            },
            "map_coefficients": map_prior,
            "null_coefficients": {
                "distribution": "independent_normal",
                "basis": config.cross_validation.null_model,
                "mean": null_mean,
                "standard_deviation": float(config.cross_validation.prior_scale),
            },
        }
    return settings, priors


def _output_path(config: MappingConfig) -> Path:
    output = Path(config.output.directory).expanduser()
    if not output.is_absolute():
        output = (config.base_dir / output).resolve()
    return output


def run_cross_validation(config: MappingConfig) -> CrossValidationReport:
    """Run deterministic K-fold predictive scoring on a supplied light curve."""

    validation_plan = validate_cross_validation(config)
    plan = _make_plan(config)
    output = _output_path(config)
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not config.output.overwrite)):
        raise ValueError(
            f"Output directory is not empty: {output}. Set output.overwrite: true or choose another directory."
        )

    apply_environment(
        plan,
        jax_platform=str(config.compute.jax_platform),
        x64=bool(config.compute.x64),
        jit=bool(config.compute.jit),
        host_devices=1,
    )
    curve = load_light_curve(config)
    bundle = _secondary_design(config, curve)
    folds = build_cv_folds(config, curve)
    null_design = _null_design(config, curve, bundle)
    fold_threads = int(plan.threads)
    worker_plan = resolve_plan(
        max_cpus=fold_threads,
        threads=fold_threads,
        workers=1,
    )
    jobs = [
        {
            "config": config,
            "curve": curve,
            "bundle": bundle,
            "null_design": null_design,
            "fold_index": fold_index,
            "held_out": held_out,
            "worker_plan": worker_plan,
            "map_seed": _fold_seed(config.project.seed, fold_index, 0),
            "null_seed": _fold_seed(config.project.seed, fold_index, 1),
        }
        for fold_index, held_out in enumerate(folds)
    ]
    results = ordered_map(_score_fold, jobs, plan)

    map_pointwise = np.full(curve.n_observations, np.nan)
    null_pointwise = np.full(curve.n_observations, np.nan)
    fold_membership = np.full(curve.n_observations, -1, dtype=np.int32)
    fold_summaries: list[dict[str, Any]] = []
    process_ids = sorted({int(result["process_id"]) for result in results})
    for result in results:
        fold_index = int(result["fold_index"])
        held_out = result["held_out"]
        map_scores = np.asarray(result["map_log_predictive_density"], dtype=float)
        null_scores = np.asarray(result["null_log_predictive_density"], dtype=float)
        if map_scores.shape != held_out.shape or null_scores.shape != held_out.shape:
            raise ValueError("a fold returned an invalid pointwise score shape.")
        if not np.all(np.isfinite(map_scores)) or not np.all(np.isfinite(null_scores)):
            raise ValueError("a fold returned a non-finite predictive score.")
        if np.any(fold_membership[held_out] != -1):
            raise RuntimeError("cross-validation fold results overlap.")
        fold_membership[held_out] = fold_index
        map_pointwise[held_out] = map_scores
        null_pointwise[held_out] = null_scores
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "held_out_observations": int(held_out.size),
                "training_observations": int(curve.n_observations - held_out.size),
                "map_elpd": float(np.sum(map_scores)),
                "null_elpd": float(np.sum(null_scores)),
                "delta_elpd": float(np.sum(map_scores - null_scores)),
                "map_seed": int(result["map_seed"]),
                "null_seed": int(result["null_seed"]),
                "process_id": int(result["process_id"]),
                "diagnostics": result["diagnostics"],
            }
        )

    evaluated = fold_membership >= 0
    if not np.any(evaluated):
        raise ValueError("cross-validation did not assign any held-out observations.")
    comparison: CVComparison = compare_pointwise_elpd(
        map_pointwise[evaluated], null_pointwise[evaluated]
    )
    settings, priors = _run_settings_and_priors(config, bundle, null_design)
    execution_plan = {
        **plan.to_dict(),
        "active_workers": len(process_ids),
        "parent_process_id": os.getpid(),
        "worker_process_ids": process_ids,
        "fold_threads": worker_plan.threads,
    }
    output.mkdir(parents=True, exist_ok=True)
    if config.output.save_resolved_config:
        write_resolved_config(config, output / "cross_validation_config.yml")
    pointwise_path = output / "cross_validation_pointwise.npz"
    np.savez_compressed(
        pointwise_path,
        time_days=np.asarray(bundle.time_days, dtype=float),
        flux=np.asarray(curve.flux, dtype=float),
        flux_err=np.asarray(curve.flux_err, dtype=float),
        fold_membership=fold_membership,
        map_log_predictive_density=map_pointwise,
        null_log_predictive_density=null_pointwise,
        pointwise_delta=(map_pointwise - null_pointwise),
    )
    summary = {
        "status": "complete",
        "project": config.project.name,
        "seed": int(config.project.seed),
        "strategy": config.cross_validation.strategy,
        "n_observations": int(curve.n_observations),
        "n_folds": int(len(folds)),
        "evaluated_observations": int(np.sum(evaluated)),
        "always_in_training_observations": int(curve.n_observations - np.sum(evaluated)),
        "map_model": validation_plan["map_model"],
        "null_model": config.cross_validation.null_model,
        "method": validation_plan["method"],
        "map_elpd": float(np.sum(map_pointwise[evaluated])),
        "null_elpd": float(np.sum(null_pointwise[evaluated])),
        "delta_elpd": comparison.delta_elpd,
        "standard_error": comparison.standard_error,
        "z_score": comparison.z_score,
        "folds": fold_summaries,
        "execution_plan": execution_plan,
        "settings": settings,
        "priors": priors,
        "limits": validation_plan["supported_limits"],
        "files": {
            "pointwise": pointwise_path.name,
            "fold_membership_note": (
                "-1 means the observation was never held out; eclipse_blocks keeps those rows in every training set."
            ),
        },
    }
    summary_path = output / "cross_validation.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return CrossValidationReport(
        status="complete",
        output_directory=output,
        summary_path=summary_path,
        pointwise_path=pointwise_path,
        n_observations=int(curve.n_observations),
        n_folds=len(folds),
        evaluated_observations=int(np.sum(evaluated)),
        map_model=validation_plan["map_model"],
        null_model=config.cross_validation.null_model,
        method=validation_plan["method"],
        map_elpd=float(np.sum(map_pointwise[evaluated])),
        null_elpd=float(np.sum(null_pointwise[evaluated])),
        delta_elpd=comparison.delta_elpd,
        standard_error=comparison.standard_error,
        z_score=(comparison.z_score if np.isfinite(comparison.z_score) else None),
        execution_plan=execution_plan,
        settings=settings,
        priors=priors,
        folds=tuple(fold_summaries),
    )


__all__ = [
    "CrossValidationReport",
    "build_cv_folds",
    "run_cross_validation",
    "validate_cross_validation",
]
