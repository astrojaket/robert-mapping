#!/usr/bin/env python3
"""Run a bounded, reproducible UNITY eclipse-mapping injection study.

The runner makes continuous, symmetric visits from the cadence and window in
each target record. It integrates the robert-mapping eclipse operator over
each exposure, injects a fixed-fraction degree-2 map, and profiles a declared
longitude/latitude/width/timing grid with free map amplitudes. The temperature
relation is an illustrative scenario rule. It is not a prediction extracted
from a GCM.

Inputs use a JSON object with a ``targets`` list. Each target needs
``name``, ``period_days``, ``a_over_rstar``, ``radius_ratio``,
``inclination_degrees``, ``teq_k``, ``depth_ppm``, ``noise_ppm``,
``cadence_seconds``, ``window_hours``, and ``instrument``. Optional top-level
keys include ``noise_levels_ppm``, ``seed``, ``sources``, and ``assumptions``.
An optional ``analysis`` object can set the grid and map assumptions.
An optional top-level ``scenarios`` list selects a subset of the named
injection scenarios. The default is all scenarios.
An optional target-level ``seed_target_index`` keeps trial seeds stable when
one target is run on its own.

Example::

    python UNITY_Theory/code/unity_injection_recovery.py \
      --inputs UNITY_Theory/data/inputs/unity_inputs.json \
      --output UNITY_Theory/generated/injection_recovery \
      --trials 30 --quick
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "UNITY_Theory"
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from robert_mapping.benchmark.recovery_cases import (  # noqa: E402
    _hotspot_coefficients,
    _uniform_coefficients,
)
from robert_mapping.physics import (  # noqa: E402
    disk_quadrature,
    evaluate_map,
    map_design_matrix,
    secondary_eclipse_design_matrix,
)
from robert_mapping.recovery import (  # noqa: E402
    compare_uniform_flexible_bic,
    fit_profile,
)


FloatArray = NDArray[np.float64]
SCENARIOS = ("temperature_scaled", "zero_offset", "constant_30_deg", "uniform_null")
FIT_SCOPES = ("full_window", "eclipse_only")
DETECTION_DELTA_BIC = -6.0
SHOWMAN_SOURCE = "https://doi.org/10.1007/s11214-020-00758-8"
SEED_PROVENANCE = (
    "Each trial seed uses NumPy SeedSequence([master_seed, seed_target_index, 0, trial]). "
    "The noise-index component stays at 0, so each target/trial uses paired standardized "
    "Gaussian draws across noise levels; scenarios reuse that draw. seed_target_index "
    "comes from the target record or the target's position in the input list."
)
TRIAL_FIELDS = (
    "run_id",
    "target",
    "instrument",
    "teq_k",
    "brightness_temperature_k",
    "noise_ppm",
    "noise_input_ppm",
    "noise_reference_seconds",
    "scenario",
    "trial",
    "fit_scope",
    "time_window_hours",
    "n_observations",
    "n_eclipse_only_observations",
    "injected_longitude_degrees",
    "injected_latitude_degrees",
    "injected_width_degrees",
    "injection_fraction",
    "injected_timing_seconds",
    "truth_map_minimum_intensity",
    "truth_map_peak_longitude_degrees",
    "truth_map_peak_latitude_degrees",
    "truth_map_peak_shift_degrees",
    "recovered_longitude_q16_degrees",
    "recovered_longitude_median_degrees",
    "recovered_longitude_q84_degrees",
    "longitude_bias_degrees",
    "longitude_interval_contains_truth",
    "longitude_interval_resolution_limited",
    "recovered_latitude_q16_degrees",
    "recovered_latitude_median_degrees",
    "recovered_latitude_q84_degrees",
    "recovered_width_q16_degrees",
    "recovered_width_median_degrees",
    "recovered_width_q84_degrees",
    "recovered_timing_q16_seconds",
    "recovered_timing_median_seconds",
    "recovered_timing_q84_seconds",
    "best_width_degrees",
    "best_latitude_degrees",
    "best_timing_seconds",
    "best_map_amplitude_scale",
    "best_map_amplitude_ppm",
    "best_uniform_map_amplitude_scale",
    "best_spot_amplitude_scale",
    "best_recovered_spot_fraction",
    "uniform_bic",
    "flexible_bic",
    "delta_bic",
    "delta_bic_detection_threshold",
    "map_detected",
    "best_map_minimum_intensity",
    "best_map_peak_longitude_degrees",
    "best_map_peak_latitude_degrees",
    "best_map_peak_shift_degrees",
    "best_map_positive",
    "residual_rms_ppm",
    "random_seed",
)


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    return result


def _number_list(value: Any, name: str, *, minimum: float | None = None) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [_finite_number(item, f"{name}[{index}]", minimum=minimum) for index, item in enumerate(value)]


def _load_inputs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The inputs JSON must contain an object")
    targets = raw.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("The inputs JSON needs a non-empty targets list")

    scenario_selection = raw.get("scenarios", list(SCENARIOS))
    if not isinstance(scenario_selection, list) or not scenario_selection:
        raise ValueError("scenarios must be a non-empty list when it is present")
    if any(not isinstance(item, str) for item in scenario_selection):
        raise ValueError("Each scenarios item must be a scenario name")
    if len(set(scenario_selection)) != len(scenario_selection):
        raise ValueError("scenarios must not contain duplicate names")
    unknown_scenarios = [item for item in scenario_selection if item not in SCENARIOS]
    if unknown_scenarios:
        raise ValueError(
            "Unknown scenario name(s): " + ", ".join(unknown_scenarios)
        )

    analysis_raw = raw.get("analysis", {})
    if not isinstance(analysis_raw, dict):
        raise ValueError("analysis must be an object when it is present")
    analysis = {
        "scenarios": list(scenario_selection),
        "harmonic_degree": int(analysis_raw.get("harmonic_degree", 2)),
        "injected_width_degrees": _finite_number(
            analysis_raw.get("injected_width_degrees", 40.0),
            "analysis.injected_width_degrees",
            minimum=0.1,
        ),
        "injection_fraction": _finite_number(
            analysis_raw.get("injection_fraction", 0.8),
            "analysis.injection_fraction",
            minimum=0.0,
        ),
        "injected_latitude_degrees": _finite_number(
            analysis_raw.get("injected_latitude_degrees", 0.0),
            "analysis.injected_latitude_degrees",
        ),
        "injected_timing_seconds": _finite_number(
            analysis_raw.get("injected_timing_seconds", 0.0),
            "analysis.injected_timing_seconds",
        ),
        "constant_offset_degrees": _finite_number(
            analysis_raw.get("constant_offset_degrees", 30.0),
            "analysis.constant_offset_degrees",
        ),
        "longitude_grid_min_degrees": _finite_number(
            analysis_raw.get("longitude_grid_min_degrees", -90.0),
            "analysis.longitude_grid_min_degrees",
        ),
        "longitude_grid_max_degrees": _finite_number(
            analysis_raw.get("longitude_grid_max_degrees", 90.0),
            "analysis.longitude_grid_max_degrees",
        ),
        "longitude_grid_step_degrees": _finite_number(
            analysis_raw.get("longitude_grid_step_degrees", 1.0),
            "analysis.longitude_grid_step_degrees",
            minimum=0.1,
        ),
        "width_grid_degrees": _number_list(
            analysis_raw.get("width_grid_degrees", [30.0, 40.0, 50.0]),
            "analysis.width_grid_degrees",
            minimum=0.1,
        ),
        "latitude_grid_degrees": _number_list(
            analysis_raw.get("latitude_grid_degrees", [-30.0, 0.0, 30.0]),
            "analysis.latitude_grid_degrees",
        ),
        "timing_grid_seconds": _number_list(
            analysis_raw.get("timing_grid_seconds", [-30.0, 0.0, 30.0]),
            "analysis.timing_grid_seconds",
        ),
        "baseline_order": int(analysis_raw.get("baseline_order", 1)),
        "baseline_intercept_ppm": _finite_number(
            analysis_raw.get("baseline_intercept_ppm", 0.0),
            "analysis.baseline_intercept_ppm",
        ),
        "baseline_slope_ppm": _finite_number(
            analysis_raw.get("baseline_slope_ppm", 0.0),
            "analysis.baseline_slope_ppm",
        ),
        "baseline_curvature_ppm": _finite_number(
            analysis_raw.get("baseline_curvature_ppm", 0.0),
            "analysis.baseline_curvature_ppm",
        ),
        "noise_reference_seconds": _finite_number(
            analysis_raw.get("noise_reference_seconds", 60.0),
            "analysis.noise_reference_seconds",
            minimum=0.1,
        ),
        "noise_levels_ppm": raw.get("noise_levels_ppm", analysis_raw.get("noise_levels_ppm")),
        "eclipse_only_loss_threshold": _finite_number(
            analysis_raw.get("eclipse_only_loss_threshold", 1.0e-10),
            "analysis.eclipse_only_loss_threshold",
            minimum=0.0,
        ),
    }
    if analysis["noise_levels_ppm"] is not None:
        analysis["noise_levels_ppm"] = _number_list(
            analysis["noise_levels_ppm"], "noise_levels_ppm", minimum=0.0
        )
        if any(value <= 0.0 for value in analysis["noise_levels_ppm"]):
            raise ValueError("noise_levels_ppm values must be positive")
    if analysis["harmonic_degree"] != 2:
        raise ValueError("This injection study requires harmonic_degree = 2")
    if not 0.0 <= analysis["injection_fraction"] < 1.0:
        raise ValueError("analysis.injection_fraction must be in [0, 1)")
    if analysis["injection_fraction"] == 0.0:
        raise ValueError("analysis.injection_fraction must be greater than zero")
    if not -90.0 <= analysis["injected_latitude_degrees"] <= 90.0:
        raise ValueError("analysis.injected_latitude_degrees must be between -90 and 90")
    if not -90.0 <= min(analysis["latitude_grid_degrees"]) <= max(
        analysis["latitude_grid_degrees"]
    ) <= 90.0:
        raise ValueError("latitude_grid_degrees must be between -90 and 90")
    if analysis["baseline_order"] not in (0, 1, 2):
        raise ValueError("analysis.baseline_order must be 0, 1, or 2")
    if analysis["longitude_grid_max_degrees"] <= analysis["longitude_grid_min_degrees"]:
        raise ValueError("longitude_grid_max_degrees must exceed the minimum")

    n_lon = int(
        np.floor(
            (analysis["longitude_grid_max_degrees"] - analysis["longitude_grid_min_degrees"])
            / analysis["longitude_grid_step_degrees"]
            + 0.5
        )
    )
    longitudes = analysis["longitude_grid_min_degrees"] + analysis[
        "longitude_grid_step_degrees"
    ] * np.arange(n_lon + 1)
    if not np.isclose(longitudes[-1], analysis["longitude_grid_max_degrees"], atol=1.0e-8):
        raise ValueError("longitude grid maximum must lie on the configured grid step")
    analysis["longitude_grid_degrees"] = [float(value) for value in longitudes]
    if max(analysis["width_grid_degrees"]) >= 180.0:
        raise ValueError("width_grid_degrees must be less than 180 degrees")

    required = (
        "name",
        "period_days",
        "a_over_rstar",
        "radius_ratio",
        "inclination_degrees",
        "teq_k",
        "depth_ppm",
        "noise_ppm",
        "cadence_seconds",
        "window_hours",
        "instrument",
    )
    names: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"targets[{index}] must be an object")
        for key in required:
            if key not in target:
                raise ValueError(f"targets[{index}] is missing {key}")
        name = str(target["name"]).strip()
        if not name or name in names:
            raise ValueError(f"target name is blank or duplicated: {name!r}")
        names.add(name)
        for key in (
            "period_days",
            "a_over_rstar",
            "radius_ratio",
            "teq_k",
            "depth_ppm",
            "noise_ppm",
            "cadence_seconds",
            "window_hours",
        ):
            _finite_number(target[key], f"targets[{index}].{key}", minimum=0.0)
        inclination = _finite_number(
            target["inclination_degrees"], f"targets[{index}].inclination_degrees"
        )
        if not 0.0 < inclination <= 180.0:
            raise ValueError(f"targets[{index}].inclination_degrees must be in (0, 180]")
        if not str(target["instrument"]).strip():
            raise ValueError(f"targets[{index}].instrument must not be blank")
        if "seed_target_index" in target:
            seed_target_index = _finite_number(
                target["seed_target_index"],
                f"targets[{index}].seed_target_index",
                minimum=0.0,
            )
            if not float(seed_target_index).is_integer():
                raise ValueError(
                    f"targets[{index}].seed_target_index must be a non-negative integer"
                )
        if _finite_number(target["period_days"], f"targets[{index}].period_days") <= 0.0:
            raise ValueError(f"targets[{index}].period_days must be positive")
        if _finite_number(target["a_over_rstar"], f"targets[{index}].a_over_rstar") <= 0.0:
            raise ValueError(f"targets[{index}].a_over_rstar must be positive")
        if _finite_number(target["radius_ratio"], f"targets[{index}].radius_ratio") <= 0.0:
            raise ValueError(f"targets[{index}].radius_ratio must be positive")
        if _finite_number(target["teq_k"], f"targets[{index}].teq_k") <= 0.0:
            raise ValueError(f"targets[{index}].teq_k must be positive")
        if _finite_number(target["depth_ppm"], f"targets[{index}].depth_ppm") <= 0.0:
            raise ValueError(f"targets[{index}].depth_ppm must be positive")
        if _finite_number(target["noise_ppm"], f"targets[{index}].noise_ppm") <= 0.0:
            raise ValueError(f"targets[{index}].noise_ppm must be positive")
        if _finite_number(target["cadence_seconds"], f"targets[{index}].cadence_seconds") <= 0.0:
            raise ValueError(f"targets[{index}].cadence_seconds must be positive")
        if _finite_number(target["window_hours"], f"targets[{index}].window_hours") <= 0.0:
            raise ValueError(f"targets[{index}].window_hours must be positive")

    return raw, analysis


def _wrap_degrees(value: float | FloatArray) -> float | FloatArray:
    return (np.asarray(value, dtype=float) + 180.0) % 360.0 - 180.0


def _circular_difference(value: float, reference: float) -> float:
    return float(_wrap_degrees(value - reference))


def _temperature_offset(teq_k: float) -> float:
    """Return the declared illustrative offset in degrees east."""

    return float(
        np.rad2deg(
            np.arctan(np.tan(np.deg2rad(40.0)) * (1500.0 / float(teq_k)) ** 3)
        )
    )


def _weighted_quantile(values: FloatArray, weights: FloatArray, quantile: float) -> float:
    unique_values, inverse = np.unique(np.asarray(values, dtype=float), return_inverse=True)
    mass = np.bincount(inverse, weights=np.asarray(weights, dtype=float))
    order = np.argsort(unique_values)
    ordered_values = unique_values[order]
    ordered_weights = mass[order]
    ordered_weights = ordered_weights / np.sum(ordered_weights)
    centres = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    return float(np.interp(quantile, centres, ordered_values))


def _weighted_circular_quantile(
    values: FloatArray, weights: FloatArray, quantile: float
) -> float:
    angle = np.deg2rad(values)
    resultant = np.sum(weights * np.exp(1j * angle))
    anchor = float(np.rad2deg(np.angle(resultant))) if abs(resultant) > 1.0e-12 else float(values[np.argmax(weights)])
    unwrapped = anchor + _wrap_degrees(values - anchor)
    return float(_wrap_degrees(_weighted_quantile(unwrapped, weights, quantile)))


def _nuisance_design(time_days: FloatArray, order: int) -> FloatArray:
    centred = np.asarray(time_days, dtype=float) - float(np.mean(time_days))
    scale = max(float(np.ptp(time_days)) / 2.0, np.finfo(float).eps)
    x = centred / scale
    return np.column_stack([x**degree for degree in range(int(order) + 1)])


def _mixture_coefficients(
    longitude_degrees: float,
    width_degrees: float,
    latitude_degrees: float,
    fraction: float,
    harmonic_degree: int,
    subobserver_latitude_radians: float,
) -> FloatArray:
    uniform = _uniform_coefficients(harmonic_degree)
    visible_design = np.asarray(
        map_design_matrix(0.0, subobserver_latitude_radians, harmonic_degree),
        dtype=float,
    )
    uniform_visible = float(visible_design @ uniform)
    uniform = uniform / uniform_visible
    hotspot = _hotspot_coefficients(
        longitude_degrees, width_degrees, harmonic_degree, latitude_degrees
    )
    hotspot_visible = float(visible_design @ hotspot)
    if not np.isfinite(hotspot_visible) or hotspot_visible <= 0.0:
        raise ValueError("A hotspot template has non-positive full-phase flux")
    hotspot = hotspot / hotspot_visible
    coefficients = (1.0 - fraction) * uniform + fraction * hotspot
    visible = float(visible_design @ coefficients)
    if not np.isfinite(visible) or visible <= 0.0:
        raise ValueError("A mixture map has non-positive full-phase flux")
    return np.asarray(coefficients / visible, dtype=float)


def _hotspot_at_viewing_latitude(
    longitude_degrees: float,
    width_degrees: float,
    latitude_degrees: float,
    harmonic_degree: int,
    subobserver_latitude_radians: float,
) -> FloatArray:
    coefficients = _hotspot_coefficients(
        longitude_degrees, width_degrees, harmonic_degree, latitude_degrees
    )
    visible = float(
        np.asarray(
            map_design_matrix(0.0, subobserver_latitude_radians, harmonic_degree),
            dtype=float,
        )
        @ coefficients
    )
    if not np.isfinite(visible) or visible <= 0.0:
        raise ValueError("A hotspot template has non-positive full-phase flux")
    return np.asarray(coefficients / visible, dtype=float)


def _maximum_positive_spot_ratio(
    uniform: FloatArray, hotspot: FloatArray, harmonic_degree: int
) -> float:
    """Return the largest spot/uniform amplitude ratio with positive pixels."""

    longitude = np.arange(-180.0, 180.5, 0.5)
    latitude = np.arange(-90.0, 90.5, 0.5)
    lon_grid, lat_grid = np.meshgrid(np.deg2rad(longitude), np.deg2rad(latitude))
    uniform_values = np.asarray(evaluate_map(uniform, lon_grid, lat_grid), dtype=float)
    hotspot_values = np.asarray(evaluate_map(hotspot, lon_grid, lat_grid), dtype=float)
    negative = hotspot_values < -1.0e-12
    if not np.any(negative):
        return float("inf")
    return float(np.min(uniform_values[negative] / -hotspot_values[negative]))


def _map_metrics(
    coefficients: FloatArray,
    harmonic_degree: int,
    *,
    step_degrees: float = 0.5,
) -> tuple[float, float, float]:
    longitude = np.arange(-180.0, 180.0 + 0.5 * step_degrees, step_degrees)
    latitude = np.arange(-90.0, 90.0 + 0.5 * step_degrees, step_degrees)
    lon_grid, lat_grid = np.meshgrid(np.deg2rad(longitude), np.deg2rad(latitude))
    values = np.asarray(evaluate_map(coefficients, lon_grid, lat_grid), dtype=float)
    index = np.unravel_index(int(np.argmax(values)), values.shape)
    peak_lon = float(longitude[index[1]])
    peak_lat = float(latitude[index[0]])
    return float(np.min(values)), peak_lon, peak_lat


def _operator_for_visit(
    time_days: FloatArray,
    target: dict[str, Any],
    timing_seconds: float,
    quadrature: Any,
    exposure_subsamples: int,
    chunk_size: int,
) -> FloatArray:
    """Return the exposure-integrated eclipse operator in bounded chunks."""

    exposure_seconds = float(target.get("exposure_seconds", target["cadence_seconds"]))
    offsets_days = (
        (np.arange(exposure_subsamples, dtype=float) + 0.5) / exposure_subsamples - 0.5
    ) * exposure_seconds / 86_400.0
    eclipse_time = float(target.get("eclipse_time_days", 0.0)) + timing_seconds / 86_400.0
    transit_time = eclipse_time - float(target["period_days"]) / 2.0
    subobserver_latitude = np.deg2rad(90.0 - float(target["inclination_degrees"]))
    output: list[FloatArray] = []
    for start in range(0, time_days.size, chunk_size):
        stop = min(start + chunk_size, time_days.size)
        sample_time = time_days[start:stop, None] + offsets_days[None, :]
        sampled_design = np.asarray(
            secondary_eclipse_design_matrix(
                sample_time,
                float(target["period_days"]),
                float(target["a_over_rstar"]),
                float(target["inclination_degrees"]),
                float(target["radius_ratio"]),
                2,
                transit_time,
                theta0=np.pi,
                subobserver_lat=subobserver_latitude,
                angle_unit="deg",
                quadrature=quadrature,
            ),
            dtype=float,
        )
        output.append(np.mean(sampled_design, axis=1))
    return np.concatenate(output, axis=0)


def _make_visit(target: dict[str, Any]) -> tuple[FloatArray, float]:
    cadence = float(target["cadence_seconds"])
    exposure = float(target.get("exposure_seconds", cadence))
    if exposure <= 0.0 or exposure > cadence:
        raise ValueError(
            f"{target['name']}: exposure_seconds must be positive and no greater than cadence_seconds"
        )
    window_seconds = float(target["window_hours"]) * 3600.0
    count = int(np.floor(window_seconds / cadence + 1.0e-10))
    if count < 3:
        raise ValueError(f"{target['name']}: window_hours must cover at least three cadence bins")
    centre_seconds = (np.arange(count, dtype=float) - (count - 1.0) / 2.0) * cadence
    event_time = float(target.get("eclipse_time_days", 0.0))
    return event_time + centre_seconds / 86_400.0, exposure


def _projected_profiles(
    observed_ppm: FloatArray,
    sigma_ppm: float,
    nuisance: FloatArray,
    candidate_curves_ppm: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Profile positive map amplitudes after projecting out baseline terms."""

    design = np.asarray(nuisance, dtype=float) / sigma_ppm
    whitened_y = np.asarray(observed_ppm, dtype=float) / sigma_ppm
    q, _ = np.linalg.qr(design, mode="reduced")
    projected_y = whitened_y - q @ (q.T @ whitened_y)
    candidate = np.asarray(candidate_curves_ppm, dtype=float).T / sigma_ppm
    projected_candidate = candidate - q @ (q.T @ candidate)
    numerator = projected_candidate.T @ projected_y
    denominator = np.sum(projected_candidate**2, axis=0)
    valid = denominator > np.finfo(float).tiny
    amplitudes = np.zeros(denominator.size, dtype=float)
    amplitudes[valid] = np.maximum(numerator[valid] / denominator[valid], 0.0)
    improvements = np.zeros_like(amplitudes)
    improvements[valid] = np.maximum(numerator[valid], 0.0) ** 2 / denominator[valid]
    chi2 = np.maximum(float(projected_y @ projected_y) - improvements, 0.0)
    fitted_map = candidate_curves_ppm * amplitudes[:, None]
    return chi2, amplitudes, projected_y, projected_candidate, fitted_map


def _profile_null_models(
    observed_ppm: FloatArray,
    sigma_ppm: float,
    nuisance: FloatArray,
    uniform_curves_ppm: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return the profiled null chi-square and amplitude at each timing."""

    results = _projected_profiles(observed_ppm, sigma_ppm, nuisance, uniform_curves_ppm)
    return results[0], results[1]


def _projected_two_map_profiles(
    observed_ppm: FloatArray,
    sigma_ppm: float,
    nuisance: FloatArray,
    uniform_curves_ppm: FloatArray,
    hotspot_curves_ppm: FloatArray,
    maximum_spot_ratio: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Profile two non-negative map amplitudes with a positivity bound.

    ``maximum_spot_ratio`` imposes ``spot / uniform <= limit`` whenever a
    degree-2 hotspot template has negative pixels. The constrained optimum
    can lie on the uniform ray, the two-map interior, or the positivity
    boundary. All cases are solved in arrays for fast repeated trials.
    """

    design = np.asarray(nuisance, dtype=float) / sigma_ppm
    whitened_y = np.asarray(observed_ppm, dtype=float) / sigma_ppm
    q, _ = np.linalg.qr(design, mode="reduced")
    projected_y = whitened_y - q @ (q.T @ whitened_y)
    uniform = np.asarray(uniform_curves_ppm, dtype=float).T / sigma_ppm
    hotspot = np.asarray(hotspot_curves_ppm, dtype=float).T / sigma_ppm
    uniform -= q @ (q.T @ uniform)
    hotspot -= q @ (q.T @ hotspot)

    yy = float(projected_y @ projected_y)
    uu = np.sum(uniform**2, axis=0)
    hh = np.sum(hotspot**2, axis=0)
    uh = np.sum(uniform * hotspot, axis=0)
    uy = uniform.T @ projected_y
    hy = hotspot.T @ projected_y
    candidate_count = uu.size
    amp_uniform = np.zeros(candidate_count, dtype=float)
    amp_spot = np.zeros(candidate_count, dtype=float)
    best_improvement = np.zeros(candidate_count, dtype=float)

    valid_u = uu > np.finfo(float).tiny
    uniform_amp = np.zeros(candidate_count, dtype=float)
    uniform_amp[valid_u] = np.maximum(uy[valid_u] / uu[valid_u], 0.0)
    uniform_improvement = np.zeros(candidate_count, dtype=float)
    uniform_improvement[valid_u] = np.maximum(uy[valid_u], 0.0) ** 2 / uu[valid_u]
    best_improvement[:] = uniform_improvement
    amp_uniform[:] = uniform_amp

    unconstrained_det = uu * hh - uh**2
    valid_det = unconstrained_det > np.finfo(float).tiny
    both_uniform = np.zeros(candidate_count, dtype=float)
    both_spot = np.zeros(candidate_count, dtype=float)
    both_uniform[valid_det] = (
        uy[valid_det] * hh[valid_det] - hy[valid_det] * uh[valid_det]
    ) / unconstrained_det[valid_det]
    both_spot[valid_det] = (
        hy[valid_det] * uu[valid_det] - uy[valid_det] * uh[valid_det]
    ) / unconstrained_det[valid_det]
    ratio_ok = np.isinf(maximum_spot_ratio)
    finite_ratio = np.isfinite(maximum_spot_ratio)
    ratio_ok[finite_ratio] = (
        both_spot[finite_ratio]
        <= maximum_spot_ratio[finite_ratio] * both_uniform[finite_ratio] + 1.0e-12
    )
    both_valid = valid_det & (both_uniform >= 0.0) & (both_spot >= 0.0) & ratio_ok
    both_improvement = np.zeros(candidate_count, dtype=float)
    both_improvement[both_valid] = (
        both_uniform[both_valid] * uy[both_valid]
        + both_spot[both_valid] * hy[both_valid]
    )
    take = both_valid & (both_improvement > best_improvement)
    best_improvement[take] = both_improvement[take]
    amp_uniform[take] = both_uniform[take]
    amp_spot[take] = both_spot[take]

    # A pure hotspot is allowed only if its degree-2 map has no negative
    # pixels. Otherwise its zero-uniform endpoint is outside the feasible
    # positive-map region.
    valid_h = hh > np.finfo(float).tiny
    hotspot_only_amp = np.zeros(candidate_count, dtype=float)
    hotspot_only_amp[valid_h] = np.maximum(hy[valid_h] / hh[valid_h], 0.0)
    hotspot_only_improvement = np.zeros(candidate_count, dtype=float)
    hotspot_only_improvement[valid_h] = (
        np.maximum(hy[valid_h], 0.0) ** 2 / hh[valid_h]
    )
    hotspot_only_ok = np.isinf(maximum_spot_ratio)
    take_h = hotspot_only_ok & (hotspot_only_improvement > best_improvement)
    best_improvement[take_h] = hotspot_only_improvement[take_h]
    amp_uniform[take_h] = 0.0
    amp_spot[take_h] = hotspot_only_amp[take_h]

    finite_limit = np.isfinite(maximum_spot_ratio)
    if np.any(finite_limit):
        boundary = uniform[:, finite_limit] + hotspot[:, finite_limit] * maximum_spot_ratio[finite_limit][None, :]
        boundary_norm = np.sum(boundary**2, axis=0)
        boundary_dot = boundary.T @ projected_y
        boundary_amp = np.zeros(boundary_norm.size, dtype=float)
        valid_boundary = boundary_norm > np.finfo(float).tiny
        boundary_amp[valid_boundary] = np.maximum(
            boundary_dot[valid_boundary] / boundary_norm[valid_boundary], 0.0
        )
        boundary_improvement = np.zeros(boundary_norm.size, dtype=float)
        boundary_improvement[valid_boundary] = (
            np.maximum(boundary_dot[valid_boundary], 0.0) ** 2
            / boundary_norm[valid_boundary]
        )
        finite_indices = np.flatnonzero(finite_limit)
        take_boundary = boundary_improvement > best_improvement[finite_indices]
        selected = finite_indices[take_boundary]
        best_improvement[selected] = boundary_improvement[take_boundary]
        amp_uniform[selected] = boundary_amp[take_boundary]
        amp_spot[selected] = (
            boundary_amp[take_boundary] * maximum_spot_ratio[selected]
        )

    chi2 = np.maximum(yy - best_improvement, 0.0)
    return chi2, amp_uniform, amp_spot


def _fit_scope(
    *,
    target: dict[str, Any],
    observed_ppm: FloatArray,
    sigma_ppm: float,
    mask: NDArray[np.bool_],
    nuisance_full: FloatArray,
    uniform_candidate_curves: FloatArray,
    hotspot_curves: FloatArray,
    uniform_curves_by_timing: FloatArray,
    candidate_longitudes: FloatArray,
    candidate_peak_longitudes: FloatArray,
    candidate_peak_latitudes: FloatArray,
    candidate_widths: FloatArray,
    candidate_latitudes: FloatArray,
    candidate_timings: FloatArray,
    uniform_coefficients: FloatArray,
    hotspot_coefficients: list[FloatArray],
    uniform_minimum_intensity: float,
    candidate_hotspot_minimums: FloatArray,
    maximum_spot_ratios: FloatArray,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    n_obs = int(indices.size)
    if n_obs <= nuisance_full.shape[1] + 1:
        raise ValueError(f"Not enough observations for {target['name']} eclipse-only fit")
    nuisance = nuisance_full[indices]
    observed = observed_ppm[indices]
    uniform_curves = uniform_candidate_curves[:, indices]
    spot_curves = hotspot_curves[:, indices]
    chi2, uniform_amplitudes, spot_amplitudes = _projected_two_map_profiles(
        observed,
        sigma_ppm,
        nuisance,
        uniform_curves,
        spot_curves,
        maximum_spot_ratios,
    )
    n_free_grid = (
        int(np.unique(candidate_longitudes).size > 1)
        + int(np.unique(candidate_widths).size > 1)
        + int(np.unique(candidate_latitudes).size > 1)
        + int(np.unique(candidate_timings).size > 1)
    )
    log_constant = n_obs * np.log(2.0 * np.pi * sigma_ppm**2)
    flexible_k = nuisance.shape[1] + 2 + n_free_grid
    flexible_bic = chi2 + log_constant + flexible_k * np.log(n_obs)
    log_likelihood = -0.5 * (chi2 + log_constant)
    log_weight = -0.5 * flexible_bic
    if not np.all(np.isfinite(flexible_bic)) or not np.all(np.isfinite(log_likelihood)):
        raise RuntimeError(f"Non-finite profile likelihood or BIC for {target['name']}")
    weight = np.exp(log_weight - np.max(log_weight))
    weight_sum = float(np.sum(weight))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise RuntimeError(f"Invalid profile weight sum for {target['name']}")
    weight /= weight_sum
    if (
        not np.all(np.isfinite(weight))
        or np.any(weight < 0.0)
        or not np.isclose(np.sum(weight), 1.0, rtol=1.0e-10, atol=1.0e-12)
    ):
        raise RuntimeError(f"Profile weights are not finite and normalized for {target['name']}")
    best_index = int(np.argmin(flexible_bic))

    uniform_chi2, _ = _profile_null_models(
        observed, sigma_ppm, nuisance, uniform_curves_by_timing[:, indices]
    )
    timing_count = int(np.unique(candidate_timings).size)
    null_k = nuisance.shape[1] + 1 + int(timing_count > 1)
    uniform_bic_by_timing = uniform_chi2 + log_constant + null_k * np.log(n_obs)
    uniform_bic = float(np.min(uniform_bic_by_timing))
    flexible_bic_best = float(flexible_bic[best_index])
    bic_comparison = compare_uniform_flexible_bic(uniform_bic, flexible_bic_best)

    # Use the public recovery solver for the selected nuisance profile. The
    # two map amplitudes come from the batched constrained solve above. This
    # keeps the positivity ratio bound active, which fit_profile alone cannot
    # express as a map-pixel constraint.
    best_map_model = (
        uniform_curves[best_index] * uniform_amplitudes[best_index]
        + spot_curves[best_index] * spot_amplitudes[best_index]
    )
    best_fit = fit_profile(
        nuisance,
        observed - best_map_model,
        sigma_ppm,
        map_columns=(),
        nuisance_columns=tuple(range(nuisance.shape[1])),
    )
    fast_chi2 = float(chi2[best_index])
    if not np.isclose(best_fit.chi2, fast_chi2, rtol=1.0e-7, atol=1.0e-6):
        raise RuntimeError(
            f"Batched profile and recovery solver disagree for {target['name']}: "
            f"{fast_chi2:.8g} vs {best_fit.chi2:.8g}"
        )

    longitude_q = [
        _weighted_circular_quantile(candidate_peak_longitudes, weight, q)
        for q in (0.16, 0.50, 0.84)
    ]
    latitude_q = [
        _weighted_quantile(candidate_peak_latitudes, weight, q) for q in (0.16, 0.50, 0.84)
    ]
    width_q = [
        _weighted_quantile(candidate_widths, weight, q) for q in (0.16, 0.50, 0.84)
    ]
    timing_q = [
        _weighted_quantile(candidate_timings, weight, q) for q in (0.16, 0.50, 0.84)
    ]
    amplitude_uniform = float(uniform_amplitudes[best_index])
    amplitude_spot = float(spot_amplitudes[best_index])
    map_min = (
        amplitude_uniform * uniform_minimum_intensity
        + amplitude_spot * float(candidate_hotspot_minimums[best_index])
    )
    if map_min < -1.0e-8:
        raise RuntimeError(
            f"Recovered degree-2 map is negative for {target['name']}: minimum {map_min:.6g}"
        )
    peak_lon = None
    peak_lat = None
    if amplitude_spot > 1.0e-12:
        peak_lon = float(candidate_peak_longitudes[best_index])
        peak_lat = float(candidate_peak_latitudes[best_index])
    predicted = nuisance_full @ best_fit.coefficients[: nuisance.shape[1]]
    predicted = predicted + (
        uniform_candidate_curves[best_index] * uniform_amplitudes[best_index]
        + hotspot_curves[best_index] * spot_amplitudes[best_index]
    )
    residual_rms = float(np.sqrt(np.mean(best_fit.residuals**2)))
    raw_longitude_q = [
        _weighted_circular_quantile(candidate_peak_longitudes, weight, q)
        for q in (0.16, 0.50, 0.84)
    ]
    longitude_q = raw_longitude_q.copy()
    raw_interval_span = abs(_circular_difference(raw_longitude_q[0], raw_longitude_q[1])) + abs(
        _circular_difference(raw_longitude_q[2], raw_longitude_q[1])
    )
    resolution_limited = raw_interval_span < float(analysis["longitude_grid_step_degrees"])
    if resolution_limited:
        half_step = float(analysis["longitude_grid_step_degrees"]) / 2.0
        longitude_q[0] = float(_wrap_degrees(raw_longitude_q[1] - half_step))
        longitude_q[2] = float(_wrap_degrees(raw_longitude_q[1] + half_step))
    amplitude_total = amplitude_uniform + amplitude_spot
    return {
        "best_index": best_index,
        "best_fit": best_fit,
        "weights": weight,
        "log_likelihood": log_likelihood,
        "flexible_bic_by_candidate": flexible_bic,
        "uniform_bic": uniform_bic,
        "flexible_bic": flexible_bic_best,
        "delta_bic": float(bic_comparison.delta_bic),
        "longitude_q": longitude_q,
        "latitude_q": latitude_q,
        "width_q": width_q,
        "timing_q": timing_q,
        "best_width": float(candidate_widths[best_index]),
        "best_latitude": float(candidate_latitudes[best_index]),
        "best_timing": float(candidate_timings[best_index]),
        "best_longitude": float(candidate_longitudes[best_index]),
        "map_amplitude": amplitude_total,
        "uniform_map_amplitude": amplitude_uniform,
        "spot_amplitude": amplitude_spot,
        "recovered_spot_fraction": 0.0
        if amplitude_total <= 0.0
        else amplitude_spot / amplitude_total,
        "map_minimum": map_min,
        "peak_longitude": peak_lon,
        "peak_latitude": peak_lat,
        "peak_shift": None
        if peak_lon is None
        else _circular_difference(peak_lon, float(candidate_longitudes[best_index])),
        "map_positive": bool(
            map_min >= -1.0e-10 and amplitude_uniform >= 0.0 and amplitude_spot >= 0.0
        ),
        "longitude_resolution_limited": bool(resolution_limited),
        "residual_rms_ppm": residual_rms,
        "predicted_full_ppm": predicted,
        "chi2": chi2,
        "log_likelihood_values": log_likelihood,
        "posterior_weights": weight,
        "profile_bic": flexible_bic,
        "n_obs": n_obs,
        "n_free_grid": n_free_grid,
    }


def _target_noise_levels(target: dict[str, Any], analysis: dict[str, Any]) -> tuple[list[float], bool]:
    levels = analysis["noise_levels_ppm"]
    if levels is None:
        local = target.get("noise_levels_ppm")
        if local is not None:
            return _number_list(local, f"{target['name']}.noise_levels_ppm", minimum=0.0), True
        return [float(target["noise_ppm"])], False
    return [float(value) for value in levels], True


def _stable_seed(master_seed: int, target_index: int, noise_index: int, trial: int) -> int:
    sequence = np.random.SeedSequence([master_seed, target_index, noise_index, trial])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "target"


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["target"], row["noise_ppm"], row["scenario"], row["fit_scope"])
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (target, noise, scenario, scope), group in groups.items():
        delta = np.asarray([float(row["delta_bic"]) for row in group])
        detected = np.asarray([bool(row["map_detected"]) for row in group])
        bias = np.asarray(
            [float(row["longitude_bias_degrees"]) for row in group if row["longitude_bias_degrees"] is not None]
        )
        covered = [row["longitude_interval_contains_truth"] for row in group]
        covered_values = [value for value in covered if value is not None]
        summaries.append(
            {
                "target": target,
                "noise_ppm": noise,
                "scenario": scenario,
                "fit_scope": scope,
                "trial_count": len(group),
                "mean_delta_bic": float(np.mean(delta)),
                "median_delta_bic": float(np.median(delta)),
                "map_detection_fraction": float(np.mean(detected)),
                "mean_longitude_bias_degrees": None if bias.size == 0 else float(np.mean(bias)),
                "longitude_bias_standard_deviation_degrees": None
                if bias.size == 0
                else float(np.std(bias, ddof=1 if bias.size > 1 else 0)),
                "longitude_interval_coverage": None
                if not covered_values
                else float(np.mean(covered_values)),
            }
        )
    return summaries


def _effective_input_assumptions(
    raw: dict[str, Any], effective_trials: int
) -> dict[str, Any]:
    """Copy input assumptions and qualify a stale single-realization note."""
    source = raw.get("assumptions", {})
    assumptions = dict(source) if isinstance(source, dict) else {}
    statement = assumptions.get("single_realization")
    if statement and effective_trials != 1:
        assumptions["single_realization"] = (
            f"Input record: {statement} This run used {effective_trials} independent trials "
            "per target, noise level, and scenario; the single-realization statement does "
            "not describe these results. See run_settings for effective values."
        )
    return assumptions


def run_study(
    input_path: Path,
    output: Path,
    *,
    trials: int = 30,
    quick: bool = False,
    exposure_subsamples: int = 8,
    disk_radial: int = 32,
    disk_azimuth: int = 128,
    chunk_size: int = 128,
    check_convergence: bool = False,
    convergence_tolerance_ppm: float = 0.5,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be at least one")
    if exposure_subsamples < 1 or disk_radial < 2 or disk_azimuth < 4 or chunk_size < 1:
        raise ValueError("quadrature counts and chunk_size must be positive")
    raw, analysis = _load_inputs(input_path)
    targets = raw["targets"][:1] if quick else raw["targets"]
    effective_trials = 1 if quick else int(trials)
    master_seed = int(raw.get("seed", 20261009))
    scenario_names = list(analysis["scenarios"])
    quadrature = disk_quadrature(disk_radial, disk_azimuth)
    harmonic_degree = 2

    analysis_resolved = dict(analysis)
    analysis_resolved.update(
        {
            "exposure_subsamples": int(exposure_subsamples),
            "disk_quadrature_radial": int(disk_radial),
            "disk_quadrature_azimuth": int(disk_azimuth),
            "operator_chunk_size_observations": int(chunk_size),
            "detection_delta_bic": DETECTION_DELTA_BIC,
            "scenarios": scenario_names,
            "fit_scopes": list(FIT_SCOPES),
            "profiles_npz_fit_scopes": ["full_window"],
            "longitude_resolution_degrees": float(analysis["longitude_grid_step_degrees"]),
            "recovery_contrast": "free non-negative uniform and hotspot amplitudes",
        }
    )

    rows: list[dict[str, Any]] = []
    profile_records: list[dict[str, Any]] = []
    examples_by_target: dict[str, list[dict[str, Any]]] = {}
    target_summaries: list[dict[str, Any]] = []
    numerical_checks: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets):
        name = str(target["name"])
        time_days, exposure_seconds = _make_visit(target)
        cadence_seconds = float(target["cadence_seconds"])
        subobserver_latitude = np.deg2rad(90.0 - float(target["inclination_degrees"]))
        uniform_coefficients = _uniform_coefficients(harmonic_degree)
        uniform_visible = float(
            np.asarray(
                map_design_matrix(0.0, subobserver_latitude, harmonic_degree), dtype=float
            )
            @ uniform_coefficients
        )
        uniform_coefficients = np.asarray(uniform_coefficients / uniform_visible, dtype=float)
        # The nuisance basis uses the full visit coordinates for both fit
        # scopes. This makes eclipse-only baseline fits comparable and lets
        # their curves be evaluated across the full visit.
        nuisance_full = _nuisance_design(time_days, int(analysis["baseline_order"]))
        injection_nuisance = _nuisance_design(time_days, 2)
        injection_baseline = (
            float(analysis["baseline_intercept_ppm"])
            + float(analysis["baseline_slope_ppm"]) * injection_nuisance[:, 1]
            + float(analysis["baseline_curvature_ppm"]) * injection_nuisance[:, 2]
        )
        timing_values = np.asarray(analysis["timing_grid_seconds"], dtype=float)
        operators: dict[float, FloatArray] = {}
        for timing in timing_values:
            operators[float(timing)] = _operator_for_visit(
                time_days,
                target,
                float(timing),
                quadrature,
                exposure_subsamples,
                chunk_size,
            )

        if check_convergence:
            offset = _temperature_offset(float(target["teq_k"]))
            truth_coefficients = _mixture_coefficients(
                offset,
                float(analysis["injected_width_degrees"]),
                float(analysis["injected_latitude_degrees"]),
                float(analysis["injection_fraction"]),
                harmonic_degree,
                subobserver_latitude,
            )
            fine_operator = _operator_for_visit(
                time_days,
                target,
                float(analysis["injected_timing_seconds"]),
                disk_quadrature(disk_radial * 2, disk_azimuth * 2),
                exposure_subsamples * 2,
                chunk_size,
            )
            coarse_operator = _operator_for_visit(
                time_days,
                target,
                float(analysis["injected_timing_seconds"]),
                quadrature,
                exposure_subsamples,
                chunk_size,
            )
            max_difference = float(
                np.max(
                    np.abs(
                        (fine_operator @ truth_coefficients - coarse_operator @ truth_coefficients)
                        * float(target["depth_ppm"])
                    )
                )
            )
            numerical_checks.append(
                {
                    "target": name,
                    "base_exposure_subsamples": exposure_subsamples,
                    "fine_exposure_subsamples": exposure_subsamples * 2,
                    "base_disk_quadrature": [disk_radial, disk_azimuth],
                    "fine_disk_quadrature": [disk_radial * 2, disk_azimuth * 2],
                    "max_absolute_curve_difference_ppm": max_difference,
                    "tolerance_ppm": float(convergence_tolerance_ppm),
                    "passed": bool(max_difference <= convergence_tolerance_ppm),
                }
            )

        # Check each injected mixture on a dense sphere. The recovery fits
        # uniform and hotspot amplitudes separately, with a positivity bound.
        candidate_longitudes: list[float] = []
        candidate_widths: list[float] = []
        candidate_latitudes: list[float] = []
        candidate_timings: list[float] = []
        candidate_peak_longitudes: list[float] = []
        candidate_peak_latitudes: list[float] = []
        candidate_hotspot_minimums: list[float] = []
        candidate_hotspot_coefficients: list[FloatArray] = []
        candidate_maximum_spot_ratios: list[float] = []
        base_hotspot_metrics: dict[tuple[float, float], tuple[float, float, float]] = {}
        base_spot_ratios: dict[tuple[float, float], float] = {}
        base_hotspots: dict[tuple[float, float], FloatArray] = {}
        fraction = float(analysis["injection_fraction"])
        uniform_minimum_intensity = _map_metrics(uniform_coefficients, harmonic_degree)[0]
        for width in analysis["width_grid_degrees"]:
            for latitude in analysis["latitude_grid_degrees"]:
                base_hotspot = _hotspot_at_viewing_latitude(
                    0.0, width, latitude, 2, subobserver_latitude
                )
                base_hotspots[(float(width), float(latitude))] = base_hotspot
                base_hotspot_metrics[(float(width), float(latitude))] = _map_metrics(
                    base_hotspot, harmonic_degree
                )
                base_spot_ratios[(float(width), float(latitude))] = (
                    _maximum_positive_spot_ratio(
                        uniform_coefficients, base_hotspot, harmonic_degree
                    )
                )
        for timing in timing_values:
            for width in analysis["width_grid_degrees"]:
                for latitude in analysis["latitude_grid_degrees"]:
                    _, base_peak_lon, base_peak_lat = base_hotspot_metrics[
                        (float(width), float(latitude))
                    ]
                    for longitude in analysis["longitude_grid_degrees"]:
                        coeff = _hotspot_at_viewing_latitude(
                            float(longitude),
                            float(width),
                            float(latitude),
                            2,
                            subobserver_latitude,
                        )
                        candidate_longitudes.append(float(longitude))
                        candidate_widths.append(float(width))
                        candidate_latitudes.append(float(latitude))
                        candidate_timings.append(float(timing))
                        candidate_peak_longitudes.append(
                            float(_wrap_degrees(float(longitude) + base_peak_lon))
                        )
                        candidate_peak_latitudes.append(float(base_peak_lat))
                        candidate_hotspot_coefficients.append(coeff)
                        ratio_limit = base_spot_ratios[(float(width), float(latitude))]
                        base_norm = float(
                            np.linalg.norm(base_hotspots[(float(width), float(latitude))])
                        )
                        candidate_norm = float(np.linalg.norm(coeff))
                        candidate_hotspot_minimums.append(
                            base_hotspot_metrics[(float(width), float(latitude))][0]
                            * candidate_norm
                            / base_norm
                        )
                        candidate_maximum_spot_ratios.append(
                            ratio_limit
                            if np.isinf(ratio_limit)
                            else 0.995 * ratio_limit * base_norm / candidate_norm
                        )
        if not candidate_hotspot_coefficients:
            raise ValueError(f"{name}: no positive recovery templates remain")

        candidate_longitude_array = np.asarray(candidate_longitudes, dtype=float)
        candidate_width_array = np.asarray(candidate_widths, dtype=float)
        candidate_latitude_array = np.asarray(candidate_latitudes, dtype=float)
        candidate_timing_array = np.asarray(candidate_timings, dtype=float)
        candidate_peak_longitude_array = np.asarray(candidate_peak_longitudes, dtype=float)
        candidate_peak_latitude_array = np.asarray(candidate_peak_latitudes, dtype=float)
        candidate_hotspot_minimum_array = np.asarray(candidate_hotspot_minimums, dtype=float)
        maximum_spot_ratio_array = np.asarray(candidate_maximum_spot_ratios, dtype=float)
        hotspot_coefficients = candidate_hotspot_coefficients
        hotspot_curves = np.empty((len(hotspot_coefficients), time_days.size), dtype=float)
        uniform_curves_by_timing = np.empty((timing_values.size, time_days.size), dtype=float)
        timing_lookup = {float(value): index for index, value in enumerate(timing_values)}
        for timing_index_value, timing in enumerate(timing_values):
            operator = operators[float(timing)]
            uniform_curves_by_timing[timing_index_value] = (
                float(target["depth_ppm"]) * (operator @ uniform_coefficients)
            )
            selection = np.flatnonzero(candidate_timing_array == float(timing))
            if selection.size:
                coefficient_block = np.asarray([hotspot_coefficients[index] for index in selection])
                hotspot_curves[selection] = float(target["depth_ppm"]) * (
                    operator @ coefficient_block.T
                ).T
        uniform_candidate_curves = np.asarray(
            [
                uniform_curves_by_timing[timing_lookup[float(timing)]]
                for timing in candidate_timing_array
            ],
            dtype=float,
        )

        injected_timing = float(analysis["injected_timing_seconds"])
        if not np.any(np.isclose(timing_values, injected_timing, rtol=0.0, atol=1.0e-10)):
            operators[injected_timing] = _operator_for_visit(
                time_days,
                target,
                injected_timing,
                quadrature,
                exposure_subsamples,
                chunk_size,
            )
        full_phase_curve = operators[injected_timing]
        local_eclipse_mask = (
            1.0 - (full_phase_curve @ uniform_coefficients)
            > float(analysis["eclipse_only_loss_threshold"])
        )
        if int(np.sum(local_eclipse_mask)) <= nuisance_full.shape[1] + 1:
            raise ValueError(
                f"{name}: window has too few in-eclipse samples for the eclipse-only recovery"
            )

        temperature_offset = _temperature_offset(float(target["teq_k"]))
        target_summary = {
            "target": name,
            "seed_target_index": int(target.get("seed_target_index", target_index)),
            "instrument": str(target["instrument"]),
            "teq_k": float(target["teq_k"]),
            "brightness_temperature_k": target.get("brightness_temperature_k"),
            "depth_ppm": float(target["depth_ppm"]),
            "window_hours": float(target["window_hours"]),
            "cadence_seconds": cadence_seconds,
            "exposure_seconds": exposure_seconds,
            "n_observations": int(time_days.size),
            "n_eclipse_only_observations": int(np.sum(local_eclipse_mask)),
            "temperature_scaled_offset_degrees": temperature_offset,
            "valid_candidate_count": len(hotspot_coefficients),
            "positivity_bounded_candidate_count": int(
                np.sum(np.isfinite(maximum_spot_ratio_array))
            ),
        }
        target_summaries.append(target_summary)
        examples_by_target[name] = []
        noise_levels, are_reference_levels = _target_noise_levels(target, analysis)
        if quick:
            noise_levels = noise_levels[:1]
        noise_reference = float(analysis["noise_reference_seconds"])
        if not are_reference_levels:
            noise_reference = exposure_seconds

        truth_by_scenario: dict[str, dict[str, Any]] = {}
        for scenario in scenario_names:
            if scenario == "temperature_scaled":
                injected_longitude = temperature_offset
            elif scenario == "zero_offset":
                injected_longitude = 0.0
            elif scenario == "constant_30_deg":
                injected_longitude = float(analysis["constant_offset_degrees"])
            else:
                injected_longitude = None
            if injected_longitude is None:
                truth_coefficients = uniform_coefficients
                truth_minimum = float(
                    _map_metrics(truth_coefficients, harmonic_degree)[0]
                )
                truth_peak_lon = None
                truth_peak_lat = None
                truth_peak_shift = None
            else:
                truth_coefficients = _mixture_coefficients(
                    injected_longitude,
                    float(analysis["injected_width_degrees"]),
                    float(analysis["injected_latitude_degrees"]),
                    fraction,
                    harmonic_degree,
                    subobserver_latitude,
                )
                truth_minimum, truth_peak_lon, truth_peak_lat = _map_metrics(
                    truth_coefficients, harmonic_degree
                )
                truth_peak_shift = _circular_difference(truth_peak_lon, injected_longitude)
                if truth_minimum < -1.0e-10:
                    raise ValueError(
                        f"{name} {scenario}: injected map is negative on the dense sphere"
                    )
                if abs(truth_peak_shift) > 0.51:
                    raise ValueError(
                        f"{name} {scenario}: degree-2 map peak shifts by "
                        f"{truth_peak_shift:.3f} degrees from the injected longitude"
                    )
            truth_by_scenario[scenario] = {
                "longitude": injected_longitude,
                "coefficients": truth_coefficients,
                "minimum": truth_minimum,
                "peak_longitude": truth_peak_lon,
                "peak_latitude": truth_peak_lat,
                "peak_shift": truth_peak_shift,
                "signal": (
                    float(target["depth_ppm"])
                    * (operators[injected_timing] @ truth_coefficients)
                    + injection_baseline
                ),
            }

        for noise_index, noise_input in enumerate(noise_levels):
            sigma_ppm = float(noise_input)
            if are_reference_levels:
                sigma_ppm *= math.sqrt(noise_reference / exposure_seconds)
            if sigma_ppm <= 0.0:
                raise ValueError(f"{name}: noise level must be positive")
            for trial in range(effective_trials):
                # Reuse the same standardized draw at each noise level. This
                # pairs noise cases and allows a focused run to reuse a
                # target's seed from a larger run by declaring its index.
                seed_target_index = int(target.get("seed_target_index", target_index))
                seed = _stable_seed(master_seed, seed_target_index, 0, trial)
                rng = np.random.default_rng(seed)
                noise = rng.normal(0.0, sigma_ppm, size=time_days.size)
                for scenario in scenario_names:
                    truth = truth_by_scenario[scenario]
                    injected_longitude = truth["longitude"]
                    truth_coefficients = truth["coefficients"]
                    truth_minimum = truth["minimum"]
                    truth_peak_lon = truth["peak_longitude"]
                    truth_peak_lat = truth["peak_latitude"]
                    truth_peak_shift = truth["peak_shift"]
                    injected_timing = float(analysis["injected_timing_seconds"])
                    truth_signal = truth["signal"]
                    observed = truth_signal + noise
                    result_by_scope: dict[str, dict[str, Any]] = {}
                    for fit_scope, mask in (
                        ("full_window", np.ones(time_days.size, dtype=bool)),
                        ("eclipse_only", local_eclipse_mask),
                    ):
                        result = _fit_scope(
                            target=target,
                            observed_ppm=observed,
                            sigma_ppm=sigma_ppm,
                            mask=mask,
                            nuisance_full=nuisance_full,
                            uniform_candidate_curves=uniform_candidate_curves,
                            hotspot_curves=hotspot_curves,
                            uniform_curves_by_timing=uniform_curves_by_timing,
                            candidate_longitudes=candidate_longitude_array,
                            candidate_peak_longitudes=candidate_peak_longitude_array,
                            candidate_peak_latitudes=candidate_peak_latitude_array,
                            candidate_widths=candidate_width_array,
                            candidate_latitudes=candidate_latitude_array,
                            candidate_timings=candidate_timing_array,
                            uniform_coefficients=uniform_coefficients,
                            hotspot_coefficients=hotspot_coefficients,
                            uniform_minimum_intensity=uniform_minimum_intensity,
                            candidate_hotspot_minimums=candidate_hotspot_minimum_array,
                            maximum_spot_ratios=maximum_spot_ratio_array,
                            analysis=analysis,
                        )
                        result_by_scope[fit_scope] = result
                        q16, q50, q84 = result["longitude_q"]
                        longitude_bias = (
                            None
                            if truth_peak_lon is None
                            else _circular_difference(q50, float(truth_peak_lon))
                        )
                        contains = (
                            None
                            if truth_peak_lon is None
                            else bool(
                                _circular_difference(q16, q50)
                                <= _circular_difference(float(truth_peak_lon), q50)
                                <= _circular_difference(q84, q50)
                            )
                        )
                        row = {
                            "run_id": f"{_slug(name)}__{noise_input:g}ppm__{scenario}__trial{trial:03d}__{fit_scope}",
                            "target": name,
                            "instrument": str(target["instrument"]),
                            "teq_k": float(target["teq_k"]),
                            "brightness_temperature_k": target.get("brightness_temperature_k"),
                            "noise_ppm": sigma_ppm,
                            "noise_input_ppm": float(noise_input),
                            "noise_reference_seconds": noise_reference,
                            "scenario": scenario,
                            "trial": int(trial),
                            "fit_scope": fit_scope,
                            "time_window_hours": float(target["window_hours"]),
                            "n_observations": result["n_obs"],
                            "n_eclipse_only_observations": int(np.sum(local_eclipse_mask)),
                            "injected_longitude_degrees": injected_longitude,
                            "injected_latitude_degrees": None
                            if injected_longitude is None
                            else float(analysis["injected_latitude_degrees"]),
                            "injected_width_degrees": None
                            if injected_longitude is None
                            else float(analysis["injected_width_degrees"]),
                            "injection_fraction": None
                            if injected_longitude is None
                            else fraction,
                            "injected_timing_seconds": injected_timing,
                            "truth_map_minimum_intensity": truth_minimum,
                            "truth_map_peak_longitude_degrees": truth_peak_lon,
                            "truth_map_peak_latitude_degrees": truth_peak_lat,
                            "truth_map_peak_shift_degrees": truth_peak_shift,
                            "recovered_longitude_q16_degrees": q16,
                            "recovered_longitude_median_degrees": q50,
                            "recovered_longitude_q84_degrees": q84,
                            "longitude_bias_degrees": longitude_bias,
                            "longitude_interval_contains_truth": contains,
                            "longitude_interval_resolution_limited": result[
                                "longitude_resolution_limited"
                            ],
                            "recovered_latitude_q16_degrees": result["latitude_q"][0],
                            "recovered_latitude_median_degrees": result["latitude_q"][1],
                            "recovered_latitude_q84_degrees": result["latitude_q"][2],
                            "recovered_width_q16_degrees": result["width_q"][0],
                            "recovered_width_median_degrees": result["width_q"][1],
                            "recovered_width_q84_degrees": result["width_q"][2],
                            "recovered_timing_q16_seconds": result["timing_q"][0],
                            "recovered_timing_median_seconds": result["timing_q"][1],
                            "recovered_timing_q84_seconds": result["timing_q"][2],
                            "best_width_degrees": result["best_width"],
                            "best_latitude_degrees": result["best_latitude"],
                            "best_timing_seconds": result["best_timing"],
                            "best_map_amplitude_scale": result["map_amplitude"],
                            "best_map_amplitude_ppm": result["map_amplitude"]
                            * float(target["depth_ppm"]),
                            "best_uniform_map_amplitude_scale": result[
                                "uniform_map_amplitude"
                            ],
                            "best_spot_amplitude_scale": result["spot_amplitude"],
                            "best_recovered_spot_fraction": result[
                                "recovered_spot_fraction"
                            ],
                            "uniform_bic": result["uniform_bic"],
                            "flexible_bic": result["flexible_bic"],
                            "delta_bic": result["delta_bic"],
                            "delta_bic_detection_threshold": DETECTION_DELTA_BIC,
                            "map_detected": bool(result["delta_bic"] < DETECTION_DELTA_BIC),
                            "best_map_minimum_intensity": result["map_minimum"],
                            "best_map_peak_longitude_degrees": result["peak_longitude"],
                            "best_map_peak_latitude_degrees": result["peak_latitude"],
                            "best_map_peak_shift_degrees": result["peak_shift"],
                            "best_map_positive": result["map_positive"],
                            "residual_rms_ppm": result["residual_rms_ppm"],
                            "random_seed": seed,
                        }
                        rows.append(row)
                        if fit_scope == "full_window":
                            # Store full-window profile arrays in NPZ only.
                            # CSV and the trial summaries contain both scopes.
                            profile_records.append(
                                {
                                    "run_id": row["run_id"],
                                    "target": name,
                                    "instrument": str(target["instrument"]),
                                    "teq_k": float(target["teq_k"]),
                                    "noise_ppm": sigma_ppm,
                                    "scenario": scenario,
                                    "trial": int(trial),
                                    "fit_scope": fit_scope,
                                    "log_likelihood": result["log_likelihood_values"],
                                    "profile_bic": result["profile_bic"],
                                    "posterior_weights": result["posterior_weights"],
                                }
                            )
                    if trial == 0:
                        for fit_scope, result in result_by_scope.items():
                            for row_index in range(time_days.size):
                                examples_by_target[name].append(
                                    {
                                        "target": name,
                                        "instrument": str(target["instrument"]),
                                        "teq_k": float(target["teq_k"]),
                                        "noise_ppm": sigma_ppm,
                                        "noise_input_ppm": float(noise_input),
                                        "scenario": scenario,
                                        "trial": trial,
                                        "fit_scope": fit_scope,
                                        "time_hours_from_eclipse": float(
                                            (time_days[row_index] - float(target.get("eclipse_time_days", 0.0)))
                                            * 24.0
                                        ),
                                        "orbital_phase_from_transit": float(
                                            (
                                                (time_days[row_index] - float(target.get("eclipse_time_days", 0.0)))
                                                / float(target["period_days"])
                                                + 0.5
                                            )
                                            % 1.0
                                        ),
                                        "in_eclipse_only_mask": bool(local_eclipse_mask[row_index]),
                                        "truth_flux_ppm": float(truth_signal[row_index]),
                                        "observed_flux_ppm": float(observed[row_index]),
                                        "sigma_ppm": sigma_ppm,
                                        "full_window_fit_flux_ppm": float(
                                            result_by_scope["full_window"]["predicted_full_ppm"][row_index]
                                        ),
                                        "eclipse_only_fit_flux_ppm": float(
                                            result_by_scope["eclipse_only"]["predicted_full_ppm"][row_index]
                                        ),
                                    }
                                )

        print(
            f"Completed target {target_index + 1}/{len(targets)}: {name}",
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    input_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path.resolve()),
        "input_snapshot": raw,
        "effective_input_assumptions": _effective_input_assumptions(raw, effective_trials),
        "resolved_analysis": analysis_resolved,
        "run_settings": {
            "trials_requested": int(trials),
            "trials_effective": int(effective_trials),
            "quick": bool(quick),
            "targets_effective": [str(item["name"]) for item in targets],
            "scenarios_effective": scenario_names,
            "noise_levels_effective_ppm": {
                str(item["name"]): _target_noise_levels(item, analysis)[0][:1]
                if quick
                else _target_noise_levels(item, analysis)[0]
                for item in targets
            },
            "master_seed": master_seed,
            "seed_provenance": SEED_PROVENANCE,
            "check_convergence": bool(check_convergence),
            "convergence_tolerance_ppm": float(convergence_tolerance_ppm),
        },
        "source_assumptions": {
            "offset_rule": "delta = atan(tan(40 degrees) * (1500 K / Teq)^3)",
            "offset_source": SHOWMAN_SOURCE,
            "offset_status": (
                "Illustrative radiative/advection scaling inspired by the published trend. "
                "The 1500 K normalization is a scenario choice. Values are not extracted "
                "Showman et al. GCM predictions."
            ),
            "map": (
                "Degree-2 map made from (1 - fraction) times a uniform map plus fraction "
                "times a projected spherical Gaussian. The default width is 40 degrees "
                "and the injected fraction is 0.8. Recovery fits separate uniform and "
                "hotspot amplitudes. A dense-sphere positivity bound limits their ratio. "
                "Map normalization uses the target's subobserver latitude."
            ),
            "noise": (
                "Independent white Gaussian noise at the configured noise_levels_ppm values. "
                "Values are per noise_reference_seconds (default 60 seconds) and are scaled "
                "by the square root of reference time over exposure time. The input file "
                "records the noise source and assumptions. No red-noise model is included."
            ),
            "visit": (
                "Symmetric continuous visit centered on the supplied eclipse time, with "
                "cadence and exposure from each target. Full-window and occulted-exposure-only "
                "fits are saved separately. The forward orbit uses transit t0 half a period "
                "before the eclipse center, and the subobserver latitude follows inclination."
            ),
            "limitations": [
                "Fixed circular orbit and supplied geometry; no orbit uncertainty.",
                "The injection and recovery use the same degree-2 Gaussian template family.",
                "Profile-grid screening with BIC weights; this is not full posterior inference.",
                "The uniform-null control has no injected hotspot and tests false map detections.",
                "Only selected scenarios are simulated; omitted scenarios are not run controls.",
            ],
        },
    }
    with (output / "run_inputs.json").open("w", encoding="utf-8") as handle:
        json.dump(input_payload, handle, indent=2, allow_nan=False)

    with (output / "recovery_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if profile_records:
        profile_log_likelihood = np.stack(
            [item["log_likelihood"] for item in profile_records]
        )
        profile_bic = np.stack([item["profile_bic"] for item in profile_records])
        posterior_weights = np.stack(
            [item["posterior_weights"] for item in profile_records]
        )
        if (
            not np.all(np.isfinite(profile_log_likelihood))
            or not np.all(np.isfinite(profile_bic))
            or not np.all(np.isfinite(posterior_weights))
            or np.any(posterior_weights < 0.0)
            or not np.allclose(
                np.sum(posterior_weights, axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12
            )
        ):
            raise RuntimeError(
                "Saved full-window profile arrays must be finite, and each weight row must sum to one"
            )
        profile_payload: dict[str, Any] = {
            "run_id": np.asarray([item["run_id"] for item in profile_records], dtype="U160"),
            "target": np.asarray([item["target"] for item in profile_records], dtype="U80"),
            "instrument": np.asarray([item["instrument"] for item in profile_records], dtype="U80"),
            "teq_k": np.asarray([item["teq_k"] for item in profile_records], dtype=float),
            "noise_ppm": np.asarray([item["noise_ppm"] for item in profile_records], dtype=float),
            "scenario": np.asarray([item["scenario"] for item in profile_records], dtype="U32"),
            "trial": np.asarray([item["trial"] for item in profile_records], dtype=np.int32),
            "fit_scope": np.asarray([item["fit_scope"] for item in profile_records], dtype="U24"),
            "candidate_longitude_degrees": candidate_longitude_array,
            "candidate_peak_longitude_degrees": candidate_peak_longitude_array,
            "candidate_latitude_degrees": candidate_latitude_array,
            "candidate_peak_latitude_degrees": candidate_peak_latitude_array,
            "candidate_width_degrees": candidate_width_array,
            "candidate_timing_seconds": candidate_timing_array,
            "profile_log_likelihood": profile_log_likelihood,
            "profile_bic": profile_bic,
            "posterior_weights": posterior_weights,
        }
        np.savez_compressed(output / "profiles.npz", **profile_payload)

    example_files: dict[str, str] = {}
    for target_name, example_rows in examples_by_target.items():
        path = output / "examples" / f"{_slug(target_name)}_curves.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(example_rows[0]) if example_rows else []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(example_rows)
        example_files[target_name] = str(path.relative_to(output))

    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": input_payload["created_utc"],
        "input_file": str(input_path.resolve()),
        "output_directory": str(output.resolve()),
        "target_count": len(targets),
        "trials_per_target_noise_scenario": int(effective_trials),
        "noise_levels_ppm": input_payload["run_settings"]["noise_levels_effective_ppm"],
        "scenario_names": scenario_names,
        "fit_scopes": list(FIT_SCOPES),
        "map_detection_delta_bic_threshold": DETECTION_DELTA_BIC,
        "source_assumptions": input_payload["source_assumptions"],
        "input_assumptions": raw.get("assumptions", {}),
        "effective_input_assumptions": _effective_input_assumptions(raw, effective_trials),
        "input_sources": raw.get("sources", []),
        "analysis": analysis_resolved,
        "targets": target_summaries,
        "trial_summaries": _summary_rows(rows),
        "numerical_convergence_checks": numerical_checks,
        "files": {
            "inputs": "run_inputs.json",
            "trials": "recovery_trials.csv",
            "profiles": "profiles.npz",
            "profiles_fit_scope": (
                "full_window only; eclipse_only profile intervals and BIC are in recovery_trials.csv"
            ),
            "profiles_npz_scopes": ["full_window"],
            "profiles_scope_note": (
                "profiles.npz stores full_window candidate likelihood, BIC, and weight arrays. "
                "Both full_window and eclipse_only intervals and BIC values are in "
                "recovery_trials.csv."
            ),
            "examples": example_files,
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return result


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not np.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="JSON target inputs")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--trials", type=_positive_int, default=30, help="Noise repeats per case (default: 30)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the first target and one repeat for each configured noise level and scenario",
    )
    parser.add_argument("--exposure-subsamples", type=_positive_int, default=8)
    parser.add_argument("--disk-radial", type=_positive_int, default=32)
    parser.add_argument("--disk-azimuth", type=_positive_int, default=128)
    parser.add_argument("--chunk-size", type=_positive_int, default=128, help="Observations per forward-operator chunk")
    parser.add_argument(
        "--check-convergence",
        action="store_true",
        help="Compare each target's injected curve with doubled quadrature counts",
    )
    parser.add_argument(
        "--convergence-tolerance-ppm",
        type=_nonnegative_float,
        default=0.5,
        help="Pass threshold for the maximum curve difference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_study(
        args.inputs,
        args.output,
        trials=args.trials,
        quick=args.quick,
        exposure_subsamples=args.exposure_subsamples,
        disk_radial=args.disk_radial,
        disk_azimuth=args.disk_azimuth,
        chunk_size=args.chunk_size,
        check_convergence=args.check_convergence,
        convergence_tolerance_ppm=args.convergence_tolerance_ppm,
    )
    print(
        f"UNITY recovery complete: {summary['target_count']} target(s), "
        f"{summary['trials_per_target_noise_scenario']} trial(s) per noise/scenario; "
        f"wrote {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
