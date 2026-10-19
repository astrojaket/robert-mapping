#!/usr/bin/env python3
"""Summarise a multi-draw UNITY noise-recovery ensemble.

The main plot mixes the saved, normalised profile weights from independent
noise draws with equal weight per draw. The companion plot shows the empirical
spread of the profile medians across those draws.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = ROOT / "UNITY_Theory"
REFERENCE_RUN_DIR = STUDY_ROOT / "reference" / "expected_noise_ensemble"
DEFAULT_INPUTS = STUDY_ROOT / "data" / "inputs" / "unity_expected_noise.json"
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "generated" / "expected_noise_ensemble"
STYLE_HELPER = Path(__file__).resolve().with_name("wasp15b_plot_style.py")
SCENARIO = "temperature_scaled"
SCOPES = ("full_window", "eclipse_only")
MODEL_COLOUR = "#6F3CC3"
DATA_COLOUR = "#111111"
OUTER_RANGE_COLOUR = "#777777"
Q1 = 0.158655
Q2 = 0.02275


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _analytic_longitude(teq_k: float) -> float:
    return math.degrees(math.atan(math.tan(math.radians(40.0)) * (1500.0 / teq_k) ** 3))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _find_targets(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        targets = value.get("targets")
        if isinstance(targets, list) and targets and all(
            isinstance(item, dict) and item.get("name") for item in targets
        ):
            return targets
        for item in value.values():
            found = _find_targets(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_targets(item)
            if found is not None:
                return found
    return None


def _read_targets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = _find_targets(payload)
    if targets is None or len(targets) != 6:
        found = 0 if targets is None else len(targets)
        raise ValueError(f"Expected six targets in {path}; found {found}")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in targets:
        name = str(item.get("name", "")).strip()
        teq = _number(item.get("teq_k"))
        noise = _number(item.get("noise_ppm"))
        if not name or name in names or teq is None or teq <= 0 or noise is None or noise <= 0:
            raise ValueError(f"Invalid target row in {path}: {item!r}")
        names.add(name)
        result.append(
            {
                "target": name,
                "teq_k": teq,
                "configured_noise_ppm": noise,
                "analytic_longitude_degrees": _analytic_longitude(teq),
            }
        )
    return result


def _normalise_draws(
    rows: list[dict[str, str]], targets: list[dict[str, Any]], expected_draws: int
) -> list[dict[str, Any]]:
    target_by_name = {item["target"]: item for item in targets}
    selected: dict[tuple[str, str, int], dict[str, str]] = {}
    trial_ids: set[int] = set()
    for raw in rows:
        if str(raw.get("scenario", "")).strip() != SCENARIO:
            continue
        target = str(raw.get("target", "")).strip()
        scope = str(raw.get("fit_scope", "")).strip()
        if target not in target_by_name or scope not in SCOPES:
            continue
        trial_raw = _number(raw.get("trial"))
        if trial_raw is None or not trial_raw.is_integer():
            raise ValueError(f"Invalid trial index for {target!r}, {scope}: {raw.get('trial')!r}")
        trial = int(trial_raw)
        trial_ids.add(trial)
        key = (target, scope, trial)
        if key in selected:
            raise ValueError(f"Duplicate ensemble row for target={target!r}, scope={scope}, trial={trial}")
        selected[key] = raw

    if len(trial_ids) != expected_draws:
        raise ValueError(
            f"Expected {expected_draws} distinct trial indices, found {len(trial_ids)}: "
            f"{sorted(trial_ids)[:12]}"
        )
    if trial_ids != set(range(expected_draws)):
        raise ValueError(f"Expected trial indices 0 through {expected_draws - 1}; found {sorted(trial_ids)}")

    result: list[dict[str, Any]] = []
    for target_info in targets:
        target = target_info["target"]
        for scope in SCOPES:
            for trial in range(expected_draws):
                raw = selected.get((target, scope, trial))
                if raw is None:
                    raise ValueError(f"Missing row for target={target!r}, scope={scope}, trial={trial}")
                median = _number(raw.get("recovered_longitude_median_degrees"))
                q16 = _number(raw.get("recovered_longitude_q16_degrees"))
                q84 = _number(raw.get("recovered_longitude_q84_degrees"))
                injected = _number(raw.get("injected_longitude_degrees"))
                if median is None or q16 is None or q84 is None or injected is None:
                    raise ValueError(
                        f"Missing recovered longitude or injected longitude for "
                        f"{target!r}, {scope}, trial {trial}"
                    )
                if q16 > q84 or median < q16 - 1e-8 or median > q84 + 1e-8:
                    raise ValueError(f"Invalid profile interval for {target!r}, {scope}, trial {trial}")
                teq = _number(raw.get("teq_k")) or target_info["teq_k"]
                if not math.isfinite(teq) or teq <= 0.0:
                    raise ValueError(f"Invalid equilibrium temperature for {target!r}, trial {trial}")
                result.append(
                    {
                        "target": target,
                        "teq_k": teq,
                        "configured_noise_ppm": target_info["configured_noise_ppm"],
                        "noise_input_ppm": _number(raw.get("noise_input_ppm")),
                        "noise_effective_ppm": _number(raw.get("noise_ppm")),
                        "scenario": SCENARIO,
                        "trial": trial,
                        "random_seed": _number(raw.get("random_seed")),
                        "fit_scope": scope,
                        "injected_longitude_degrees": injected,
                        "analytic_longitude_degrees": _analytic_longitude(teq),
                        "recovered_longitude_q16_degrees": q16,
                        "recovered_longitude_median_degrees": median,
                        "recovered_longitude_q84_degrees": q84,
                        "interval_halfwidth_degrees": 0.5 * (q84 - q16),
                        "map_detected": _boolean(raw.get("map_detected")),
                        "delta_bic": _number(raw.get("delta_bic")),
                        "residual_rms_ppm": _number(raw.get("residual_rms_ppm")),
                    }
                )
    return result


def _wrap_degrees(value: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(value, dtype=float) + 180.0) % 360.0 - 180.0


def _circular_mixture_ranges(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Aggregate the profile grid once, then calculate central circular quantiles."""
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    unique_values, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    total = float(np.sum(mass))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Profile mixture weights must have positive finite sum")
    mass = mass / total
    resultant = np.sum(mass * np.exp(1j * np.deg2rad(unique_values)))
    if abs(resultant) > 1.0e-12:
        anchor = float(np.rad2deg(np.angle(resultant)))
    else:
        anchor = float(unique_values[int(np.argmax(mass))])
    unwrapped = anchor + _wrap_degrees(unique_values - anchor)
    order = np.argsort(unwrapped)
    unwrapped = np.asarray(unwrapped, dtype=float)[order]
    mass = mass[order]
    centres = np.cumsum(mass) - 0.5 * mass

    def quantile(q: float) -> float:
        return float(_wrap_degrees(np.interp(q, centres, unwrapped)))

    return {
        "median": quantile(0.5),
        "q1_lower": quantile(Q1),
        "q1_upper": quantile(1.0 - Q1),
        "q2_lower": quantile(Q2),
        "q2_upper": quantile(1.0 - Q2),
    }


def _load_profile_mixtures(
    path: Path | None,
    targets: list[dict[str, Any]],
    expected_draws: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None or not path.is_file():
        return [], {"available": False, "reason": "profiles.npz was not found" if path else "not supplied"}

    required = {
        "target",
        "scenario",
        "trial",
        "fit_scope",
        "candidate_peak_longitude_degrees",
        "posterior_weights",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            return [], {"available": False, "reason": f"profiles.npz lacks fields: {sorted(missing)}"}
        names = archive["target"].astype(str)
        scenarios = archive["scenario"].astype(str)
        scopes = archive["fit_scope"].astype(str)
        trials = np.asarray(archive["trial"], dtype=int)
        candidate_peaks = np.asarray(archive["candidate_peak_longitude_degrees"], dtype=float)
        weights = np.asarray(archive["posterior_weights"], dtype=float)

    if candidate_peaks.ndim == 1:
        candidate_peaks = np.broadcast_to(candidate_peaks, weights.shape)
    if candidate_peaks.shape != weights.shape or weights.ndim != 2 or weights.shape[0] != len(names):
        raise ValueError("Profile candidate peak longitudes and weights have incompatible shapes")
    if not np.all(np.isfinite(candidate_peaks)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Profile candidates and weights must be finite, and weights must be non-negative")
    row_sums = np.sum(weights, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-9, atol=1e-11):
        raise ValueError("Each saved profile weight row must sum to one")

    result: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    by_name = {item["target"]: item for item in targets}
    for target_info in targets:
        target = target_info["target"]
        selected = np.flatnonzero((names == target) & (scenarios == SCENARIO) & (scopes == "full_window"))
        selected = selected[np.argsort(trials[selected])]
        selected_trials = trials[selected]
        if selected.size != expected_draws or not np.array_equal(selected_trials, np.arange(expected_draws)):
            raise ValueError(
                f"Expected {expected_draws} full_window profile rows for {target}; "
                f"found trials {selected_trials[:12].tolist()} (n={selected.size})"
            )
        mix_values = candidate_peaks[selected]
        # Give each noise realization equal total mass, then retain its saved
        # normalized BIC-weighted distribution over nuisance candidates.
        mix_weights = weights[selected] / expected_draws
        quantiles = _circular_mixture_ranges(mix_values, mix_weights)
        summary = {
            "target": target,
            "fit_scope": "full_window",
            "teq_k": target_info["teq_k"],
            "n_noise_draws": int(selected.size),
            "n_profile_candidates_per_draw": int(weights.shape[1]),
            "mixture_median_degrees": quantiles["median"],
            "mixture_q1_lower_degrees": quantiles["q1_lower"],
            "mixture_q1_upper_degrees": quantiles["q1_upper"],
            "mixture_q2_lower_degrees": quantiles["q2_lower"],
            "mixture_q2_upper_degrees": quantiles["q2_upper"],
            "mixture_definition": (
                "Equal-weight mixture across noise draws of each draw's normalized approximate "
                "BIC-weighted profile distribution, using candidate_peak_longitude_degrees. "
                "Includes conditional profile-fit uncertainty and noise variation; not full posterior sampling."
            ),
        }
        result.append(summary)
        counts[target] = int(selected.size)
    return result, {
        "available": True,
        "path": str(path),
        "profile_rows_by_target": counts,
        "fit_scope": "full_window",
        "weight_field": "posterior_weights (saved approximate BIC-weighted profile weights)",
        "longitude_field": "candidate_peak_longitude_degrees",
        "mixture_rule": "Each noise draw has equal total mixture weight; its normalized profile weights are retained.",
        "interpretation": (
            "The mixture includes conditional profile-fit uncertainty as well as noise variation. "
            "It is not full posterior sampling and does not represent repeated real visits combined."
        ),
    }


def _central_range(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.quantile(values, 0.5)),
        "q1_lower": float(np.quantile(values, Q1)),
        "q1_upper": float(np.quantile(values, 1.0 - Q1)),
        "q2_lower": float(np.quantile(values, Q2)),
        "q2_upper": float(np.quantile(values, 1.0 - Q2)),
    }


def _target_summaries(
    targets: list[dict[str, Any]], results: list[dict[str, Any]], expected_draws: int
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for target_info in targets:
        target = target_info["target"]
        for scope in SCOPES:
            rows = [row for row in results if row["target"] == target and row["fit_scope"] == scope]
            values = np.asarray([row["recovered_longitude_median_degrees"] for row in rows], dtype=float)
            centre = _central_range(values)
            detected = [row["map_detected"] for row in rows]
            detections_known = [value for value in detected if value is not None]
            injected = np.asarray([row["injected_longitude_degrees"] for row in rows], dtype=float)
            input_noise = np.asarray(
                [row["noise_input_ppm"] for row in rows if row["noise_input_ppm"] is not None], dtype=float
            )
            effective_noise = np.asarray(
                [row["noise_effective_ppm"] for row in rows if row["noise_effective_ppm"] is not None], dtype=float
            )
            summaries.append(
                {
                    "target": target,
                    "fit_scope": scope,
                    "teq_k": target_info["teq_k"],
                    "configured_noise_ppm": target_info["configured_noise_ppm"],
                    "noise_input_ppm_median": None if input_noise.size == 0 else float(np.median(input_noise)),
                    "noise_effective_ppm_median": None if effective_noise.size == 0 else float(np.median(effective_noise)),
                    "expected_draws": int(expected_draws),
                    "valid_recovered_medians": int(values.size),
                    "injected_longitude_mean_degrees": float(np.mean(injected)),
                    "analytic_longitude_degrees": target_info["analytic_longitude_degrees"],
                    "recovered_median_across_draws_degrees": centre["median"],
                    "recovered_mean_across_draws_degrees": float(np.mean(values)),
                    "recovered_scatter_sd_across_draws_degrees": float(np.std(values, ddof=0)),
                    "mean_bias_vs_injected_degrees": float(np.mean(values - injected)),
                    "median_bias_vs_injected_degrees": float(np.median(values - injected)),
                    "q1_lower_degrees": centre["q1_lower"],
                    "q1_upper_degrees": centre["q1_upper"],
                    "q2_lower_degrees": centre["q2_lower"],
                    "q2_upper_degrees": centre["q2_upper"],
                    "map_detection_count": int(sum(value is True for value in detections_known)),
                    "map_detection_known_count": int(len(detections_known)),
                    "map_detection_fraction": (
                        None if len(detections_known) == 0 else sum(value is True for value in detections_known) / len(detections_known)
                    ),
                }
            )
    return summaries


def _unweighted_slope(temperature: np.ndarray, value: np.ndarray) -> float:
    x = (np.asarray(temperature, dtype=float) - float(np.mean(temperature))) / 1000.0
    design = np.column_stack((np.ones(x.size), x))
    estimate, *_ = np.linalg.lstsq(design, np.asarray(value, dtype=float), rcond=None)
    return float(estimate[1])


def _weighted_slope(temperature: np.ndarray, value: np.ndarray, halfwidth: np.ndarray) -> dict[str, float] | None:
    temperature = np.asarray(temperature, dtype=float)
    value = np.asarray(value, dtype=float)
    sigma = np.asarray(halfwidth, dtype=float)
    usable = np.isfinite(temperature) & np.isfinite(value) & np.isfinite(sigma) & (sigma > 0.0)
    if int(np.sum(usable)) < 3:
        return None
    temperature = temperature[usable]
    value = value[usable]
    sigma = sigma[usable]
    weight = 1.0 / np.square(sigma)
    centre = float(np.average(temperature, weights=weight))
    x = (temperature - centre) / 1000.0
    design = np.column_stack((np.ones(x.size), x))
    precision = design.T @ (weight[:, None] * design)
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return None
    estimate = covariance @ (design.T @ (weight * value))
    slope_error = float(np.sqrt(max(0.0, covariance[1, 1])))
    if not math.isfinite(slope_error) or slope_error <= 0.0:
        return None
    return {
        "slope_deg_per_1000k": float(estimate[1]),
        "slope_error_deg_per_1000k": slope_error,
        "n_targets": float(x.size),
    }


def _trend_summaries(
    targets: list[dict[str, Any]], results: list[dict[str, Any]], expected_draws: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_order = [item["target"] for item in targets]
    by_target = {item["target"]: item for item in targets}
    rows_out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": 1,
        "expected_draws": int(expected_draws),
        "quantile_probabilities": {
            "one_sigma_lower": Q1,
            "one_sigma_upper": 1.0 - Q1,
            "two_sigma_lower": Q2,
            "two_sigma_upper": 1.0 - Q2,
        },
        "slope_units": "degrees per 1000 K",
        "scopes": {},
    }
    for scope in SCOPES:
        scope_results = [row for row in results if row["fit_scope"] == scope]
        by_trial: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in scope_results:
            by_trial[int(row["trial"])][row["target"]] = row
        trial_slopes: list[dict[str, Any]] = []
        temperatures = np.asarray([by_target[name]["teq_k"] for name in target_order], dtype=float)
        input_values = np.asarray(
            [
                float(np.mean([row["injected_longitude_degrees"] for row in scope_results if row["target"] == name]))
                for name in target_order
            ],
            dtype=float,
        )
        input_slope = _unweighted_slope(temperatures, input_values)
        for trial in range(expected_draws):
            group = by_trial.get(trial, {})
            if set(group) != set(target_order):
                raise ValueError(f"Trial {trial}, {scope} does not contain all six targets")
            values = np.asarray([group[name]["recovered_longitude_median_degrees"] for name in target_order], dtype=float)
            halfwidths = np.asarray([group[name]["interval_halfwidth_degrees"] for name in target_order], dtype=float)
            unweighted = _unweighted_slope(temperatures, values)
            weighted = _weighted_slope(temperatures, values, halfwidths)
            row = {
                "trial": trial,
                "fit_scope": scope,
                "n_targets": len(target_order),
                "input_unweighted_slope_deg_per_1000k": input_slope,
                "recovered_unweighted_slope_deg_per_1000k": unweighted,
                "recovered_unweighted_slope_error_from_input_deg_per_1000k": unweighted - input_slope,
                "recovered_weighted_slope_deg_per_1000k": None if weighted is None else weighted["slope_deg_per_1000k"],
                "recovered_weighted_slope_error_deg_per_1000k": None if weighted is None else weighted["slope_error_deg_per_1000k"],
                "weighted_n_targets": 0 if weighted is None else int(weighted["n_targets"]),
                "weighted_fit_halfwidth_source": "0.5 * (recovered q84 - recovered q16) per target and realization",
            }
            rows_out.append(row)
            trial_slopes.append(row)

        scope_summary: dict[str, Any] = {}
        for method, field in (
            ("unweighted", "recovered_unweighted_slope_deg_per_1000k"),
            ("weighted", "recovered_weighted_slope_deg_per_1000k"),
        ):
            values = np.asarray(
                [row[field] for row in trial_slopes if row[field] is not None and math.isfinite(row[field])], dtype=float
            )
            if values.size == 0:
                scope_summary[method] = {"n_realizations": 0, "available": False}
                continue
            interval = _central_range(values)
            scope_summary[method] = {
                "n_realizations": int(values.size),
                "median_slope_deg_per_1000k": interval["median"],
                "one_sigma_range_deg_per_1000k": [interval["q1_lower"], interval["q1_upper"]],
                "two_sigma_range_deg_per_1000k": [interval["q2_lower"], interval["q2_upper"]],
                "fraction_negative": float(np.mean(values < 0.0)),
                "negative_realization_count": int(np.sum(values < 0.0)),
                "available": True,
            }
        scope_summary["input_unweighted_slope_deg_per_1000k"] = input_slope
        scope_summary["interpretation"] = (
            "Ranges and negative fractions describe finite noise realizations. The weighted fit uses "
            "each target's per-realization profile-interval halfwidth; this is a descriptive summary, "
            "not a posterior detection probability."
        )
        summary["scopes"][scope] = scope_summary
    return rows_out, summary


def _apply_style() -> Any:
    if not STYLE_HELPER.is_file():
        raise FileNotFoundError(f"WASP-15b plot-style helper was not found: {STYLE_HELPER}")
    spec = importlib.util.spec_from_file_location("wasp15b_plot_style", STYLE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plot-style helper: {STYLE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_style()
    return module


def _annotation_offsets(targets: list[dict[str, Any]]) -> dict[str, tuple[float, float, str, str]]:
    # Paired offsets keep the close-temperature systems apart in the compact panel.
    offsets = {
        "HAT-P-1 b": (5.0, 8.0, "left", "bottom"),
        "HAT-P-30 b": (-5.0, -10.0, "right", "top"),
        "HAT-P-13 b": (5.0, 14.0, "left", "bottom"),
        "KELT-7 b": (5.0, 8.0, "left", "bottom"),
        "WASP-178 b": (10.0, -22.0, "left", "top"),
        "WASP-12 b": (10.0, 22.0, "left", "bottom"),
    }
    return {item["target"]: offsets.get(item["target"], (4.0, 5.0, "left", "bottom")) for item in targets}


def _plot(
    output_dir: Path,
    targets: list[dict[str, Any]],
    empirical: list[dict[str, Any]],
    mixture: list[dict[str, Any]],
    *,
    kind: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    style = _apply_style()
    if kind == "marginalised":
        rows_by_name = {row["target"]: row for row in mixture}
        lower1, upper1 = "mixture_q1_lower_degrees", "mixture_q1_upper_degrees"
        lower2, upper2 = "mixture_q2_lower_degrees", "mixture_q2_upper_degrees"
        centre = "mixture_median_degrees"
        point_label = "Noise-averaged recovery"
        output_stem = "unity_expected_noise_marginalised"
    else:
        rows_by_name = {
            row["target"]: row
            for row in empirical
            if row["fit_scope"] == "full_window"
        }
        lower1, upper1 = "q1_lower_degrees", "q1_upper_degrees"
        lower2, upper2 = "q2_lower_degrees", "q2_upper_degrees"
        centre = "recovered_median_across_draws_degrees"
        point_label = "Median across noise draws"
        output_stem = "unity_expected_noise_ensemble"

    sorted_targets = sorted(targets, key=lambda item: item["teq_k"])
    temperature = np.asarray([item["teq_k"] for item in sorted_targets], dtype=float)
    x_min = float(np.min(temperature)) - 90.0
    x_max = float(np.max(temperature)) + 430.0
    x_curve = np.linspace(x_min, x_max, 600)
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.97)
    ax.set_box_aspect(1)
    ax.plot(
        x_curve,
        [_analytic_longitude(value) for value in x_curve],
        color=MODEL_COLOUR,
        linewidth=1.2,
        zorder=1,
    )

    offsets = _annotation_offsets(sorted_targets)
    for target in sorted_targets:
        name = target["target"]
        row = rows_by_name[name]
        x = float(target["teq_k"])
        y = float(row[centre])
        y1_low, y1_high = float(row[lower1]), float(row[upper1])
        y2_low, y2_high = float(row[lower2]), float(row[upper2])
        # The outer range is light and thin. The inner range and median are dark.
        cap_halfwidth = 5.2
        ax.vlines(x, y2_low, y2_high, color=OUTER_RANGE_COLOUR, linewidth=0.65, alpha=0.9, zorder=2)
        ax.hlines([y2_low, y2_high], x - cap_halfwidth, x + cap_halfwidth,
                  color=OUTER_RANGE_COLOUR, linewidth=0.55, alpha=0.9, zorder=2)
        ax.vlines(x, y1_low, y1_high, color=DATA_COLOUR, linewidth=0.95, zorder=3)
        ax.hlines([y1_low, y1_high], x - cap_halfwidth * 0.65, x + cap_halfwidth * 0.65,
                  color=DATA_COLOUR, linewidth=0.8, zorder=3)
        ax.plot(
            x,
            y,
            marker="o",
            linestyle="none",
            markersize=3.7,
            markerfacecolor=DATA_COLOUR,
            markeredgecolor=DATA_COLOUR,
            markeredgewidth=0.6,
            zorder=4,
        )
        dx, dy, ha, va = offsets[name]
        ax.annotate(
            name,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.0,
            color=DATA_COLOUR,
            annotation_clip=False,
            zorder=5,
        )

    ax.set_xlim(x_min, x_max)
    all_low = [float(row[lower2]) for row in rows_by_name.values()]
    all_high = [float(row[upper2]) for row in rows_by_name.values()]
    curve_y = [_analytic_longitude(value) for value in x_curve]
    y_min = min(all_low + curve_y)
    y_max = max(all_high + curve_y)
    y_pad = max(2.0, 0.06 * (y_max - y_min))
    ax.set_ylim(max(-5.0, y_min - y_pad), min(95.0, y_max + y_pad))
    ax.set_xlabel("Equilibrium temperature (K)", labelpad=4.0)
    ax.set_ylabel("Eastward hotspot longitude (°)", labelpad=4.5)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=DATA_COLOUR, markersize=3.7, label=point_label),
        Line2D([], [], color=DATA_COLOUR, linewidth=0.9, label="1σ range"),
        Line2D([], [], color=OUTER_RANGE_COLOUR, linewidth=0.65, label="2σ range"),
        Line2D([], [], color=MODEL_COLOUR, linewidth=1.2, label="Injected relation"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=1,
        frameon=False,
        borderaxespad=0.35,
        borderpad=0.15,
        labelspacing=0.22,
        handletextpad=0.45,
        handlelength=1.4,
    )
    style.style_axis(ax)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_artists = [artist for artist in ax.texts if artist.get_text() in offsets]
    axes_box = ax.get_window_extent(renderer)
    point_adjustments: dict[str, tuple[float, float]] = {}
    # Keep direct labels inside the panel at the final display size.
    for artist in label_artists:
        offset_x, offset_y = artist.get_position()
        for _ in range(4):
            box = Text.get_window_extent(artist, renderer)
            shift_x = 0.0
            shift_y = 0.0
            if box.x0 < axes_box.x0 + 1.0:
                shift_x += axes_box.x0 + 1.0 - box.x0
            if box.x1 > axes_box.x1 - 1.0:
                shift_x -= box.x1 - (axes_box.x1 - 1.0)
            if box.y0 < axes_box.y0 + 1.0:
                shift_y += axes_box.y0 + 1.0 - box.y0
            if box.y1 > axes_box.y1 - 1.0:
                shift_y -= box.y1 - (axes_box.y1 - 1.0)
            if abs(shift_x) < 0.25 and abs(shift_y) < 0.25:
                break
            offset_x += shift_x * 72.0 / fig.dpi
            offset_y += shift_y * 72.0 / fig.dpi
            artist.set_position((offset_x, offset_y))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        final_box = Text.get_window_extent(artist, renderer)
        if (
            final_box.x0 < axes_box.x0
            or final_box.x1 > axes_box.x1
            or final_box.y0 < axes_box.y0
            or final_box.y1 > axes_box.y1
        ):
            plt.close(fig)
            raise ValueError(f"Target label is outside the plot panel ({kind}): {artist.get_text()}")
        point_adjustments[artist.get_text()] = (float(offset_x), float(offset_y))

    def _label_boxes() -> list[tuple[str, Any]]:
        return [
            (artist.get_text(), Text.get_window_extent(artist, renderer).expanded(1.03, 1.08))
            for artist in label_artists
        ]

    # If data-driven shifts make two labels touch, move one a short distance
    # up or down while keeping it inside the axes and clear of all labels.
    for _ in range(12):
        boxes = _label_boxes()
        overlap_indices: tuple[int, int] | None = None
        for index, (_, box_a) in enumerate(boxes):
            for other_index in range(index + 1, len(boxes)):
                if box_a.overlaps(boxes[other_index][1]):
                    overlap_indices = (index, other_index)
                    break
            if overlap_indices:
                break
        if overlap_indices is None:
            break
        _, moving_index = overlap_indices
        moving = label_artists[moving_index]
        old_x, old_y = moving.get_position()
        moved = False
        for delta_y in (12.0, -12.0, 18.0, -18.0, 24.0, -24.0, 30.0, -30.0):
            moving.set_position((old_x, old_y + delta_y))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            moving_box = Text.get_window_extent(moving, renderer).expanded(1.03, 1.08)
            inside = (
                moving_box.x0 >= axes_box.x0
                and moving_box.x1 <= axes_box.x1
                and moving_box.y0 >= axes_box.y0
                and moving_box.y1 <= axes_box.y1
            )
            clear = all(
                not moving_box.overlaps(Text.get_window_extent(artist, renderer).expanded(1.03, 1.08))
                for index, artist in enumerate(label_artists)
                if index != moving_index
            )
            if inside and clear:
                point_adjustments[moving.get_text()] = (float(old_x), float(old_y + delta_y))
                moved = True
                break
        if not moved:
            moving.set_position((old_x, old_y))
            break
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_boxes = _label_boxes()
    label_overlaps: list[tuple[str, str]] = []
    for index, (name_a, box_a) in enumerate(label_boxes):
        for name_b, box_b in label_boxes[index + 1 :]:
            if box_a.overlaps(box_b):
                label_overlaps.append((name_a, name_b))
    if label_overlaps:
        plt.close(fig)
        raise ValueError(f"Target labels overlap in the final-size plot ({kind}): {label_overlaps}")

    # Use short vertical leaders for the two close-temperature WASP targets.
    # Add them after label placement so the line ends at the final text offset.
    target_by_name = {item["target"]: item for item in targets}
    for artist in label_artists:
        name = artist.get_text()
        if name not in {"WASP-178 b", "WASP-12 b"}:
            continue
        offset = artist.get_position()
        ax.annotate(
            "",
            (float(target_by_name[name]["teq_k"]), float(rows_by_name[name][centre])),
            xytext=offset,
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": "#666666",
                "linewidth": 0.45,
                "shrinkA": 1.0,
                "shrinkB": 2.5,
                "connectionstyle": "arc3,rad=0",
            },
            annotation_clip=False,
            zorder=3.6,
        )

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{output_stem}.{suffix}", dpi=600)
    metadata = {
        "plot": output_stem,
        "canvas_inches": [3.5, 3.5],
        "saved_with_tight_bbox": False,
        "target_labels": [artist.get_text() for artist in label_artists],
        "label_offsets_points": point_adjustments,
        "label_overlap_pairs": label_overlaps,
        "label_overlap_check": "Passed on the 3.5-inch square canvas using the Agg renderer; labels are inside the plot panel.",
    }
    plt.close(fig)
    return metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    trials_path: Path,
    inputs_path: Path,
    profiles_path: Path | None,
    expected_draws: int,
    target_summaries: list[dict[str, Any]],
    profile_summary: dict[str, Any],
    trend_summary: dict[str, Any],
) -> None:
    lines = [
        "UNITY expected-noise ensemble summary",
        f"Recovery trials CSV: {trials_path}",
        f"Input JSON: {inputs_path}",
        f"Profile arrays: {profiles_path if profiles_path else 'not supplied'}",
        f"Independent noise draws per target: {expected_draws}",
        "",
        "Empirical ranges are across recovered profile medians from noise realizations. They are not posterior intervals and are not standard errors of the mean.",
        "All realizations are included, whether or not the map-detection flag is true.",
        "The simulation uses fixed geometry and depth, approximate blackbody depths, target-specific PandExo white noise only (29.6 to 89.9 ppm per 60 seconds), profile fits rather than full posterior sampling, and an illustrative Showman-inspired normalization rather than exact GCM predictions.",
    ]
    if profile_summary.get("available"):
        lines.extend(
            (
                "",
                "The primary marginalised plot and equal-weight profile-mixture CSV combine each draw's normalized approximate BIC-weighted profile distribution with equal weight per noise draw.",
                "This profile mixture includes conditional profile-fit uncertainty as well as noise variation. It is not full Bayesian posterior sampling and does not describe combining multiple real visits.",
            )
        )
    else:
        lines.extend(("", f"Profile mixture unavailable: {profile_summary.get('reason', 'not available')}"))

    for scope in SCOPES:
        lines.extend(("", f"{scope} target summaries:"))
        for row in target_summaries:
            if row["fit_scope"] != scope:
                continue
            lines.append(
                f"- {row['target']}: median={row['recovered_median_across_draws_degrees']:.2f} deg; "
                f"1σ range=[{row['q1_lower_degrees']:.2f}, {row['q1_upper_degrees']:.2f}] deg; "
                f"2σ range=[{row['q2_lower_degrees']:.2f}, {row['q2_upper_degrees']:.2f}] deg; "
                f"scatter={row['recovered_scatter_sd_across_draws_degrees']:.2f} deg; "
                f"mean bias={row['mean_bias_vs_injected_degrees']:.2f} deg; "
                f"map detections={row['map_detection_count']}/{row['map_detection_known_count']}"
            )
        trend = trend_summary["scopes"][scope]
        lines.append(
            "Unweighted recovered trend: "
            f"median={trend['unweighted']['median_slope_deg_per_1000k']:.3f} deg per 1000 K; "
            f"1σ range={trend['unweighted']['one_sigma_range_deg_per_1000k']}; "
            f"2σ range={trend['unweighted']['two_sigma_range_deg_per_1000k']}; "
            f"input-relation slope={trend['input_unweighted_slope_deg_per_1000k']:.3f} deg per 1000 K; "
            f"negative fraction={trend['unweighted']['fraction_negative']:.3f}."
        )
        weighted = trend["weighted"]
        if weighted.get("available"):
            lines.append(
                "Weighted recovered trend: "
                f"median={weighted['median_slope_deg_per_1000k']:.3f} deg per 1000 K; "
                f"1σ range={weighted['one_sigma_range_deg_per_1000k']}; "
                f"2σ range={weighted['two_sigma_range_deg_per_1000k']}; "
                f"negative fraction={weighted['fraction_negative']:.3f}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=REFERENCE_RUN_DIR / "recovery_trials.csv")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--profiles", type=Path, default=REFERENCE_RUN_DIR / "profiles.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--expected-draws", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.expected_draws <= 0:
        raise ValueError("--expected-draws must be a positive integer")
    if not args.trials.is_file():
        raise FileNotFoundError(f"Recovery-trials CSV was not found: {args.trials}")
    if not args.inputs.is_file():
        raise FileNotFoundError(f"Input JSON was not found: {args.inputs}")

    targets = _read_targets(args.inputs)
    results = _normalise_draws(_read_csv(args.trials), targets, args.expected_draws)
    empirical = _target_summaries(targets, results, args.expected_draws)
    mixtures, mixture_meta = _load_profile_mixtures(args.profiles, targets, args.expected_draws)
    trend_rows, trend_summary = _trend_summaries(targets, results, args.expected_draws)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "recovery_draws.csv", results)
    _write_csv(args.output_dir / "target_summary.csv", empirical)
    _write_csv(args.output_dir / "trend_per_realization.csv", trend_rows)
    if mixtures:
        _write_csv(args.output_dir / "equal_weight_profile_mixture.csv", mixtures)
    (args.output_dir / "trend_summary.json").write_text(
        json.dumps(trend_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    plot_meta: dict[str, Any] = {}
    if mixtures:
        plot_meta["marginalised"] = _plot(args.output_dir, targets, empirical, mixtures, kind="marginalised")
    plot_meta["ensemble"] = _plot(args.output_dir, targets, empirical, mixtures, kind="ensemble")
    metadata = {
        "schema_version": 1,
        "trials_csv": str(args.trials),
        "inputs_json": str(args.inputs),
        "profiles_npz": str(args.profiles) if args.profiles and args.profiles.is_file() else None,
        "expected_draws_per_target_and_scope": args.expected_draws,
        "target_count": len(targets),
        "fit_scopes": list(SCOPES),
        "all_draws_included_regardless_of_map_detection": True,
        "quantile_probabilities": {
            "one_sigma_lower": Q1,
            "one_sigma_upper": 1.0 - Q1,
            "two_sigma_lower": Q2,
            "two_sigma_upper": 1.0 - Q2,
        },
        "empirical_ranges": (
            "Central empirical ranges across recovered profile medians from independent noise draws. "
            "Not posterior intervals and not standard errors of the mean."
        ),
        "profile_mixture": mixture_meta,
        "plots": plot_meta,
        "limitations": [
            "Fixed geometry and depth.",
            "Approximate blackbody band depths.",
            "Target-specific PandExo white-noise only.",
            "Profile fits are not full posterior sampling.",
            "Illustrative Showman-inspired normalization, not exact GCM predictions.",
        ],
    }
    (args.output_dir / "ensemble_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(
        args.output_dir / "ensemble_report.txt",
        args.trials,
        args.inputs,
        args.profiles if args.profiles and args.profiles.is_file() else None,
        args.expected_draws,
        empirical,
        mixture_meta,
        trend_summary,
    )
    captions = {
        "caption_marginalised.txt": (
            f"UNITY expected-noise recovery under the temperature-scaled injected relation. Black points show the median of the equal-weight mixture of each independent noise draw's normalized approximate BIC-weighted profile distribution; inner and outer bars show the central 1σ and 2σ ranges. The purple curve is the analytic injected relation. The finite {args.expected_draws}-draw mixture includes conditional profile-fit uncertainty and noise variation; its 2σ tails are simulation estimates. It is not a full Bayesian posterior and does not describe combining multiple real visits. The simulation uses fixed geometry and depth, approximate blackbody depths, target-specific PandExo white noise only, profile fits rather than full posterior sampling, and an illustrative Showman-inspired normalization rather than exact GCM predictions.\n"
        ),
        "caption_ensemble.txt": (
            f"UNITY expected-noise recovery under the temperature-scaled injected relation. Black points show the median of {args.expected_draws} recovered profile medians per target; inner and outer bars show the central 1σ and 2σ empirical ranges across independent noise realizations. The purple curve is the analytic injected relation. These empirical noise-realization ranges are simulation estimates, not posterior intervals or standard errors of the mean; the 2σ tails are limited by the finite {args.expected_draws}-draw ensemble. The simulation uses fixed geometry and depth, approximate blackbody depths, target-specific PandExo white noise only, profile fits rather than full posterior sampling, and an illustrative Showman-inspired normalization rather than exact GCM predictions.\n"
        ),
    }
    for filename, caption in captions.items():
        (args.output_dir / filename).write_text(caption, encoding="utf-8")
    full_summary = [row for row in empirical if row["fit_scope"] == "full_window"]
    eclipse_summary = [row for row in empirical if row["fit_scope"] == "eclipse_only"]
    full_detections = sum(row["map_detection_count"] for row in full_summary)
    full_detection_known = sum(row["map_detection_known_count"] for row in full_summary)
    eclipse_detections = sum(row["map_detection_count"] for row in eclipse_summary)
    eclipse_detection_known = sum(row["map_detection_known_count"] for row in eclipse_summary)
    target_by_name = {item["target"]: item for item in targets}
    mixture_truth_count = sum(
        row["mixture_q1_lower_degrees"]
        <= target_by_name[row["target"]]["analytic_longitude_degrees"]
        <= row["mixture_q1_upper_degrees"]
        for row in mixtures
    )
    full_trend = trend_summary["scopes"]["full_window"]["unweighted"]
    full_input_slope = trend_summary["scopes"]["full_window"]["input_unweighted_slope_deg_per_1000k"]
    methodology = (
        "We simulated eclipse-centred JWST/NIRSpec G395M/G395H visits for six hot Jupiters. We used fixed circular geometry, fixed depths, and ideal 60-second samples. We set each band depth from a photon-weighted blackbody over 2.9–5.1 µm, with brightness temperature at 1.1 times the equilibrium temperature. We set target-specific white noise from saved PandExo calculations (29.6–89.9 ppm per 60 seconds) and generated "
        f"{args.expected_draws} independent draws. We fitted each draw over the full visit and the eclipse-only samples with the saved profile grid for longitude, width, latitude, and timing. We averaged the normalized approximate BIC-weighted profile distributions with equal weight per draw. This mixture includes profile-fit uncertainty and noise variation; it is not a full posterior or a combination of repeated visits. The injected relation was δ = atan[tan(40°)(1500 K / Teq)^3], a Showman-inspired illustration with a chosen normalization, not an exact GCM prediction. The full-visit recovered slope median was {full_trend['median_slope_deg_per_1000k']:.1f} degrees per 1000 K; the injected-relation slope was {full_input_slope:.1f}, and the recovered slope was negative in {full_trend['negative_realization_count']} of {full_trend['n_realizations']} draws. The map-detection flag was true in {full_detections} of {full_detection_known} full-visit fits and {eclipse_detections} of {eclipse_detection_known} eclipse-only fits. All {mixture_truth_count} injected longitudes fell within the mixture's 1σ range. These are simulation results, not formal false-alarm probabilities.\n"
    )
    (args.output_dir / "methodology.txt").write_text(methodology, encoding="utf-8")
    print(f"Wrote ensemble summary and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
