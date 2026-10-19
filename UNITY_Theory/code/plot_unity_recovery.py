#!/usr/bin/env python3
"""Plot UNITY injection and recovery results and summarize population trends.

The per-target fits use profile grids.  This tool does not fit a population
trend during those fits.  It fits the trend after recovery, using the six
recovered medians for each noise, injection, scope, and trial group.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = ROOT / "UNITY_Theory"
STYLE_HELPER = Path(__file__).resolve().with_name("wasp15b_plot_style.py")
DEFAULT_OUTPUT = STUDY_ROOT / "generated" / "injection_recovery"
DEFAULT_INPUTS = STUDY_ROOT / "data" / "inputs" / "unity_inputs.json"
REFERENCE_RUN_DIR = STUDY_ROOT / "reference" / "study"
DEFAULT_TRIALS = REFERENCE_RUN_DIR / "recovery_trials.csv"
DEFAULT_RUN_INPUTS = REFERENCE_RUN_DIR / "run_inputs.json"
DEFAULT_SUMMARY = REFERENCE_RUN_DIR / "summary.json"
DEFAULT_PROFILES = REFERENCE_RUN_DIR / "profiles.npz"
SCENARIO_ORDER = (
    "temperature_scaled",
    "zero_offset",
    "constant_30_deg",
    "uniform_null",
)
SCENARIO_LABELS = {
    "temperature_scaled": "Temperature-scaled injection",
    "zero_offset": "Zero-degree hotspot",
    "constant_30_deg": "Constant +30-degree hotspot",
    "uniform_null": "Uniform-planet null",
}
SCOPE_ORDER = ("full_window", "eclipse_only")
MODEL_COLOUR = "#6F3CC3"
DATA_COLOUR = "#111111"
_STYLE_MODULE: Any | None = None
TREND_REFERENCE_OFFSET_DEGREES = 40.0
TREND_REFERENCE_TEMPERATURE_K = 1500.0


def _number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value is None or str(value).strip() == "":
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def _truth(row: dict[str, Any], *names: str) -> bool | None:
    for name in names:
        value = row.get(name)
        if value is None or str(value).strip() == "":
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _find_key(value: Any, names: set[str]) -> Any | None:
    """Find the first named key in nested JSON without depending on its shape."""

    if isinstance(value, dict):
        for key, item in value.items():
            if key in names and item is not None:
                return item
        for item in value.values():
            found = _find_key(item, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_key(item, names)
            if found is not None:
                return found
    return None


def _candidate_grid_step(path: Path) -> float | None:
    """Read a longitude grid step from the saved profile archive, if available."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                if "candidate_longitude" not in key.lower():
                    continue
                values = np.asarray(archive[key], dtype=float).ravel()
                values = np.unique(values[np.isfinite(values)])
                differences = np.diff(values)
                differences = differences[differences > 1.0e-10]
                if differences.size:
                    return float(np.median(differences))
    except (OSError, ValueError, TypeError):
        return None
    return None


def _target_order(input_data: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    configured = input_data.get("targets", [])
    names = [str(item.get("name")) for item in configured if isinstance(item, dict) and item.get("name")]
    for row in rows:
        name = str(row.get("target", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def _target_metadata(input_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for target in input_data.get("targets", []):
        if isinstance(target, dict) and target.get("name"):
            result[str(target["name"])] = target
    return result


def _normalise_rows(
    raw_rows: list[dict[str, str]],
    input_data: dict[str, Any],
    detection_threshold: float,
) -> list[dict[str, Any]]:
    target_meta = _target_metadata(input_data)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        target = str(raw.get("target", "")).strip()
        metadata = target_meta.get(target, {})
        teq = _number(raw, "teq_k", "equilibrium_temperature_k")
        if teq is None:
            teq = _number(metadata, "teq_k")
        scenario = str(raw.get("scenario", "")).strip()
        scope = str(raw.get("fit_scope", "")).strip()
        delta_bic = _number(raw, "delta_bic", "delta_bic_flexible_minus_uniform")
        detected = _truth(raw, "detected", "map_detected")
        if detected is None and delta_bic is not None:
            detected = delta_bic < detection_threshold
        row: dict[str, Any] = {
            "run_id": str(raw.get("run_id", "")).strip(),
            "target": target,
            "teq_k": teq,
            "noise_ppm": _number(raw, "noise_ppm"),
            "scenario": scenario,
            "trial": str(raw.get("trial", raw.get("replicate", ""))).strip(),
            "fit_scope": scope,
            "injected": _number(raw, "injected_longitude_degrees", "injected_offset_degrees"),
            "q16": _number(raw, "recovered_longitude_q16_degrees", "longitude_q16_degrees"),
            "median": _number(
                raw,
                "recovered_longitude_median_degrees",
                "recovered_longitude_degrees",
                "longitude_median_degrees",
            ),
            "q84": _number(raw, "recovered_longitude_q84_degrees", "longitude_q84_degrees"),
            "bias": _number(raw, "longitude_bias_degrees", "bias_degrees"),
            "coverage": _truth(
                raw, "longitude_coverage", "interval_contains_injection",
                "longitude_interval_contains_truth",
            ),
            "resolution_limited": _truth(raw, "longitude_interval_resolution_limited"),
            "delta_bic": delta_bic,
            "detected": detected if detected is not None else _truth(raw, "map_detected"),
            "seed": str(raw.get("random_seed", "")).strip(),
            "raw": raw,
        }
        if row["bias"] is None and row["median"] is not None and row["injected"] is not None:
            row["bias"] = row["median"] - row["injected"]
        if not row["target"] or not row["scenario"] or not row["fit_scope"] or not row["trial"]:
            continue
        if row["teq_k"] is None:
            continue
        rows.append(row)
    return rows


def _sort_noise(values: Iterable[float | None]) -> list[float]:
    return sorted({float(value) for value in values if value is not None and math.isfinite(float(value))})


def _sort_trials(values: Iterable[str]) -> list[str]:
    def key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(set(values), key=key)


def _complete_groups(
    rows: list[dict[str, Any]],
    target_order: list[str],
    *,
    scenario: str,
    noise: float,
    scope: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["scenario"] != scenario or row["fit_scope"] != scope or row["noise_ppm"] != noise:
            continue
        grouped[row["trial"]][row["target"]] = row
    return {
        trial: [by_target[target] for target in target_order]
        for trial, by_target in grouped.items()
        if all(target in by_target for target in target_order)
    }


def _interval_sigma(row: dict[str, Any], grid_step: float | None) -> float | None:
    q16, q84 = row["q16"], row["q84"]
    if q16 is None or q84 is None or q84 < q16:
        return None
    estimate = 0.5 * (q84 - q16)
    if grid_step is not None and grid_step > 0.0:
        # A single occupied grid cell is not evidence for zero uncertainty.
        estimate = max(estimate, grid_step / math.sqrt(12.0))
    return estimate if estimate > 0.0 and math.isfinite(estimate) else None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _weighted_slope(
    group: list[dict[str, Any]],
    grid_step: float | None,
    value_key: str = "median",
) -> dict[str, float] | None:
    if len(group) < 3:
        return None
    y = np.asarray([row[value_key] if row[value_key] is not None else np.nan for row in group])
    temperature = np.asarray([row["teq_k"] for row in group], dtype=float)
    sigma = np.asarray(
        [np.nan if _interval_sigma(row, grid_step) is None else _interval_sigma(row, grid_step) for row in group],
        dtype=float,
    )
    valid = np.isfinite(y) & np.isfinite(temperature) & np.isfinite(sigma) & (sigma > 0.0)
    if int(np.sum(valid)) < 3:
        return None
    y = y[valid]
    temperature = temperature[valid]
    sigma = sigma[valid]
    centred = (temperature - float(np.mean(temperature))) / 1000.0
    design = np.column_stack((np.ones(centred.size), centred))
    precision = design.T @ ((1.0 / sigma**2)[:, None] * design)
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError:
        return None
    estimate = covariance @ (design.T @ ((1.0 / sigma**2) * y))
    slope = float(estimate[1])
    slope_sigma = float(np.sqrt(max(0.0, covariance[1, 1])))
    if not math.isfinite(slope_sigma) or slope_sigma <= 0.0:
        return None
    probability_negative = _normal_cdf(-slope / slope_sigma)
    return {
        "intercept_at_mean_teq_degrees": float(estimate[0]),
        "slope_degrees_per_1000k": slope,
        "slope_sigma_degrees_per_1000k": slope_sigma,
        "negative_slope_probability": probability_negative,
        "negative_slope_detected_95pct": float(probability_negative >= 0.95),
        "n_targets": float(centred.size),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _apply_style() -> Any:
    global _STYLE_MODULE
    if not STYLE_HELPER.is_file():
        raise FileNotFoundError(f"WASP-15b plot-style helper was not found: {STYLE_HELPER}")
    spec = importlib.util.spec_from_file_location("wasp15b_plot_style", STYLE_HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plot-style helper: {STYLE_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_style()
    _STYLE_MODULE = module
    return module


def _style_figure_axes(figure: Any) -> None:
    if _STYLE_MODULE is None:
        return
    for axis in figure.axes:
        _STYLE_MODULE.style_axis(axis)


def _save_figure(figure: Any, output: Path, stem: str) -> None:
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"{stem}.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )


def _plot_offsets(
    output: Path,
    rows: list[dict[str, Any]],
    target_order: list[str],
    scenarios: list[str],
    noises: list[float],
    scopes: list[str],
    representative: str,
    grid_step: float | None,
    grid_min: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scenarios = [name for name in SCENARIO_ORDER if name in scenarios]
    if not scenarios or not noises or not scopes:
        return
    for scope in scopes:
        fig, axes = plt.subplots(
            len(scenarios), len(noises),
            figsize=(7.6, max(6.0, 1.62 * len(scenarios))),
            squeeze=False,
            sharex="col",
            sharey="row",
        )
        target_markers = ("o", "s", "^", "D", "v", "P")
        for row_index, scenario in enumerate(scenarios):
            for col_index, noise in enumerate(noises):
                ax = axes[row_index, col_index]
                groups = _complete_groups(
                    rows, target_order, scenario=scenario, noise=noise, scope=scope
                )
                group = groups.get(representative)
                if group is None:
                    # If trial zero is absent, use the first complete fixed-seed trial.
                    available = _sort_trials(groups)
                    group = groups[available[0]] if available else []
                group = sorted(group, key=lambda item: item["teq_k"])
                temps = np.asarray([item["teq_k"] for item in group], dtype=float)
                for target_index, item in enumerate(group):
                    if item["median"] is None or item["q16"] is None or item["q84"] is None:
                        continue
                    lower = max(0.0, item["median"] - item["q16"])
                    upper = max(0.0, item["q84"] - item["median"])
                    ax.errorbar(
                        item["teq_k"], item["median"], yerr=[[lower], [upper]],
                        fmt=target_markers[target_order.index(item["target"]) % len(target_markers)],
                        color=DATA_COLOUR, ecolor=DATA_COLOUR, markersize=3.3,
                        capsize=1.7, elinewidth=0.7, linewidth=0.7, zorder=3,
                    )
                injected = np.asarray(
                    [np.nan if item["injected"] is None else item["injected"] for item in group],
                    dtype=float,
                )
                known = np.isfinite(injected)
                if np.any(known):
                    off_grid_truth = (
                        scenario == "temperature_scaled"
                        and grid_step is not None
                        and bool(np.all(
                            np.abs((injected[known] - grid_min) / grid_step
                                   - np.round((injected[known] - grid_min) / grid_step)) > 1.0e-6
                        ))
                    )
                    ax.plot(
                        temps[known], injected[known],
                        color=MODEL_COLOUR, marker="s", markersize=3.0,
                        markerfacecolor="white", markeredgewidth=0.8, linewidth=1.0,
                        zorder=2,
                    )
                else:
                    ax.axhline(
                        0.0, color=MODEL_COLOUR, linestyle="--", linewidth=0.9,
                    )
                if row_index == len(scenarios) - 1:
                    ax.set_xlabel(f"Equilibrium temperature (K)\n{noise:g} ppm per 60 s")
                else:
                    ax.tick_params(labelbottom=False)
                ax.set_ylim(-90.0, 90.0)
                ax.set_yticks([-90, -45, 0, 45, 90])
                if not group:
                    ax.text(0.5, 0.5, "No complete six-target trial", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7.0)
        off_grid_truth = (
            grid_step is not None
            and any(
                item["scenario"] == "temperature_scaled" and item["injected"] is not None
                and abs((item["injected"] - grid_min) / grid_step
                        - round((item["injected"] - grid_min) / grid_step)) > 1.0e-6
                for item in rows
            )
        )
        legend_handles = [
            Line2D([], [], color=DATA_COLOUR, marker="o", markersize=3.3, linewidth=0,
                   label="Recovered median; 1σ profile interval (not posterior)"),
            Line2D([], [], color=MODEL_COLOUR, marker="s", markerfacecolor="white",
                   markeredgewidth=0.8, linewidth=1.0,
                   label="Injected truth" + (" (off-grid)" if off_grid_truth else "")),
            Line2D([], [], color=MODEL_COLOUR, linestyle="--", linewidth=0.9,
                   label="Zero reference; uniform null has no hotspot"),
        ]
        legend_handles.extend(
            Line2D([], [], color=DATA_COLOUR, marker=target_markers[index % len(target_markers)],
                   markersize=3.3, linewidth=0, label=target.replace(" b", ""))
            for index, target in enumerate(target_order)
        )
        fig.legend(legend_handles, [item.get_label() for item in legend_handles],
                   loc="lower center", ncol=4, bbox_to_anchor=(0.59, 0.005),
                   borderaxespad=0.0, handletextpad=0.35, columnspacing=0.9)
        fig.supylabel("Hotspot longitude (degrees east)", x=0.15)
        fig.subplots_adjust(left=0.25, right=0.99, top=0.995, bottom=0.21,
                            hspace=0.16, wspace=0.15)
        for idx, scenario in enumerate(scenarios):
            position = axes[idx, 0].get_position()
            fig.text(0.035, 0.5 * (position.y0 + position.y1),
                     SCENARIO_LABELS.get(scenario, scenario), rotation=90,
                     ha="center", va="center", fontsize=7.0)
        _style_figure_axes(fig)
        _save_figure(fig, output, f"offset_recovery_{scope}")
        plt.close(fig)


def _temperature_scaled_longitude(temperature_k: np.ndarray) -> np.ndarray:
    angle = np.deg2rad(TREND_REFERENCE_OFFSET_DEGREES)
    return np.rad2deg(
        np.arctan(np.tan(angle) * (TREND_REFERENCE_TEMPERATURE_K / temperature_k) ** 3)
    )


def _plot_main_temperature_trend(
    output: Path,
    rows: list[dict[str, Any]],
    target_order: list[str],
    noises: list[float],
) -> bool:
    """Plot the preselected six-target, 50/100 ppm trial for the grant."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    requested_noises = [noise for noise in (50.0, 100.0) if noise in noises]
    if requested_noises != [50.0, 100.0]:
        return False
    groups = {
        noise: _complete_groups(
            rows,
            target_order,
            scenario="temperature_scaled",
            noise=noise,
            scope="full_window",
        ).get("0")
        for noise in requested_noises
    }
    if any(group is None for group in groups.values()):
        return False

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), squeeze=False, sharey=True)
    axes_row = axes[0]
    target_markers = ("o", "s", "^", "D", "v", "P")
    for ax, noise in zip(axes_row, requested_noises):
        group = sorted(groups[noise], key=lambda item: item["teq_k"])
        temperature = np.asarray([item["teq_k"] for item in group], dtype=float)
        curve_temperature = np.linspace(1250.0, 2650.0, 600)
        ax.plot(
            curve_temperature,
            _temperature_scaled_longitude(curve_temperature),
            color=MODEL_COLOUR,
            linewidth=1.2,
            zorder=1,
        )
        for item in group:
            if item["median"] is None or item["q16"] is None or item["q84"] is None:
                continue
            lower = max(0.0, item["median"] - item["q16"])
            upper = max(0.0, item["q84"] - item["median"])
            ax.errorbar(
                item["teq_k"],
                item["median"],
                yerr=[[lower], [upper]],
                fmt=target_markers[target_order.index(item["target"]) % len(target_markers)],
                color=DATA_COLOUR,
                ecolor=DATA_COLOUR,
                markerfacecolor=DATA_COLOUR if item["detected"] else "white",
                markersize=3.7,
                capsize=2.0,
                elinewidth=0.75,
                linewidth=0.75,
                zorder=3,
            )
        noise_handle = Line2D([], [], linestyle="none", color="none",
                              label=f"{noise:g} ppm per 60 s")
        ax.legend(handles=[noise_handle], loc="upper right", handlelength=0,
                  handletextpad=0.0, borderaxespad=0.45)
        if any(not item["detected"] for item in group):
            ax.text(0.97, 0.91, "Open symbols: map not detected", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)
        ax.set_xlim(1250.0, 2650.0)
        ax.set_ylim(0.0, 75.0)
        ax.set_xticks([1400, 1800, 2200, 2600])
        ax.set_yticks([0, 15, 30, 45, 60, 75])

    legend_handles = [
        Line2D([], [], color=DATA_COLOUR, marker="o", markersize=3.7, linewidth=0,
               label="Recovered median; 1σ profile interval (not posterior)"),
        Line2D([], [], color=MODEL_COLOUR, linewidth=1.2,
               label="Injected analytic trend (40 degrees at 1500 K)"),
    ]
    legend_handles.extend(
        Line2D([], [], color=DATA_COLOUR, marker=target_markers[index % len(target_markers)],
               markersize=3.7, linewidth=0, label=target.replace(" b", ""))
        for index, target in enumerate(target_order)
    )
    fig.legend(legend_handles, [item.get_label() for item in legend_handles],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.015),
               borderaxespad=0.0, handletextpad=0.35, columnspacing=0.9)
    fig.supylabel("Eastward hot-spot longitude (degrees)", x=0.025)
    fig.supxlabel("Equilibrium temperature (K)", y=0.17)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.29, wspace=0.10)
    _style_figure_axes(fig)
    _save_figure(fig, output, "unity_hotspot_trend")
    plt.close(fig)
    return True


def _plot_bias(
    output: Path,
    rows: list[dict[str, Any]],
    target_order: list[str],
    scenarios: list[str],
    noises: list[float],
    scopes: list[str],
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    injected_scenarios = [
        name for name in SCENARIO_ORDER
        if name in scenarios and name != "uniform_null"
    ]
    if not injected_scenarios or not noises or not scopes:
        return
    rng = np.random.default_rng(seed)
    for scope in scopes:
        fig, axes = plt.subplots(
            len(injected_scenarios), len(noises),
            figsize=(8.2, max(5.7, 1.72 * len(injected_scenarios))),
            squeeze=False,
            sharex="col",
            sharey="row",
        )
        for row_index, scenario in enumerate(injected_scenarios):
            for col_index, noise in enumerate(noises):
                ax = axes[row_index, col_index]
                bias_by_target: list[list[float]] = []
                for target in target_order:
                    values = [
                        item["bias"] for item in rows
                        if item["target"] == target and item["scenario"] == scenario
                        and item["noise_ppm"] == noise and item["fit_scope"] == scope
                        and item["bias"] is not None
                    ]
                    bias_by_target.append([float(value) for value in values])
                positions = np.arange(1, len(target_order) + 1, dtype=float)
                nonempty = [(pos, vals) for pos, vals in zip(positions, bias_by_target) if vals]
                if nonempty:
                    ax.boxplot(
                        [vals for _, vals in nonempty],
                        positions=[pos for pos, _ in nonempty],
                        widths=0.48, showfliers=False, patch_artist=False,
                        boxprops={"color": DATA_COLOUR, "linewidth": 0.7},
                        whiskerprops={"color": DATA_COLOUR, "linewidth": 0.7},
                        capprops={"color": DATA_COLOUR, "linewidth": 0.7},
                        medianprops={"color": MODEL_COLOUR, "linewidth": 1.0},
                    )
                for pos, vals in zip(positions, bias_by_target):
                    if vals:
                        jitter = rng.uniform(-0.16, 0.16, len(vals))
                        ax.scatter(pos + jitter, vals, s=6, color=DATA_COLOUR,
                                   alpha=0.58, linewidths=0, zorder=2)
                ax.axhline(0.0, color=MODEL_COLOUR, linestyle="--", linewidth=0.8)
                if row_index == len(injected_scenarios) - 1:
                    ax.set_xticks(positions, [name.replace(" b", "") for name in target_order],
                                  rotation=30, ha="right")
                    ax.set_xlabel(f"Target; {noise:g} ppm per 60 s")
                else:
                    ax.tick_params(labelbottom=False)
                    ax.set_xticks(positions)
        fig.supylabel("Recovered minus injected longitude (degrees)", x=0.15)
        fig.subplots_adjust(left=0.25, right=0.99, top=0.995, bottom=0.18,
                            hspace=0.20, wspace=0.16)
        for idx, scenario in enumerate(injected_scenarios):
            position = axes[idx, 0].get_position()
            fig.text(0.035, 0.5 * (position.y0 + position.y1),
                     SCENARIO_LABELS.get(scenario, scenario), rotation=90,
                     ha="center", va="center", fontsize=7.0)
        _style_figure_axes(fig)
        _save_figure(fig, output, f"recovery_bias_{scope}")
        plt.close(fig)


def _plot_delta_bic(
    output: Path,
    rows: list[dict[str, Any]],
    target_order: list[str],
    scenarios: list[str],
    noises: list[float],
    scopes: list[str],
    detection_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    scenarios = [name for name in SCENARIO_ORDER if name in scenarios]
    if not scenarios or not noises or not scopes:
        return
    for scope in scopes:
        fig, axes = plt.subplots(
            len(scenarios), len(noises),
            figsize=(8.2, max(5.7, 1.72 * len(scenarios))),
            squeeze=False,
            sharex="col",
            sharey="row",
        )
        for row_index, scenario in enumerate(scenarios):
            for col_index, noise in enumerate(noises):
                ax = axes[row_index, col_index]
                for position, target in enumerate(target_order, start=1):
                    values = [
                        item["delta_bic"] for item in rows
                        if item["target"] == target and item["scenario"] == scenario
                        and item["noise_ppm"] == noise and item["fit_scope"] == scope
                        and item["delta_bic"] is not None
                    ]
                    if values:
                        jitter = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.zeros(1)
                        ax.scatter(position + jitter, values, s=7, color=DATA_COLOUR,
                                   alpha=0.62, linewidths=0, zorder=2)
                        ax.plot([position - 0.20, position + 0.20],
                                [float(np.median(values))] * 2,
                                color=MODEL_COLOUR, linewidth=1.0, zorder=3)
                ax.axhline(detection_threshold, color=MODEL_COLOUR, linestyle="--",
                           linewidth=0.8, label=f"Map detection threshold: {detection_threshold:g}")
                ax.axhline(0.0, color=DATA_COLOUR, linestyle=":", linewidth=0.55)
                if scenario != "uniform_null":
                    ax.set_yscale("symlog", linthresh=10)
                if row_index == len(scenarios) - 1:
                    ax.set_xticks(np.arange(1, len(target_order) + 1),
                                  [name.replace(" b", "") for name in target_order],
                                  rotation=30, ha="right")
                    ax.set_xlabel(f"Target; {noise:g} ppm per 60 s")
                else:
                    ax.tick_params(labelbottom=False)
                    ax.set_xticks(np.arange(1, len(target_order) + 1))
                if row_index == 0 and col_index == 0:
                    handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=1,
                       bbox_to_anchor=(0.5, 0.005), borderaxespad=0.0)
        fig.supylabel(r"$\Delta$BIC (map minus uniform)", x=0.15)
        fig.subplots_adjust(left=0.25, right=0.99, top=0.995, bottom=0.19,
                            hspace=0.20, wspace=0.16)
        for idx, scenario in enumerate(scenarios):
            position = axes[idx, 0].get_position()
            fig.text(0.035, 0.5 * (position.y0 + position.y1),
                     SCENARIO_LABELS.get(scenario, scenario), rotation=90,
                     ha="center", va="center", fontsize=7.0)
        _style_figure_axes(fig)
        _save_figure(fig, output, f"map_delta_bic_{scope}")
        plt.close(fig)


def _plot_slopes(
    output: Path,
    slope_rows: list[dict[str, Any]],
    scenarios: list[str],
    noises: list[float],
    scopes: list[str],
) -> None:
    import matplotlib.pyplot as plt

    scenarios = [name for name in SCENARIO_ORDER if name in scenarios]
    if not scenarios or not noises or not scopes or not slope_rows:
        return
    for scope in scopes:
        fig, axes = plt.subplots(1, len(noises), figsize=(7.4, 3.3), squeeze=False, sharey=True)
        from matplotlib.lines import Line2D

        for col_index, noise in enumerate(noises):
            ax = axes[0, col_index]
            positions = np.arange(1, len(scenarios) + 1, dtype=float)
            values_by_scenario: list[list[float]] = []
            for scenario in scenarios:
                values = [
                    float(row["slope_degrees_per_1000k"])
                    for row in slope_rows
                    if row["scenario"] == scenario and row["noise_ppm"] == noise
                    and row["fit_scope"] == scope and row["slope_degrees_per_1000k"] is not None
                ]
                values_by_scenario.append(values)
            nonempty = [(pos, vals) for pos, vals in zip(positions, values_by_scenario) if vals]
            if nonempty:
                ax.boxplot(
                    [vals for _, vals in nonempty], positions=[pos for pos, _ in nonempty],
                    widths=0.52, showfliers=False, patch_artist=False,
                    boxprops={"color": DATA_COLOUR, "linewidth": 0.7},
                    whiskerprops={"color": DATA_COLOUR, "linewidth": 0.7},
                    capprops={"color": DATA_COLOUR, "linewidth": 0.7},
                    medianprops={"color": MODEL_COLOUR, "linewidth": 1.0},
                )
            for pos, vals in zip(positions, values_by_scenario):
                if vals:
                    jitter = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else np.zeros(1)
                    ax.scatter(pos + jitter, vals, s=5, color=DATA_COLOUR,
                               alpha=0.45, linewidths=0, zorder=2)
            ax.axhline(0.0, color=DATA_COLOUR, linestyle="--", linewidth=0.8)
            for pos, scenario in zip(positions, scenarios):
                truth_slopes = [
                    float(row["injected_truth_slope_degrees_per_1000k"])
                    for row in slope_rows
                    if row["scenario"] == scenario and row["noise_ppm"] == noise
                    and row["fit_scope"] == scope
                    and row["injected_truth_slope_degrees_per_1000k"] not in (None, "")
                ]
                if scenario == "temperature_scaled" and truth_slopes:
                    low, centre, high = np.quantile(truth_slopes, [0.16, 0.5, 0.84])
                    ax.fill_between([pos - 0.29, pos + 0.29], [low, low], [high, high],
                                    color=MODEL_COLOUR, alpha=0.12, linewidth=0, zorder=1)
                    ax.plot(pos, centre, marker="D", markersize=3.1,
                            markerfacecolor="white", markeredgecolor=MODEL_COLOUR,
                            markeredgewidth=0.8, linestyle="none", zorder=4)
                elif scenario in {"zero_offset", "constant_30_deg"} and truth_slopes:
                    ax.plot(pos, float(np.median(truth_slopes)), marker="D", markersize=3.1,
                            markerfacecolor="white", markeredgecolor=MODEL_COLOUR,
                            markeredgewidth=0.8, linestyle="none", zorder=4)
            ax.set_xticks(positions, [name.replace("_", "\n") for name in scenarios])
            ax.set_xlabel(f"Injection case; {noise:g} ppm per 60 s")
        legend_handles = [
            Line2D([], [], color=DATA_COLOUR, marker="o", markersize=3,
                   linewidth=0, label="Recovered trial slope"),
            Line2D([], [], color=MODEL_COLOUR, marker="D", markerfacecolor="white",
                   markeredgewidth=0.8, linewidth=0, label="Matched-weight input slope"),
        ]
        fig.legend(legend_handles, [item.get_label() for item in legend_handles],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015),
                   borderaxespad=0.0)
        fig.supylabel("Population slope (degrees per 1000 K)", x=0.055)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.25, wspace=0.12)
        _style_figure_axes(fig)
        _save_figure(fig, output, f"population_slope_recovery_{scope}")
        plt.close(fig)


def _build_summary_tables(
    output: Path,
    rows: list[dict[str, Any]],
    target_order: list[str],
    scenarios: list[str],
    noises: list[float],
    scopes: list[str],
    grid_step: float | None,
    expected_targets: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_summary: list[dict[str, Any]] = []
    for scenario in scenarios:
        for noise in noises:
            for scope in scopes:
                for target in target_order:
                    selected = [
                        item for item in rows if item["scenario"] == scenario
                        and item["noise_ppm"] == noise and item["fit_scope"] == scope
                        and item["target"] == target
                    ]
                    biases = [item["bias"] for item in selected if item["bias"] is not None]
                    widths = [item["q84"] - item["q16"] for item in selected
                              if item["q84"] is not None and item["q16"] is not None]
                    delta_values = [item["delta_bic"] for item in selected if item["delta_bic"] is not None]
                    coverages = [item["coverage"] for item in selected if item["coverage"] is not None]
                    resolution_flags = [
                        item["resolution_limited"] for item in selected
                        if item["resolution_limited"] is not None
                    ]
                    detections = [item["detected"] for item in selected if item["detected"] is not None]
                    target_summary.append({
                        "scenario": scenario, "noise_ppm": noise, "fit_scope": scope,
                        "target": target, "teq_k": next((item["teq_k"] for item in selected), ""),
                        "n_trials": len(selected), "n_bias_values": len(biases),
                        "bias_mean_degrees": float(np.mean(biases)) if biases else "",
                        "bias_median_degrees": float(np.median(biases)) if biases else "",
                        "bias_q16_degrees": float(np.quantile(biases, 0.16)) if biases else "",
                        "bias_q84_degrees": float(np.quantile(biases, 0.84)) if biases else "",
                        "bias_sd_degrees": float(np.std(biases, ddof=1)) if len(biases) > 1 else "",
                        "profile_interval_width_median_degrees": float(np.median(widths)) if widths else "",
                        "profile_interval_width_q16_degrees": float(np.quantile(widths, 0.16)) if widths else "",
                        "profile_interval_width_q84_degrees": float(np.quantile(widths, 0.84)) if widths else "",
                        "interval_coverage_fraction": float(np.mean(coverages)) if coverages else "",
                        "interval_resolution_limited_fraction": (
                            float(np.mean(resolution_flags)) if resolution_flags else ""
                        ),
                        "map_detection_fraction": float(np.mean(detections)) if detections else "",
                        "delta_bic_median": float(np.median(delta_values)) if delta_values else "",
                        "delta_bic_q16": float(np.quantile(delta_values, 0.16)) if delta_values else "",
                        "delta_bic_q84": float(np.quantile(delta_values, 0.84)) if delta_values else "",
                    })

    population_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for noise in noises:
            for scope in scopes:
                groups = _complete_groups(rows, target_order, scenario=scenario, noise=noise, scope=scope)
                for trial, group in groups.items():
                    fit = _weighted_slope(group, grid_step)
                    if fit is None:
                        continue
                    truth_fit = (
                        _weighted_slope(group, grid_step, value_key="injected")
                        if all(item["injected"] is not None for item in group)
                        else None
                    )
                    truth_slope = (
                        truth_fit["slope_degrees_per_1000k"] if truth_fit is not None else None
                    )
                    all_six_detected = (
                        len(group) == expected_targets
                        and all(item["detected"] is True for item in group)
                    )
                    any_detected = any(item["detected"] is True for item in group)
                    population_rows.append({
                        "scenario": scenario, "noise_ppm": noise, "fit_scope": scope,
                        "trial": trial,
                        "n_targets": int(fit["n_targets"]),
                        "intercept_at_mean_teq_degrees": fit["intercept_at_mean_teq_degrees"],
                        "slope_degrees_per_1000k": fit["slope_degrees_per_1000k"],
                        "slope_sigma_degrees_per_1000k": fit["slope_sigma_degrees_per_1000k"],
                        "negative_slope_probability": fit["negative_slope_probability"],
                        "negative_slope_detected_95pct": bool(fit["negative_slope_detected_95pct"]),
                        "injected_truth_slope_degrees_per_1000k": "" if truth_slope is None else truth_slope,
                        "recovered_minus_truth_slope_degrees_per_1000k": (
                            "" if truth_slope is None else fit["slope_degrees_per_1000k"] - truth_slope
                        ),
                        "all_six_map_detected": all_six_detected,
                        "any_map_detected": any_detected,
                    })

    _write_csv(
        output / "target_recovery_summary.csv", target_summary,
        ["scenario", "noise_ppm", "fit_scope", "target", "teq_k", "n_trials",
         "n_bias_values", "bias_mean_degrees", "bias_median_degrees", "bias_sd_degrees",
         "bias_q16_degrees", "bias_q84_degrees",
         "profile_interval_width_median_degrees", "profile_interval_width_q16_degrees",
         "profile_interval_width_q84_degrees", "interval_coverage_fraction",
         "interval_resolution_limited_fraction",
         "map_detection_fraction", "delta_bic_median", "delta_bic_q16", "delta_bic_q84"],
    )
    _write_csv(
        output / "population_slope_recovery.csv", population_rows,
        ["scenario", "noise_ppm", "fit_scope", "trial", "n_targets",
         "intercept_at_mean_teq_degrees", "slope_degrees_per_1000k",
         "slope_sigma_degrees_per_1000k", "negative_slope_probability",
         "negative_slope_detected_95pct", "injected_truth_slope_degrees_per_1000k",
         "recovered_minus_truth_slope_degrees_per_1000k",
         "all_six_map_detected", "any_map_detected"],
    )

    population_summary: list[dict[str, Any]] = []
    for scenario in scenarios:
        for noise in noises:
            for scope in scopes:
                selected = [
                    row for row in population_rows if row["scenario"] == scenario
                    and row["noise_ppm"] == noise and row["fit_scope"] == scope
                ]
                slopes = [float(row["slope_degrees_per_1000k"]) for row in selected]
                probabilities = [float(row["negative_slope_probability"]) for row in selected]
                truth_slopes = [
                    float(row["injected_truth_slope_degrees_per_1000k"])
                    for row in selected if row["injected_truth_slope_degrees_per_1000k"] != ""
                ]
                slope_errors = [
                    float(row["recovered_minus_truth_slope_degrees_per_1000k"])
                    for row in selected if row["recovered_minus_truth_slope_degrees_per_1000k"] != ""
                ]
                truth_slope = float(np.median(truth_slopes)) if truth_slopes else ""
                all_six = [bool(row["all_six_map_detected"]) for row in selected]
                any_detect = [bool(row["any_map_detected"]) for row in selected]
                population_summary.append({
                    "scenario": scenario, "noise_ppm": noise, "fit_scope": scope,
                    "n_complete_six_target_trials": len(selected),
                    "injected_truth_slope_degrees_per_1000k": truth_slope,
                    "injected_truth_slope_q16_degrees_per_1000k": (
                        float(np.quantile(truth_slopes, 0.16)) if truth_slopes else ""
                    ),
                    "injected_truth_slope_q84_degrees_per_1000k": (
                        float(np.quantile(truth_slopes, 0.84)) if truth_slopes else ""
                    ),
                    "recovered_slope_median_degrees_per_1000k": float(np.median(slopes)) if slopes else "",
                    "recovered_slope_q16_degrees_per_1000k": float(np.quantile(slopes, 0.16)) if slopes else "",
                    "recovered_slope_q84_degrees_per_1000k": float(np.quantile(slopes, 0.84)) if slopes else "",
                    "recovered_minus_truth_slope_median_degrees_per_1000k": (
                        float(np.median(slope_errors)) if slope_errors else ""
                    ),
                    "recovered_minus_truth_slope_q16_degrees_per_1000k": (
                        float(np.quantile(slope_errors, 0.16)) if slope_errors else ""
                    ),
                    "recovered_minus_truth_slope_q84_degrees_per_1000k": (
                        float(np.quantile(slope_errors, 0.84)) if slope_errors else ""
                    ),
                    "negative_slope_fraction_95pct": float(np.mean([p >= 0.95 for p in probabilities])) if probabilities else "",
                    "mean_negative_slope_probability": float(np.mean(probabilities)) if probabilities else "",
                    "all_six_map_detection_fraction": float(np.mean(all_six)) if all_six else "",
                    "any_map_detection_fraction": float(np.mean(any_detect)) if any_detect else "",
                })
    _write_csv(
        output / "population_slope_summary.csv", population_summary,
        ["scenario", "noise_ppm", "fit_scope", "n_complete_six_target_trials",
         "injected_truth_slope_degrees_per_1000k",
         "injected_truth_slope_q16_degrees_per_1000k", "injected_truth_slope_q84_degrees_per_1000k",
         "recovered_slope_median_degrees_per_1000k",
         "recovered_slope_q16_degrees_per_1000k", "recovered_slope_q84_degrees_per_1000k",
         "recovered_minus_truth_slope_median_degrees_per_1000k",
         "recovered_minus_truth_slope_q16_degrees_per_1000k",
         "recovered_minus_truth_slope_q84_degrees_per_1000k",
         "negative_slope_fraction_95pct", "mean_negative_slope_probability",
         "all_six_map_detection_fraction", "any_map_detection_fraction"],
    )
    return population_rows, population_summary


def _methodology_text(
    *,
    output: Path,
    trials_path: Path,
    inputs_path: Path,
    run_inputs: dict[str, Any],
    summary: dict[str, Any],
    input_data: dict[str, Any],
    rows: list[dict[str, Any]],
    target_order: list[str],
    noises: list[float],
    scopes: list[str],
    grid_step: float | None,
    threshold: float,
    threshold_source: str,
    population_summary: list[dict[str, Any]],
    main_trend_written: bool,
    missing: list[str],
    off_grid_count: int | None,
    injected_count: int,
    expected_target_count: int,
) -> str:
    assumptions = input_data.get("assumptions", {})
    trend = assumptions.get("trend", "Temperature-scaled input values are read from each trial row.")
    showman = input_data.get("showman_source", "")
    master_seed = _find_key(run_inputs, {"master_seed", "seed"})
    figure_caption = (
        "Figure caption for unity_hotspot_trend: Six independent full-window hot-spot longitude "
        f"fits for preselected trial 0 (master seed {master_seed or 'not recorded'}) at 50 and "
        "100 ppm per ideal 60-second white-noise bin. The purple line shows "
        "delta = atan[tan(40 degrees) * (1500 K / Teq)^3], an illustrative Showman-inspired "
        "scaling with a chosen normalization, not an offset from a specific GCM. Black markers "
        "show recovered medians; bars show saved q16-q84 profile-grid intervals, not posterior "
        "intervals. Open symbols mark cases that do not meet the map-detection rule, "
        "delta BIC < -6. The model uses fixed circular geometry and symmetric continuous visits."
    )
    scenarios = [name for name in SCENARIO_ORDER if name in {item["scenario"] for item in rows}]
    trial_counts = []
    for scenario in scenarios:
        for noise in noises:
            for scope in scopes:
                count = len({item["trial"] for item in rows if item["scenario"] == scenario
                             and item["noise_ppm"] == noise and item["fit_scope"] == scope})
                trial_counts.append(f"{scenario}, {noise:g} ppm, {scope}: {count}")
    lines = [
        "UNITY injection and recovery: methods and limits",
        "",
        f"Output folder: {output}",
        f"Trial table: {trials_path}",
        f"Input record: {inputs_path}",
        f"Targets in the trial table: {len(target_order)} ({', '.join(target_order)})",
        f"Noise cases: {', '.join(f'{value:g} ppm per ideal 60-second bin' for value in noises) or 'not found'}.",
        f"Fit scopes: {', '.join(scopes) or 'not found'}.",
        f"Trials by case: {'; '.join(trial_counts) if trial_counts else 'not available'}.",
        f"Saved longitude grid step: {grid_step:g} degrees." if grid_step else "Saved longitude grid step: not found; off-grid status is unverified.",
        f"Map detection threshold: delta BIC < {threshold:g} ({threshold_source}).",
        "",
        "Each target is fitted independently. The population slope is fitted only after the target fits, from complete groups of recovered target medians for each scenario, noise case, fit scope, and trial. The input trend is not a constraint or prior in any target fit.",
        "",
        f"A representative plot uses one common trial number across all {expected_target_count} target(s), selected as trial 0 when complete or otherwise the lowest complete trial. Its vertical bars are the saved q16 to q84 profile-grid intervals, marginalized over unique longitude values and with a minimum width of one longitude grid step. These are profile intervals, not posterior intervals. The recovered offset is conditional on the mapped model; a small delta BIC can mean that the map is not detected.",
        "",
        "The population fit is weighted least squares: recovered longitude = intercept + slope * ((Teq - mean(Teq))/1000 K). Each target weight uses an approximate 1-sigma value equal to half the q84-q16 profile interval. If the interval has zero width, the uncertainty is floored at the standard deviation of one uniform grid cell (grid step / sqrt(12)). The fit assumes independent target errors. The reported slope uncertainty uses the inverse weighted design matrix. A negative trend is marked detected when its Gaussian approximation gives P(slope < 0) >= 0.95. This nominal threshold is not a posterior probability. The reported false-detection fractions in the zero, constant-offset, and uniform-planet control cases show how often this approximation gives a negative trend when none is injected." if expected_target_count >= 3 else f"A population slope needs at least three targets. This run has {expected_target_count}, so no population slope was calculated.",
        "For injected longitude trends, each realization's input slope is also fitted with the same target weights as the recovered slope. This avoids comparing the weighted recovered slope with a different unweighted fit to the nonlinear input trend. The slope error is recovered slope minus this matched-weight input slope. The uniform-planet null has no injected longitude and therefore no input slope." if expected_target_count >= 3 else "No population slope figures or estimates are available for this target count.",
        "",
        "Bias is recovered median minus injected longitude. Ensemble bias, interval coverage, interval width, and the fraction of intervals limited by the longitude grid are calculated across repeat trials for each target, scenario, noise case, and fit scope. The representative figure shows one trial only; it does not show an ensemble interval as if it were one realization.",
        "",
        "Map evidence uses delta BIC = BIC(mapped) - BIC(uniform), so negative values support the mapped model. The all-target detection fraction is the fraction of complete target groups in which all saved target fits pass their map-detection rule. The uniform-null case has no injected hotspot longitude; its recovered location remains conditional on fitting a map.",
        "",
        f"Temperature-scaled injection: {trend}",
        "This scenario is inspired by Showman et al. (2020), but it is an illustrative analytic scaling with a chosen normalization. It is not an offset predicted by a specific GCM or a full population prediction. Source: " + str(showman),
        *([figure_caption] if main_trend_written else []),
        "",
        "Study assumptions: " + str(assumptions.get("noise", "Noise assumptions are recorded in unity_inputs.json.")),
        "Depth assumption: " + str(assumptions.get("depth", "Depth assumptions are recorded in unity_inputs.json.")),
        "Geometry assumption: " + str(assumptions.get("geometry", "Geometry assumptions are recorded in unity_inputs.json.")),
        "Visit assumption: " + str(assumptions.get("visit", "Visit assumptions are recorded in unity_inputs.json.")),
        "Science scope: " + str(assumptions.get("science", "See the input record for the scope.")),
        "",
        "The analysis uses the configured white-noise cases recorded in the input file. It does not include observed time stamps, gaps, orbit uncertainty, eccentricity recovery, correlated noise, or instrument systematics unless the runner metadata states otherwise. Do not read the recovery fraction as a prediction of real-data performance without those terms.",
        "",
        f"Off-grid check: {off_grid_count}/{injected_count} unique temperature-scaled injected values were off the saved regular longitude grid." if off_grid_count is not None else "Off-grid check: unable to verify because the grid step was not saved.",
        "",
        "Population slope results (slope units are degrees per 1000 K):"
        if expected_target_count >= 3
        else "Population slope results are not available for this target count.",
    ]
    if any(row["n_complete_six_target_trials"] > 0 for row in population_summary):
        for row in population_summary:
            if row["n_complete_six_target_trials"] == 0:
                continue
            truth_detail = ""
            if row["injected_truth_slope_degrees_per_1000k"] != "":
                truth_detail = (
                    f"Matched-weight input slope median {row['injected_truth_slope_degrees_per_1000k']:.3g}"
                    f" (q16-q84 {row['injected_truth_slope_q16_degrees_per_1000k']:.3g} to "
                    f"{row['injected_truth_slope_q84_degrees_per_1000k']:.3g}); recovered-minus-input slope median "
                    f"{row['recovered_minus_truth_slope_median_degrees_per_1000k']:.3g}"
                    f" (q16-q84 {row['recovered_minus_truth_slope_q16_degrees_per_1000k']:.3g} to "
                    f"{row['recovered_minus_truth_slope_q84_degrees_per_1000k']:.3g})"
                )
            truth_clause = f"; {truth_detail};" if truth_detail else ""
            lines.append(
                f"- {row['scenario']}, {row['noise_ppm']:g} ppm, {row['fit_scope']}: "
                f"median recovered slope {row['recovered_slope_median_degrees_per_1000k']:.3g} "
                f"(q16-q84 {row['recovered_slope_q16_degrees_per_1000k']:.3g} to "
                f"{row['recovered_slope_q84_degrees_per_1000k']:.3g}){truth_clause} "
                f"negative-slope detection fraction {row['negative_slope_fraction_95pct']:.3f}; "
                f"all-six map detection fraction {row['all_six_map_detection_fraction']:.3f}; "
                f"n={row['n_complete_six_target_trials']} trials."
            )
        for scope in scopes:
            controls = [row for row in population_summary
                        if row["fit_scope"] == scope and row["scenario"] in {"uniform_null", "zero_offset", "constant_30_deg"}]
            if controls:
                detail = "; ".join(
                    f"{row['scenario']} at {row['noise_ppm']:g} ppm: "
                    f"{row['negative_slope_fraction_95pct']:.3f}"
                    for row in controls if row["negative_slope_fraction_95pct"] != ""
                )
                if detail:
                    lines.append(f"False detections of a negative trend in zero-slope controls, {scope}: {detail}.")
                else:
                    lines.append(
                        f"False detections of a negative trend in zero-slope controls, {scope}: "
                        "not available because no complete six-target trials were found."
                    )
    else:
        if expected_target_count >= 3:
            lines.append(
                f"No complete {expected_target_count}-target trials were available for the weighted slope fit."
            )
    lines += ["", "Generated products:"]
    if main_trend_written:
        lines.append("- unity_hotspot_trend: compact 50/100 ppm full-window figure for preselected trial 0.")
    lines += [
              "- Offset recovery figures: one common trial, with injected truths and recovered 1σ profile intervals.",
              "- Bias figures: repeat-trial bias distributions by target.",
              "- Map evidence figures: delta-BIC distributions by target and case.",
              "- target_recovery_summary.csv: per-target bias, precision, coverage, and detection rates."]
    if any(row["n_complete_six_target_trials"] > 0 for row in population_summary):
        lines += [
            "- Population slope figures and CSV tables: per-trial weighted slopes and repeat-trial summaries.",
            "- population_slope_recovery.csv and population_slope_summary.csv: slope estimates and trend-detection rates.",
        ]
    else:
        lines.append("- Population slope figures and estimates: not generated because no valid slope fit was available.")
    if missing:
        lines += ["", "Missing or incomplete inputs:"]
        lines.extend(f"- {item}" for item in missing)
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"study output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS,
                        help=f"trial CSV (default: {DEFAULT_TRIALS})")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS,
                        help=f"target/noise input JSON (default: {DEFAULT_INPUTS})")
    parser.add_argument("--run-inputs", type=Path, default=DEFAULT_RUN_INPUTS,
                        help=f"runner metadata JSON (default: {DEFAULT_RUN_INPUTS})")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                        help=f"runner summary JSON (default: {DEFAULT_SUMMARY})")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES,
                        help=f"profile-grid NPZ (default: {DEFAULT_PROFILES})")
    parser.add_argument("--representative-trial", default="0",
                        help="trial ID for the representative figure (default: 0)")
    parser.add_argument("--expected-target-count", type=_positive_int_arg, default=6,
                        help="expected number of targets (default: 6)")
    parser.add_argument("--detection-threshold", type=float, default=None,
                        help="override the saved delta-BIC map threshold")
    return parser


def _positive_int_arg(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials_path = args.trials.expanduser().resolve()
    inputs_path = args.inputs.expanduser().resolve()
    run_inputs_path = args.run_inputs.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    profiles_path = args.profiles.expanduser().resolve()
    input_data = _read_json(inputs_path)
    run_inputs = _read_json(run_inputs_path)
    summary = _read_json(summary_path)
    metadata = {"input": input_data, "run_inputs": run_inputs, "summary": summary}
    threshold_value = args.detection_threshold
    threshold_source = "command-line override"
    if threshold_value is None:
        saved_threshold = _find_key(metadata, {
            "detection_delta_bic", "map_detection_delta_bic", "delta_bic_threshold",
            "delta_bic_detection_threshold",
        })
        if saved_threshold is not None:
            try:
                threshold_value = float(saved_threshold)
                threshold_source = "saved runner metadata"
            except (TypeError, ValueError):
                threshold_value = None
    if threshold_value is None:
        threshold_value = -6.0
        threshold_source = "repository default; no saved threshold was found"
    threshold = float(threshold_value)

    missing: list[str] = []
    if not trials_path.is_file():
        missing.append(f"Trial table not found: {trials_path}")
    if not inputs_path.is_file():
        missing.append(f"Target input record not found: {inputs_path}")
    if not run_inputs_path.is_file():
        missing.append(f"Runner input record not found: {run_inputs_path}")
    raw_rows = _read_csv(trials_path) if trials_path.is_file() else []
    grid_step = _find_key(metadata, {"longitude_grid_step_degrees", "longitude_step_degrees"})
    try:
        grid_step = float(grid_step) if grid_step is not None else None
    except (TypeError, ValueError):
        grid_step = None
    if grid_step is None:
        grid_step = _candidate_grid_step(profiles_path)
    grid_min_raw = _find_key(metadata, {"longitude_grid_min_degrees", "longitude_min_degrees"})
    try:
        grid_min = float(grid_min_raw) if grid_min_raw is not None else -90.0
    except (TypeError, ValueError):
        grid_min = -90.0
    if not raw_rows:
        missing.append("Trial table has no rows.")
    required = {
        "target": {"target"},
        "scenario": {"scenario"},
        "trial": {"trial", "replicate"},
        "fit_scope": {"fit_scope"},
        "temperature": {"teq_k", "equilibrium_temperature_k"},
        "median": {"recovered_longitude_median_degrees", "recovered_longitude_degrees", "longitude_median_degrees"},
        "q16": {"recovered_longitude_q16_degrees", "longitude_q16_degrees"},
        "q84": {"recovered_longitude_q84_degrees", "longitude_q84_degrees"},
        "delta_bic": {"delta_bic", "delta_bic_flexible_minus_uniform"},
    }
    columns = set(raw_rows[0]) if raw_rows else set()
    for label, aliases in required.items():
        if not columns.intersection(aliases):
            missing.append(f"Trial CSV does not contain a required {label} column (one of {sorted(aliases)}).")
    rows = _normalise_rows(raw_rows, input_data, threshold)
    target_order = _target_order(input_data, raw_rows)
    expected_targets = args.expected_target_count
    found_targets = len(target_order)
    if found_targets != expected_targets:
        target_label = "target" if expected_targets == 1 else "targets"
        missing.append(f"Expected {expected_targets} {target_label}, found {found_targets}.")
    noises = _sort_noise(item["noise_ppm"] for item in rows)
    scopes = [scope for scope in SCOPE_ORDER if scope in {item["fit_scope"] for item in rows}]
    scopes += sorted({item["fit_scope"] for item in rows} - set(scopes))
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in {item["scenario"] for item in rows}]
    trial_ids = {item["trial"] for item in rows}
    representative = args.representative_trial
    if representative not in trial_ids and trial_ids:
        representative = _sort_trials(trial_ids)[0]
        missing.append(f"Requested representative trial {args.representative_trial!r} was absent; using {representative!r}.")

    _apply_style()
    population_rows, population_summary = _build_summary_tables(
        output, rows, target_order, scenarios, noises, scopes, grid_step, expected_targets
    )
    for scope in scopes:
        complete_ids = set()
        for scenario in scenarios:
            for noise in noises:
                complete_ids.update(_complete_groups(rows, target_order,
                                                      scenario=scenario, noise=noise, scope=scope))
        chosen = representative if representative in complete_ids else (_sort_trials(complete_ids)[0] if complete_ids else None)
        if chosen is not None:
            representative_rows = [item["raw"] for item in rows
                                    if item["fit_scope"] == scope and item["trial"] == chosen]
            columns_out = list(representative_rows[0]) if representative_rows else []
            if columns_out:
                _write_csv(output / f"representative_trials_{scope}.csv", representative_rows, columns_out)
        else:
            missing.append(
                f"No complete representative {expected_targets}-target group was found for {scope}."
            )
        _plot_offsets(output, rows, target_order, scenarios, noises, [scope],
                      chosen or representative, grid_step, grid_min)
        _plot_bias(output, rows, target_order, scenarios, noises, [scope],
                   int(_find_key(metadata, {"seed"}) or 0))
        _plot_delta_bic(output, rows, target_order, scenarios, noises, [scope], threshold)
        _plot_slopes(output, population_rows, scenarios, noises, [scope])

    main_trend_written = (
        _plot_main_temperature_trend(output, rows, target_order, noises)
        if expected_targets == 6
        else False
    )
    if expected_targets == 6 and not main_trend_written:
        missing.append(
            "Main trend figure was not generated: trial 0 needs complete six-target temperature-scaled full-window fits at 50 and 100 ppm."
        )

    injected_values = sorted({
        round(float(item["injected"]), 8) for item in rows
        if item["injected"] is not None and item["scenario"] == "temperature_scaled"
    })
    off_grid_count: int | None = None
    if grid_step is not None and injected_values:
        off_grid_count = sum(
            abs(((value - grid_min) / grid_step) - round((value - grid_min) / grid_step)) > 1.0e-6
            for value in injected_values
        )
    methodology = _methodology_text(
        output=output, trials_path=trials_path, inputs_path=inputs_path,
        run_inputs=run_inputs, summary=summary, input_data=input_data, rows=rows,
        target_order=target_order, noises=noises, scopes=scopes, grid_step=grid_step,
        threshold=threshold, threshold_source=threshold_source,
        population_summary=population_summary, main_trend_written=main_trend_written,
        missing=missing,
        off_grid_count=off_grid_count, injected_count=len(injected_values),
        expected_target_count=expected_targets,
    )
    (output / "methodology.txt").write_text(methodology, encoding="utf-8")
    print(f"Wrote UNITY recovery plots and tables to {output}")
    if missing:
        print("Input limits were written to methodology.txt:")
        for item in missing:
            print(f"- {item}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
