#!/usr/bin/env python3
"""Plot one temperature-scaled UNITY recovery trial with target-specific noise."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


STYLE_HELPER = Path(__file__).resolve().with_name("wasp15b_plot_style.py")
SCENARIO = "temperature_scaled"
SCOPES = ("full_window", "eclipse_only")
MARKERS = ("o", "s", "^", "D", "v", "P")
MODEL_COLOUR = "#6F3CC3"
DATA_COLOUR = "#111111"


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
    angle_40_rad = math.radians(40.0)
    return math.degrees(
        math.atan(math.tan(angle_40_rad) * (1500.0 / teq_k) ** 3)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _find_targets(value: Any) -> list[dict[str, Any]] | None:
    """Find the target list in either raw inputs or a runner input snapshot."""

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
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    targets = _find_targets(payload)
    if targets is None:
        raise ValueError(f"No target list was found in inputs JSON: {path}")
    if len(targets) != 6:
        raise ValueError(f"Expected six input targets; found {len(targets)}")

    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(targets):
        name = str(item.get("name", "")).strip()
        teq_k = _number(item.get("teq_k"))
        noise_ppm = _number(item.get("noise_ppm"))
        if not name or name in names:
            raise ValueError(f"Target {index + 1} has a blank or repeated name: {name!r}")
        if teq_k is None or teq_k <= 0.0:
            raise ValueError(f"Target {name!r} needs a positive teq_k value")
        if noise_ppm is None or noise_ppm <= 0.0:
            raise ValueError(f"Target {name!r} needs a positive noise_ppm value")
        names.add(name)
        result.append({"target": name, "teq_k": teq_k, "configured_noise_ppm": noise_ppm})

    return result


def _normalise_results(
    trials: list[dict[str, str]], targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_target = {item["target"]: item for item in targets}
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for row in trials:
        if str(row.get("scenario", "")).strip() != SCENARIO:
            continue
        scope = str(row.get("fit_scope", "")).strip()
        target = str(row.get("target", "")).strip()
        if scope not in SCOPES or target not in by_target:
            continue
        key = (target, scope)
        if key in selected:
            raise ValueError(
                f"Expected one {SCENARIO} row for {target!r}, {scope}; found more than one. "
                "Use a trials CSV with one trial per target and scope."
            )
        selected[key] = row

    results: list[dict[str, Any]] = []
    for target_info in targets:
        target = target_info["target"]
        for scope in SCOPES:
            raw = selected.get((target, scope))
            if raw is None:
                raise ValueError(f"Missing {SCENARIO} row for target {target!r}, scope {scope}")

            teq_k = _number(raw.get("teq_k")) or target_info["teq_k"]
            if teq_k <= 0.0:
                raise ValueError(f"Target {target!r} has a non-positive teq_k in the trials CSV")
            injected = _number(raw.get("injected_longitude_degrees"))
            q16 = _number(raw.get("recovered_longitude_q16_degrees"))
            median = _number(raw.get("recovered_longitude_median_degrees"))
            q84 = _number(raw.get("recovered_longitude_q84_degrees"))
            if q16 is not None and q84 is not None and q84 < q16:
                raise ValueError(f"Target {target!r}, {scope} has q84 below q16")
            if median is not None and q16 is not None and median < q16:
                raise ValueError(f"Target {target!r}, {scope} has median below q16")
            if median is not None and q84 is not None and median > q84:
                raise ValueError(f"Target {target!r}, {scope} has median above q84")

            results.append(
                {
                    "target": target,
                    "teq_k": teq_k,
                    "configured_noise_ppm": target_info["configured_noise_ppm"],
                    "noise_input_ppm": _number(raw.get("noise_input_ppm")),
                    "noise_effective_ppm": _number(raw.get("noise_ppm")),
                    "scenario": SCENARIO,
                    "trial": raw.get("trial", ""),
                    "fit_scope": scope,
                    "injected_longitude_degrees": injected,
                    "analytic_longitude_degrees": _analytic_longitude(teq_k),
                    "recovered_longitude_q16_degrees": q16,
                    "recovered_longitude_median_degrees": median,
                    "recovered_longitude_q84_degrees": q84,
                    "interval_halfwidth_degrees": (
                        None if q16 is None or q84 is None else 0.5 * (q84 - q16)
                    ),
                    "map_detected": _boolean(raw.get("map_detected")),
                    "delta_bic": _number(raw.get("delta_bic")),
                }
            )
    return results


def _weighted_slope(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    usable = [
        row
        for row in rows
        if row["recovered_longitude_median_degrees"] is not None
        and row["interval_halfwidth_degrees"] is not None
        and row["interval_halfwidth_degrees"] > 0.0
    ]
    if len(usable) < 3:
        return None

    temperature = np.asarray([row["teq_k"] for row in usable], dtype=float)
    value = np.asarray(
        [row["recovered_longitude_median_degrees"] for row in usable], dtype=float
    )
    sigma = np.asarray([row["interval_halfwidth_degrees"] for row in usable], dtype=float)
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


def _format(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


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


def _plot(output_dir: Path, targets: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = _apply_style()
    full_rows = [row for row in results if row["fit_scope"] == "full_window"]
    full_rows.sort(key=lambda row: row["teq_k"])
    temperatures = np.asarray([row["teq_k"] for row in full_rows], dtype=float)
    curve_temperature = np.linspace(float(temperatures.min()), float(temperatures.max()), 600)

    figure, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.plot(
        curve_temperature,
        [_analytic_longitude(value) for value in curve_temperature],
        color=MODEL_COLOUR,
        linewidth=1.2,
        zorder=1,
    )

    target_index = {item["target"]: index for index, item in enumerate(targets)}
    for row in full_rows:
        median = row["recovered_longitude_median_degrees"]
        if median is None:
            continue
        index = target_index[row["target"]]
        marker = MARKERS[index % len(MARKERS)]
        q16 = row["recovered_longitude_q16_degrees"]
        q84 = row["recovered_longitude_q84_degrees"]
        yerr = None
        if q16 is not None and q84 is not None:
            yerr = [[max(0.0, median - q16)], [max(0.0, q84 - median)]]
        axis.errorbar(
            row["teq_k"],
            median,
            yerr=yerr,
            fmt=marker,
            color=DATA_COLOUR,
            ecolor=DATA_COLOUR,
            markerfacecolor="white" if row["map_detected"] is False else DATA_COLOUR,
            markeredgecolor=DATA_COLOUR,
            markeredgewidth=0.75,
            markersize=4.2,
            capsize=1.8,
            elinewidth=0.7,
            linewidth=0.7,
            zorder=3,
        )

    legend_handles = [
        Line2D([], [], color=DATA_COLOUR, marker="o", markersize=4.2, linewidth=0,
               label="Recovered median"),
        Line2D([], [], color=DATA_COLOUR, linewidth=0.7,
               label="q16–q84 profile interval"),
        Line2D([], [], color=MODEL_COLOUR, linewidth=1.2,
               label="Injected analytic trend"),
    ]
    legend_handles.extend(
        Line2D(
            [], [], color=DATA_COLOUR, marker=MARKERS[index % len(MARKERS)],
            markersize=4.2, linewidth=0, label=item["target"],
        )
        for index, item in enumerate(targets)
    )
    if any(row["map_detected"] is False for row in full_rows):
        legend_handles.append(
            Line2D([], [], color=DATA_COLOUR, marker="o", markerfacecolor="white",
                   markeredgecolor=DATA_COLOUR, markersize=4.2, linewidth=0,
                   label="Open: map not detected")
        )
    axis.set_xlabel("Equilibrium temperature (K)")
    axis.set_ylabel("Eastward hotspot longitude (degrees)")
    axis.margins(x=0.04, y=0.08)
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.30)
    style.style_axis(axis)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output_dir / f"unity_expected_noise.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(figure)


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    columns = [
        "target",
        "teq_k",
        "configured_noise_ppm",
        "noise_input_ppm",
        "noise_effective_ppm",
        "scenario",
        "trial",
        "fit_scope",
        "injected_longitude_degrees",
        "analytic_longitude_degrees",
        "recovered_longitude_q16_degrees",
        "recovered_longitude_median_degrees",
        "recovered_longitude_q84_degrees",
        "interval_halfwidth_degrees",
        "map_detected",
        "delta_bic",
        "weighted_slope_deg_per_1000k",
        "weighted_slope_error_deg_per_1000k",
        "weighted_slope_n_targets",
    ]
    slope_by_scope = {
        scope: _weighted_slope([row for row in results if row["fit_scope"] == scope])
        for scope in SCOPES
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            output = dict(row)
            slope = slope_by_scope[row["fit_scope"]]
            output["weighted_slope_deg_per_1000k"] = (
                "" if slope is None else slope["slope_deg_per_1000k"]
            )
            output["weighted_slope_error_deg_per_1000k"] = (
                "" if slope is None else slope["slope_error_deg_per_1000k"]
            )
            output["weighted_slope_n_targets"] = "" if slope is None else int(slope["n_targets"])
            writer.writerow(output)


def _write_report(
    path: Path,
    trials_path: Path,
    inputs_path: Path,
    targets: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    lines = [
        "UNITY temperature-scaled injection and recovery",
        f"Trials CSV: {trials_path}",
        f"Inputs JSON: {inputs_path}",
        "",
        "One trial is reported for each target and fit scope. Noise values are in ppm.",
        "Input noise is the configured or runner input value; effective noise is the value used by the fit.",
    ]
    for scope in SCOPES:
        scope_rows = [row for row in results if row["fit_scope"] == scope]
        lines.extend(("", f"{scope} results:"))
        for row in scope_rows:
            input_noise = row["noise_input_ppm"]
            if input_noise is None:
                input_noise = row["configured_noise_ppm"]
            truth = row["injected_longitude_degrees"]
            truth_text = _format(truth)
            if truth is None:
                truth_text = "NA"
            detected = row["map_detected"]
            detection_text = "unknown" if detected is None else ("yes" if detected else "no")
            lines.append(
                f"- {row['target']}: Teq={_format(row['teq_k'], 1)} K; "
                f"noise input={_format(input_noise)} ppm; "
                f"effective noise={_format(row['noise_effective_ppm'])} ppm; "
                f"injected truth={truth_text} deg; "
                f"analytic truth={_format(row['analytic_longitude_degrees'])} deg; "
                f"recovered median={_format(row['recovered_longitude_median_degrees'])} deg; "
                f"q16–q84=[{_format(row['recovered_longitude_q16_degrees'])}, "
                f"{_format(row['recovered_longitude_q84_degrees'])}] deg; "
                f"map detected={detection_text}; trial={row['trial']}"
            )
        slope = _weighted_slope(scope_rows)
        if slope is None:
            lines.append("Approximate weighted slope: unavailable (fewer than three usable targets).")
        else:
            lines.append(
                "Approximate weighted slope: "
                f"{slope['slope_deg_per_1000k']:.3f} +/- "
                f"{slope['slope_error_deg_per_1000k']:.3f} degrees per 1000 K "
                f"(n={int(slope['n_targets'])}; weights use q16–q84 interval halfwidths)."
            )
    lines.extend(
        (
            "",
            "Limit: one noise sample per target cannot establish interval coverage or map detection rates.",
            "The q16–q84 ranges are profile intervals, not posterior intervals.",
            "The approximate weighted slope fits recovered medians with inverse squared interval-halfwidth weights.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True, help="Runner recovery_trials.csv")
    parser.add_argument("--inputs", type=Path, required=True, help="Runner inputs JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plot and summaries")
    args = parser.parse_args()

    targets = _read_targets(args.inputs)
    results = _normalise_results(_read_csv(args.trials), targets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _plot(args.output_dir, targets, results)
    _write_csv(args.output_dir / "unity_expected_noise_results.csv", results)
    _write_report(
        args.output_dir / "unity_expected_noise_report.txt",
        args.trials,
        args.inputs,
        targets,
        results,
    )
    print(f"Wrote plot, result CSV, and report to {args.output_dir}")


if __name__ == "__main__":
    main()
