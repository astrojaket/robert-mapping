#!/usr/bin/env python3
"""Build portable UNITY case folders from saved study results.

This script does not run a fit. It selects saved rows, rebuilds the map
coefficients from the saved fit values, and writes a numbered result folder
for each noise case.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STUDY_ROOT.parent
CODE_ROOT = STUDY_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from unity_injection_recovery import (  # noqa: E402
    _map_metrics,
    _mixture_coefficients,
    _temperature_offset,
    _uniform_coefficients,
    _wrap_degrees,
)
from robert_mapping.physics import evaluate_map, map_design_matrix  # noqa: E402
from wasp15b_plot_style import apply_style, style_axis  # noqa: E402


CASE_DEFINITIONS = {
    "JWST_Noise": {
        "bundle": "expected_noise_ensemble",
        "input_name": "unity_expected_noise.json",
        "kind": "per_target",
    },
    "50ppm": {
        "bundle": "study",
        "input_name": "unity_inputs.json",
        "kind": "fixed",
        "noise_ppm": 50.0,
    },
    "100ppm": {
        "bundle": "study",
        "input_name": "unity_inputs.json",
        "kind": "fixed",
        "noise_ppm": 100.0,
    },
}
FINEGRID_BUNDLE = "wasp12_finegrid"
MAP_LABEL_PREFIX = "y"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(
    path: Path,
    fields: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, *, default: int | None = None) -> int | None:
    number = _float(value)
    if number is None or not number.is_integer():
        return default
    return int(number)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _same_noise(value: Any, expected: float) -> bool:
    parsed = _float(value)
    return parsed is not None and math.isclose(parsed, expected, rel_tol=0.0, abs_tol=1.0e-8)


def _target_metadata(summary: dict[str, Any], input_snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    seed_by_target: dict[str, int] = {}
    for index, target in enumerate(input_snapshot.get("targets", [])):
        if not isinstance(target, dict) or "name" not in target:
            continue
        seed_by_target[str(target["name"])] = _int(
            target.get("seed_target_index"), default=index
        ) or 0
    for target in summary.get("targets", []):
        if not isinstance(target, dict) or "target" not in target:
            continue
        name = str(target["target"])
        seed_by_target[name] = _int(target.get("seed_target_index"), default=seed_by_target.get(name, 0)) or 0
    return {name: {"seed_target_index": seed} for name, seed in seed_by_target.items()}


def _bundle_paths(source_root: Path, bundle_name: str) -> tuple[Path, Path, Path, Path]:
    bundle = source_root / bundle_name
    run_inputs = bundle / "run_inputs.json"
    summary = bundle / "summary.json"
    trials = bundle / "recovery_trials.csv"
    examples = bundle / "examples"
    missing = [path for path in (run_inputs, summary, trials) if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Saved results are missing: {joined}. Run the study before organizing cases."
        )
    return run_inputs, summary, trials, examples


def _noise_map(
    case_name: str,
    target_names: list[str],
    input_snapshot: dict[str, Any],
    run_settings: dict[str, Any],
) -> dict[str, float]:
    definition = CASE_DEFINITIONS.get(case_name)
    if definition is None:
        raise ValueError(f"Unknown case: {case_name}")
    if definition["kind"] == "fixed":
        return {name: float(definition["noise_ppm"]) for name in target_names}

    targets = input_snapshot.get("targets", [])
    raw_by_name = {
        str(target.get("name")): target
        for target in targets
        if isinstance(target, dict) and target.get("name") is not None
    }
    configured = run_settings.get("noise_levels_effective_ppm", {})
    resolved: dict[str, float] = {}
    for name in target_names:
        value: Any = None
        if isinstance(configured, dict) and name in configured:
            options = configured[name]
            if isinstance(options, list) and len(options) == 1:
                value = options[0]
        target = raw_by_name.get(name, {})
        if value is None:
            values = target.get("noise_levels_ppm")
            if isinstance(values, list) and len(values) == 1:
                value = values[0]
        if value is None:
            value = target.get("noise_ppm")
        noise = _float(value)
        if noise is None or noise <= 0.0:
            raise ValueError(f"Could not find the JWST noise for {name} in the saved inputs")
        resolved[name] = noise
    return resolved


def _prepare_case_input(
    input_snapshot: dict[str, Any],
    target_names: list[str],
    noise_by_target: dict[str, float],
    scenario_names: list[str],
    case_name: str,
    expected_trials: int = 100,
) -> dict[str, Any]:
    result = json.loads(json.dumps(input_snapshot))
    all_targets = result.get("targets", [])
    result["targets"] = [
        target
        for target in all_targets
        if isinstance(target, dict) and str(target.get("name")) in set(target_names)
    ]
    for target in result["targets"]:
        name = str(target["name"])
        target["noise_ppm"] = noise_by_target[name]
        target["noise_levels_ppm"] = [noise_by_target[name]]
    result["scenarios"] = list(scenario_names)
    assumptions = result.get("assumptions")
    if isinstance(assumptions, dict):
        assumptions.pop("single_realization", None)
        assumptions["realizations"] = (
            f"{expected_trials} saved draws per target, using the preserved seed_target_index values. "
            "All saved draws are included; no trial is selected by fit outcome."
        )
    if case_name in {"50ppm", "100ppm"}:
        result["noise_levels_ppm"] = [float(CASE_DEFINITIONS[case_name]["noise_ppm"])]
    elif "noise_levels_ppm" in result:
        result.pop("noise_levels_ppm")
    return result


def _selector(case_name: str, noise_by_target: dict[str, float]) -> dict[str, Any]:
    if case_name == "JWST_Noise":
        return {
            "column": "noise_input_ppm",
            "kind": "per_target_equal",
            "values_ppm": noise_by_target,
        }
    return {
        "column": "noise_input_ppm",
        "kind": "equal",
        "value_ppm": float(CASE_DEFINITIONS[case_name]["noise_ppm"]),
    }


def _selected_rows(
    rows: list[dict[str, str]],
    target_names: list[str],
    noise_by_target: dict[str, float],
) -> list[dict[str, str]]:
    allowed = set(target_names)
    return [
        row
        for row in rows
        if row.get("target") in allowed
        and row.get("target") in noise_by_target
        and _same_noise(row.get("noise_input_ppm"), noise_by_target[row["target"]])
    ]


def _check_trial_groups(
    rows: list[dict[str, str]],
    *,
    expected_trials: int,
    source_path: Path,
) -> None:
    groups: dict[tuple[str, str, str], set[int]] = {}
    for row in rows:
        trial = _int(row.get("trial"))
        if trial is None:
            raise ValueError(f"Bad trial value in {source_path}: {row.get('trial')!r}")
        key = (str(row.get("target")), str(row.get("scenario")), str(row.get("fit_scope")))
        groups.setdefault(key, set()).add(trial)
    expected = set(range(expected_trials))
    incomplete = {key: sorted(value) for key, value in groups.items() if value != expected}
    if not groups:
        raise ValueError(f"The noise selector found no saved rows in {source_path}")
    if incomplete:
        example = next(iter(incomplete.items()))
        raise ValueError(
            f"Expected trials 0..{expected_trials - 1} for each target, scenario, and fit scope; "
            f"{example[0]} has {len(example[1])} trial(s) in {source_path}"
        )


def _harmonic_labels(ydeg: int) -> list[str]:
    labels: list[str] = []
    for ell in range(ydeg + 1):
        for order in range(-ell, ell + 1):
            signed = "m0" if order == 0 else (f"mpos{order}" if order > 0 else f"mneg{abs(order)}")
            labels.append(f"{MAP_LABEL_PREFIX}_l{ell}_{signed}")
    return labels


@lru_cache(maxsize=4096)
def _cached_uniform(subobserver_latitude: float, ydeg: int) -> tuple[float, ...]:
    coefficients = np.asarray(_uniform_coefficients(ydeg), dtype=float)
    visible = float(
        np.asarray(map_design_matrix(0.0, subobserver_latitude, ydeg), dtype=float)
        @ coefficients
    )
    if not math.isfinite(visible) or visible <= 0.0:
        raise ValueError("The uniform map has invalid full-phase flux")
    return tuple(float(value) for value in coefficients / visible)


@lru_cache(maxsize=8192)
def _cached_hotspot(
    longitude: float,
    width: float,
    latitude: float,
    subobserver_latitude: float,
    ydeg: int,
) -> tuple[float, ...]:
    from unity_injection_recovery import _hotspot_at_viewing_latitude

    values = _hotspot_at_viewing_latitude(
        longitude, width, latitude, ydeg, subobserver_latitude
    )
    return tuple(float(value) for value in np.asarray(values, dtype=float))


def _truth_map(
    target: dict[str, Any],
    scenario: str,
    analysis: dict[str, Any],
    ydeg: int,
    fraction_override: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    subobserver_latitude = math.radians(90.0 - float(target["inclination_degrees"]))
    width = float(analysis["injected_width_degrees"])
    latitude = float(analysis["injected_latitude_degrees"])
    fraction = float(analysis["injection_fraction"]) if fraction_override is None else fraction_override
    if scenario == "temperature_scaled":
        longitude: float | None = _temperature_offset(float(target["teq_k"]))
    elif scenario == "zero_offset":
        longitude = 0.0
    elif scenario == "constant_30_deg":
        longitude = float(analysis["constant_offset_degrees"])
    elif scenario == "uniform_null":
        longitude = None
    else:
        raise ValueError(f"Unknown saved scenario: {scenario}")

    if longitude is None:
        coefficients = np.asarray(_uniform_coefficients(ydeg), dtype=float)
        visible = float(
            np.asarray(map_design_matrix(0.0, subobserver_latitude, ydeg), dtype=float)
            @ coefficients
        )
        coefficients = coefficients / visible
        applied_fraction = 0.0
    else:
        coefficients = np.asarray(
            _mixture_coefficients(
                longitude,
                width,
                latitude,
                fraction,
                ydeg,
                subobserver_latitude,
            ),
            dtype=float,
        )
        applied_fraction = fraction
    minimum, peak_lon, peak_lat = _map_metrics(coefficients, ydeg)
    peak_shift = None if longitude is None else float(_wrap_degrees(peak_lon - longitude))
    return coefficients, {
        "injected_longitude_degrees": longitude,
        "injected_latitude_degrees": None if longitude is None else latitude,
        "injected_width_degrees": None if longitude is None else width,
        "source_injection_fraction": None if longitude is None else fraction,
        "injection_fraction_applied": applied_fraction,
        "minimum_intensity": minimum,
        "peak_longitude_degrees": None if longitude is None else peak_lon,
        "peak_latitude_degrees": None if longitude is None else peak_lat,
        "peak_shift_degrees": peak_shift,
        "full_phase_flux_scale": 1.0,
        "map_units": "surface intensity for unit planetary full-phase flux; multiply by depth_ppm for stellar-ppm scale",
    }


def _recovered_map(row: dict[str, Any], target: dict[str, Any], ydeg: int) -> tuple[np.ndarray, dict[str, Any]]:
    uniform_scale = float(row["best_uniform_map_amplitude_scale"])
    spot_scale = float(row["best_spot_amplitude_scale"])
    width = float(row["best_width_degrees"])
    latitude = float(row["best_latitude_degrees"])
    subobserver_latitude = math.radians(90.0 - float(target["inclination_degrees"]))
    uniform = np.asarray(_cached_uniform(subobserver_latitude, ydeg), dtype=float)

    saved_peak = _float(row.get("best_map_peak_longitude_degrees"))
    saved_shift = _float(row.get("best_map_peak_shift_degrees"))
    if spot_scale > 1.0e-12:
        if saved_peak is None or saved_shift is None:
            raise ValueError(f"{row.get('run_id')}: a non-zero spot has no saved peak and shift")
        hotspot_longitude = float(_wrap_degrees(saved_peak - saved_shift))
        hotspot = np.asarray(
            _cached_hotspot(
                hotspot_longitude,
                width,
                latitude,
                subobserver_latitude,
                ydeg,
            ),
            dtype=float,
        )
    else:
        hotspot_longitude = None
        hotspot = np.zeros_like(uniform)
    coefficients = uniform_scale * uniform + spot_scale * hotspot
    if coefficients.shape != ((ydeg + 1) ** 2,) or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{row.get('run_id')}: invalid retrieved map coefficients")
    return coefficients, {
        "best_uniform_map_amplitude_scale": uniform_scale,
        "best_spot_amplitude_scale": spot_scale,
        "best_spot_longitude_degrees": hotspot_longitude,
        "best_width_degrees": width,
        "best_latitude_degrees": latitude,
        "full_phase_flux_scale": uniform_scale + spot_scale,
        "depth_ppm": float(row["depth_ppm"]),
        "map_units": "surface intensity in planetary full-phase flux scale; multiply by depth_ppm for stellar-ppm scale",
    }


def _coefficient_row(
    row_metadata: dict[str, Any],
    coefficients: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    result = dict(row_metadata)
    result.update({label: float(value) for label, value in zip(labels, coefficients, strict=True)})
    return result


def _make_injection_table(
    targets: list[dict[str, Any]],
    scenario_names: list[str],
    analysis: dict[str, Any],
    ydeg: int,
    injection_fraction_by_target_scenario: dict[tuple[str, str], float | None],
) -> list[dict[str, Any]]:
    labels = _harmonic_labels(ydeg)
    rows: list[dict[str, Any]] = []
    for target in targets:
        for scenario in scenario_names:
            coefficients, details = _truth_map(
                target,
                scenario,
                analysis,
                ydeg,
                fraction_override=injection_fraction_by_target_scenario.get((target["name"], scenario)),
            )
            row = {
                "target": target["name"],
                "instrument": target.get("instrument", ""),
                "teq_k": target.get("teq_k"),
                "brightness_temperature_k": target.get("brightness_temperature_k"),
                "depth_ppm": target.get("depth_ppm"),
                "scenario": scenario,
                "harmonic_degree": ydeg,
                **details,
            }
            rows.append(_coefficient_row(row, coefficients, labels))
    return rows


def _mean(values: list[float]) -> float | str:
    return float(np.mean(values)) if values else ""


def _quantile(values: list[float], q: float) -> float | str:
    return float(np.quantile(values, q)) if values else ""


def _make_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["target"], row["scenario"], row["fit_scope"])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2])):
        group = groups[key]
        delta = [value for value in (_float(row.get("delta_bic")) for row in group) if value is not None]
        bias = [value for value in (_float(row.get("longitude_bias_degrees")) for row in group) if value is not None]
        spot_fraction = [
            value
            for value in (_float(row.get("best_recovered_spot_fraction")) for row in group)
            if value is not None
        ]
        detected = [_bool(row.get("map_detected")) for row in group]
        detected_count = sum(detected)
        recovered_longitude = [
            value
            for value in (_float(row.get("recovered_longitude_median_degrees")) for row in group)
            if value is not None
        ]
        first = group[0]
        bias_std = float(np.std(bias, ddof=0)) if bias else ""
        output.append(
            {
                "target": key[0],
                "noise_input_ppm": first.get("noise_input_ppm"),
                "noise_ppm": first.get("noise_ppm"),
                "scenario": key[1],
                "fit_scope": key[2],
                "trial_count": len({row.get("trial") for row in group}),
                "row_count": len(group),
                "detected_trials": detected_count,
                "not_detected_trials": len(detected) - detected_count,
                "mean_delta_bic": _mean(delta),
                "median_delta_bic": _quantile(delta, 0.5),
                "map_detection_fraction": _mean([float(value) for value in detected]),
                "mean_longitude_bias_degrees": _mean(bias),
                "median_longitude_bias_degrees": _quantile(bias, 0.5),
                "longitude_bias_std_degrees": bias_std,
                "longitude_bias_1sigma_low_degrees": _quantile(bias, 0.158655),
                "longitude_bias_1sigma_high_degrees": _quantile(bias, 0.841345),
                "longitude_bias_2sigma_low_degrees": _quantile(bias, 0.02275),
                "longitude_bias_2sigma_high_degrees": _quantile(bias, 0.97725),
                "recovered_longitude_1sigma_low_degrees": _quantile(recovered_longitude, 0.158655),
                "recovered_longitude_1sigma_high_degrees": _quantile(recovered_longitude, 0.841345),
                "recovered_longitude_2sigma_low_degrees": _quantile(recovered_longitude, 0.02275),
                "recovered_longitude_2sigma_high_degrees": _quantile(recovered_longitude, 0.97725),
                "median_recovered_longitude_degrees": _quantile(recovered_longitude, 0.5),
                "mean_recovered_spot_fraction": _mean(spot_fraction),
            }
        )
    return output


def _relative_files(paths: list[Path], start: Path) -> list[str]:
    return [Path(os.path.relpath(path.resolve(), start.resolve())).as_posix() for path in paths]


def _profile_source(
    *,
    source_label: str,
    source_root: Path,
    bundle_name: str,
    output_dir: Path,
    case_name: str,
    selected_rows: list[dict[str, str]],
    noise_by_target: dict[str, float],
) -> dict[str, Any]:
    bundle = source_root / bundle_name
    if source_label == "reference":
        archive_folder = STUDY_ROOT / "archive" / bundle_name
        candidates = sorted(archive_folder.glob("profiles.npz*"))
        source_bundle = archive_folder
        restore_command = f"python UNITY_Theory/archive_profiles.py --restore --study {bundle_name}"
    else:
        candidates = sorted(bundle.glob("profiles.npz*"))
        if not candidates:
            archive_folder = STUDY_ROOT / "archive" / bundle_name
            candidates = sorted(archive_folder.glob("profiles.npz*"))
            source_bundle = archive_folder
        else:
            source_bundle = bundle
        restore_command = None
    return {
        "schema_version": 1,
        "purpose": "Pointer to the centralized lossless fit-profile archive. No profile binary is copied into a case folder.",
        "source_case": case_name,
        "relative_to": "case root",
        "source_bundle": Path(os.path.relpath(source_bundle.resolve(), output_dir.resolve())).as_posix(),
        "profiles_files": _relative_files(candidates, output_dir),
        "profiles_found": bool(candidates),
        "restore_command": restore_command,
        "noise_selector": _selector(case_name, noise_by_target),
        "selected_row_count": len(selected_rows),
        "selected_targets": sorted({row["target"] for row in selected_rows}),
        "selected_scenarios": sorted({row["scenario"] for row in selected_rows}),
        "selected_fit_scopes": sorted({row["fit_scope"] for row in selected_rows}),
        "selected_trials": sorted({_int(row["trial"], default=-1) for row in selected_rows}),
    }


def _case_readme(case_name: str, target_count: int, scenarios: list[str], trials: int) -> str:
    if case_name == "JWST_Noise":
        noise_text = "This case uses the saved JWST noise value for each target."
        input_path = "UNITY_Theory/JWST_Noise/01_initialisation/input.json"
    else:
        noise_text = f"This case uses a noise input of {CASE_DEFINITIONS[case_name]['noise_ppm']:g} ppm for each target."
        input_path = f"UNITY_Theory/{case_name}/01_initialisation/input.json"
    output_path = f"UNITY_Theory/generated/{case_name}"
    simulation_command = (
        "python UNITY_Theory/code/unity_injection_recovery.py "
        f"--inputs {input_path} --output {output_path} --trials {trials} "
        "--exposure-subsamples 8 --disk-radial 32 --disk-azimuth 128 --chunk-size 128"
    )
    return f"""# {case_name}

This folder shows the saved run from input settings through injected and retrieved maps. {noise_text}

The organizer does not run fits. It selects the saved rows for this noise case. The saved trial number stays in every table. The trial 0 light curves and maps use the saved row `trial = 0`; they do not use a row chosen by its fit result.

The saved `seed_target_index` stays with each target. The run uses paired standard noise draws across noise levels and scenarios. The trial number and seed index keep that pairing when you rerun one target.

## Folders

- `01_initialisation/` has the case input and saved run settings.
- `02_injection/` has the injected degree-2 map coefficients for each target and scenario.
- `03_light_curves/` has one CSV file per target. These files contain trial-0 rows for both fit scopes.
- `04_retrieval/recovery_trials.csv` has every saved draw and fit scope for this noise case.
- `04_retrieval/retrieved_map_coefficients.csv` has the nine best-fit harmonic coefficients for every recovery row.
- `04_retrieval/profiles_source.json` points to the shared fit-profile archive.
- `05_summary/` has the case summary, map figures, and figure caption.

The main result has {target_count} targets, {trials} draws per target, and these scenarios: {', '.join(scenarios)}. It has one row for each draw and fit scope. The case summary gives detection counts out of all draws. It also gives central 1σ and 2σ ranges for longitude bias and recovered longitude, plus the bias standard deviation. The map coefficients use the canonical degree-2 spherical-harmonic basis. Injected maps have unit planetary full-phase flux. Retrieved maps use the saved uniform and hotspot amplitudes. The `depth_ppm` value converts map intensity to stellar-ppm scale. The right map is the conditional best fit from trial 0, even when its saved detection result is false.

Run the organizer again with `python UNITY_Theory/run.py organize --source reference`. After a new complete run, use `python UNITY_Theory/run.py organize --source generated`.

To run this case from its saved input alone, use:

```sh
{simulation_command}
```

After running this case alone, rebuild its maps from that output with:

```sh
python UNITY_Theory/code/build_case_folders.py --source generated --case {case_name} --bundle-directory {output_path} --output-root UNITY_Theory/generated/cases --expected-trials {trials}
```
"""


def _validate_map_metrics(
    row: dict[str, Any],
    coefficients: np.ndarray,
    *,
    truth: bool,
    ydeg: int,
) -> dict[str, Any]:
    minimum, peak_lon, peak_lat = _map_metrics(coefficients, ydeg)
    if truth:
        saved_min = _float(row.get("truth_map_minimum_intensity"))
        saved_peak_lon = _float(row.get("truth_map_peak_longitude_degrees"))
        saved_peak_lat = _float(row.get("truth_map_peak_latitude_degrees"))
        peak_key = "truth"
    else:
        saved_min = _float(row.get("best_map_minimum_intensity"))
        saved_peak_lon = _float(row.get("best_map_peak_longitude_degrees"))
        saved_peak_lat = _float(row.get("best_map_peak_latitude_degrees"))
        peak_key = "best-fit"
    if saved_min is None or not math.isclose(minimum, saved_min, rel_tol=0.0, abs_tol=2.0e-6):
        raise ValueError(
            f"{row.get('run_id')}: reconstructed {peak_key} minimum {minimum:.12g} "
            f"does not match saved value {saved_min}"
        )
    if saved_peak_lon is not None and abs(float(_wrap_degrees(peak_lon - saved_peak_lon))) > 0.500001:
        raise ValueError(
            f"{row.get('run_id')}: reconstructed {peak_key} longitude {peak_lon:g} "
            f"does not match saved value {saved_peak_lon:g} within 0.5 degree"
        )
    if saved_peak_lat is not None and abs(peak_lat - saved_peak_lat) > 0.500001:
        raise ValueError(
            f"{row.get('run_id')}: reconstructed {peak_key} latitude {peak_lat:g} "
            f"does not match saved value {saved_peak_lat:g} within 0.5 degree"
        )
    return {
        "minimum_intensity": minimum,
        "peak_longitude_degrees": peak_lon,
        "peak_latitude_degrees": peak_lat,
        "saved_minimum_intensity": saved_min,
        "saved_peak_longitude_degrees": saved_peak_lon,
        "saved_peak_latitude_degrees": saved_peak_lat,
    }


def _write_retrieved_maps(
    selected_rows: list[dict[str, str]],
    targets_by_name: dict[str, dict[str, Any]],
    seed_by_target: dict[str, int],
    ydeg: int,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    labels = _harmonic_labels(ydeg)
    map_rows: list[dict[str, Any]] = []
    coefficients_by_run: dict[str, np.ndarray] = {}
    validation: dict[str, Any] = {
        "schema_version": 1,
        "checked_trial_zero_rows": 0,
        "checked_finite_coefficient_rows": 0,
        "maximum_trial_zero_minimum_absolute_error": 0.0,
        "maximum_trial_zero_peak_error_degrees": 0.0,
        "metric_tolerance": {
            "minimum_absolute": 2.0e-6,
            "peak_degrees": 0.500001,
        },
    }
    for row in selected_rows:
        target = targets_by_name[row["target"]]
        coefficients, details = _recovered_map(row, target, ydeg)
        run_id = row.get("run_id", "")
        coefficients_by_run[run_id] = coefficients
        metadata = {
            "run_id": run_id,
            "target": row["target"],
            "seed_target_index": seed_by_target[row["target"]],
            "instrument": row.get("instrument", ""),
            "teq_k": row.get("teq_k"),
            "depth_ppm": row.get("depth_ppm"),
            "noise_input_ppm": row.get("noise_input_ppm"),
            "noise_ppm": row.get("noise_ppm"),
            "scenario": row.get("scenario"),
            "trial": row.get("trial"),
            "random_seed": row.get("random_seed"),
            "fit_scope": row.get("fit_scope"),
            "harmonic_degree": ydeg,
            **details,
        }
        map_rows.append(_coefficient_row(metadata, coefficients, labels))
        validation["checked_finite_coefficient_rows"] += 1
        if _int(row.get("trial")) == 0:
            metrics = _validate_map_metrics(row, coefficients, truth=False, ydeg=ydeg)
            error = abs(metrics["minimum_intensity"] - float(metrics["saved_minimum_intensity"]))
            validation["maximum_trial_zero_minimum_absolute_error"] = max(
                validation["maximum_trial_zero_minimum_absolute_error"], error
            )
            saved_lon = metrics["saved_peak_longitude_degrees"]
            if saved_lon is not None and float(row["best_spot_amplitude_scale"]) > 1.0e-12:
                peak_error = abs(float(_wrap_degrees(metrics["peak_longitude_degrees"] - saved_lon)))
                validation["maximum_trial_zero_peak_error_degrees"] = max(
                    validation["maximum_trial_zero_peak_error_degrees"], peak_error
                )
            validation["checked_trial_zero_rows"] += 1
    _write_csv(output_path, list(map_rows[0]) if map_rows else ["run_id"], map_rows)
    validation["result"] = "passed"
    return map_rows, coefficients_by_run, validation


def _write_light_curves(
    rows: list[dict[str, str]],
    examples_dir: Path,
    output_dir: Path,
) -> list[str]:
    if not examples_dir.is_dir():
        return []
    targets = sorted({row["target"] for row in rows})
    output_files: list[str] = []
    for target in targets:
        source_path = examples_dir / f"{_slug(target)}_curves.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"No trial-0 light curves for {target}: {source_path}")
        fields, examples = _read_csv(source_path)
        selected_pairs = {
            (item["scenario"], item["fit_scope"], float(item["noise_input_ppm"]))
            for item in rows
            if item.get("target") == target
        }
        target_rows = [
            row for row in examples
            if row.get("target") == target
            and row.get("trial") == "0"
            and _float(row.get("noise_input_ppm")) is not None
            and (
                row.get("scenario"),
                row.get("fit_scope"),
                float(row["noise_input_ppm"]),
            ) in selected_pairs
        ]
        slug = _slug(target)
        destination = output_dir / f"{slug}_trial000.csv"
        _write_csv(destination, fields, target_rows)
        output_files.append(destination.name)
    return output_files


def _slug(text: str) -> str:
    return "_".join("".join(character.lower() if character.isalnum() else " " for character in text).split())


def _write_truth_table(
    injection_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _write_csv(output_path, list(injection_rows[0]) if injection_rows else ["target"], injection_rows)


def _make_validation(
    selected_rows: list[dict[str, str]],
    injection_by_target_scenario: dict[tuple[str, str], np.ndarray],
    targets_by_name: dict[str, dict[str, Any]],
    analysis: dict[str, Any],
    ydeg: int,
    recovery_validation: dict[str, Any],
) -> dict[str, Any]:
    checked = 0
    max_truth_minimum_error = 0.0
    max_truth_peak_error = 0.0
    for row in selected_rows:
        if _int(row.get("trial")) != 0:
            continue
        target = targets_by_name[row["target"]]
        key = (row["target"], row["scenario"])
        coefficients = injection_by_target_scenario[key]
        metrics = _validate_map_metrics(row, coefficients, truth=True, ydeg=ydeg)
        max_truth_minimum_error = max(
            max_truth_minimum_error,
            abs(metrics["minimum_intensity"] - float(metrics["saved_minimum_intensity"])),
        )
        saved_lon = metrics["saved_peak_longitude_degrees"]
        saved_lat = metrics["saved_peak_latitude_degrees"]
        # Uniform-null maps have no unique peak in the saved results.
        if row.get("scenario") != "uniform_null" and saved_lon is not None:
            max_truth_peak_error = max(
                max_truth_peak_error,
                abs(float(_wrap_degrees(metrics["peak_longitude_degrees"] - saved_lon))),
                abs(metrics["peak_latitude_degrees"] - float(saved_lat)),
            )
        checked += 1
    return {
        "schema_version": 1,
        "result": "passed",
        "checked_truth_trial_zero_rows": checked,
        "maximum_truth_minimum_absolute_error": max_truth_minimum_error,
        "maximum_truth_peak_error_degrees": max_truth_peak_error,
        "map_metrics": recovery_validation,
        "reconstruction": {
            "truth": "Canonical _mixture_coefficients; uniform_null uses _uniform_coefficients with the full-phase normalization.",
            "recovery": "Normalized uniform template times best_uniform_map_amplitude_scale plus _hotspot_at_viewing_latitude times best_spot_amplitude_scale.",
            "peak_longitude": "best_map_peak_longitude_degrees minus best_map_peak_shift_degrees, wrapped to [-180, 180).",
            "map_grid": "Canonical _map_metrics at 0.5 degree spacing.",
        },
    }


def _plot_map_pairs(
    output_dir: Path,
    target_list: list[dict[str, Any]],
    selected_rows: list[dict[str, str]],
    injection_by_target_scenario: dict[tuple[str, str], np.ndarray],
    coefficients_by_run: dict[str, np.ndarray],
    scenario_name: str,
) -> list[dict[str, Any]]:
    apply_style()
    longitude = np.arange(-180.0, 180.01, 2.0)
    latitude = np.arange(-90.0, 90.01, 2.0)
    lon_grid, lat_grid = np.meshgrid(np.deg2rad(longitude), np.deg2rad(latitude))
    target_rows = []
    for target in target_list:
        candidates = [
            row for row in selected_rows
            if row["target"] == target["name"]
            and row.get("scenario") == scenario_name
            and row.get("fit_scope") == "full_window"
            and _int(row.get("trial")) == 0
        ]
        if not candidates:
            raise ValueError(f"No trial-0 full-window map for {target['name']} / {scenario_name}")
        target_rows.append((target, candidates[0]))
    pair_metadata = [
        {
            "target": target["name"],
            "run_id": row.get("run_id"),
            "scenario": row.get("scenario"),
            "trial": row.get("trial"),
            "fit_scope": row.get("fit_scope"),
            "noise_input_ppm": row.get("noise_input_ppm"),
            "noise_ppm": row.get("noise_ppm"),
            "delta_bic": row.get("delta_bic"),
            "delta_bic_detection_threshold": row.get("delta_bic_detection_threshold"),
            "map_detected": row.get("map_detected"),
            "displayed_map": "conditional best-fit map from the saved fit, including when map_detected is false",
        }
        for target, row in target_rows
    ]

    figure = plt.figure(figsize=(8.4, 11.8), constrained_layout=False)
    grid = figure.add_gridspec(
        len(target_rows), 3, width_ratios=(1.0, 1.0, 0.045),
        left=0.13, right=0.94, top=0.985, bottom=0.045, hspace=0.38, wspace=0.18,
    )
    for row_index, (target, trial_row) in enumerate(target_rows):
        truth = injection_by_target_scenario[(target["name"], scenario_name)]
        recovered = coefficients_by_run[trial_row.get("run_id", "")]
        depth_ppm = float(target["depth_ppm"])
        truth_image = np.asarray(evaluate_map(truth, lon_grid, lat_grid), dtype=float) * depth_ppm
        recovered_image = np.asarray(evaluate_map(recovered, lon_grid, lat_grid), dtype=float) * depth_ppm
        low = float(min(np.min(truth_image), np.min(recovered_image)))
        high = float(max(np.max(truth_image), np.max(recovered_image)))
        if math.isclose(low, high, rel_tol=0.0, abs_tol=1.0e-14):
            high = low + max(abs(low) * 1.0e-8, 1.0e-8)
        axes = [figure.add_subplot(grid[row_index, col]) for col in (0, 1)]
        for axis, image in zip(axes, (truth_image, recovered_image), strict=True):
            mesh = axis.pcolormesh(
                longitude,
                latitude,
                image,
                cmap="Purples",
                vmin=low,
                vmax=high,
                shading="nearest",
                rasterized=True,
            )
            style_axis(axis)
            axis.set_xlim(-180, 180)
            axis.set_ylim(-90, 90)
            axis.set_xticks([-180, -90, 0, 90, 180])
            axis.set_yticks([-90, -45, 0, 45, 90])
            axis.set_xlabel("Longitude (deg)")
            axis.set_aspect("equal", adjustable="box")
        axes[0].set_ylabel("Latitude (deg)")
        axes[1].set_yticklabels([])
        colour_axis = figure.add_subplot(grid[row_index, 2])
        colourbar = figure.colorbar(mesh, cax=colour_axis)
        colourbar.ax.tick_params(labelsize=7, width=0.7, length=2.5, direction="in")
        colourbar.set_label("Scaled intensity (ppm)", fontsize=7.0, labelpad=2)
        row_y = 1.0 - (row_index + 0.5) / len(target_rows)
        figure.text(0.018, row_y, str(target["name"]), rotation=90, ha="center", va="center", fontsize=7.5)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"injection_recovery_maps.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    target_order = "\n".join(f"{index}. {target['name']}" for index, (target, _) in enumerate(target_rows, start=1))
    status_text = "\n".join(
        f"{row['target']}: delta BIC = {row['delta_bic']}; map_detected = {row['map_detected']}. "
        "The displayed right-hand map is the conditional best fit, including for a non-detection."
        for row in pair_metadata
    )
    caption = (
        "Injected and retrieved map pairs for the temperature_scaled scenario. "
        "The left map in each row is the injected map. The right map is the ‘Best-fit map, trial 0’: the conditional best-fit map from the saved `trial = 0`, `fit_scope = full_window` row. "
        "The same purple colour scale is used for both maps in each row. Colour shows reconstructed surface intensity multiplied by the target planet depth, in stellar-ppm scale; it is not an observed light-curve sample. "
        "Longitude is measured east from the substellar point. Row order, top to bottom:\n" + target_order + "\n"
        "Detection status for each right-hand map:\n" + status_text + "\n"
    )
    (output_dir / "injection_recovery_maps_caption.txt").write_text(caption, encoding="utf-8")
    return pair_metadata


def _build_one_case(
    *,
    case_name: str,
    source_label: str,
    source_root: Path,
    output_dir: Path,
    expected_trials: int,
    make_figures: bool,
    write_readme: bool = True,
    bundle_override: str | None = None,
    input_name_override: str | None = None,
) -> dict[str, Any]:
    bundle_name = bundle_override or (
        CASE_DEFINITIONS[case_name]["bundle"] if case_name in CASE_DEFINITIONS else FINEGRID_BUNDLE
    )
    run_input_path, summary_path, trial_path, examples_dir = _bundle_paths(source_root, bundle_name)
    run_inputs = _read_json(run_input_path)
    summary = _read_json(summary_path)
    input_snapshot = run_inputs.get("input_snapshot")
    if not isinstance(input_snapshot, dict):
        raise ValueError(f"Saved run inputs have no input_snapshot object: {run_input_path}")
    analysis = run_inputs.get("resolved_analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError(f"Saved run inputs have no resolved_analysis object: {run_input_path}")
    run_settings = run_inputs.get("run_settings", {})
    if not isinstance(run_settings, dict):
        run_settings = {}
    ydeg = _int(analysis.get("harmonic_degree"), default=2) or 2
    if ydeg != 2:
        raise ValueError(f"Expected degree-2 UNITY maps, found harmonic degree {ydeg} in {run_input_path}")
    scenario_names = [str(value) for value in summary.get("scenario_names", analysis.get("scenarios", []))]
    if not scenario_names:
        raise ValueError(f"No scenarios in saved summary: {summary_path}")
    targets_by_name = {
        str(target.get("name")): target
        for target in input_snapshot.get("targets", [])
        if isinstance(target, dict) and target.get("name") is not None
    }
    if bundle_name == FINEGRID_BUNDLE:
        # The supplementary run contains only WASP-12 b.
        target_names = [name for name in targets_by_name if name == "WASP-12 b"]
        if not target_names:
            target_names = sorted(targets_by_name)
        noise_by_target = {name: float(case_name.split("ppm")[0]) for name in target_names}
        output_case_name = case_name
    else:
        target_names = list(targets_by_name)
        noise_by_target = _noise_map(case_name, target_names, input_snapshot, run_settings)
        output_case_name = case_name
    target_list = [targets_by_name[name] for name in target_names]
    seed_metadata = _target_metadata(summary, input_snapshot)
    seed_by_target = {name: seed_metadata[name]["seed_target_index"] for name in target_names}

    source_fields, source_rows = _read_csv(trial_path)
    selected_rows = _selected_rows(source_rows, target_names, noise_by_target)
    _check_trial_groups(selected_rows, expected_trials=expected_trials, source_path=trial_path)
    for row in selected_rows:
        row["seed_target_index"] = seed_by_target[row["target"]]
        row["depth_ppm"] = float(targets_by_name[row["target"]]["depth_ppm"])

    output_dir.mkdir(parents=True, exist_ok=True)
    initialisation = output_dir / "01_initialisation"
    injection = output_dir / "02_injection"
    light_curves = output_dir / "03_light_curves"
    retrieval = output_dir / "04_retrieval"
    summary_dir = output_dir / "05_summary"
    for folder in (initialisation, injection, light_curves, retrieval, summary_dir):
        folder.mkdir(parents=True, exist_ok=True)

    case_input = _prepare_case_input(
        input_snapshot, target_names, noise_by_target, scenario_names, output_case_name, expected_trials
    )
    for target in case_input["targets"]:
        target["seed_target_index"] = seed_by_target[str(target["name"])]
    _write_json(initialisation / "input.json", case_input)
    selector = _selector(output_case_name, noise_by_target)
    settings = {
        "schema_version": 1,
        "case": output_case_name,
        "source": source_label,
        "source_bundle": f"{source_label}/{bundle_name}",
        "source_run_inputs": f"{source_label}/{bundle_name}/run_inputs.json",
        "source_recovery_trials": f"{source_label}/{bundle_name}/recovery_trials.csv",
        "input_file": input_name_override or CASE_DEFINITIONS.get(case_name, {}).get("input_name", "wasp12_finegrid.json"),
        "noise_selector": selector,
        "noise_input_ppm_by_target": noise_by_target,
        "targets": target_names,
        "seed_target_index_by_target": seed_by_target,
        "scenarios": scenario_names,
        "fit_scopes": sorted({row["fit_scope"] for row in selected_rows}),
        "trials_requested": expected_trials,
        "trials_per_target_scenario_fit_scope": expected_trials,
        "selected_recovery_rows": len(selected_rows),
        "master_seed": run_settings.get("master_seed", input_snapshot.get("seed")),
        "resolved_analysis": analysis,
        "fit_recomputed": False,
        "map_reconstruction": "The organizer uses the saved fit amplitudes and canonical UNITY map functions. It does not run a fit.",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _write_json(initialisation / "settings.json", settings)

    injection_fraction_by_target_scenario: dict[tuple[str, str], float | None] = {}
    for row in selected_rows:
        key = (row["target"], row["scenario"])
        fraction = _float(row.get("injection_fraction"))
        if key in injection_fraction_by_target_scenario:
            saved_fraction = injection_fraction_by_target_scenario[key]
            if saved_fraction is None:
                if fraction is not None:
                    raise ValueError(f"Inconsistent injection_fraction values for {key}")
            elif fraction is None or not math.isclose(saved_fraction, fraction, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"Inconsistent injection_fraction values for {key}")
        else:
            injection_fraction_by_target_scenario[key] = fraction
    injection_rows = _make_injection_table(
        target_list,
        scenario_names,
        analysis,
        ydeg,
        injection_fraction_by_target_scenario,
    )
    _write_truth_table(injection_rows, injection / "injected_map_coefficients.csv")
    injection_by_target_scenario = {
        (row["target"], row["scenario"]): np.asarray(
            [row[label] for label in _harmonic_labels(ydeg)], dtype=float
        )
        for row in injection_rows
    }

    example_files = _write_light_curves(selected_rows, examples_dir, light_curves)

    recovery_fields = list(source_fields)
    for extra_field in ("seed_target_index", "depth_ppm"):
        if extra_field not in recovery_fields:
            recovery_fields.append(extra_field)
    _write_csv(retrieval / "recovery_trials.csv", recovery_fields, selected_rows)
    map_rows, coefficients_by_run, map_validation = _write_retrieved_maps(
        selected_rows,
        targets_by_name,
        seed_by_target,
        ydeg,
        retrieval / "retrieved_map_coefficients.csv",
    )
    validation = _make_validation(
        selected_rows,
        injection_by_target_scenario,
        targets_by_name,
        analysis,
        ydeg,
        map_validation,
    )
    _write_json(retrieval / "map_reconstruction_check.json", validation)
    profiles = _profile_source(
        source_label=source_label,
        source_root=source_root,
        bundle_name=bundle_name,
        output_dir=output_dir,
        case_name=output_case_name,
        selected_rows=selected_rows,
        noise_by_target=noise_by_target,
    )
    _write_json(retrieval / "profiles_source.json", profiles)

    summary_rows = _make_summary(selected_rows)
    _write_csv(summary_dir / "case_summary.csv", list(summary_rows[0]), summary_rows)
    temperature_rows = [
        row for row in selected_rows
        if row.get("scenario") == "temperature_scaled"
        and row.get("fit_scope") == "full_window"
        and _int(row.get("trial")) == 0
    ]
    map_pair_rows = []
    for target in target_list:
        candidates = [row for row in temperature_rows if row.get("target") == target["name"]]
        if not candidates:
            continue
        row = candidates[0]
        map_pair_rows.append(
            {
                "target": target["name"],
                "run_id": row.get("run_id"),
                "scenario": row.get("scenario"),
                "trial": row.get("trial"),
                "fit_scope": row.get("fit_scope"),
                "noise_input_ppm": row.get("noise_input_ppm"),
                "noise_ppm": row.get("noise_ppm"),
                "delta_bic": row.get("delta_bic"),
                "delta_bic_detection_threshold": row.get("delta_bic_detection_threshold"),
                "map_detected": row.get("map_detected"),
                "displayed_map": "conditional best-fit map from the saved fit, including when map_detected is false",
            }
        )
    if map_pair_rows:
        _write_csv(summary_dir / "map_pair_metadata.csv", list(map_pair_rows[0]), map_pair_rows)
    if make_figures:
        _plot_map_pairs(
            summary_dir,
            target_list,
            selected_rows,
            injection_by_target_scenario,
            coefficients_by_run,
            "temperature_scaled",
        )
    if write_readme:
        (output_dir / "README.md").write_text(
            _case_readme(output_case_name, len(target_list), scenario_names, expected_trials),
            encoding="utf-8",
        )
    return {
        "case": output_case_name,
        "output": str(output_dir),
        "target_count": len(target_list),
        "scenario_count": len(scenario_names),
        "trials_per_group": expected_trials,
        "recovery_rows": len(selected_rows),
        "retrieved_map_rows": len(map_rows),
        "trial_zero_map_checks": validation["checked_truth_trial_zero_rows"],
        "profile_files": len(profiles["profiles_files"]),
        "light_curve_files": len(example_files),
        "validation": validation["result"],
    }


def _build_finegrid_supplement(
    *,
    source_label: str,
    source_root: Path,
    output_root: Path,
    noise_ppm: float,
    expected_trials: int,
    make_figures: bool,
) -> dict[str, Any]:
    case_name = f"{int(noise_ppm)}ppm"
    output = output_root / case_name / "supplementary" / "wasp12_finegrid"
    result = _build_one_case(
        case_name=case_name,
        source_label=source_label,
        source_root=source_root,
        output_dir=output,
        expected_trials=expected_trials,
        make_figures=make_figures,
        write_readme=False,
        bundle_override=FINEGRID_BUNDLE,
        input_name_override="wasp12_finegrid.json",
    )
    (output / "README.md").write_text(
        f"""# WASP-12 b fine-grid supplement at {int(noise_ppm)} ppm

This folder has the saved one-target fine-grid results filtered to `noise_input_ppm = {int(noise_ppm)}`. It contains {expected_trials} draws per scenario and fit scope. It does not run fits. Its source profile archive is linked in `04_retrieval/profiles_source.json`.

The normal six-target case folder is the primary result. This fine-grid run is a one-target supplement for WASP-12 b.
""",
        encoding="utf-8",
    )
    result["case"] = f"{case_name}/supplementary/wasp12_finegrid"
    return result


def build_case_folders(
    *,
    source: str = "reference",
    output_root: Path | str | None = None,
    source_root: Path | str | None = None,
    case_names: list[str] | None = None,
    expected_trials: int = 100,
    make_figures: bool = True,
    include_finegrid: bool = True,
) -> list[dict[str, Any]]:
    """Build case folders from a reference or generated source bundle."""
    if source not in {"reference", "generated"}:
        raise ValueError("source must be 'reference' or 'generated'")
    if expected_trials < 1:
        raise ValueError("expected_trials must be positive")
    if source_root is None:
        source_base = STUDY_ROOT / ("reference" if source == "reference" else "generated")
    else:
        source_base = Path(source_root).resolve()
    if output_root is None:
        destination_root = STUDY_ROOT if source == "reference" else STUDY_ROOT / "generated" / "cases"
    else:
        destination_root = Path(output_root).resolve()
    names = list(CASE_DEFINITIONS) if case_names is None else case_names
    unknown = [name for name in names if name not in CASE_DEFINITIONS]
    if unknown:
        raise ValueError("Unknown case name(s): " + ", ".join(unknown))
    results = [
        _build_one_case(
            case_name=name,
            source_label=source,
            source_root=source_base,
            output_dir=destination_root / name,
            expected_trials=expected_trials,
            make_figures=make_figures,
        )
        for name in names
    ]
    if include_finegrid and any(name in {"50ppm", "100ppm"} for name in names):
        for noise in (50.0, 100.0):
            case_name = f"{int(noise)}ppm"
            if case_name in names:
                results.append(
                    _build_finegrid_supplement(
                        source_label=source,
                        source_root=source_base,
                        output_root=destination_root,
                        noise_ppm=noise,
                        expected_trials=expected_trials,
                        make_figures=False,
                    )
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("reference", "generated"), default="reference")
    parser.add_argument("--output-root", type=Path, help="Root for case folders")
    parser.add_argument("--case", choices=tuple(CASE_DEFINITIONS), action="append", help="Build one case; repeat to build more than one")
    parser.add_argument("--expected-trials", type=int, default=100, help="Number of complete saved draws required per target, scenario, and fit scope")
    parser.add_argument("--bundle-directory", type=Path,
                        help="Standalone runner output; requires exactly one --case and --source generated")
    args = parser.parse_args()
    if args.expected_trials < 1:
        parser.error("--expected-trials must be positive")
    if args.bundle_directory is not None:
        if args.source != "generated" or not args.case or len(args.case) != 1:
            parser.error("--bundle-directory requires --source generated and exactly one --case")
        destination = args.output_root or STUDY_ROOT / "generated" / "cases"
        bundle = args.bundle_directory.resolve()
        results = [_build_one_case(
            case_name=args.case[0], source_label="generated", source_root=bundle.parent,
            output_dir=destination / args.case[0], expected_trials=args.expected_trials,
            make_figures=True, bundle_override=bundle.name,
            input_name_override="standalone run input_snapshot",
        )]
    else:
        results = build_case_folders(
            source=args.source,
            output_root=args.output_root,
            case_names=args.case,
            expected_trials=args.expected_trials,
        )
    for result in results:
        print(
            f"{result['case']}: {result['recovery_rows']} recovery rows, "
            f"{result['retrieved_map_rows']} retrieved maps; {result['validation']}"
        )


if __name__ == "__main__":
    main()
