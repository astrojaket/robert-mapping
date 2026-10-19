#!/usr/bin/env python3
"""Extract PandExo white-light noise for the UNITY NIRSpec targets.

The saved online results are trusted only after a restricted unpickle.  The
calculation uses the PandExo ``fml`` electron counts and variance, summed over
2.9--5.1 micron and scaled by wall time to a 60-second sample.  NumPy is
required.  Pysynphot is optional and enables the PHOENIX shape audit.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = ROOT / "UNITY_Theory"
DEFAULT_MANIFEST = (
    STUDY_ROOT / "data" / "expected_noise_calculations" / "online" / "manifest.json"
)
DEFAULT_EVIDENCE = (
    STUDY_ROOT
    / "data"
    / "expected_noise_inputs"
    / "pid10786_detector_magnitudes_evidence.json"
)
DEFAULT_OUTPUT_DIR = STUDY_ROOT / "generated" / "expected_noise_calculations"
DEFAULT_JSON = DEFAULT_OUTPUT_DIR / "expected_noise.json"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "expected_noise_report.txt"
WAVE_MIN_UM = 2.9
WAVE_MAX_UM = 5.1
CADENCE_SECONDS = 60.0


class RestrictedUnpickler(pickle.Unpickler):
    """Load only the NumPy types used by PandExo result dictionaries."""

    _allowed = {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(f"Blocked pickle global: {module}.{name}")
        if name == "ndarray":
            return np.ndarray
        if name == "dtype":
            return np.dtype
        try:
            multiarray = np._core.multiarray
        except AttributeError:  # NumPy 1.x
            multiarray = np.core.multiarray
        if name == "scalar":
            return multiarray.scalar
        return multiarray._reconstruct


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = RestrictedUnpickler(handle).load()
    if not isinstance(value, dict):
        raise ValueError(f"Expected a PandExo result dictionary in {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _resolve_pickle_path(pickle_path: str | Path, manifest_dir: Path) -> Path:
    """Resolve relative manifest paths before trying the legacy basename layout."""
    path = Path(pickle_path)
    if not path.is_absolute():
        path = manifest_dir / path
    if path.is_file():
        return path
    fallback = manifest_dir / Path(pickle_path).name
    return fallback if fallback.is_file() else path


def _finite_scalar(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def _close(actual: float, expected: float, *, atol: float = 1.0e-6) -> bool:
    return abs(actual - expected) <= atol


def _match_evidence(manifest_name: str, evidence: dict[str, Any]) -> dict[str, Any]:
    short_name = manifest_name.removesuffix(" b")
    for item in evidence.get("targets", []):
        if item.get("target") == short_name:
            return item
    raise ValueError(f"No input evidence row matches {manifest_name!r}")


def _array(raw: dict[str, Any], key: str, n: int) -> np.ndarray:
    value = np.asarray(raw[key], dtype=float)
    if value.ndim != 1 or value.size != n or not np.all(np.isfinite(value)):
        raise ValueError(f"RawData[{key!r}] is not a finite vector of length {n}")
    return value


def _side_metrics(
    raw: dict[str, Any],
    side: str,
    side_index: int,
    mask: np.ndarray,
    n_integrations: int,
    elapsed_per_integration_s: float,
    on_source_s: float,
) -> dict[str, Any]:
    if n_integrations <= 0 or elapsed_per_integration_s <= 0.0:
        raise ValueError(f"Invalid {side} integration count or elapsed time")
    wall_s = n_integrations * elapsed_per_integration_s
    scale = CADENCE_SECONDS / wall_s
    electrons = _array(raw, f"electrons_{side}", mask.size)
    variance = _array(raw, f"var_{side}", mask.size)
    band_electrons = float(np.sum(electrons[mask]))
    band_variance = float(np.sum(variance[mask]))
    rn_pair = raw["rn[out,in]"]
    bkg_pair = raw["bkg[out,in]"]
    rn_per_channel_total = _finite_scalar(rn_pair[side_index], f"{side} read variance")
    bkg = np.asarray(bkg_pair[side_index], dtype=float)
    if bkg.shape != mask.shape or not np.all(np.isfinite(bkg)):
        raise ValueError(f"{side} background vector does not match the spectral grid")
    background = float(np.sum(bkg[mask]))
    read_variance = rn_per_channel_total * int(np.count_nonzero(mask))
    closure_total = band_electrons + background + read_variance
    closure_fraction = (band_variance - closure_total) / band_variance
    if abs(closure_fraction) > 1.0e-8:
        raise ValueError(
            f"{side} variance does not close: relative residual {closure_fraction:.3g}"
        )

    electrons_60 = band_electrons * scale
    variance_60 = band_variance * scale
    background_60 = background * scale
    read_variance_60 = read_variance * scale
    return {
        "integrations": int(n_integrations),
        "wall_time_seconds": wall_s,
        "on_source_time_seconds": float(on_source_s),
        "duty_cycle_fraction": float(on_source_s) / wall_s,
        "scaling_factor_to_60s_wall_time": scale,
        "band_electrons_total": band_electrons,
        "band_variance_electron2_total": band_variance,
        "background_electrons_total": background,
        "read_variance_electron2_total": read_variance,
        "band_electrons_per_60s_wall_time": electrons_60,
        "band_variance_electron2_per_60s_wall_time": variance_60,
        "background_electrons_per_60s_wall_time": background_60,
        "read_variance_electron2_per_60s_wall_time": read_variance_60,
        "white_noise_relative_to_state_flux_ppm_60s": (
            math.sqrt(variance_60) / electrons_60 * 1.0e6
        ),
        "variance_closure_relative_residual": closure_fraction,
    }


def _audit_phoenix(
    saved: dict[str, Any],
    teff_k: float,
    metallicity_dex: float,
    logg_cgs: float,
) -> dict[str, Any]:
    """Compare saved star spectrum with the matching local PHOENIX grid model."""
    try:
        # PySynphot 2.0 calls this NumPy 1.x name. NumPy 2 removed it, while
        # np.all has the same behavior. This is an in-process compatibility
        # alias; it does not change the installed NumPy or PySynphot packages.
        if not hasattr(np, "alltrue"):
            np.alltrue = np.all
        with warnings.catch_warnings(record=True) as import_warnings:
            warnings.simplefilter("always")
            import pysynphot as syn
    except Exception as exc:
        return {"status": "not_run", "reason": f"pysynphot could not load: {type(exc).__name__}: {exc}"}

    original = saved.get("OriginalInput", {})
    wave_um = np.asarray(original.get("model_wave"), dtype=float)
    saved_star_mjy = np.asarray(original.get("star_spec"), dtype=float)
    if (
        wave_um.ndim != 1
        or saved_star_mjy.shape != wave_um.shape
        or wave_um.size == 0
        or not np.all(np.isfinite(wave_um))
        or not np.all(np.isfinite(saved_star_mjy))
    ):
        return {"status": "failed", "reason": "Saved PHOENIX spectrum is invalid"}

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reference = syn.Icat("phoenix", teff_k, metallicity_dex, logg_cgs)
            reference.convert("nm")
            reference.convert("jy")
        ref_jy = np.interp(wave_um * 1000.0, reference.wave, reference.flux)
        ref_mjy = ref_jy * 1000.0
    except Exception as exc:  # Missing local grid data must be visible in the report.
        return {"status": "not_run", "reason": f"Local PHOENIX grid could not load: {exc}"}

    jmask = (wave_um >= 1.22) & (wave_um <= 1.28)
    band = (wave_um >= WAVE_MIN_UM) & (wave_um <= WAVE_MAX_UM)
    valid = jmask & np.isfinite(ref_mjy) & (ref_mjy > 0.0)
    if not np.any(valid) or not np.any(band):
        return {"status": "failed", "reason": "Saved spectrum lacks J or science-band samples"}
    normalization = float(np.median(saved_star_mjy[valid] / ref_mjy[valid]))
    ratio = saved_star_mjy[band] / (ref_mjy[band] * normalization)
    max_abs = float(np.max(np.abs(ratio - 1.0)))
    return {
        "status": "passed" if max_abs < 1.0e-9 else "failed",
        "reference_model": "pysynphot Icat('phoenix', Teff, [Fe/H], logg)",
        "reference_grid_parameters": {
            "teff_k": teff_k,
            "metallicity_dex": metallicity_dex,
            "logg_cgs": logg_cgs,
        },
        "normalization": "one free multiplicative scale fit by median ratio in 1.22--1.28 micron J band",
        "max_absolute_fractional_shape_difference_2p9_to_5p1_um": max_abs,
        "pysynphot_warnings": [
            str(item.message) for item in [*import_warnings, *caught]
        ],
    }


def _target_result(
    item: dict[str, Any], evidence: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    name = str(item["target"])
    pickle_path = _resolve_pickle_path(item["pickle_path"], manifest_dir)
    if not pickle_path.is_file():
        raise FileNotFoundError(f"PandExo pickle is missing for {name}: {pickle_path}")
    saved = _load_pickle(pickle_path)
    input_evidence = _match_evidence(name, evidence)
    setup = input_evidence["science_detector_setup"]
    mags = input_evidence["stellar_magnitudes"]
    stellar = input_evidence["stellar_parameters_from_archive"]
    ui = saved["input"]
    pand = saved["PandeiaOutTrans"]
    config = pand["input"]["configuration"]
    instrument = config["instrument"]
    detector = config["detector"]
    exposure = pand["information"]["exposure_specification"]
    timing = saved["timing"]
    raw = saved["RawData"]
    wave = np.asarray(raw["wave"], dtype=float)
    if wave.ndim != 1 or not np.all(np.isfinite(wave)):
        raise ValueError(f"{name}: invalid RawData wavelength grid")
    mask = (wave >= WAVE_MIN_UM) & (wave <= WAVE_MAX_UM)
    if int(np.count_nonzero(mask)) == 0:
        raise ValueError(f"{name}: no spectral bins fall in the white-light band")

    expected = {
        "instrument": str(setup["instrument"]).lower(),
        "mode": "bots",
        "filter": str(setup["filter"]).lower(),
        "aperture": "s1600a1",
        "disperser": str(setup["grating"]).lower(),
        "subarray": str(setup["subarray"]).lower(),
        "readout_pattern": str(setup["readout_pattern"]).lower(),
        "ngroup": int(setup["groups_per_integration"]),
    }
    actual = {
        "instrument": str(instrument["instrument"]).lower(),
        "mode": str(instrument["mode"]).lower(),
        "filter": str(instrument["filter"]).lower(),
        "aperture": str(instrument["aperture"]).lower(),
        "disperser": str(instrument["disperser"]).lower(),
        "subarray": str(detector["subarray"]).lower(),
        "readout_pattern": str(detector["readout_pattern"]).lower(),
        "ngroup": int(detector["ngroup"]),
    }
    if actual != expected:
        raise ValueError(f"{name}: accepted instrument setup mismatch: {actual} != {expected}")
    if str(ui.get("Calculation Type", "")).lower() != "fml":
        raise ValueError(f"{name}: saved PandExo calculation is not fml")
    if str(ui.get("Primary/Secondary", "")).lower() != "fp/f*":
        raise ValueError(f"{name}: saved result is not a secondary-eclipse flux-ratio setup")
    if int(exposure.get("nsuperstripe", -1)) != 1:
        raise ValueError(f"{name}: expected one NIRSpec superstripe")

    target_mag = _finite_scalar(ui["Target Mag"], f"{name} target magnitude")
    target_mag_delta = target_mag - float(mags["J_mag"])
    if abs(target_mag_delta) > 0.001:
        raise ValueError(f"{name}: submitted target magnitude differs from catalog J by {target_mag_delta:.4g} mag")
    scene_spectrum = pand["input"]["scene"][0]["spectrum"]
    scene = np.asarray(scene_spectrum["sed"]["spectrum"], dtype=float)
    original_wave = np.asarray(saved["OriginalInput"]["model_wave"], dtype=float)
    original_star = np.asarray(saved["OriginalInput"]["star_spec"], dtype=float)
    expected_scene = np.vstack((original_wave, original_star))
    if scene.shape != expected_scene.shape or not np.all(np.isfinite(scene)):
        raise ValueError(f"{name}: Pandeia scene spectrum does not match saved PHOENIX input shape")
    scene_relative_error = float(
        np.max(np.abs(scene - expected_scene) / np.maximum(np.abs(expected_scene), 1.0e-300))
    )
    if scene_relative_error > 1.0e-12:
        raise ValueError(f"{name}: Pandeia scene spectrum differs from saved stellar input")
    extinction = scene_spectrum["extinction"]
    extinction_value = _finite_scalar(extinction["value"], f"{name} extinction value")
    if extinction_value != 0.0:
        raise ValueError(f"{name}: a non-zero extinction was applied in the saved Pandeia scene")
    if str(scene_spectrum["normalization"]["type"]).lower() != "none":
        raise ValueError(f"{name}: unexpected extra Pandeia spectrum normalization")

    elapsed = _finite_scalar(timing["Time/Integration incl reset (sec)"], f"{name} elapsed time")
    measurement = _finite_scalar(
        timing["Measurement Time per Integration (sec)"], f"{name} measurement time"
    )
    n_out = int(timing["Num Integrations Out of Transit"])
    n_in = int(timing["Num Integrations In Transit"])
    out = _side_metrics(
        raw,
        "out",
        0,
        mask,
        n_out,
        elapsed,
        _finite_scalar(timing["On Source Time Out of Transit"], f"{name} out on-source time"),
    )
    inn = _side_metrics(
        raw,
        "in",
        1,
        mask,
        n_in,
        elapsed,
        _finite_scalar(timing["On Source Time In Transit"], f"{name} in on-source time"),
    )
    # PandExo's secondary-eclipse convention: "out" is the stellar baseline
    # during eclipse; "in" contains star plus the supplied Fp/F* flux.
    star_60 = out["band_electrons_per_60s_wall_time"]
    noise_relative_star = math.sqrt(out["band_variance_electron2_per_60s_wall_time"]) / star_60
    physical_out_relative_star = (
        math.sqrt(inn["band_variance_electron2_per_60s_wall_time"]) / star_60
    )
    physical_out_relative_total = (
        math.sqrt(inn["band_variance_electron2_per_60s_wall_time"])
        / inn["band_electrons_per_60s_wall_time"]
    )
    warnings_engine = saved.get("warning", {})
    warning_summary = {
        str(key): str(value) for key, value in warnings_engine.items()
    } if isinstance(warnings_engine, dict) else {"value": str(warnings_engine)}
    phx = _audit_phoenix(
        saved,
        float(stellar["st_teff_K"]),
        float(stellar["st_met_dex"]),
        float(stellar["st_logg_cgs"]),
    )
    if phx.get("status") == "failed":
        raise ValueError(f"{name}: PHOENIX input audit failed: {phx}")

    return {
        "name": name,
        "noise_ppm_60s": noise_relative_star * 1.0e6,
        "noise_definition": "star-only / in-eclipse white noise, relative to Fstar; used for the recovery input",
        "physical_out_of_eclipse_noise_relative_to_stellar_flux_ppm_60s": physical_out_relative_star * 1.0e6,
        "physical_out_of_eclipse_noise_relative_to_total_flux_ppm_60s": physical_out_relative_total * 1.0e6,
        "physical_out_of_eclipse_change_vs_star_only_percent": (
            physical_out_relative_star / noise_relative_star - 1.0
        ) * 100.0,
        "white_light_band_um": [WAVE_MIN_UM, WAVE_MAX_UM],
        "n_valid_spectral_bins": int(np.count_nonzero(mask)),
        "wavelength_coverage_um": [float(wave[mask][0]), float(wave[mask][-1])],
        "stellar_input": {
            "catalog_j_mag": float(mags["J_mag"]),
            "catalog_j_mag_error": float(mags["J_mag_plus_error"]),
            "catalog_k_mag": float(mags["K_mag"]),
            "catalog_k_mag_error": float(mags["K_mag_plus_error"]),
            "submitted_pandexo_target_mag": target_mag,
            "submitted_minus_catalog_j_mag": target_mag_delta,
            "pandeia_scene_matches_saved_phoenix_spectrum_max_fractional_error": scene_relative_error,
            "pandeia_extinction_value_mag": extinction_value,
            "pandeia_scene_normalization": str(scene_spectrum["normalization"]["type"]),
            "stellar_parameter_source": "archive evidence JSON; scalar Teff/logg/metallicity are not stored in the online result, so the saved spectrum is audited against this PHOENIX model",
            "teff_k": float(stellar["st_teff_K"]),
            "logg_cgs": float(stellar["st_logg_cgs"]),
            "metallicity_dex": float(stellar["st_met_dex"]),
            "phoenix_sed_shape_audit": phx,
        },
        "instrument": {
            "instrument": actual["instrument"],
            "mode": actual["mode"],
            "disperser": actual["disperser"],
            "filter": actual["filter"],
            "aperture": actual["aperture"],
            "subarray": actual["subarray"],
            "readout_pattern": actual["readout_pattern"],
            "groups_per_integration": actual["ngroup"],
            "calculation_type": str(ui["Calculation Type"]),
            "primary_secondary": str(ui["Primary/Secondary"]),
            "nirspec_superstripes": int(exposure["nsuperstripe"]),
        },
        "timing": {
            "elapsed_wall_seconds_per_integration_including_reset": elapsed,
            "measurement_seconds_per_integration": measurement,
            "integration_duty_cycle_fraction": measurement / elapsed,
            "accepted_apt_total_integrations": int(setup["total_integrations"]),
            "pandexo_timing_total_integrations": n_out + n_in,
            "pandexo_minus_accepted_integration_count": (
                n_out + n_in - int(setup["total_integrations"])
            ),
            "observing_efficiency_percent_from_pandexo": _finite_scalar(
                timing["Observing Efficiency (%)"], f"{name} observing efficiency"
            ),
            "star_only_in_eclipse_state": out,
            "physical_out_of_eclipse_star_plus_planet_state": inn,
        },
        "pandexo_warnings": warning_summary,
        "engine_result": {
            "pickle": str(pickle_path),
            "calculation_id": item.get("calculation_id"),
            "result_url": item.get("result_url"),
            "calculation": "fml; online engine/reference-data version is not exposed by the saved result",
        },
    }


def _report(payload: dict[str, Any]) -> str:
    lines = [
        "UNITY expected NIRSpec white-light noise",
        f"Date: {payload['retrieval_date']}",
        f"Band: {WAVE_MIN_UM:.1f}--{WAVE_MAX_UM:.1f} micron; reference: {CADENCE_SECONDS:.0f} s wall time",
        "",
        "The main value is the star-only, in-eclipse noise relative to Fstar. The saved recovery uses a fixed Fstar flux scale.",
        "For each 60-second wall-time sample, scale the total PandExo counts and variance by 60/(N_integrations * elapsed wall seconds per integration).",
        "Then compute 1e6 * sqrt(sum variance) / sum stellar electrons. This includes source shot noise, PandExo background, and read variance.",
        "The elapsed integration time includes reset. The measurement time excludes reset, so the scale includes detector duty loss.",
        "PandExo's transit-timing split can differ by one integration from the accepted APT total; the report records both. This does not change per-60-second noise because each state is scaled by its own integration count and wall time.",
        "The secondary-eclipse convention in PandExo uses `out` for the star-only eclipse state and `in` for star plus supplied Fp/F* outside eclipse.",
        "The two physical out-of-eclipse comparison values use the `in` counts/variance, divided by total flux or by the star-only Fstar reference.",
        "Variance closure is checked against source counts + reported background counts + per-channel read variance times the selected bin count.",
        "The saved Pandeia scene matches the serialized J-normalized PHOENIX spectrum; the scene adds no second normalization and applies zero extinction. Stellar Teff, logg, and metallicity come from the input evidence file because the online result does not store those scalar settings.",
        "No additional noise floor or correlated-noise term was added. Exact deployed online engine/reference-data versions are not present in the saved result.",
        "",
        "Target values (noise_ppm_60s is the recovery input):",
    ]
    if any(
        "Extinction files not found" in warning
        for target in payload["targets"]
        for warning in target["stellar_input"]["phoenix_sed_shape_audit"].get("pysynphot_warnings", [])
    ):
        lines.insert(
            14,
            "Local PySynphot reported that its optional extinction-files directory is missing. The PHOENIX grid loaded, the shape audit passed, and the saved Pandeia scene uses zero extinction.",
        )
    for target in payload["targets"]:
        lines.append(
            f"- {target['name']}: {target['noise_ppm_60s']:.6f} ppm; "
            f"outside eclipse relative to Fstar {target['physical_out_of_eclipse_noise_relative_to_stellar_flux_ppm_60s']:.6f} ppm "
            f"({target['physical_out_of_eclipse_change_vs_star_only_percent']:+.4f}%)."
        )
        lines.append(
            f"  J={target['stellar_input']['catalog_j_mag']:.3f}, "
            f"Teff={target['stellar_input']['teff_k']:.0f} K, "
            f"logg={target['stellar_input']['logg_cgs']:.3f}, "
            f"[Fe/H]={target['stellar_input']['metallicity_dex']:+.2f}; "
            f"PHOENIX shape audit={target['stellar_input']['phoenix_sed_shape_audit']['status']}; "
            f"duty={target['timing']['integration_duty_cycle_fraction']:.2%}; "
            f"saturation: {target['pandexo_warnings'].get('% full well high?', 'not reported')}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = _json(args.manifest)
    evidence = _json(args.evidence)
    rows = manifest.get("targets")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("The online manifest must contain six target results")
    targets = [_target_result(item, evidence, args.manifest.parent) for item in rows]
    payload = {
        "retrieval_date": str(manifest.get("retrieval_date", "unknown")),
        "service": manifest.get("service"),
        "normalization": manifest.get("normalization"),
        "reference_cadence_seconds": CADENCE_SECONDS,
        "white_light_band_um": [WAVE_MIN_UM, WAVE_MAX_UM],
        "calculation_method": {
            "summary": "PandExo first-minus-last (fml) noise from online NIRSpec BOTS result pickles",
            "reference_flux": "Fstar; the star-only secondary-eclipse state is the recovery white-light normalization",
            "formula": "noise_ppm_60s = 1e6 * sqrt(sum(var_out[band]) * 60 / wall_time_out) / (sum(electrons_out[band]) * 60 / wall_time_out)",
            "wall_time": "N integrations times elapsed seconds per integration, including reset",
            "variance_components": "Source shot variance + PandExo reported background variance contribution + read variance; closure is checked per target",
            "other_terms": "No added noise floor or correlated noise. Reset duty loss is included by scaling to wall time.",
            "secondary_eclipse_states": "PandExo f_unit=fp/f*: out is star-only/in-eclipse; in is star plus supplied planet flux/out-of-eclipse.",
        },
        "stellar_input_audit": "Saved spectrum is compared with local pysynphot PHOENIX spectra using archive Teff, logg, and metallicity. The free scale is fitted in J; the submitted PandExo target magnitude is checked against the catalog J magnitude.",
        "engine_version_note": "The online result pickle does not expose the deployed PandExo engine or reference-data version.",
        "targets": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report.write_text(_report(payload), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")
    for target in targets:
        print(f"{target['name']}: {target['noise_ppm_60s']:.6f} ppm")


if __name__ == "__main__":
    main()
