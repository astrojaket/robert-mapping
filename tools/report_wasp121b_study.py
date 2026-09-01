"""Build the combined real-data report for the WASP-121b study.

This script does not sample. It reads completed mapped and uniform-control fits,
calculates a common approximate BIC at the posterior-mean noise parameters, and
makes the small set of figures needed to review the study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "wasp121b_study"
PURPLE = "mediumpurple"
DARK_PURPLE = "#4b286d"

RUNS = {
    "NRS1": {
        "mapped": ROOT / "results" / "wasp121b_nrs1_production",
        "uniform": ROOT / "results" / "wasp121b_nrs1_uniform",
        "data": ROOT / "literature_data" / "WASP-121b" / "JWST-NIRSpec-G395H" / "prepared" / "white_nrs1.npz",
        "wavelength": 3.21,
        "bandpass": "2.70-3.72 micron",
        "published": {"median": 3.36, "lower": 3.25, "upper": 3.47, "citation": "Mikal-Evans et al. (2023)"},
        "published_reanalysis": {"median": 2.95, "lower": 2.83, "upper": 3.06, "citation": "Evans-Soma et al. (2025)"},
        "ou_sensitivity": ROOT / "results" / "wasp121b_nrs1_ou_sensitivity",
        "delta_parameters": 2,
        "map_is_physical": False,
    },
    "NRS2": {
        "mapped": ROOT / "results" / "wasp121b_nrs2_production",
        "uniform": ROOT / "results" / "wasp121b_nrs2_uniform",
        "data": ROOT / "literature_data" / "WASP-121b" / "JWST-NIRSpec-G395H" / "prepared" / "white_nrs2.npz",
        "wavelength": 4.485,
        "bandpass": "3.82-5.15 micron",
        "published": {"median": 2.66, "lower": 2.54, "upper": 2.78, "citation": "Mikal-Evans et al. (2023)"},
        "published_reanalysis": {"median": 2.39, "lower": 2.26, "upper": 2.51, "citation": "Evans-Soma et al. (2025)"},
        "ou_sensitivity": ROOT / "results" / "wasp121b_nrs2_ou_sensitivity",
        "delta_parameters": 2,
        "map_is_physical": False,
    },
    "MIRI": {
        "mapped": ROOT / "results" / "wasp121b_miri_lrs_degree1_production",
        "uniform": ROOT / "results" / "wasp121b_miri_lrs_uniform",
        "data": ROOT / "literature_data" / "WASP-121b" / "JWST-MIRI-LRS" / "prepared" / "white_light_curve.npz",
        "wavelength": 8.12,
        "bandpass": "5.00-12.00 micron",
        "published": {"median": 4.8, "lower": 2.0, "upper": 7.5, "citation": "MIRI phase-curve paper (2026)"},
        "delta_parameters": 3,
        "map_is_physical": True,
    },
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing study product: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _normal_log_likelihood(residual: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum(-0.5 * (residual / sigma) ** 2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)))


def _effective_sigma(fit: dict[str, Any], flux_error: np.ndarray) -> np.ndarray:
    jitter = fit.get("white_jitter_mean")
    if jitter is not None:
        return np.sqrt(flux_error**2 + float(jitter) ** 2)
    scale = float(fit.get("error_scale_mean", 1.0))
    return flux_error * scale


def _acf(values: np.ndarray, maximum_lag: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=float) - float(np.mean(values))
    denominator = float(np.dot(values, values))
    if denominator <= 0.0:
        return np.zeros(maximum_lag + 1)
    return np.array([
        float(np.dot(values[: values.size - lag], values[lag:]) / denominator)
        if lag else 1.0
        for lag in range(maximum_lag + 1)
    ])


def _ou_standardized_innovations(
    residual: np.ndarray,
    sigma: np.ndarray,
    time: np.ndarray,
    amplitude: float,
    timescale: float,
    jitter: float,
) -> np.ndarray:
    """Return residuals after the fitted time correlation is removed."""

    time_seconds = (np.asarray(time, dtype=float) - float(time[0])) * 86400.0
    dt = np.concatenate((np.diff(time_seconds), np.zeros(1)))
    decay = np.exp(-dt / max(float(timescale), 1.0e-12))
    process_variance = float(amplitude) ** 2 * (-np.expm1(-2.0 * dt / max(float(timescale), 1.0e-12)))
    observation_variance = np.asarray(sigma, dtype=float) ** 2 + float(jitter) ** 2
    state_mean = 0.0
    state_variance = float(amplitude) ** 2
    innovations = np.empty_like(residual, dtype=float)
    for index, value in enumerate(np.asarray(residual, dtype=float)):
        prediction_variance = max(state_variance, 1.0e-30)
        innovation_variance = max(prediction_variance + observation_variance[index], 1.0e-30)
        innovation = value - state_mean
        innovations[index] = innovation / np.sqrt(innovation_variance)
        gain = prediction_variance / innovation_variance
        updated_mean = state_mean + gain * innovation
        updated_variance = max((1.0 - gain) * prediction_variance, 0.0)
        state_mean = decay[index] * updated_mean
        state_variance = decay[index] ** 2 * updated_variance + process_variance[index]
    return innovations


def _load_run(name: str, settings: dict[str, Any]) -> dict[str, Any]:
    mapped = Path(settings["mapped"])
    uniform = Path(settings["uniform"])
    data_path = Path(settings["data"])
    report = _json(mapped / "production_report.json")
    mapped_fit = _json(mapped / "fit_summary.json")
    uniform_fit = _json(uniform / "fit_summary.json")
    with np.load(data_path) as data:
        time = np.asarray(data["time"], dtype=float)
        flux_error = np.asarray(data["flux_err"], dtype=float)
        jitter_x = np.asarray(data["jitter_x"], dtype=float) if "jitter_x" in data else None
        jitter_y = np.asarray(data["jitter_y"], dtype=float) if "jitter_y" in data else None
    mapped_residual = np.asarray(np.load(mapped / "residuals.npy"), dtype=float)
    uniform_residual = np.asarray(np.load(uniform / "residuals.npy"), dtype=float)
    mapped_ll = _normal_log_likelihood(mapped_residual, _effective_sigma(mapped_fit, flux_error))
    uniform_ll = _normal_log_likelihood(uniform_residual, _effective_sigma(uniform_fit, flux_error))
    delta_bic = -2.0 * (mapped_ll - uniform_ll) + float(settings["delta_parameters"]) * np.log(time.size)
    hotspot = report["hotspot_longitude_degrees_east"]
    positivity = report.get("positivity_conditioned", {})
    result = {
        "name": name,
        "bandpass": settings["bandpass"],
        "effective_wavelength_micron": settings["wavelength"],
        "n_observations": int(time.size),
        "hotspot_offset_degrees_east": {key: float(hotspot[key]) for key in ("q16", "median", "q84")},
        "published_hotspot_offset_degrees_east": settings["published"],
        "published_reanalysis_hotspot_offset_degrees_east": settings.get("published_reanalysis"),
        "residual_rms_ppm": float(mapped_fit["residual_rms"]) * 1.0e6,
        "uniform_residual_rms_ppm": float(uniform_fit["residual_rms"]) * 1.0e6,
        "delta_bic_map_minus_uniform": float(delta_bic),
        "bic_note": "Approximate BIC at posterior-mean noise parameters; negative values prefer the mapped model.",
        "divergences": int(mapped_fit["divergences"]),
        "maximum_rhat": float(mapped_fit["maximum_rhat"]),
        "minimum_ess": float(mapped_fit["minimum_effective_sample_size"]),
        "posterior_draws": int(mapped_fit["n_samples"]),
        "positive_draw_fraction": float(positivity.get("accepted_fraction", 0.0)),
        "map_is_physical": bool(settings["map_is_physical"]),
        "map_warning": None if settings["map_is_physical"] else "The direct-harmonic map has no globally non-negative posterior draws. Use its longitude profile as a contrast diagnostic, not as a physical temperature map.",
        "lag1_residual_correlation": float(_acf(mapped_residual, 1)[1]),
        "source_files": {"mapped": str(mapped), "uniform": str(uniform), "data": str(data_path)},
    }
    result["_arrays"] = {
        "time": time,
        "residual_ppm": mapped_residual * 1.0e6,
        "acf": _acf(mapped_residual),
        "jitter_x": jitter_x,
        "jitter_y": jitter_y,
    }
    sensitivity_path = settings.get("ou_sensitivity")
    if sensitivity_path is not None:
        sensitivity_path = Path(sensitivity_path)
        sensitivity_fit = _json(sensitivity_path / "fit_summary.json")
        sensitivity_report = _json(sensitivity_path / "production_report.json")
        sensitivity_residual = np.asarray(np.load(sensitivity_path / "residuals.npy"), dtype=float)
        sensitivity_hotspot = sensitivity_report["hotspot_longitude_degrees_east"]
        innovations = _ou_standardized_innovations(
            sensitivity_residual,
            flux_error,
            time,
            float(sensitivity_fit["ou_amplitude_mean"]),
            float(sensitivity_fit["ou_timescale_mean"]),
            float(sensitivity_fit["jitter_mean"]),
        )
        result["ou_sensitivity"] = {
            "hotspot_offset_degrees_east": {key: float(sensitivity_hotspot[key]) for key in ("q16", "median", "q84")},
            "ou_amplitude_ppm": float(sensitivity_fit["ou_amplitude_mean"]) * 1.0e6,
            "ou_timescale_seconds": float(sensitivity_fit["ou_timescale_mean"]),
            "independent_jitter_ppm": float(sensitivity_fit["jitter_mean"]) * 1.0e6,
            "divergences": int(sensitivity_fit["divergences"]),
            "maximum_rhat": float(sensitivity_fit["maximum_rhat"]),
            "minimum_ess": float(sensitivity_fit["minimum_effective_sample_size"]),
            "posterior_draws": int(sensitivity_fit["n_samples"]),
            "source": str(sensitivity_path),
        }
        result["_arrays"]["ou_innovation_acf"] = _acf(innovations)
    return result


def _plot_overview(runs: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    wavelengths = np.array([run["effective_wavelength_micron"] for run in runs])

    axis = axes[0, 0]
    for run in runs:
        hot = run["hotspot_offset_degrees_east"]
        published = run["published_hotspot_offset_degrees_east"]
        x = run["effective_wavelength_micron"]
        axis.errorbar(x, hot["median"], yerr=[[hot["median"] - hot["q16"]], [hot["q84"] - hot["median"]]], fmt="o", color=PURPLE, capsize=3, label="robert-mapping" if run is runs[0] else None)
        axis.errorbar(x, published["median"], yerr=[[published["median"] - published["lower"]], [published["upper"] - published["median"]]], fmt="D", color=DARK_PURPLE, markerfacecolor="white", capsize=3, label="published" if run is runs[0] else None)
        reanalysis = run.get("published_reanalysis_hotspot_offset_degrees_east")
        if reanalysis is not None:
            axis.errorbar(x, reanalysis["median"], yerr=[[reanalysis["median"] - reanalysis["lower"]], [reanalysis["upper"] - reanalysis["median"]]], fmt="^", color="#7b4ba1", markerfacecolor="white", capsize=3, label="published reanalysis" if run is runs[0] else None)
        sensitivity = run.get("ou_sensitivity")
        if sensitivity is not None:
            ou_hot = sensitivity["hotspot_offset_degrees_east"]
            axis.errorbar(x, ou_hot["median"], yerr=[[ou_hot["median"] - ou_hot["q16"]], [ou_hot["q84"] - ou_hot["median"]]], fmt="s", color="#b19cd9", markeredgecolor=DARK_PURPLE, capsize=3, label="time-correlated noise" if run is runs[0] else None)
    axis.axhline(0.0, color="0.4", linestyle="--", linewidth=0.9)
    axis.set(xlabel="Effective wavelength (micron)", ylabel="Hot-spot offset (degrees east)", title="Longitude offsets")
    axis.legend(frameon=False)

    axis = axes[0, 1]
    positions = np.arange(len(runs))
    width = 0.35
    axis.bar(positions - width / 2, [run["residual_rms_ppm"] for run in runs], width, color=PURPLE, label="mapped")
    axis.bar(positions + width / 2, [run["uniform_residual_rms_ppm"] for run in runs], width, color=DARK_PURPLE, alpha=0.82, label="uniform")
    axis.set_xticks(positions, [run["name"] for run in runs])
    axis.set(ylabel="Residual RMS (ppm)", title="Mapped and uniform residuals")
    axis.set_yscale("log")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    evidence_strength = [-run["delta_bic_map_minus_uniform"] for run in runs]
    bars = axis.bar(positions, evidence_strength, color=PURPLE)
    axis.axhline(10.0, color=DARK_PURPLE, linestyle="--", linewidth=1.0, label="strong preference (10)")
    axis.set_yscale("log")
    axis.bar_label(bars, labels=[f"{value:.1f}" for value in evidence_strength], padding=3, fontsize=8)
    axis.set_xticks(positions, [run["name"] for run in runs])
    axis.set(ylabel="-Delta BIC (map preferred)", title="Mapping evidence")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    axis.axis("off")
    rows = [[run["name"], f'{run["divergences"]}', f'{run["maximum_rhat"]:.4f}', f'{run["minimum_ess"]:.0f}', f'{run["positive_draw_fraction"]:.3f}'] for run in runs]
    table = axis.table(cellText=rows, colLabels=["Data", "Div.", "R-hat", "min ESS", "positive"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    axis.set_title("Sampling and positivity checks", pad=14)

    for axis in axes.flat[:3]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18, axis="y")
    figure.suptitle("WASP-121b real-data eclipse-map study", fontsize=14)
    figure.savefig(output / "wasp121b_study_overview.png", dpi=220)
    figure.savefig(output / "wasp121b_study_overview.pdf")
    plt.close(figure)


def _plot_residuals(runs: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(len(runs), 2, figsize=(11.5, 8.5), constrained_layout=True)
    for row, run in enumerate(runs):
        arrays = run["_arrays"]
        time = arrays["time"]
        residual = arrays["residual_ppm"]
        axis = axes[row, 0]
        axis.scatter((time - time[0]) * 24.0, residual, s=3, color=PURPLE, alpha=0.5, rasterized=True)
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        axis.set(ylabel=f'{run["name"]}\nResidual (ppm)')
        if row == len(runs) - 1:
            axis.set_xlabel("Hours from first retained exposure")
        axis = axes[row, 1]
        lags = np.arange(arrays["acf"].size)
        axis.vlines(lags[1:], 0.0, arrays["acf"][1:], color=PURPLE, linewidth=1.1)
        if "ou_innovation_acf" in arrays:
            axis.plot(lags[1:], arrays["ou_innovation_acf"][1:], color=DARK_PURPLE, linewidth=1.1, label="after time-correlation removal")
        limit = 1.96 / np.sqrt(run["n_observations"])
        axis.axhspan(-limit, limit, color=PURPLE, alpha=0.15)
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        axis.set(ylabel="ACF")
        if row == len(runs) - 1:
            axis.set_xlabel("Lag (exposures)")
        if row == 0 and "ou_innovation_acf" in arrays:
            axis.legend(frameon=False, fontsize=8)
    axes[0, 0].set_title("Residual time series")
    axes[0, 1].set_title("Residual autocorrelation")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("WASP-121b residual checks", fontsize=14)
    figure.savefig(output / "wasp121b_residual_diagnostics.png", dpi=220)
    figure.savefig(output / "wasp121b_residual_diagnostics.pdf")
    plt.close(figure)


def _plot_map_montage(runs: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(len(runs), 3, figsize=(13.0, 10.2), constrained_layout=True)
    columns = [("brightness_map_median.png", "Brightness map"), ("temperature_map.png", "Temperature display"), ("longitude_profile.png", "Longitude profile")]
    for row, run in enumerate(runs):
        directory = Path(run["source_files"]["mapped"])
        for column, (filename, title) in enumerate(columns):
            axis = axes[row, column]
            image = mpimg.imread(directory / filename)
            axis.imshow(image)
            axis.axis("off")
            if row == 0:
                axis.set_title(title)
            if column == 0:
                status = "physical positive map" if run["map_is_physical"] else "harmonic contrast diagnostic"
                axis.text(-0.04, 0.5, f'{run["name"]}\n{status}', transform=axis.transAxes, rotation=90, ha="right", va="center", fontsize=10)
    figure.suptitle("WASP-121b maps and longitude profiles", fontsize=14)
    figure.savefig(output / "wasp121b_map_montage.png", dpi=180)
    figure.savefig(output / "wasp121b_map_montage.pdf")
    plt.close(figure)


def make_report(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = [_load_run(name, settings) for name, settings in RUNS.items()]
    _plot_overview(runs, output)
    _plot_residuals(runs, output)
    _plot_map_montage(runs, output)
    clean_runs = []
    for run in runs:
        clean = {key: value for key, value in run.items() if key != "_arrays"}
        clean_runs.append(clean)
    summary = {
        "status": "complete_with_scope_limits",
        "target": "WASP-121b",
        "scope": "Fit-ready real JWST white-light products: NIRSpec NRS1, NIRSpec NRS2, and MIRI LRS.",
        "not_run": {
            "NIRISS_SOSS": "Processed provenance, masks, regressors, and exposure metadata are incomplete.",
            "HST_WFC3_G141": "The local products still need a documented JD_UTC to BJD_TDB conversion; the 2019 spectral timestamps are invalid.",
            "TESS": "Only a phase-binned curve is present and it does not include eclipse centre.",
            "SMARTS_K": "The time scale, weights, and night/dither systematics need an audit.",
            "other_HST_and_Spitzer": "Source reductions are not present locally.",
            "spectral_maps": "Deferred because the NIRSpec direct-harmonic white maps fail global positivity and individual native channels need scientifically defined wavelength binning.",
        },
        "runs": clean_runs,
        "files": {
            "overview": "wasp121b_study_overview.png",
            "residuals": "wasp121b_residual_diagnostics.png",
            "maps": "wasp121b_map_montage.png",
        },
    }
    (output / "wasp121b_study_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    summary = make_report(args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
