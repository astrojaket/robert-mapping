"""Frozen, starry-free comparisons with the accepted HAT-P-32b run.

The old HAT-P-32b run saved enough products to make a useful reference without
installing ``starry``.  This module compares those products with the public
harmonic basis and the numerical eclipse operator in :mod:`robert_mapping`.

Two geometry choices are reported deliberately:

``literal``
    Uses the ``a_over_rstar`` value saved in ``run_config.json`` (6.05).

``reference_aligned``
    Fits only the effective ``a/Rstar`` and map phase to the saved, noiseless
    planet curve. This isolates the numerical occultation calculation from
    inconsistencies in the legacy run metadata. It is not a new science fit.

The module never imports ``starry`` or PyMC3.  It is a frozen reference check,
not a replacement for posterior sampling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from robert_mapping.physics import (
    disk_quadrature,
    evaluate_map,
    secondary_eclipse_design_matrix,
)


FloatArray = NDArray[np.float64]


_G_SI = 6.67430e-11
_M_SUN_KG = 1.98847e30
_R_SUN_M = 6.957e8
_M_JUP_OVER_M_SUN = 0.000954588


@dataclass(frozen=True)
class CurveComparison:
    """Summary of one planet-only light-curve comparison."""

    n_observations: int
    phase_min: float
    phase_max: float
    reference_peak_ppm: float
    robert_peak_ppm: float
    rmse_ppm: float
    mean_absolute_error_ppm: float
    maximum_absolute_error_ppm: float
    correlation: float
    normalized_rmse_fraction: float


@dataclass(frozen=True)
class MapComparison:
    """Summary of the coefficient-map comparison."""

    n_longitudes: int
    n_latitudes: int
    coefficient_count: int
    coefficient_maximum_absolute_difference: float
    map_rmse_percent: float
    map_maximum_absolute_difference_percent: float
    map_correlation: float
    legacy_peak_longitude_degrees: float
    legacy_peak_latitude_degrees: float
    robert_peak_longitude_degrees: float
    robert_peak_latitude_degrees: float
    legacy_peak_intensity_percent: float
    robert_peak_intensity_percent: float


@dataclass(frozen=True)
class FrozenReferenceReport:
    """Machine-readable result of the frozen HAT-P-32b comparison."""

    status: str
    target: str
    reference_directory: str
    output_directory: str
    quadrature: dict[str, int]
    files: dict[str, str]
    file_sha256: dict[str, str]
    legacy_metadata: dict[str, Any]
    geometry: dict[str, Any]
    map_comparison: MapComparison
    literal_curve_comparison: CurveComparison
    reference_aligned_curve_comparison: CurveComparison
    notes: tuple[str, ...]


def _required(reference: Path, name: str) -> Path:
    path = reference / name
    if not path.is_file():
        raise FileNotFoundError(f"Frozen reference file does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert NumPy values in the legacy metadata to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _read_observation(path: Path) -> dict[str, FloatArray]:
    try:
        table = np.genfromtxt(path, names=True, delimiter=",", dtype=float)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read frozen HAT-P-32b CSV {path}: {exc}") from exc
    required = (
        "time_bjd_tdb",
        "time_from_eclipse_hours",
        "flux_true",
        "planet_flux_true",
        "flux_uncertainty",
    )
    if table.dtype.names is None:
        raise ValueError(f"Frozen HAT-P-32b CSV has no named columns: {path}")
    missing = [name for name in required if name not in table.dtype.names]
    if missing:
        raise ValueError(f"Frozen HAT-P-32b CSV is missing column(s): {', '.join(missing)}")
    arrays = {name: np.asarray(table[name], dtype=float) for name in required}
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("Frozen HAT-P-32b observation columns must be one-dimensional")
    if len({value.size for value in arrays.values()}) != 1:
        raise ValueError("Frozen HAT-P-32b observation columns have different lengths")
    if arrays["time_bjd_tdb"].size == 0:
        raise ValueError("Frozen HAT-P-32b observation has no rows")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Frozen HAT-P-32b observation contains non-finite values")
    return arrays


def _derived_a_over_rstar(system: dict[str, Any]) -> float:
    """Infer ``a/Rstar`` in the same way as the old omitted ``starry`` input."""

    period_seconds = float(system["period_days"]) * 86_400.0
    stellar_mass = float(system["stellar_mass_msun"])
    planet_mass = float(system["planet_mass_mjup"]) * _M_JUP_OVER_M_SUN
    total_mass_kg = (stellar_mass + planet_mass) * _M_SUN_KG
    semi_major_m = (
        _G_SI * total_mass_kg * (period_seconds / (2.0 * np.pi)) ** 2
    ) ** (1.0 / 3.0)
    return float(semi_major_m / (float(system["stellar_radius_rsun"]) * _R_SUN_M))


def _curve_metrics(
    reference: FloatArray,
    prediction: FloatArray,
    phase: FloatArray,
) -> CurveComparison:
    residual = np.asarray(prediction, dtype=float) - np.asarray(reference, dtype=float)
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).eps)
    correlation = float(np.corrcoef(reference, prediction)[0, 1])
    return CurveComparison(
        n_observations=int(reference.size),
        phase_min=float(np.min(phase)),
        phase_max=float(np.max(phase)),
        reference_peak_ppm=float(np.max(reference) * 1.0e6),
        robert_peak_ppm=float(np.max(prediction) * 1.0e6),
        rmse_ppm=float(np.sqrt(np.mean(residual**2)) * 1.0e6),
        mean_absolute_error_ppm=float(np.mean(np.abs(residual)) * 1.0e6),
        maximum_absolute_error_ppm=float(np.max(np.abs(residual)) * 1.0e6),
        correlation=correlation,
        normalized_rmse_fraction=float(np.sqrt(np.mean(residual**2)) / scale),
    )


def _map_comparison(
    config: dict[str, Any], map_data: dict[str, FloatArray]
) -> tuple[MapComparison, FloatArray, FloatArray]:
    injection = config["injection"]
    coefficients = np.asarray(injection["starry_coefficients"], dtype=float)
    if coefficients.ndim != 1:
        raise ValueError("Saved starry_coefficients must be one-dimensional")
    if coefficients.size != 9:
        raise ValueError("The frozen HAT-P-32b reference must contain degree-2 coefficients")
    longitude = np.deg2rad(np.asarray(map_data["longitude_deg"], dtype=float))
    latitude = np.deg2rad(np.asarray(map_data["latitude_deg"], dtype=float))
    grid_lon, grid_lat = np.meshgrid(longitude, latitude, indexing="xy")
    values = np.asarray(evaluate_map(coefficients, grid_lon, grid_lat), dtype=float)
    scale = float(injection["starry_map_amplitude"]) * np.pi * 100.0
    robert_map = values * scale
    legacy_map = np.asarray(map_data["injected_specific_intensity_percent"], dtype=float)
    if robert_map.shape != legacy_map.shape:
        raise ValueError(
            "Saved map grid does not match the expected longitude/latitude shape: "
            f"{legacy_map.shape} versus {robert_map.shape}"
        )
    difference = robert_map - legacy_map
    legacy_peak = np.unravel_index(int(np.argmax(legacy_map)), legacy_map.shape)
    robert_peak = np.unravel_index(int(np.argmax(robert_map)), robert_map.shape)
    coefficient_difference = coefficients - np.asarray(injection["starry_coefficients"], dtype=float)
    metric = MapComparison(
        n_longitudes=int(longitude.size),
        n_latitudes=int(latitude.size),
        coefficient_count=int(coefficients.size),
        coefficient_maximum_absolute_difference=float(np.max(np.abs(coefficient_difference))),
        map_rmse_percent=float(np.sqrt(np.mean(difference**2))),
        map_maximum_absolute_difference_percent=float(np.max(np.abs(difference))),
        map_correlation=float(np.corrcoef(legacy_map.ravel(), robert_map.ravel())[0, 1]),
        legacy_peak_longitude_degrees=float(np.asarray(map_data["longitude_deg"])[legacy_peak[1]]),
        legacy_peak_latitude_degrees=float(np.asarray(map_data["latitude_deg"])[legacy_peak[0]]),
        robert_peak_longitude_degrees=float(np.asarray(map_data["longitude_deg"])[robert_peak[1]]),
        robert_peak_latitude_degrees=float(np.asarray(map_data["latitude_deg"])[robert_peak[0]]),
        legacy_peak_intensity_percent=float(legacy_map[legacy_peak]),
        robert_peak_intensity_percent=float(robert_map[robert_peak]),
    )
    return metric, legacy_map, robert_map


def _planet_prediction(
    time: FloatArray,
    system: dict[str, Any],
    coefficients: FloatArray,
    amplitude: float,
    a_over_rstar: float,
    quadrature,
    *,
    theta0_radians: float = np.pi,
) -> FloatArray:
    design = secondary_eclipse_design_matrix(
        time,
        float(system["period_days"]),
        float(a_over_rstar),
        float(system["inclination_deg"]),
        float(system["rp_over_rstar"]),
        2,
        float(system["derived_transit_epoch_bjd_tdb"]),
        theta0=float(theta0_radians),
        angle_unit="deg",
        quadrature=quadrature,
    )
    return float(amplitude) * np.asarray(design, dtype=float).dot(coefficients)


def _align_reference_geometry(
    time: FloatArray,
    reference_flux: FloatArray,
    system: dict[str, Any],
    coefficients: FloatArray,
    amplitude: float,
    literal_a: float,
) -> tuple[float, float, float]:
    """Fit two reference coordinates without changing map coefficients."""

    coarse_quadrature = disk_quadrature(16, 64)

    def objective(parameters: FloatArray) -> float:
        a_over_rstar, theta0 = np.asarray(parameters, dtype=float)
        if not (1.0 < a_over_rstar < 20.0 and 0.0 < theta0 < 2.0 * np.pi):
            return 1.0e12
        prediction = _planet_prediction(
            time,
            system,
            coefficients,
            amplitude,
            a_over_rstar,
            coarse_quadrature,
            theta0_radians=theta0,
        )
        return float(np.mean((prediction - reference_flux) ** 2) * 1.0e12)

    start = np.asarray((literal_a * 1.21, np.pi + 0.33), dtype=float)
    fit = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1.0e-9, "fatol": 1.0e-9},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"Could not align the frozen starry curve: {fit.message}")
    return float(fit.x[0]), float(fit.x[1]), float(np.sqrt(fit.fun))


def _save_plots(
    output: Path,
    observation: dict[str, FloatArray],
    map_data: dict[str, FloatArray],
    robert_map: FloatArray,
    literal_prediction: FloatArray,
    aligned_prediction: FloatArray,
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    purple = "mediumpurple"
    dark_purple = "#6a3d9a"
    phase_hours = observation["time_from_eclipse_hours"]
    reference = observation["planet_flux_true"] * 1.0e6
    literal = literal_prediction * 1.0e6
    aligned = aligned_prediction * 1.0e6

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(phase_hours, reference, color="black", linewidth=1.2, label="Saved starry planet flux")
    axes[0].plot(phase_hours, literal, color=purple, linewidth=1.2, label="robert literal geometry")
    axes[0].plot(phase_hours, aligned, color=dark_purple, linestyle="--", linewidth=1.2, label="robert reference-aligned geometry")
    axes[0].set_ylabel("Planet flux (ppm)")
    axes[0].set_title("HAT-P-32b frozen planet light curve")
    axes[0].legend(loc="best")
    axes[1].plot(phase_hours, literal - reference, color=purple, label="literal residual")
    axes[1].plot(phase_hours, aligned - reference, color=dark_purple, linestyle="--", label="reference-aligned residual")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Time from eclipse centre (hours)")
    axes[1].set_ylabel("Residual (ppm)")
    axes[1].legend(loc="best")
    figure.savefig(output / "frozen_hatp32_lightcurve.png", dpi=180)
    figure.savefig(output / "frozen_hatp32_lightcurve.pdf")
    plt.close(figure)

    legacy_map = np.asarray(map_data["injected_specific_intensity_percent"], dtype=float)
    difference = robert_map - legacy_map
    extent = [
        float(np.min(map_data["longitude_deg"])),
        float(np.max(map_data["longitude_deg"])),
        float(np.min(map_data["latitude_deg"])),
        float(np.max(map_data["latitude_deg"])),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    image = axes[0].imshow(legacy_map, origin="lower", extent=extent, aspect="auto", cmap="Purples_r")
    axes[0].set_title("Saved starry map")
    axes[1].imshow(robert_map, origin="lower", extent=extent, aspect="auto", cmap="Purples_r", vmin=image.get_clim()[0], vmax=image.get_clim()[1])
    axes[1].set_title("robert map")
    difference_image = axes[2].imshow(difference, origin="lower", extent=extent, aspect="auto", cmap="PuOr")
    axes[2].set_title("robert − saved")
    for axis in axes:
        axis.set_xlabel("Longitude (degrees)")
        axis.set_ylabel("Latitude (degrees)")
    figure.colorbar(image, ax=axes[:2], label="Specific intensity (%)")
    figure.colorbar(difference_image, ax=axes[2], label="Difference (%)")
    figure.savefig(output / "frozen_hatp32_map_comparison.png", dpi=180)
    figure.savefig(output / "frozen_hatp32_map_comparison.pdf")
    plt.close(figure)


def run_hatp32_frozen_reference(
    reference_directory: str | Path,
    output_directory: str | Path,
    *,
    n_radial: int = 32,
    n_azimuth: int = 128,
    save_plots: bool = True,
) -> FrozenReferenceReport:
    """Compare robert-mapping with the saved HAT-P-32b starry products.

    Parameters are intentionally explicit.  The external reference directory
    is not copied into this repository and starry is never imported.
    """

    reference = Path(reference_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = _required(reference, "run_config.json")
    observation_path = _required(reference, "synthetic_observation.csv")
    map_path = _required(reference, "map_data.npz")
    posterior_path = reference / "posterior_samples.npz"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    observation = _read_observation(observation_path)
    with np.load(map_path, allow_pickle=False) as archive:
        map_data = {key: np.asarray(archive[key], dtype=float) for key in archive.files}
    required_map = {
        "longitude_deg",
        "latitude_deg",
        "injected_specific_intensity_percent",
    }
    missing_map = sorted(required_map - set(map_data))
    if missing_map:
        raise ValueError(f"Frozen HAT-P-32b map data are missing: {', '.join(missing_map)}")

    system = config["system"]
    injection = config["injection"]
    coefficients = np.asarray(injection["starry_coefficients"], dtype=float)
    amplitude = float(injection["starry_map_amplitude"])
    if coefficients.shape != (9,):
        raise ValueError("Frozen HAT-P-32b coefficients must contain nine degree-2 values")
    derived_a = _derived_a_over_rstar(system)
    literal_a = float(system["a_over_rstar"])
    quadrature = disk_quadrature(int(n_radial), int(n_azimuth))
    time = observation["time_bjd_tdb"]
    literal_prediction = _planet_prediction(
        time, system, coefficients, amplitude, literal_a, quadrature
    )
    aligned_a, aligned_theta0, coarse_rmse_ppm = _align_reference_geometry(
        time,
        observation["planet_flux_true"],
        system,
        coefficients,
        amplitude,
        literal_a,
    )
    aligned_prediction = _planet_prediction(
        time,
        system,
        coefficients,
        amplitude,
        aligned_a,
        quadrature,
        theta0_radians=aligned_theta0,
    )
    phase = (time - float(system["eclipse_mid_bjd_tdb"])) / float(system["period_days"])
    map_metric, legacy_map, robert_map = _map_comparison(config, map_data)

    files = {
        "run_config": config_path.name,
        "synthetic_observation": observation_path.name,
        "map_data": map_path.name,
    }
    hashes = {name: _sha256(path) for name, path in (("run_config", config_path), ("synthetic_observation", observation_path), ("map_data", map_path))}
    if posterior_path.is_file():
        files["posterior_samples"] = posterior_path.name
        hashes["posterior_samples"] = _sha256(posterior_path)
        with np.load(posterior_path, allow_pickle=False) as posterior:
            posterior_count = int(posterior["flux_model"].shape[0]) if "flux_model" in posterior else None
    else:
        posterior_count = None

    output_arrays = output / "frozen_hatp32_comparison.npz"
    np.savez_compressed(
        output_arrays,
        time_bjd_tdb=time,
        time_from_eclipse_hours=observation["time_from_eclipse_hours"],
        orbital_phase=phase,
        planet_flux_true=observation["planet_flux_true"],
        robert_planet_flux_literal=literal_prediction,
        robert_planet_flux_reference_aligned=aligned_prediction,
        longitude_deg=np.asarray(map_data["longitude_deg"], dtype=float),
        latitude_deg=np.asarray(map_data["latitude_deg"], dtype=float),
        legacy_injected_map_percent=legacy_map,
        robert_map_percent=robert_map,
        map_difference_percent=robert_map - legacy_map,
    )
    literal_metrics = _curve_metrics(observation["planet_flux_true"], literal_prediction, phase)
    aligned_metrics = _curve_metrics(observation["planet_flux_true"], aligned_prediction, phase)
    metadata = _json_safe(config)
    metadata["posterior_sample_count"] = posterior_count
    geometry = {
        "saved_a_over_rstar": literal_a,
        "kepler_derived_a_over_rstar": derived_a,
        "reference_aligned_a_over_rstar": aligned_a,
        "reference_aligned_theta0_radians": aligned_theta0,
        "reference_aligned_theta0_degrees": float(np.rad2deg(aligned_theta0)),
        "coarse_alignment_rmse_ppm": coarse_rmse_ppm,
        "phase_definition": "(time_bjd_tdb - eclipse_mid_bjd_tdb) / period_days",
        "old_script_note": "The old runner did not pass a to starry.Secondary; starry inferred it from masses, radius, and period.",
        "literal_comparison": "Uses saved system.a_over_rstar.",
        "reference_aligned_comparison": (
            "Fits only effective a/Rstar and theta0 to the saved noiseless "
            "planet curve; map coefficients and amplitude stay fixed."
        ),
    }
    report = FrozenReferenceReport(
        status=(
            "passed"
            if map_metric.map_maximum_absolute_difference_percent < 1.0e-10
            and aligned_metrics.rmse_ppm < 5.0
            else "failed"
        ),
        target=str(config.get("target", "HAT-P-32b")),
        reference_directory=str(reference),
        output_directory=str(output),
        quadrature={"n_radial": int(n_radial), "n_azimuth": int(n_azimuth)},
        files=files,
        file_sha256=hashes,
        legacy_metadata=metadata,
        geometry=geometry,
        map_comparison=map_metric,
        literal_curve_comparison=literal_metrics,
        reference_aligned_curve_comparison=aligned_metrics,
        notes=(
            "The map comparison is an exact coefficient-basis comparison and does not import starry.",
            "Planet-only light-curve differences include the current fixed quadrature and old starry geometry details.",
            "The aligned curve is a two-coordinate implementation check, not a new orbital inference.",
            "No temperature conversion is attempted because the frozen HAT-P-32b products contain intensity, not a calibrated passband radiance.",
        ),
    )
    (output / "frozen_hatp32_report.json").write_text(
        json.dumps(_json_safe(asdict(report)), indent=2) + "\n", encoding="utf-8"
    )
    if save_plots:
        _save_plots(output, observation, map_data, robert_map, literal_prediction, aligned_prediction)
    return report


def run_frozen_reference(
    reference_directory: str | Path,
    output_directory: str | Path,
    **kwargs: Any,
) -> FrozenReferenceReport:
    """Run the currently supported frozen reference target."""

    return run_hatp32_frozen_reference(reference_directory, output_directory, **kwargs)


__all__ = [
    "CurveComparison",
    "FrozenReferenceReport",
    "MapComparison",
    "run_frozen_reference",
    "run_hatp32_frozen_reference",
]
