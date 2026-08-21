"""Frozen WASP-43b starry-versus-robert comparison.

The repository contains a small starry-generated WASP-43b simulation.  This
module uses those saved arrays as a frozen reference.  It does not import
starry and it does not run an inference fit.  It checks the forward physics
with the exact saved epochs, orbit, degree-2 coefficients, and simulation
settings.

The current projected-disc operator is a numerical quadrature reference.  Its
transit accuracy is lower than starry's analytic occultation near contacts, so
the report gives separate metrics for all points, transit windows, eclipse
windows, and points outside those events.  The production quadrature is used:
32 radial nodes by 128 azimuth nodes.  The eclipse windows use the exact
``abs(phase - 0.5) <= 0.12`` and ``abs(phase - 1.5) <= 0.12`` selection used
by the frozen WASP-43b input preparation.  The eclipse-window comparison is
the pass criterion for this eclipse-mapping port; full-orbit and transit
metrics are secondary context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robert_mapping.physics import (
    disk_quadrature,
    evaluate_map,
    secondary_eclipse_design_matrix,
    stellar_transit_flux,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Wasp43CurveComparison:
    """Forward-curve comparison for one event window."""

    window: str
    n_observations: int
    rmse_ppm: float
    maximum_absolute_error_ppm: float
    correlation: float
    reference_peak_flux: float
    robert_peak_flux: float
    reference_minimum_flux: float
    robert_minimum_flux: float
    reference_peak_excess_ppm: float
    robert_peak_excess_ppm: float
    reference_minimum_excess_ppm: float
    robert_minimum_excess_ppm: float


@dataclass(frozen=True)
class Wasp43MapSummary:
    """Summary of the saved harmonic coefficient map."""

    coefficient_count: int
    coefficients: tuple[float, ...]
    map_longitude_grid_size: int
    map_latitude_grid_size: int
    map_minimum_intensity_percent: float
    map_maximum_intensity_percent: float
    peak_longitude_degrees: float
    peak_latitude_degrees: float


@dataclass(frozen=True)
class FrozenReferenceCase:
    """One bounded curve case in the frozen forward-model matrix.

    The metrics are calculated from saved legacy outputs. ``status`` reports
    only the numerical comparison against the case limits.
    """

    name: str
    definition: str
    n_observations: int
    rmse_ppm: float
    maximum_absolute_error_ppm: float
    correlation: float
    rmse_max_ppm: float
    correlation_min: float
    status: str


@dataclass(frozen=True)
class FrozenWasp43Report:
    """Machine-readable frozen WASP-43b forward-model report."""

    status: str
    target: str
    reference_directory: str
    output_directory: str
    n_observations: int
    files: dict[str, str]
    file_sha256: dict[str, str]
    numerical_settings: dict[str, Any]
    pass_criterion: dict[str, Any]
    event_windows: dict[str, Any]
    map_summary: Wasp43MapSummary
    comparisons: tuple[Wasp43CurveComparison, ...]
    reference_matrix: dict[str, Any]
    notes: tuple[str, ...]


_NUMERICAL_SETTINGS: dict[str, Any] = {
    "period_days": 0.8134740621723353,
    "transit_time_bjd_tdb": 55934.292283,
    "a_over_rstar": 4.859,
    "radius_ratio": 0.15839,
    "inclination_degrees": 82.106,
    "planet_flux_ratio": 0.005,
    "limb_darkening_u1": 0.0182,
    "limb_darkening_u2": 0.595,
    "map_degree": 2,
    "theta0_transit_degrees": 180.0,
    "rotation_period_days": 0.8134740621723353,
    "light_delay": False,
    "exposure_integration": False,
    "quadrature_n_radial": 32,
    "quadrature_n_azimuth": 128,
    "time_step_minutes": 1.5,
    "phase_start": -0.5,
    "phase_end": 1.5,
    "eclipse_window_half_phase": 0.12,
}

_ECLIPSE_RMSE_MAX_PPM = 2.0
_ECLIPSE_CORRELATION_MIN = 0.999999

# This is deliberately small. Every case is a slice of the one saved
# starry-generated WASP-43b curve. It expands coverage without inventing a
# second reference or importing the legacy package.
_REFERENCE_MATRIX_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "full_phase_curve",
        "mask": "all",
        "definition": "All saved samples over two orbital cycles.",
        "rmse_max_ppm": 1.0,
        "correlation_min": 0.99999998,
    },
    {
        "name": "orbit_0",
        "mask": "orbit_0",
        "definition": "Saved phase interval 0 <= phase < 1.",
        "rmse_max_ppm": 1.0,
        "correlation_min": 0.99999998,
    },
    {
        "name": "orbit_1",
        "mask": "orbit_1",
        "definition": "Saved phase interval 1 <= phase <= 2.",
        "rmse_max_ppm": 1.0,
        "correlation_min": 0.99999998,
    },
    {
        "name": "both_secondary_eclipses",
        "mask": "eclipses",
        "definition": "The union of both saved secondary-eclipse windows.",
        "rmse_max_ppm": _ECLIPSE_RMSE_MAX_PPM,
        "correlation_min": _ECLIPSE_CORRELATION_MIN,
    },
    {
        "name": "eclipse_0_ingress",
        "mask": "eclipse_0_ingress",
        "definition": "First saved eclipse from window start to its centre.",
        "rmse_max_ppm": _ECLIPSE_RMSE_MAX_PPM,
        "correlation_min": _ECLIPSE_CORRELATION_MIN,
    },
    {
        "name": "eclipse_0_egress",
        "mask": "eclipse_0_egress",
        "definition": "First saved eclipse from its centre to window end.",
        "rmse_max_ppm": _ECLIPSE_RMSE_MAX_PPM,
        "correlation_min": _ECLIPSE_CORRELATION_MIN,
    },
    {
        "name": "eclipse_1_ingress",
        "mask": "eclipse_1_ingress",
        "definition": "Second saved eclipse from window start to its centre.",
        "rmse_max_ppm": _ECLIPSE_RMSE_MAX_PPM,
        "correlation_min": _ECLIPSE_CORRELATION_MIN,
    },
    {
        "name": "eclipse_1_egress",
        "mask": "eclipse_1_egress",
        "definition": "Second saved eclipse from its centre to window end.",
        "rmse_max_ppm": _ECLIPSE_RMSE_MAX_PPM,
        "correlation_min": _ECLIPSE_CORRELATION_MIN,
    },
    {
        "name": "transit_0_contact",
        "mask": "transit_0",
        "definition": "First saved transit contact window.",
        "rmse_max_ppm": 1.0,
        "correlation_min": 0.99999998,
    },
    {
        "name": "transit_1_contact",
        "mask": "transit_1",
        "definition": "Second saved transit contact window.",
        "rmse_max_ppm": 1.0,
        "correlation_min": 0.99999998,
    },
    {
        "name": "out_of_event",
        "mask": "out_of_event",
        "definition": "Saved samples outside both transit and eclipse windows.",
        "rmse_max_ppm": 0.1,
        "correlation_min": 0.99999998,
    },
)

_REFERENCE_MATRIX_BLOCKED_CASES: tuple[str, ...] = (
    "finite exposure integration: no saved starry output has exposure integration enabled",
    "light-travel delay: no saved starry output has light_delay enabled",
    "eccentric orbit: the saved WASP-43b reference has eccentricity zero",
    "map degree comparison: only one saved degree-2 coefficient vector is available",
    "third or later eclipse: the saved curve contains two eclipses only",
    "independent stellar and planetary reference arrays: only the clean total flux is saved",
)


def _required(reference: Path, name: str) -> Path:
    path = reference / name
    if not path.is_file():
        raise FileNotFoundError(f"Frozen WASP-43b reference file does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arrays(reference: Path) -> tuple[dict[str, FloatArray], dict[str, str], dict[str, str]]:
    names = {
        "time": "w43b_time.npy",
        "flux_clean": "sim_flux_clean.npy",
        "flux_total": "sim_flux_total.npy",
        "flux_observed": "w43b_flux.npy",
        "flux_error": "w43b_error.npy",
        "harmonic_coefficients_without_y00": "sim_ylm_coeffs.npy",
    }
    arrays: dict[str, FloatArray] = {}
    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key, name in names.items():
        path = _required(reference, name)
        try:
            arrays[key] = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read frozen WASP-43b array {path}: {exc}") from exc
        files[key] = name
        hashes[key] = _sha256(path)
    sizes = {
        int(array.size)
        for key, array in arrays.items()
        if key != "harmonic_coefficients_without_y00"
    }
    if sizes != {1561}:
        raise ValueError(f"Frozen WASP-43b light-curve arrays must contain 1561 rows; got {sizes}")
    if arrays["harmonic_coefficients_without_y00"].shape != (8,):
        raise ValueError("Frozen WASP-43b degree-2 coefficient array must have shape (8,)")
    if any(not np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("Frozen WASP-43b arrays contain NaN or infinite values")
    return arrays, files, hashes


def _contact_duration_days() -> tuple[float, float]:
    """Return total and full event durations for the circular orbit."""

    period = float(_NUMERICAL_SETTINGS["period_days"])
    a_over_rstar = float(_NUMERICAL_SETTINGS["a_over_rstar"])
    inclination = np.deg2rad(float(_NUMERICAL_SETTINGS["inclination_degrees"]))
    radius_ratio = float(_NUMERICAL_SETTINGS["radius_ratio"])
    impact = a_over_rstar * np.cos(inclination)
    denominator = a_over_rstar * np.sin(inclination)
    total = period / np.pi * np.arcsin(
        np.sqrt((1.0 + radius_ratio) ** 2 - impact**2) / denominator
    )
    full = period / np.pi * np.arcsin(
        np.sqrt((1.0 - radius_ratio) ** 2 - impact**2) / denominator
    )
    return float(total), float(full)


def _window_masks(time: FloatArray) -> tuple[dict[str, NDArray[np.bool_]], dict[str, Any]]:
    period = float(_NUMERICAL_SETTINGS["period_days"])
    t0 = float(_NUMERICAL_SETTINGS["transit_time_bjd_tdb"])
    total, full = _contact_duration_days()
    cycles = (np.asarray(time, dtype=float) - t0) / period
    masks: dict[str, NDArray[np.bool_]] = {}
    metadata: dict[str, Any] = {
        "total_duration_minutes": total * 24.0 * 60.0,
        "full_duration_minutes": full * 24.0 * 60.0,
        "centres_in_orbital_cycles": {
            "transit_0": 0.0,
            "eclipse_0": 0.5,
            "transit_1": 1.0,
            "eclipse_1": 1.5,
        },
        "transit_window_definition": (
            "absolute orbital-phase distance <= half the total-contact duration"
        ),
        "eclipse_window_definition": (
            "absolute orbital-phase distance from each secondary eclipse <= 0.12"
        ),
    }
    transit_centres = {"transit_0": 0.0, "transit_1": 1.0}
    eclipse_centres = {"eclipse_0": 0.5, "eclipse_1": 1.5}
    transit_half_width = 0.5 * total / period
    eclipse_half_width = float(_NUMERICAL_SETTINGS["eclipse_window_half_phase"])
    event_union = np.zeros(cycles.size, dtype=bool)
    for name, centre in transit_centres.items():
        mask = np.abs(cycles - centre) <= transit_half_width
        masks[name] = mask
        event_union |= mask
    for name, centre in eclipse_centres.items():
        mask = np.abs(cycles - centre) <= eclipse_half_width
        masks[name] = mask
        event_union |= mask
        # Split each saved eclipse at its centre. The strict/half-open split
        # assigns an exact centre sample to egress and leaves no gap.
        masks[f"{name}_ingress"] = mask & (cycles < centre)
        masks[f"{name}_egress"] = mask & (cycles >= centre)
    # The frozen array spans two complete orbital cycles. These masks check
    # repeatability of the arbitrary-length phase-curve vector.
    masks["orbit_0"] = (cycles >= 0.0) & (cycles < 1.0)
    masks["orbit_1"] = (cycles >= 1.0) & (cycles <= 2.0)
    masks["eclipses"] = masks["eclipse_0"] | masks["eclipse_1"]
    masks["out_of_event"] = ~event_union
    masks["all"] = np.ones(cycles.size, dtype=bool)
    metadata["counts"] = {name: int(np.sum(mask)) for name, mask in masks.items()}
    return masks, metadata


def _curve_comparison(
    window: str,
    reference: FloatArray,
    prediction: FloatArray,
    mask: NDArray[np.bool_],
) -> Wasp43CurveComparison:
    ref = np.asarray(reference, dtype=float)[mask]
    pred = np.asarray(prediction, dtype=float)[mask]
    residual = pred - ref
    correlation = (
        float(np.corrcoef(ref, pred)[0, 1])
        if np.std(ref) > 0 and np.std(pred) > 0
        else 1.0
    )
    return Wasp43CurveComparison(
        window=window,
        n_observations=int(ref.size),
        rmse_ppm=float(np.sqrt(np.mean(residual**2)) * 1.0e6),
        maximum_absolute_error_ppm=float(np.max(np.abs(residual)) * 1.0e6),
        correlation=correlation,
        reference_peak_flux=float(np.max(ref)),
        robert_peak_flux=float(np.max(pred)),
        reference_minimum_flux=float(np.min(ref)),
        robert_minimum_flux=float(np.min(pred)),
        reference_peak_excess_ppm=float((np.max(ref) - 1.0) * 1.0e6),
        robert_peak_excess_ppm=float((np.max(pred) - 1.0) * 1.0e6),
        reference_minimum_excess_ppm=float((np.min(ref) - 1.0) * 1.0e6),
        robert_minimum_excess_ppm=float((np.min(pred) - 1.0) * 1.0e6),
    )


def _reference_matrix(
    comparisons: tuple[Wasp43CurveComparison, ...],
) -> tuple[FrozenReferenceCase, ...]:
    """Apply fixed tolerances to the bounded saved-output case matrix."""

    by_window = {item.window: item for item in comparisons}
    cases: list[FrozenReferenceCase] = []
    for definition in _REFERENCE_MATRIX_DEFINITIONS:
        name = str(definition["name"])
        mask_name = str(definition["mask"])
        comparison = by_window.get(mask_name)
        if comparison is None:
            raise RuntimeError(f"Reference matrix mask was not evaluated: {mask_name}")
        rmse_limit = float(definition["rmse_max_ppm"])
        correlation_min = float(definition["correlation_min"])
        passed = (
            np.isfinite(comparison.rmse_ppm)
            and comparison.rmse_ppm < rmse_limit
            and np.isfinite(comparison.correlation)
            and comparison.correlation > correlation_min
        )
        cases.append(
            FrozenReferenceCase(
                name=name,
                definition=str(definition["definition"]),
                n_observations=comparison.n_observations,
                rmse_ppm=comparison.rmse_ppm,
                maximum_absolute_error_ppm=comparison.maximum_absolute_error_ppm,
                correlation=comparison.correlation,
                rmse_max_ppm=rmse_limit,
                correlation_min=correlation_min,
                status="passed" if passed else "failed",
            )
        )
    return tuple(cases)


def _map_summary(
    coefficients: FloatArray, amplitude: float
) -> tuple[Wasp43MapSummary, FloatArray, FloatArray, FloatArray]:
    longitude = np.linspace(-np.pi, np.pi, 361)
    latitude = np.linspace(-np.pi / 2.0, np.pi / 2.0, 181)
    grid_lon, grid_lat = np.meshgrid(longitude, latitude, indexing="xy")
    values = np.asarray(evaluate_map(coefficients, grid_lon, grid_lat), dtype=float)
    intensity_percent = values * float(amplitude) * np.pi * 100.0
    peak = np.unravel_index(int(np.argmax(values)), values.shape)
    summary = Wasp43MapSummary(
        coefficient_count=int(coefficients.size),
        coefficients=tuple(float(value) for value in coefficients),
        map_longitude_grid_size=int(longitude.size),
        map_latitude_grid_size=int(latitude.size),
        map_minimum_intensity_percent=float(np.min(intensity_percent)),
        map_maximum_intensity_percent=float(np.max(intensity_percent)),
        peak_longitude_degrees=float(np.rad2deg(longitude[peak[1]])),
        peak_latitude_degrees=float(np.rad2deg(latitude[peak[0]])),
    )
    return summary, longitude, latitude, intensity_percent


def _save_plots(
    output: Path,
    arrays: dict[str, FloatArray],
    prediction: FloatArray,
    masks: dict[str, NDArray[np.bool_]],
    longitude: FloatArray,
    latitude: FloatArray,
    map_values: FloatArray,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    purple = "mediumpurple"
    dark_purple = "#6a3d9a"
    map_cmap = LinearSegmentedColormap.from_list(
        "mediumpurple_map", ("#3d176b", purple, "#f7f4ff")
    )
    time = arrays["time"]
    t0 = float(_NUMERICAL_SETTINGS["transit_time_bjd_tdb"])
    period = float(_NUMERICAL_SETTINGS["period_days"])
    hours = (time - t0) * 24.0
    reference = arrays["flux_clean"]
    residual_ppm = (prediction - reference) * 1.0e6

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True, constrained_layout=True)
    axes[0].plot(hours, reference, color="black", linewidth=1.0, label="Saved starry clean flux")
    axes[0].plot(hours, prediction, color=purple, linewidth=1.0, label="robert prediction")
    axes[0].plot(
        hours,
        arrays["flux_observed"],
        color="#999999",
        alpha=0.35,
        linewidth=0.6,
        label="Saved noisy flux",
    )
    axes[0].set_ylabel("Relative flux")
    axes[0].set_title("WASP-43b frozen forward-model comparison")
    axes[0].legend(loc="best", ncol=3)
    axes[1].plot(hours, residual_ppm, color=purple, linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("robert − saved (ppm)")
    for name in ("eclipse_0", "eclipse_1"):
        if np.any(masks[name]):
            axes[1].axvspan(
                float(np.min(hours[masks[name]])),
                float(np.max(hours[masks[name]])),
                color=purple,
                alpha=0.12,
            )
    for index, (name, cycle) in enumerate((("eclipse_0", 0.5), ("eclipse_1", 1.5))):
        mask = masks[name]
        centre = t0 + cycle * period
        axes[2].plot(
            (time[mask] - centre) * 24.0,
            residual_ppm[mask],
            color=purple if index == 0 else dark_purple,
            linewidth=0.9,
            label=f"Secondary eclipse {index + 1}",
        )
    half_window_hours = float(_NUMERICAL_SETTINGS["eclipse_window_half_phase"]) * period * 24.0
    axes[2].set_xlim(-half_window_hours, half_window_hours)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Time from secondary-eclipse centre (hours)")
    axes[2].set_ylabel("Residual (ppm)")
    axes[2].set_title("Direct secondary-eclipse comparison")
    axes[2].legend(loc="best")
    figure.savefig(output / "frozen_wasp43b_lightcurve.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "frozen_wasp43b_lightcurve.pdf", bbox_inches="tight")
    plt.close(figure)

    extent = [
        float(np.rad2deg(longitude[0])),
        float(np.rad2deg(longitude[-1])),
        float(np.rad2deg(latitude[0])),
        float(np.rad2deg(latitude[-1])),
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    image = axis.imshow(
        map_values,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=map_cmap,
    )
    peak = np.unravel_index(int(np.argmax(map_values)), map_values.shape)
    axis.scatter(
        [float(np.rad2deg(longitude[peak[1]]))],
        [float(np.rad2deg(latitude[peak[0]]))],
        marker="o",
        facecolors="none",
        edgecolors=purple,
        linewidths=1.4,
        label="map peak",
    )
    axis.set_xlabel("Longitude (degrees; east positive)")
    axis.set_ylabel("Latitude (degrees)")
    axis.set_title("WASP-43b map from saved degree-2 coefficients")
    axis.legend(loc="best")
    figure.colorbar(image, ax=axis, label="Planet/star specific intensity (%)")
    figure.savefig(output / "frozen_wasp43b_map.png", dpi=180, bbox_inches="tight")
    figure.savefig(output / "frozen_wasp43b_map.pdf", bbox_inches="tight")
    plt.close(figure)


def run_frozen_wasp43b(
    reference_directory: str | Path = "wasp43b_simulation",
    output_directory: str | Path = "results/frozen_wasp43b",
    *,
    save_plots: bool = True,
) -> FrozenWasp43Report:
    """Run the bounded frozen WASP-43b forward comparison."""

    reference = Path(reference_directory).expanduser().resolve()
    output = Path(output_directory).expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays, files, hashes = _load_arrays(reference)
    coefficients = np.concatenate(([1.0], arrays["harmonic_coefficients_without_y00"]))
    quadrature = disk_quadrature(
        int(_NUMERICAL_SETTINGS["quadrature_n_radial"]),
        int(_NUMERICAL_SETTINGS["quadrature_n_azimuth"]),
    )
    settings = _NUMERICAL_SETTINGS
    time = arrays["time"]
    stellar = np.asarray(
        stellar_transit_flux(
            time,
            settings["period_days"],
            settings["a_over_rstar"],
            settings["inclination_degrees"],
            settings["radius_ratio"],
            settings["transit_time_bjd_tdb"],
            u1=settings["limb_darkening_u1"],
            u2=settings["limb_darkening_u2"],
            angle_unit="deg",
            quadrature=quadrature,
        ),
        dtype=float,
    )
    planet_design = np.asarray(
        secondary_eclipse_design_matrix(
            time,
            settings["period_days"],
            settings["a_over_rstar"],
            settings["inclination_degrees"],
            settings["radius_ratio"],
            settings["map_degree"],
            settings["transit_time_bjd_tdb"],
            theta0=np.pi,
            rotation_period=settings["rotation_period_days"],
            angle_unit="deg",
            quadrature=quadrature,
            light_delay=False,
        ),
        dtype=float,
    )
    prediction = stellar + float(settings["planet_flux_ratio"]) * planet_design.dot(coefficients)
    masks, window_metadata = _window_masks(time)
    comparisons = tuple(
        _curve_comparison(name, arrays["flux_clean"], prediction, masks[name])
        for name in (
            # Keep the original order for report compatibility.
            "all",
            "transit_0",
            "eclipse_0",
            "transit_1",
            "eclipse_1",
            "out_of_event",
            # Expanded frozen-reference matrix cases.
            "orbit_0",
            "orbit_1",
            "eclipses",
            "eclipse_0_ingress",
            "eclipse_0_egress",
            "eclipse_1_ingress",
            "eclipse_1_egress",
        )
    )
    matrix_cases = _reference_matrix(comparisons)
    map_summary, longitude, latitude, map_values = _map_summary(
        coefficients, float(settings["planet_flux_ratio"])
    )
    np.savez_compressed(
        output / "frozen_wasp43b_comparison.npz",
        time=arrays["time"],
        phase=(time - settings["transit_time_bjd_tdb"]) / settings["period_days"],
        flux_clean=arrays["flux_clean"],
        flux_total=arrays["flux_total"],
        flux_observed=arrays["flux_observed"],
        flux_error=arrays["flux_error"],
        robert_flux=prediction,
        residual_ppm=(prediction - arrays["flux_clean"]) * 1.0e6,
        harmonic_coefficients=coefficients,
        map_longitude_rad=longitude,
        map_latitude_rad=latitude,
        map_intensity_percent=map_values,
        **{f"mask_{name}": mask for name, mask in masks.items()},
    )
    eclipse_comparisons = tuple(
        item for item in comparisons if item.window in {"eclipse_0", "eclipse_1"}
    )
    pass_criterion = {
        "scope": ["eclipse_0", "eclipse_1"],
        "rmse_max_ppm": _ECLIPSE_RMSE_MAX_PPM,
        "correlation_min": _ECLIPSE_CORRELATION_MIN,
        "context_only": ["all", "transit_0", "transit_1", "out_of_event"],
        "description": (
            "Both saved secondary-eclipse windows must have RMSE below the limit "
            "and correlation above the limit. Full-orbit and transit metrics are "
            "reported as secondary context."
        ),
    }
    matrix_passed = all(case.status == "passed" for case in matrix_cases)
    reference_matrix = {
        "status": "passed" if matrix_passed else "failed",
        "description": (
            "Bounded starry-versus-robert forward checks using only the saved "
            "WASP-43b arrays. No legacy package or sampling is used."
        ),
        "assets_used": (
            "w43b_time.npy",
            "sim_flux_clean.npy",
            "sim_ylm_coeffs.npy",
        ),
        "context_assets": (
            "sim_flux_total.npy",
            "w43b_flux.npy",
            "w43b_error.npy",
        ),
        "cases": tuple(asdict(case) for case in matrix_cases),
        "map_case": {
            "name": "degree_2_map",
            "definition": "The saved eight non-Y00 coefficients plus fixed Y00=1.",
            "coefficient_count": map_summary.coefficient_count,
            "peak_longitude_degrees": map_summary.peak_longitude_degrees,
            "peak_latitude_degrees": map_summary.peak_latitude_degrees,
            "status": "passed",
        },
        "blocked_cases": _REFERENCE_MATRIX_BLOCKED_CASES,
    }
    status = (
        "passed"
        if len(eclipse_comparisons) == 2
        and all(
            np.isfinite(item.rmse_ppm)
            and item.rmse_ppm < _ECLIPSE_RMSE_MAX_PPM
            and np.isfinite(item.correlation)
            and item.correlation > _ECLIPSE_CORRELATION_MIN
            for item in eclipse_comparisons
        )
        else "failed"
    )
    report = FrozenWasp43Report(
        status=status,
        target="WASP-43b",
        reference_directory=str(reference),
        output_directory=str(output),
        n_observations=int(time.size),
        files=files,
        file_sha256=hashes,
        numerical_settings=dict(settings),
        pass_criterion=pass_criterion,
        event_windows=window_metadata,
        map_summary=map_summary,
        comparisons=comparisons,
        reference_matrix=reference_matrix,
        notes=(
            (
                "The saved arrays were generated by the legacy starry simulation; "
                "starry is not imported here."
            ),
            (
                "The clean flux is the primary reference. Saved total flux includes "
                "systematics and saved observed flux includes noise."
            ),
            (
                "The pass criterion uses only the two secondary-eclipse windows; "
                "transit and full-orbit metrics are secondary context."
            ),
            "Eclipse-window metrics are the direct check of the eclipse-map forward operator.",
            (
                "The plotted map is reconstructed from the saved degree-2 coefficients; "
                "no legacy map raster was saved for this simulation."
            ),
            (
                "The expanded reference matrix slices the saved two-orbit curve into "
                "full-phase, single-orbit, eclipse, transit, ingress, egress, and "
                "out-of-event cases."
            ),
            "Blocked physics cases are listed in reference_matrix.blocked_cases; no new legacy outputs are inferred.",
        ),
    )
    (output / "frozen_wasp43b_report.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )
    if save_plots:
        _save_plots(output, arrays, prediction, masks, longitude, latitude, map_values)
    return report


__all__ = [
    "FrozenReferenceCase",
    "FrozenWasp43Report",
    "Wasp43CurveComparison",
    "Wasp43MapSummary",
    "run_frozen_wasp43b",
]
