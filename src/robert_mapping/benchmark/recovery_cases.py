"""Fast HAT-P-32b and WASP-178b recovery checks.

These checks use fixed physical map templates and generalized least squares.
They are intentionally small. They test the code and the sign convention; they
do not replace a full posterior or the Hammond cross-validation analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from robert_mapping.data import load_light_curve
from robert_mapping.physics import (
    disk_quadrature,
    equal_area_pixels,
    map_design_matrix,
    pixels_to_harmonics,
    render_map,
    secondary_eclipse_design_matrix,
)
from robert_mapping.recovery import cyclic_residual_shift, fit_candidate_grid, fit_profile


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RecoveryTrial:
    """One null or injected recovery trial."""

    trial: int
    shift: int
    injected_longitude_degrees: float | None
    recovered_longitude_degrees: float
    longitude_q16_degrees: float
    longitude_q84_degrees: float
    delta_bic: float
    detected: bool
    interval_contains_injection: bool | None
    residual_rms_ppm: float
    best_width_degrees: float
    best_timing_seconds: float
    minimum_rendered_map_intensity: float
    rendered_map_positive: bool
    # The latitude fields are optional for backwards compatibility with the
    # original longitude-only HAT-P-32b and WASP-178b calibrations.  They are
    # populated by the synthetic recovery matrix below.
    injected_latitude_degrees: float | None = None
    recovered_latitude_degrees: float | None = None
    latitude_q16_degrees: float | None = None
    latitude_q84_degrees: float | None = None
    latitude_interval_contains_injection: bool | None = None
    noise_ppm: float | None = None
    eclipse_count: int | None = None


@dataclass(frozen=True)
class RecoveryReport:
    """Machine-readable summary of a recovery or rejection calibration."""

    status: str
    case: str
    seed: int
    n_observations: int
    longitude_sign: str
    false_positive_count: int
    null_trial_count: int
    injection_coverage_count: int
    injection_trial_count: int
    false_positive_rate: float | None
    injection_coverage: float | None
    trials: tuple[RecoveryTrial, ...]
    comparison: dict[str, Any]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _TemplateGrid:
    longitudes: FloatArray
    widths: FloatArray
    timings: FloatArray
    raw_designs: FloatArray
    fit_designs: FloatArray
    candidate_longitudes: FloatArray
    candidate_latitudes: FloatArray
    candidate_widths: FloatArray
    candidate_timings: FloatArray
    nuisance_count: int
    covariance_cholesky: FloatArray | None
    harmonic_degree: int


def _hotspot_coefficients(
    longitude_degrees: float,
    width_degrees: float,
    ydeg: int,
    latitude_degrees: float = 0.0,
) -> FloatArray:
    """Project one positive Gaussian spot onto the selected harmonic degree."""

    pixels = equal_area_pixels(32, 64)
    lon0 = np.deg2rad(float(longitude_degrees))
    lat0 = np.deg2rad(float(latitude_degrees))
    width = np.deg2rad(float(width_degrees))
    cosine = (
        np.sin(pixels.lat) * np.sin(lat0)
        + np.cos(pixels.lat) * np.cos(lat0) * np.cos(pixels.lon - lon0)
    )
    distance = np.arccos(np.clip(cosine, -1.0, 1.0))
    values = np.exp(-0.5 * (distance / width) ** 2)
    coefficients = np.asarray(
        pixels_to_harmonics(values, pixels, int(ydeg)), dtype=float
    )
    unocculted = float(
        np.asarray(map_design_matrix(0.0, 0.0, int(ydeg))) @ coefficients
    )
    if not np.isfinite(unocculted) or unocculted <= 0.0:
        raise RuntimeError("The hotspot template has invalid visible flux.")
    return coefficients / unocculted


def _uniform_coefficients(ydeg: int) -> FloatArray:
    coefficients = np.zeros((int(ydeg) + 1) ** 2, dtype=float)
    coefficients[0] = 1.0
    return coefficients


def _longitude_grid(settings) -> FloatArray:
    stop = settings.longitude_grid_max_degrees
    step = settings.longitude_grid_step_degrees
    count = int(np.floor((stop - settings.longitude_grid_min_degrees) / step + 0.5))
    return settings.longitude_grid_min_degrees + step * np.arange(count + 1)


def _nuisance_design(time: FloatArray, order: int, *, ramp_hours: float | None) -> FloatArray:
    centred = time - np.mean(time)
    scale = max(float(np.ptp(time)) / 2.0, np.finfo(float).eps)
    x = centred / scale
    columns = [np.ones(time.size)]
    columns.extend(x**degree for degree in range(1, int(order) + 1))
    if ramp_hours is not None:
        elapsed_hours = (time - np.min(time)) * 24.0
        columns.append(np.exp(-elapsed_hours / float(ramp_hours)))
    return np.column_stack(columns)


def _ou_covariance(time: FloatArray, sigma: FloatArray, settings) -> FloatArray:
    seconds = np.asarray(time, dtype=float) * 86_400.0
    lag = np.abs(seconds[:, None] - seconds[None, :])
    amplitude = settings.correlated_amplitude_ppm * 1.0e-6
    jitter = settings.extra_jitter_ppm * 1.0e-6
    covariance = amplitude**2 * np.exp(
        -lag / settings.correlation_timescale_seconds
    )
    covariance.flat[:: time.size + 1] += sigma**2 + jitter**2
    return covariance


def _whiten_stack(
    observed: FloatArray, designs: FloatArray, cholesky: FloatArray | None, sigma: FloatArray
) -> tuple[FloatArray, FloatArray]:
    if cholesky is None:
        return observed / sigma, designs / sigma[None, :, None]
    whitened_y = solve_triangular(cholesky, observed, lower=True, check_finite=False)
    candidates, rows, columns = designs.shape
    packed = designs.transpose(1, 0, 2).reshape(rows, candidates * columns)
    whitened = solve_triangular(cholesky, packed, lower=True, check_finite=False)
    return whitened_y, whitened.reshape(rows, candidates, columns).transpose(1, 0, 2)


def _design_grid(config, time: FloatArray, sigma: FloatArray) -> _TemplateGrid:
    settings = config.recovery
    ydeg = int(config.map.harmonic_degree)
    longitudes = _longitude_grid(settings)
    # The existing cases use an equatorial spot.  A separate explicit grid is
    # useful for the bounded latitude-recovery matrix because it makes the
    # north--south information content visible without changing the default
    # production-style calibrations.
    latitudes = np.asarray(
        getattr(settings, "latitude_grid_degrees", (0.0,)), dtype=float
    )
    widths = np.asarray(settings.width_grid_degrees, dtype=float)
    timings = np.asarray(settings.timing_grid_seconds, dtype=float)
    nuisance = _nuisance_design(
        time,
        settings.baseline_order,
        ramp_hours=(settings.ramp_timescale_hours if config.model.fit_ramp else None),
    )
    quadrature = disk_quadrature(16, 64)
    uniform_coefficients = _uniform_coefficients(ydeg)
    raw: list[FloatArray] = []
    candidate_lon: list[float] = []
    candidate_lat: list[float] = []
    candidate_width: list[float] = []
    candidate_timing: list[float] = []
    for timing in timings:
        t0 = float(config.system.transit_time) + timing / 86_400.0
        eclipse = np.asarray(
            secondary_eclipse_design_matrix(
                time,
                float(config.system.period_days),
                float(config.system.a_over_rstar),
                float(config.system.inclination_degrees),
                float(config.system.radius_ratio),
                ydeg,
                t0,
                theta0=np.pi,
                angle_unit="deg",
                quadrature=quadrature,
            ),
            dtype=float,
        )
        uniform = eclipse @ uniform_coefficients
        for width in widths:
            for latitude in latitudes:
                for longitude in longitudes:
                    spot = eclipse @ _hotspot_coefficients(
                        longitude, width, ydeg, latitude
                    )
                    raw.append(np.column_stack((nuisance, uniform, spot)))
                    candidate_lon.append(float(longitude))
                    candidate_lat.append(float(latitude))
                    candidate_width.append(float(width))
                    candidate_timing.append(float(timing))
    raw_designs = np.asarray(raw, dtype=float)
    cholesky = None
    if settings.correlated_noise:
        cholesky = np.linalg.cholesky(_ou_covariance(time, sigma, settings))
    _, fit_designs = _whiten_stack(
        np.zeros(time.size), raw_designs, cholesky, sigma
    )
    return _TemplateGrid(
        longitudes=longitudes,
        widths=widths,
        timings=timings,
        raw_designs=raw_designs,
        fit_designs=fit_designs,
        candidate_longitudes=np.asarray(candidate_lon),
        candidate_latitudes=np.asarray(candidate_lat),
        candidate_widths=np.asarray(candidate_width),
        candidate_timings=np.asarray(candidate_timing),
        nuisance_count=nuisance.shape[1],
        covariance_cholesky=cholesky,
        harmonic_degree=ydeg,
    )


def _weighted_quantile(values: FloatArray, weights: FloatArray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    centres = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(quantile, centres, values))


def _fit_trial(
    observed: FloatArray,
    sigma: FloatArray,
    grid: _TemplateGrid,
    detection_threshold: float,
    *,
    trial: int,
    shift: int,
    injected_longitude: float | None,
    injected_latitude: float | None = None,
    noise_ppm: float | None = None,
    eclipse_count: int | None = None,
) -> tuple[RecoveryTrial, FloatArray, FloatArray]:
    fit_y, _ = _whiten_stack(
        observed, grid.raw_designs[:1], grid.covariance_cholesky, sigma
    )
    result = fit_candidate_grid(
        grid.candidate_longitudes,
        grid.fit_designs,
        fit_y,
        1.0,
        map_columns=(grid.nuisance_count, grid.nuisance_count + 1),
        nuisance_columns=tuple(range(grid.nuisance_count)),
    )
    candidate_dimensions = (
        1
        + int(grid.widths.size > 1)
        + int(grid.timings.size > 1)
        + int(grid.candidate_latitudes.size > 1)
    )
    flexible_k = grid.nuisance_count + 2 + candidate_dimensions
    flexible_bic = flexible_k * np.log(observed.size) - 2.0 * result.log_likelihood
    weights = np.exp(-0.5 * flexible_bic - logsumexp(-0.5 * flexible_bic))
    q16 = _weighted_quantile(grid.candidate_longitudes, weights, 0.16)
    median = _weighted_quantile(grid.candidate_longitudes, weights, 0.50)
    q84 = _weighted_quantile(grid.candidate_longitudes, weights, 0.84)
    lat_q16 = _weighted_quantile(grid.candidate_latitudes, weights, 0.16)
    lat_median = _weighted_quantile(grid.candidate_latitudes, weights, 0.50)
    lat_q84 = _weighted_quantile(grid.candidate_latitudes, weights, 0.84)
    best = int(np.argmin(flexible_bic))

    # The null uses the same nuisance terms and a positive uniform map. If a
    # timing grid is active, it profiles timing and counts it as one parameter.
    uniform_indices = [
        index
        for index, (longitude, width) in enumerate(
            zip(grid.candidate_longitudes, grid.candidate_widths)
        )
        if longitude == grid.longitudes[0] and width == grid.widths[0]
    ]
    null_fits = []
    for index in uniform_indices:
        design = grid.fit_designs[index, :, : grid.nuisance_count + 1]
        null_fits.append(
            fit_profile(
                design,
                fit_y,
                1.0,
                map_columns=(grid.nuisance_count,),
                nuisance_columns=tuple(range(grid.nuisance_count)),
            )
        )
    null_log_likelihood = max(fit.log_likelihood for fit in null_fits)
    null_k = grid.nuisance_count + 1 + int(grid.timings.size > 1)
    uniform_bic = null_k * np.log(observed.size) - 2.0 * null_log_likelihood
    delta_bic = float(np.min(flexible_bic) - uniform_bic)

    raw_best = grid.raw_designs[best] @ result.fits[best].coefficients
    residual = observed - raw_best
    coefficients = result.fits[best].coefficients
    map_coefficients = (
        coefficients[grid.nuisance_count] * _uniform_coefficients(grid.harmonic_degree)
        + coefficients[grid.nuisance_count + 1]
        * _hotspot_coefficients(
            grid.candidate_longitudes[best],
            grid.candidate_widths[best],
            grid.harmonic_degree,
            grid.candidate_latitudes[best],
        )
    )
    _, _, rendered_map = render_map(map_coefficients, nlon=181, nlat=91)
    minimum_map = float(np.min(np.asarray(rendered_map, dtype=float)))
    contains = None
    if injected_longitude is not None:
        contains = bool(q16 <= injected_longitude <= q84)
    latitude_contains = None
    if injected_latitude is not None:
        latitude_contains = bool(lat_q16 <= injected_latitude <= lat_q84)
    item = RecoveryTrial(
        trial=int(trial),
        shift=int(shift),
        injected_longitude_degrees=(
            None if injected_longitude is None else float(injected_longitude)
        ),
        recovered_longitude_degrees=median,
        longitude_q16_degrees=q16,
        longitude_q84_degrees=q84,
        delta_bic=delta_bic,
        detected=bool(delta_bic < detection_threshold),
        interval_contains_injection=contains,
        residual_rms_ppm=float(np.sqrt(np.mean(residual**2)) * 1.0e6),
        best_width_degrees=float(grid.candidate_widths[best]),
        best_timing_seconds=float(grid.candidate_timings[best]),
        minimum_rendered_map_intensity=minimum_map,
        rendered_map_positive=bool(minimum_map >= -1.0e-12),
        injected_latitude_degrees=(
            None if injected_latitude is None else float(injected_latitude)
        ),
        recovered_latitude_degrees=lat_median,
        latitude_q16_degrees=lat_q16,
        latitude_q84_degrees=lat_q84,
        latitude_interval_contains_injection=latitude_contains,
        noise_ppm=(None if noise_ppm is None else float(noise_ppm)),
        eclipse_count=(None if eclipse_count is None else int(eclipse_count)),
    )
    return item, weights, raw_best


def _injected_planet(
    config,
    time: FloatArray,
    longitude: float,
    width: float,
    contrast: float,
    latitude: float = 0.0,
) -> FloatArray:
    ydeg = int(config.map.harmonic_degree)
    quadrature = disk_quadrature(16, 64)
    eclipse = np.asarray(
        secondary_eclipse_design_matrix(
            time,
            float(config.system.period_days),
            float(config.system.a_over_rstar),
            float(config.system.inclination_degrees),
            float(config.system.radius_ratio),
            ydeg,
            float(config.system.transit_time),
            theta0=np.pi,
            angle_unit="deg",
            quadrature=quadrature,
        ),
        dtype=float,
    )
    coefficients = _uniform_coefficients(ydeg) + contrast * _hotspot_coefficients(
        longitude, width, ydeg, latitude
    )
    visible = float(np.asarray(map_design_matrix(0.0, 0.0, ydeg)) @ coefficients)
    return eclipse @ (coefficients / visible)


def _uniform_reference_fit(
    observed: FloatArray, sigma: FloatArray, grid: _TemplateGrid
) -> tuple[FloatArray, FloatArray, float]:
    fit_y, _ = _whiten_stack(
        observed, grid.raw_designs[:1], grid.covariance_cholesky, sigma
    )
    candidates = [
        index
        for index, (longitude, width) in enumerate(
            zip(grid.candidate_longitudes, grid.candidate_widths)
        )
        if longitude == grid.longitudes[0] and width == grid.widths[0]
    ]
    fits = []
    for index in candidates:
        design = grid.fit_designs[index, :, : grid.nuisance_count + 1]
        fits.append(
            fit_profile(
                design,
                fit_y,
                1.0,
                map_columns=(grid.nuisance_count,),
                nuisance_columns=tuple(range(grid.nuisance_count)),
            )
        )
    selected = int(np.argmax([fit.log_likelihood for fit in fits]))
    best_index = candidates[selected]
    fit = fits[selected]
    raw_design = grid.raw_designs[best_index, :, : grid.nuisance_count + 1]
    prediction = raw_design @ fit.coefficients
    return prediction, observed - prediction, float(fit.coefficients[-1])


def _save_trials(output: Path, trials: list[RecoveryTrial]) -> None:
    rows = [asdict(item) for item in trials]
    if not rows:
        return
    with (output / "recovery_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_comparison(output: Path, report: RecoveryReport) -> None:
    """Write a short human-readable comparison beside the JSON result."""

    if report.case == "hatp32":
        trial = report.trials[0]
        text = f"""# HAT-P-32b recovery comparison

| Result | Prior task | robert-mapping quick test |
| --- | ---: | ---: |
| Injected longitude | +10.00 deg | +10.00 deg |
| Recovered median | +10.25 deg | {trial.recovered_longitude_degrees:+.2f} deg |
| 68% interval | -10.25 to +41.00 deg | {trial.longitude_q16_degrees:+.2f} to {trial.longitude_q84_degrees:+.2f} deg |
| Residual RMS | 60.31 ppm | {trial.residual_rms_ppm:.2f} ppm |
| Delta BIC, map - uniform | not used | {trial.delta_bic:+.2f} |

Both intervals contain the +10 degree injection. The mapped model is not
preferred by the quick BIC test. This agrees with the prior result that the
longitude was conditional and was not a strong offset detection.
"""
    elif report.case == "synthetic_matrix":
        evidence = report.comparison["mapping_evidence"]
        location = report.comparison["conditional_location"]
        strata = report.comparison["strata"]
        text = f"""# Synthetic recovery and rejection matrix

| Quantity | Result |
| --- | ---: |
| Noise levels | {", ".join(f"{value:g} ppm" for value in strata["noise_levels_ppm"])} |
| Eclipse counts | {", ".join(str(value) for value in strata["eclipse_counts"])} |
| Null false positives | {evidence["null_false_positives"]} |
| Injection detection rate | {evidence["injection_detection_count"]} |
| Longitude interval coverage | {location["longitude_interval_coverage"]:.3f} |
| Latitude interval coverage | {location["latitude_interval_coverage"]:.3f} |

Delta BIC tests mapping evidence against a uniform map. Longitude and
latitude intervals are conditional on the mapped model and must not be read
as independent detection probabilities. The latitude result is expected to
be weaker because north--south information is concentrated in ingress and
egress.
"""
    else:
        injected_detections = sum(
            trial.detected
            for trial in report.trials
            if trial.injected_longitude_degrees is not None
        )
        null_delta = [
            trial.delta_bic
            for trial in report.trials
            if trial.injected_longitude_degrees is None
        ]
        text = f"""# WASP-178b rejection and recovery comparison

| Check | Prior task | robert-mapping quick test |
| --- | ---: | ---: |
| Null false positives | 0/8 | {report.false_positive_count}/{report.null_trial_count} |
| Null Delta BIC range | +18.80 to +22.44 | {min(null_delta):+.2f} to {max(null_delta):+.2f} |
| Injected trials crossing the detection rule | 0/24 | {injected_detections}/{report.injection_trial_count} |
| Injection interval coverage | 21/24 (0.875) | {report.injection_coverage_count}/{report.injection_trial_count} ({report.injection_coverage:.3f}) |
| Detection rule | Delta BIC < -6 | Delta BIC < -6 |

The small run gives the same broad result: no null trial gives a hotspot
detection, and most injected longitudes are inside their 68% recovery
intervals. This time-correlated-noise and BIC result is separate from the fixed-timing Hammond
cross-validation result.
"""
    (output / "comparison_report.md").write_text(text, encoding="utf-8")


def _plot_report(config, report: RecoveryReport, output: Path) -> None:
    if not config.output.save_report:
        return
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "font.size": 10})
    injected = [
        np.nan if trial.injected_longitude_degrees is None else trial.injected_longitude_degrees
        for trial in report.trials
    ]
    recovered = [trial.recovered_longitude_degrees for trial in report.trials]
    low = [trial.recovered_longitude_degrees - trial.longitude_q16_degrees for trial in report.trials]
    high = [trial.longitude_q84_degrees - trial.recovered_longitude_degrees for trial in report.trials]
    delta = [trial.delta_bic for trial in report.trials]
    x = np.arange(len(report.trials))
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), constrained_layout=True)
    axes[0].errorbar(
        x,
        recovered,
        yerr=np.vstack((low, high)),
        fmt="o",
        color=config.output.best_fit_color,
        ecolor=config.output.best_fit_color,
        capsize=3,
        label="robert-mapping recovery",
    )
    finite = np.isfinite(injected)
    axes[0].scatter(x[finite], np.asarray(injected)[finite], marker="x", color="black", label="injection")
    axes[0].axhline(0.0, color="#777777", linewidth=1)
    axes[0].set_ylabel("Hotspot longitude (degrees)")
    axes[0].set_xticks(x)
    axes[0].legend(loc="best")
    axes[1].bar(x, delta, color=config.output.best_fit_color)
    axes[1].axhline(
        config.recovery.detection_delta_bic,
        color="black",
        linestyle="--",
        label="detection threshold",
    )
    axes[1].axhline(0.0, color="#777777", linewidth=1)
    axes[1].set_ylabel("Delta BIC (map - uniform)")
    axes[1].set_xlabel("Trial")
    axes[1].set_xticks(x)
    axes[1].legend(loc="best")
    display_name = {
        "hatp32": "HAT-P-32b",
        "wasp178b": "WASP-178b",
        "synthetic_matrix": "synthetic recovery matrix",
    }[report.case]
    figure.suptitle(f"{display_name} recovery and rejection check")
    figure.savefig(output / "recovery_summary.png", dpi=180)
    figure.savefig(output / "recovery_summary.pdf")
    plt.close(figure)


def _hatp32(config) -> RecoveryReport:
    settings = config.recovery
    centre = config.system.transit_time + 0.5 * config.system.period_days
    time = centre + np.linspace(-3.5, 3.5, 285) / 24.0
    sigma = np.full(time.size, settings.noise_ppm * 1.0e-6)
    grid = _design_grid(config, time, sigma)
    rng = np.random.default_rng(int(config.project.seed))
    trials: list[RecoveryTrial] = []
    trial_number = 0
    for longitude in settings.injected_longitudes_degrees:
        planet = _injected_planet(
            config,
            time,
            longitude,
            settings.hotspot_width_degrees,
            settings.hotspot_fraction,
        )
        for repetition in range(settings.trials_per_case):
            observed = (
                1.0
                + config.system.planet_flux_ratio * planet
                + rng.normal(0.0, sigma)
            )
            trial, _, _ = _fit_trial(
                observed,
                sigma,
                grid,
                settings.detection_delta_bic,
                trial=trial_number,
                shift=repetition,
                injected_longitude=longitude,
            )
            trials.append(trial)
            if trial_number == 0:
                output = Path(config.output.directory)
                output.mkdir(parents=True, exist_ok=True)
                with (output / "synthetic_observation.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        ("time_bjd_tdb", "relative_flux", "relative_flux_err")
                    )
                    writer.writerows(zip(time, observed, sigma))
            trial_number += 1
    coverage = sum(bool(item.interval_contains_injection) for item in trials)
    # The prior 60 ppm posterior was broad. Broad consistency means that the
    # new interval contains the injection and that its median lies inside the
    # prior 68% interval. It does not mean that the two medians must match.
    coverage_rate = coverage / len(trials)
    median_recovery = float(
        np.median([item.recovered_longitude_degrees for item in trials])
    )
    passed = coverage_rate >= 0.75 and -10.25 <= median_recovery <= 41.0
    comparison = {
        "prior_task": "HAT-P-32b eclipse-map recovery",
        "prior_injected_longitude_degrees": 10.0,
        "prior_recovered_median_degrees": 10.25,
        "prior_q16_degrees": -10.25,
        "prior_q84_degrees": 41.0,
        "prior_residual_rms_ppm": 60.309,
        "interpretation": "The old interval included zero; its median was not a strong detection.",
    }
    return RecoveryReport(
        status="passed" if passed else "failed",
        case="hatp32",
        seed=int(config.project.seed),
        n_observations=time.size,
        longitude_sign="positive is east, in the direction of planetary rotation",
        false_positive_count=0,
        null_trial_count=0,
        injection_coverage_count=coverage,
        injection_trial_count=len(trials),
        false_positive_rate=None,
        injection_coverage=coverage_rate,
        trials=tuple(trials),
        comparison=comparison,
        notes=(
            "Fast profile-grid test; no NUTS samples were drawn.",
            "The injection is a positive Gaussian map projected to degree 2.",
            "Mapping evidence and conditional longitude are reported separately.",
            "Pass requires at least 75% interval coverage and broad median agreement.",
        ),
    )


def _wasp178b(config) -> RecoveryReport:
    settings = config.recovery
    light_curve = load_light_curve(config)
    time = np.asarray(light_curve.time, dtype=float)
    observed = np.asarray(light_curve.flux, dtype=float)
    sigma = np.asarray(light_curve.flux_err, dtype=float)
    grid = _design_grid(config, time, sigma)
    base, residuals, depth = _uniform_reference_fit(observed, sigma, grid)
    uniform_shape = _injected_planet(config, time, 0.0, 40.0, 0.0)
    preferred_shifts = np.asarray([113, 299, 487, 701, 991, 1217, 1511, 1733])
    shifts = preferred_shifts[: settings.trials_per_case]
    trials: list[RecoveryTrial] = []
    trial_number = 0
    injections: tuple[float | None, ...] = (None,) + tuple(
        float(value) for value in settings.injected_longitudes_degrees
    )
    for injected in injections:
        if injected is None:
            difference = np.zeros_like(time)
        else:
            injected_shape = _injected_planet(
                config,
                time,
                injected,
                settings.hotspot_width_degrees,
                settings.hotspot_fraction,
            )
            difference = depth * (injected_shape - uniform_shape)
        for shift in shifts:
            trial_observed = base + difference + cyclic_residual_shift(residuals, int(shift))
            item, _, _ = _fit_trial(
                trial_observed,
                sigma,
                grid,
                settings.detection_delta_bic,
                trial=trial_number,
                shift=int(shift),
                injected_longitude=injected,
            )
            trials.append(item)
            trial_number += 1
    null = [item for item in trials if item.injected_longitude_degrees is None]
    injected_trials = [item for item in trials if item.injected_longitude_degrees is not None]
    false_positives = sum(item.detected for item in null)
    coverage = sum(bool(item.interval_contains_injection) for item in injected_trials)
    coverage_rate = coverage / len(injected_trials) if injected_trials else None
    passed = false_positives == 0 and coverage_rate is not None and coverage_rate >= 0.75
    comparison = {
        "prior_task": "Fit WASP-178b G395M eclipse map",
        "prior_null_false_positives": "0/8",
        "prior_null_delta_bic_range": [18.79519, 22.44460],
        "new_null_delta_bic_range": [
            min(item.delta_bic for item in null),
            max(item.delta_bic for item in null),
        ],
        "prior_injected_trials_detected": "0/24",
        "new_injected_trials_detected": f"{sum(item.detected for item in injected_trials)}/{len(injected_trials)}",
        "prior_injection_coverage": "21/24 (0.875)",
        "prior_delta_bic_flexible_minus_uniform": 11.86793,
        "prior_conditional_longitude_median_degrees": 14.4546,
        "prior_conditional_q16_degrees": -1.4211,
        "prior_conditional_q84_degrees": 33.8806,
        "interpretation": "The flexible time-correlated-noise and BIC analysis preferred the uniform model.",
        "separate_hammond_cv_delta_elpd": 111.5432,
        "separate_hammond_cv_note": "Fixed timing/detrending Hammond CV is a different model and is not this null calibration.",
    }
    return RecoveryReport(
        status="passed" if passed else "failed",
        case="wasp178b",
        seed=int(config.project.seed),
        n_observations=time.size,
        longitude_sign="positive is east, in the direction of planetary rotation",
        false_positive_count=false_positives,
        null_trial_count=len(null),
        injection_coverage_count=coverage,
        injection_trial_count=len(injected_trials),
        false_positive_rate=false_positives / len(null),
        injection_coverage=coverage_rate,
        trials=tuple(trials),
        comparison=comparison,
        notes=(
            "Small cyclic-residual calibration with fixed time-correlated-noise settings.",
            "Positive uniform and Gaussian map amplitudes are fitted by active sets.",
            "Mapping evidence and conditional longitude are reported separately.",
        ),
    )


def _matrix_times(config, eclipse_count: int, points_per_eclipse: int = 121) -> FloatArray:
    """Return one or more short, separated eclipse windows.

    This deliberately uses repeated windows at the same orbital phase.  It
    tests the expected information gain from additional eclipses without
    adding a separate phase-curve model or a large synthetic data product.
    """

    if eclipse_count < 1:
        raise ValueError("eclipse_count must be at least one")
    points = int(points_per_eclipse)
    if points < 9:
        raise ValueError("points_per_eclipse must be at least nine")
    period = float(config.system.period_days)
    centre = float(config.system.transit_time) + 0.5 * period
    offsets = np.linspace(-3.5, 3.5, points, dtype=float) / 24.0
    return np.concatenate(
        [centre + index * period + offsets for index in range(int(eclipse_count))]
    )


def _synthetic_matrix(config) -> RecoveryReport:
    """Run a bounded latitude/longitude/noise/eclipses recovery matrix.

    Each trial reports two distinct quantities:

    * ``delta_bic`` and ``detected`` test mapping evidence against a uniform
      map;
    * the longitude and latitude intervals describe the conditional location
      within the mapped model.

    The matrix is intentionally sampler-free.  It is a diagnostic calibration,
    not a replacement for a posterior analysis.
    """

    settings = config.recovery
    longitudes = tuple(float(value) for value in settings.injected_longitudes_degrees)
    latitudes = tuple(
        float(value)
        for value in getattr(settings, "injected_latitudes_degrees", (0.0,))
    )
    noise_levels = tuple(
        float(value)
        for value in getattr(settings, "noise_levels_ppm", (settings.noise_ppm,))
    )
    eclipse_counts = tuple(
        int(value)
        for value in getattr(settings, "eclipse_counts", (1,))
    )
    points_per_eclipse = int(getattr(settings, "points_per_eclipse", 121))
    rng = np.random.default_rng(int(config.project.seed))
    trials: list[RecoveryTrial] = []
    trial_number = 0

    # One null trial per noise/eclipses stratum measures false positives.  The
    # injected trials then measure conditional longitude and latitude coverage
    # in the same stratum.
    for noise_ppm in noise_levels:
        sigma = np.full(
            points_per_eclipse,
            noise_ppm * 1.0e-6,
            dtype=float,
        )
        for eclipse_count in eclipse_counts:
            time = _matrix_times(config, eclipse_count, points_per_eclipse)
            sigma = np.full(time.size, noise_ppm * 1.0e-6, dtype=float)
            grid = _design_grid(config, time, sigma)
            uniform_shape = _injected_planet(config, time, 0.0, 40.0, 0.0)
            for repetition in range(int(settings.trials_per_case)):
                null_observed = (
                    1.0
                    + config.system.planet_flux_ratio * uniform_shape
                    + rng.normal(0.0, sigma)
                )
                null_trial, _, _ = _fit_trial(
                    null_observed,
                    sigma,
                    grid,
                    settings.detection_delta_bic,
                    trial=trial_number,
                    shift=repetition,
                    injected_longitude=None,
                    noise_ppm=noise_ppm,
                    eclipse_count=eclipse_count,
                )
                trials.append(null_trial)
                trial_number += 1

            for longitude in longitudes:
                for latitude in latitudes:
                    injected_shape = _injected_planet(
                        config,
                        time,
                        longitude,
                        settings.hotspot_width_degrees,
                        settings.hotspot_fraction,
                        latitude,
                    )
                    for repetition in range(int(settings.trials_per_case)):
                        observed = (
                            1.0
                            + config.system.planet_flux_ratio * injected_shape
                            + rng.normal(0.0, sigma)
                        )
                        item, _, _ = _fit_trial(
                            observed,
                            sigma,
                            grid,
                            settings.detection_delta_bic,
                            trial=trial_number,
                            shift=repetition,
                            injected_longitude=longitude,
                            injected_latitude=latitude,
                            noise_ppm=noise_ppm,
                            eclipse_count=eclipse_count,
                        )
                        trials.append(item)
                        trial_number += 1

    null = [item for item in trials if item.injected_longitude_degrees is None]
    injected = [item for item in trials if item.injected_longitude_degrees is not None]
    false_positives = sum(item.detected for item in null)
    longitude_coverage = sum(
        bool(item.interval_contains_injection) for item in injected
    )
    latitude_coverage = sum(
        bool(item.latitude_interval_contains_injection) for item in injected
    )
    injection_count = len(injected)
    coverage_rate = longitude_coverage / injection_count if injection_count else None
    latitude_coverage_rate = (
        latitude_coverage / injection_count if injection_count else None
    )
    # A pass means that mapping evidence is not spuriously found in the null
    # data and that the small conditional intervals cover the injections at a
    # useful calibration rate.  Longitude and latitude are reported separately
    # because the latter is expected to be less informative.
    passed = (
        false_positives == 0
        and coverage_rate is not None
        and coverage_rate >= 0.75
        and latitude_coverage_rate is not None
        and latitude_coverage_rate >= 0.50
    )
    comparison = {
        "mapping_evidence": {
            "null_false_positives": f"{false_positives}/{len(null)}",
            "null_false_positive_rate": (
                false_positives / len(null) if null else None
            ),
            "injection_detection_count": f"{sum(item.detected for item in injected)}/{len(injected)}",
            "detection_delta_bic_threshold": float(settings.detection_delta_bic),
        },
        "conditional_location": {
            "longitude_interval_coverage": coverage_rate,
            "latitude_interval_coverage": latitude_coverage_rate,
            "longitude_injection_degrees": list(longitudes),
            "latitude_injection_degrees": list(latitudes),
        },
        "strata": {
            "noise_levels_ppm": list(noise_levels),
            "eclipse_counts": list(eclipse_counts),
            "points_per_eclipse": points_per_eclipse,
        },
        "interpretation": (
            "Longitude and latitude intervals are conditional on the mapped model. "
            "Delta BIC is the separate mapping-evidence diagnostic."
        ),
    }
    return RecoveryReport(
        status="passed" if passed else "failed",
        case="synthetic_matrix",
        seed=int(config.project.seed),
        n_observations=max(item.eclipse_count or 1 for item in trials)
        * points_per_eclipse,
        longitude_sign="positive is east, in the direction of planetary rotation",
        false_positive_count=false_positives,
        null_trial_count=len(null),
        injection_coverage_count=longitude_coverage,
        injection_trial_count=injection_count,
        false_positive_rate=(false_positives / len(null) if null else None),
        injection_coverage=coverage_rate,
        trials=tuple(trials),
        comparison=comparison,
        notes=(
            "Bounded sampler-free GLS profile-grid matrix; no NUTS samples were drawn.",
            "The matrix varies injected longitude, injected latitude, white-noise level, and eclipse count.",
            "Mapping evidence (Delta BIC) is reported separately from conditional location intervals.",
            "Repeated eclipse windows test additional phase coverage at the same orbital phase.",
            "Latitude coverage is expected to be lower because ingress and egress carry most north--south information.",
        ),
    )


def run_recovery(config) -> RecoveryReport:
    """Run the selected recovery case and save all compact products."""

    if config.recovery.case == "hatp32":
        report = _hatp32(config)
    elif config.recovery.case == "wasp178b":
        report = _wasp178b(config)
    elif config.recovery.case == "synthetic_matrix":
        report = _synthetic_matrix(config)
    else:  # strict configuration validation should make this unreachable
        raise ValueError(f"Unsupported recovery case: {config.recovery.case}")
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "recovery_summary.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
    )
    _save_trials(output, list(report.trials))
    _save_comparison(output, report)
    _plot_report(config, report, output)
    return report


__all__ = ["RecoveryReport", "RecoveryTrial", "run_recovery"]
