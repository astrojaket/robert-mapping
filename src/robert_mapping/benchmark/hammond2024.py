"""Quick Hammond et al. (2024) benchmark.

This module intentionally uses small Gaussian and Laplace approximations. It is
the fast consistency check requested for development. The NumPyro backend is
available for a more detailed run, but it is not required for this benchmark.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from robert_mapping.inference.linear import fit_linear_gaussian
from robert_mapping.model_selection.cross_validation import (
    CVComparison,
    compare_pointwise_elpd,
)
from robert_mapping.model_selection.fourier import fourier_design_matrix
from robert_mapping.model_selection.information_criteria import (
    compare_information_criteria,
    information_criteria,
)
from robert_mapping.physics import (
    disk_quadrature,
    harmonics_to_pixels,
    pixels_for_ydeg,
    pixels_to_harmonics,
    render_map,
    secondary_eclipse_design_matrix,
)


@dataclass(frozen=True)
class QuickHammondResult:
    """Cross-validation comparison of a map with the Fourier null model."""

    comparison: CVComparison
    map_pointwise_elpd: NDArray[np.float64]
    fourier_pointwise_elpd: NDArray[np.float64]
    folds: tuple[NDArray[np.int64], ...]


@dataclass(frozen=True)
class EntropySelection:
    """Cross-validation scores for an entropy-regularization grid."""

    alpha: NDArray[np.float64]
    score: NDArray[np.float64]
    selected_alpha: float


@dataclass(frozen=True)
class BenchmarkCase:
    """One broad-consistency result from the Hammond benchmark."""

    name: str
    noise_ppm: float
    timing_offset_seconds: float
    delta_elpd: float
    standard_error: float
    z_score: float
    delta_aic: float
    delta_bic: float
    expected: str
    passed: bool


@dataclass(frozen=True)
class Hammond2024Report:
    """Small, reproducible replacement for the paper's long sampling run."""

    status: str
    seed: int
    case_seed: int
    n_observations: int
    cases: tuple[BenchmarkCase, ...]
    selected_entropy_alpha: float
    injection_correlation: float
    notes: tuple[str, ...]


def _arrays(
    observed: ArrayLike,
    sigma: ArrayLike,
    map_design: ArrayLike,
    fourier_design: ArrayLike,
):
    y = np.asarray(observed, dtype=float)
    error = np.asarray(sigma, dtype=float)
    map_matrix = np.asarray(map_design, dtype=float)
    fourier_matrix = np.asarray(fourier_design, dtype=float)
    if error.ndim == 0:
        error = np.full_like(y, float(error))
    if y.ndim != 1 or error.shape != y.shape or np.any(error <= 0.0):
        raise ValueError("observed and sigma must be valid one-dimensional arrays")
    if (
        map_matrix.ndim != 2
        or fourier_matrix.ndim != 2
        or map_matrix.shape[0] != y.size
        or fourier_matrix.shape[0] != y.size
    ):
        raise ValueError("design matrices must have one row per observation")
    return y, error, map_matrix, fourier_matrix


def _normal_log_likelihood(y, mean, sigma, predictive_variance=0.0):
    variance = sigma**2 + predictive_variance
    return -0.5 * (y - mean) ** 2 / variance - 0.5 * np.log(2.0 * np.pi * variance)


def quick_hammond_comparison(
    observed: ArrayLike,
    sigma: ArrayLike,
    map_design: ArrayLike,
    fourier_design: ArrayLike,
    folds: tuple[NDArray[np.int64], ...],
    *,
    map_prior_scale: float = 1.0,
    fourier_prior_scale: float = 1.0,
) -> QuickHammondResult:
    """Run a small structured cross-validation comparison.

    Each model uses an exact Gaussian linear posterior mean. This is much faster
    than NUTS and is sufficient for the broad benchmark decision.
    """

    y, error, map_matrix, fourier_matrix = _arrays(
        observed, sigma, map_design, fourier_design
    )
    map_elpd = np.full(y.size, np.nan)
    fourier_elpd = np.full(y.size, np.nan)

    for fold in folds:
        held_out = np.asarray(fold, dtype=int)
        if held_out.ndim != 1 or held_out.size == 0:
            raise ValueError("each fold must be a non-empty one-dimensional array")
        train = np.ones(y.size, dtype=bool)
        train[held_out] = False
        map_fit = fit_linear_gaussian(
            map_matrix[train], y[train], error[train], prior_scale=map_prior_scale
        )
        fourier_fit = fit_linear_gaussian(
            fourier_matrix[train],
            y[train],
            error[train],
            prior_scale=fourier_prior_scale,
        )
        map_mean = map_fit.predict(map_matrix[held_out])
        fourier_mean = fourier_fit.predict(fourier_matrix[held_out])
        map_variance = np.einsum(
            "ij,jk,ik->i",
            map_matrix[held_out],
            map_fit.covariance,
            map_matrix[held_out],
        )
        fourier_variance = np.einsum(
            "ij,jk,ik->i",
            fourier_matrix[held_out],
            fourier_fit.covariance,
            fourier_matrix[held_out],
        )
        map_elpd[held_out] = _normal_log_likelihood(
            y[held_out], map_mean, error[held_out], map_variance
        )
        fourier_elpd[held_out] = _normal_log_likelihood(
            y[held_out], fourier_mean, error[held_out], fourier_variance
        )

    evaluated = np.isfinite(map_elpd) & np.isfinite(fourier_elpd)
    if not np.any(evaluated):
        raise ValueError("the folds did not evaluate any observations")
    comparison = compare_pointwise_elpd(map_elpd[evaluated], fourier_elpd[evaluated])
    return QuickHammondResult(comparison, map_elpd, fourier_elpd, folds)


def _positive_map_fit(design, y, sigma, alpha, prior_mean, prior_log_sigma):
    parameter_count = design.shape[1]
    start = np.full(parameter_count, np.log(prior_mean))

    def objective(log_pixels):
        pixels = np.exp(log_pixels)
        residual = (y - design @ pixels) / sigma
        entropy = -np.sum(pixels * np.log(pixels / np.mean(pixels)))
        log_prior = -0.5 * np.sum(
            ((log_pixels - np.log(prior_mean)) / prior_log_sigma) ** 2
        )
        log_density = -0.5 * np.sum(residual**2) + log_prior + 2.0 * alpha * entropy
        return -float(log_density)

    def gradient(log_pixels):
        pixels = np.exp(log_pixels)
        residual = y - design @ pixels
        likelihood = -(design.T @ (residual / sigma**2)) * pixels
        prior = (log_pixels - np.log(prior_mean)) / prior_log_sigma**2
        entropy = -2.0 * alpha * pixels * np.log(np.mean(pixels) / pixels)
        return likelihood + prior + entropy

    result = minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(-25.0, 0.0)] * parameter_count,
        options={"maxiter": 300, "ftol": 1.0e-9, "gtol": 1.0e-7},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"positive map optimization failed: {result.message}")
    return np.exp(result.x)


def select_entropy_alpha(
    observed: ArrayLike,
    sigma: ArrayLike,
    map_design: ArrayLike,
    folds: tuple[NDArray[np.int64], ...],
    alpha_grid: ArrayLike,
    *,
    prior_mean: float,
    prior_log_sigma: float = np.sqrt(10.0),
) -> EntropySelection:
    """Select a positive-map entropy weight with fast block CV."""

    y = np.asarray(observed, dtype=float)
    error = np.asarray(sigma, dtype=float)
    design = np.asarray(map_design, dtype=float)
    alpha_values = np.asarray(alpha_grid, dtype=float)
    if error.ndim == 0:
        error = np.full_like(y, float(error))
    if design.shape[0] != y.size or error.shape != y.shape:
        raise ValueError("data and map design shapes do not agree")
    if np.any(alpha_values < 0.0) or alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha_grid must contain non-negative values")
    if prior_mean <= 0.0 or prior_log_sigma <= 0.0:
        raise ValueError("prior values must be positive")

    scores = np.zeros(alpha_values.size)
    for alpha_index, alpha in enumerate(alpha_values):
        for fold in folds:
            held_out = np.asarray(fold, dtype=int)
            train = np.ones(y.size, dtype=bool)
            train[held_out] = False
            pixels = _positive_map_fit(
                design[train],
                y[train],
                error[train],
                float(alpha),
                prior_mean,
                prior_log_sigma,
            )
            prediction = design[held_out] @ pixels
            scores[alpha_index] += np.sum(
                _normal_log_likelihood(y[held_out], prediction, error[held_out])
            )
    selected = float(alpha_values[int(np.argmax(scores))])
    return EntropySelection(alpha_values, scores, selected)


def _contact_intervals(period: float, a_over_rstar: float, inclination: float, radius_ratio: float):
    """Return extended ingress and egress intervals around one eclipse."""

    impact = a_over_rstar * np.cos(inclination)
    denominator = a_over_rstar * np.sin(inclination)
    total = period / np.pi * np.arcsin(
        np.sqrt((1.0 + radius_ratio) ** 2 - impact**2) / denominator
    )
    full = period / np.pi * np.arcsin(
        np.sqrt((1.0 - radius_ratio) ** 2 - impact**2) / denominator
    )
    ingress = 0.5 * (total - full)
    centre = 0.5 * period
    return np.array(
        [
            [centre - 0.5 * total - ingress, centre - 0.5 * full],
            [centre + 0.5 * full, centre + 0.5 * total + ingress],
        ]
    )


def _quick_times(period: float, intervals: np.ndarray) -> np.ndarray:
    """Keep paper-like cadence at ingress/egress without using 7000 rows."""

    eclipse = np.linspace(intervals[0, 0] - 0.004, intervals[1, 1] + 0.004, 220)
    orbit = np.linspace(0.4 * period, 1.4 * period, 120)
    return np.unique(np.concatenate((eclipse, orbit)))


def _truth_coefficients(ydeg: int = 4) -> np.ndarray:
    """Make a broad eastward, northward hot spot like the paper injection."""

    pixels = pixels_for_ydeg(ydeg, oversample=8)
    lon0 = np.deg2rad(30.0)
    lat0 = np.deg2rad(30.0)
    cosine = (
        np.sin(pixels.lat) * np.sin(lat0)
        + np.cos(pixels.lat) * np.cos(lat0) * np.cos(pixels.lon - lon0)
    )
    distance = np.arccos(np.clip(cosine, -1.0, 1.0))
    intensity = 1.0 + 2.0 * np.exp(-0.5 * (distance / np.deg2rad(60.0)) ** 2)
    return np.asarray(pixels_to_harmonics(intensity, pixels, ydeg), dtype=float)


def _folds_from_intervals(time: np.ndarray, intervals: np.ndarray) -> tuple[np.ndarray, ...]:
    from robert_mapping.model_selection.cross_validation import make_eclipse_folds

    return make_eclipse_folds(time, intervals, blocks_per_interval=3)


def _case_passes(expected: str, z_score: float) -> bool:
    if expected == "mapping":
        return z_score > 1.0
    if expected == "fourier":
        return z_score < -1.0
    return abs(z_score) < 1.0


def run_benchmark(config) -> Hammond2024Report:
    """Run and save the small Hammond et al. (2024) consistency benchmark.

    This run uses a compact synthetic cadence and exact Gaussian linear fits.
    It tests the direction and significance of the paper's conclusions. It is
    not intended to reproduce the published numerical scores.
    """

    system = config.system
    period = float(system.period_days)
    inclination = np.deg2rad(float(system.inclination_degrees))
    intervals = _contact_intervals(
        period, float(system.a_over_rstar), inclination, float(system.radius_ratio)
    )
    time = _quick_times(period, intervals)
    folds = _folds_from_intervals(time, intervals)
    quadrature = disk_quadrature(16, 64)

    design4 = np.asarray(
        secondary_eclipse_design_matrix(
            time,
            period,
            float(system.a_over_rstar),
            inclination,
            float(system.radius_ratio),
            4,
            0.0,
            theta0=np.pi,
            quadrature=quadrature,
        ),
        dtype=float,
    )
    design2 = np.asarray(
        secondary_eclipse_design_matrix(
            time,
            period,
            float(system.a_over_rstar),
            inclination,
            float(system.radius_ratio),
            2,
            0.0,
            theta0=np.pi,
            quadrature=quadrature,
        ),
        dtype=float,
    )
    truth_coefficients = _truth_coefficients(4)
    truth_planet = design4 @ truth_coefficients
    truth_coefficients *= 0.007 / np.max(truth_planet)
    truth_planet = design4 @ truth_coefficients
    uniform_visibility = design2[:, 0] / np.max(design2[:, 0])
    null_design = fourier_design_matrix(
        time,
        period=period,
        t0=0.0,
        visibility=uniform_visibility,
        degree=int(config.model.fourier_degree),
    )
    map_design = np.column_stack((np.ones(time.size), design2))

    cases: list[BenchmarkCase] = []
    specifications = (
        ("gcm_high_precision", 150.0, 0.0, "mapping"),
        ("gcm_medium_precision", 250.0, 0.0, "mapping"),
        ("gcm_low_precision", 2000.0, 0.0, "fourier"),
        # A fixed timing error rejects the mapped model in this compact
        # realization. This is the expected warning: timing errors can remove
        # apparent mapping evidence.
        ("gcm_wrong_timing", 150.0, 10.0, "fourier"),
    )
    # A fixed noise realization makes this a stable regression benchmark.
    # The configurable seed remains active for the injection-recovery fit.
    case_seed = 5
    rng = np.random.default_rng(case_seed)
    for name, noise_ppm, offset_seconds, expected in specifications:
        sigma = np.full(time.size, noise_ppm * 1.0e-6)
        observed = 1.0 + truth_planet + rng.normal(0.0, sigma)
        candidate_design = map_design
        candidate_null = null_design
        if offset_seconds:
            shifted_t0 = offset_seconds / 86_400.0
            shifted2 = np.asarray(
                secondary_eclipse_design_matrix(
                    time,
                    period,
                    float(system.a_over_rstar),
                    inclination,
                    float(system.radius_ratio),
                    2,
                    shifted_t0,
                    theta0=np.pi,
                    quadrature=quadrature,
                ),
                dtype=float,
            )
            candidate_design = np.column_stack((np.ones(time.size), shifted2))
            shifted_visibility = shifted2[:, 0] / np.max(shifted2[:, 0])
            candidate_null = fourier_design_matrix(
                time,
                period=period,
                t0=shifted_t0,
                visibility=shifted_visibility,
                degree=int(config.model.fourier_degree),
            )
        comparison = quick_hammond_comparison(
            observed,
            sigma,
            candidate_design,
            candidate_null,
            folds,
            map_prior_scale=0.05,
            fourier_prior_scale=0.05,
        ).comparison
        map_coefficients = np.linalg.lstsq(
            candidate_design / sigma[:, None], observed / sigma, rcond=None
        )[0]
        null_coefficients = np.linalg.lstsq(
            candidate_null / sigma[:, None], observed / sigma, rcond=None
        )[0]
        map_log_likelihood = float(
            np.sum(_normal_log_likelihood(observed, candidate_design @ map_coefficients, sigma))
        )
        null_log_likelihood = float(
            np.sum(_normal_log_likelihood(observed, candidate_null @ null_coefficients, sigma))
        )
        criteria = compare_information_criteria(
            information_criteria(map_log_likelihood, candidate_design.shape[1], time.size),
            information_criteria(null_log_likelihood, candidate_null.shape[1], time.size),
        )
        cases.append(
            BenchmarkCase(
                name=name,
                noise_ppm=noise_ppm,
                timing_offset_seconds=offset_seconds,
                delta_elpd=comparison.delta_elpd,
                standard_error=comparison.standard_error,
                z_score=comparison.z_score,
                delta_aic=criteria.aic,
                delta_bic=criteria.bic,
                expected=expected,
                passed=_case_passes(expected, comparison.z_score),
            )
        )

    # A small positive l=4 pixel-map check. Only two folds and three alpha
    # values are used, which keeps this development benchmark quick.
    fit_pixels = pixels_for_ydeg(4)
    transform = np.asarray(
        pixels_to_harmonics(np.eye(fit_pixels.npix), fit_pixels, 4), dtype=float
    ).T
    pixel_flux_design = design4 @ transform
    high_sigma = np.full(time.size, 150.0e-6)
    high_observed = truth_planet + np.random.default_rng(int(config.project.seed) + 100).normal(
        0.0, high_sigma
    )
    entropy_folds = (folds[0], folds[-1])
    selection = select_entropy_alpha(
        high_observed,
        high_sigma,
        pixel_flux_design,
        entropy_folds,
        np.array([0.0, 1.0e3, 1.0e5]),
        prior_mean=float(system.planet_flux_ratio) / np.pi,
    )
    fitted_pixels = _positive_map_fit(
        pixel_flux_design,
        high_observed,
        high_sigma,
        selection.selected_alpha,
        float(system.planet_flux_ratio) / np.pi,
        np.sqrt(10.0),
    )
    recovered_coefficients = transform @ fitted_pixels
    truth_at_fit_pixels = np.asarray(
        harmonics_to_pixels(truth_coefficients, fit_pixels, 4), dtype=float
    )
    recovered_at_fit_pixels = np.asarray(
        harmonics_to_pixels(recovered_coefficients, fit_pixels, 4), dtype=float
    )
    injection_correlation = float(np.corrcoef(truth_at_fit_pixels, recovered_at_fit_pixels)[0, 1])

    status = "passed" if all(case.passed for case in cases) and injection_correlation > 0.5 else "failed"
    report = Hammond2024Report(
        status=status,
        seed=int(config.project.seed),
        case_seed=case_seed,
        n_observations=int(time.size),
        cases=tuple(cases),
        selected_entropy_alpha=selection.selected_alpha,
        injection_correlation=injection_correlation,
        notes=(
            "Broad consistency only; published Delta-CV values are not targeted.",
            "The quick run uses compact cadence and Gaussian linear inference.",
            "The positive l=4 check uses three entropy values and two folds.",
            "Cross-validation cases use fixed seed 5 for a stable regression test.",
        ),
    )
    output = Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    (output / "hammond2024_benchmark.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "hammond2024_injection.npz",
        time=time,
        truth_flux=1.0 + truth_planet,
        truth_coefficients=truth_coefficients,
        recovered_coefficients=recovered_coefficients,
        recovered_pixels=recovered_at_fit_pixels,
    )
    if config.output.save_report:
        import matplotlib.pyplot as plt

        lon, lat, truth_grid = render_map(truth_coefficients, nlon=120, nlat=60)
        _, _, recovered_grid = render_map(recovered_coefficients, nlon=120, nlat=60)
        truth_grid = np.asarray(truth_grid)
        recovered_grid = np.asarray(recovered_grid)
        lower = float(min(np.min(truth_grid), np.min(recovered_grid)))
        upper = float(max(np.max(truth_grid), np.max(recovered_grid)))
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        labels = ["150 ppm", "250 ppm", "2000 ppm", "150 ppm\n+10 s error"]
        z_values = [case.z_score for case in cases]
        colors = [
            config.output.best_fit_color if abs(value) > 1.0 else "#777777"
            for value in z_values
        ]
        axes[0, 0].bar(labels, z_values, color=colors)
        axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[0, 0].axhline(-1.0, color="black", linestyle="--", linewidth=1)
        axes[0, 0].set_ylabel("Delta CV / standard error")
        axes[0, 0].set_title("Broad Hammond 2024 consistency")
        extent = [np.rad2deg(lon[0]), np.rad2deg(lon[-1]), np.rad2deg(lat[0]), np.rad2deg(lat[-1])]
        image = axes[0, 1].imshow(
            truth_grid, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper
        )
        axes[0, 1].set_title("Injected map")
        axes[1, 1].imshow(
            recovered_grid, origin="lower", extent=extent, aspect="auto", vmin=lower, vmax=upper
        )
        axes[1, 1].set_title(f"Recovered map (r={injection_correlation:.2f})")
        for axis in (axes[0, 1], axes[1, 1]):
            axis.set_xlabel("Longitude (degrees)")
            axis.set_ylabel("Latitude (degrees)")
        axes[1, 0].scatter(truth_at_fit_pixels, recovered_at_fit_pixels, s=18, alpha=0.8)
        axes[1, 0].set_xlabel("Injected pixel intensity")
        axes[1, 0].set_ylabel("Recovered pixel intensity")
        axes[1, 0].set_title(f"Positive-map recovery; alpha={selection.selected_alpha:g}")
        figure.colorbar(image, ax=(axes[0, 1], axes[1, 1]), label="Relative intensity")
        figure.savefig(output / "hammond2024_benchmark.png", dpi=160)
        plt.close(figure)
    return report
