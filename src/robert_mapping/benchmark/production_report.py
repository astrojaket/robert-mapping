"""Create complete plot and summary products from a sampled map fit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from robert_mapping.config import MappingConfig
from robert_mapping.data import load_light_curve
from robert_mapping.physics import render_map

from .diagnostic_plots import (
    plot_brightness_map,
    plot_longitude_profile,
    plot_map_peak_posterior,
    plot_north_south_asymmetry,
    plot_white_light_curve,
)
from .temperature import BandpassTemperatureConverter, blackbody_stellar_radiance


_POLE_PEAK_FRACTION_THRESHOLD = 0.20
# A rendered contrast is dimensionless.  This is deliberately much smaller
# than any useful posterior contrast, so that numerical round-off at zero is
# accepted while genuinely negative map regions are rejected.
_POSITIVITY_CONDITIONING_TOLERANCE = 1.0e-12
# Percentiles and map peaks are not useful when only a handful of posterior
# draws survive the conditioning.  This threshold affects reporting only; it
# never changes the sampler or the original posterior products.
_MIN_CONDITIONED_DRAWS = 100


def _temperature_converter(config: MappingConfig) -> tuple[BandpassTemperatureConverter, str]:
    """Return the benchmark bandpass and its plain-language assumption."""

    name = config.project.name.lower()
    radius_ratio = float(config.system.radius_ratio)
    if "wasp178" in name:
        phoenix = Path(os.environ.get("ROBERT_MAPPING_WASP178_PHOENIX", ""))
        spectrum_path = Path(os.environ.get("ROBERT_MAPPING_WASP178_SPECTRUM", ""))
        if phoenix.is_file() and spectrum_path.is_file():
            with np.load(phoenix, allow_pickle=False) as archive:
                wavelength = np.asarray(archive["wavelength_micron"], dtype=float)
                stellar = np.asarray(archive["radiance_W_m3_sr"], dtype=float)
            spectrum = np.genfromtxt(spectrum_path, delimiter=",", names=True)
            weights = np.asarray(spectrum["baseline_flux"], dtype=float)
            assumption = (
                "2.87-5.18 micron NRS1 band; configured PHOENIX stellar radiance"
            )
        else:
            wavelength = np.linspace(2.87, 5.18, 151)
            stellar = blackbody_stellar_radiance(
                wavelength, 9350.0, wavelength_unit="micron"
            )
            weights = np.ones_like(wavelength)
            assumption = (
                "2.87-5.18 micron NRS1 band; portable 9350 K blackbody fallback"
            )
        return (
            BandpassTemperatureConverter(
                wavelength, stellar, weights, radius_ratio, wavelength_unit="micron"
            ),
            assumption,
        )
    if "hatp32" in name:
        wavelength = np.linspace(1.1, 1.7, 121)
        stellar = blackbody_stellar_radiance(
            wavelength, 6001.0, wavelength_unit="micron"
        )
        return (
            BandpassTemperatureConverter(
                wavelength,
                stellar,
                np.ones_like(wavelength),
                radius_ratio,
                wavelength_unit="micron",
            ),
            "assumed 1.1-1.7 micron WFC3-like band; 6001 K blackbody star",
        )
    wavelength = np.linspace(5.25, 11.75, 151)
    stellar = blackbody_stellar_radiance(
        wavelength, 4500.0, wavelength_unit="micron"
    )
    return (
        BandpassTemperatureConverter(
            wavelength,
            stellar,
            np.ones_like(wavelength),
            radius_ratio,
            wavelength_unit="micron",
        ),
        "5.25-11.75 micron MIRI/LRS band; 4500 K blackbody star",
    )


def _save_trace_plot(
    harmonic_by_chain: np.ndarray, output: Path, color: str
) -> None:
    import matplotlib.pyplot as plt

    count = min(9, harmonic_by_chain.shape[-1])
    figure, axes = plt.subplots(3, 3, figsize=(10.5, 7.5), constrained_layout=True)
    for coefficient, axis in enumerate(axes.ravel()):
        if coefficient >= count:
            axis.axis("off")
            continue
        for chain in range(harmonic_by_chain.shape[0]):
            axis.plot(
                harmonic_by_chain[chain, :, coefficient],
                color=color if chain == 0 else "#4b286d",
                alpha=0.7,
                linewidth=0.55,
                label=f"Chain {chain + 1}",
            )
        axis.set_title(f"Harmonic coefficient {coefficient}")
        axis.set_xlabel("Draw")
    axes[0, 0].legend(frameon=False)
    figure.savefig(output / "posterior_trace.png", dpi=220)
    figure.savefig(output / "posterior_trace.pdf")
    plt.close(figure)


def _save_offset_histogram(
    offsets: np.ndarray,
    output: Path,
    color: str,
    *,
    stem: str = "hotspot_longitude",
    title: str = "Conditional hotspot longitude",
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    axis.hist(offsets, bins=np.arange(-91.0, 92.0, 4.0), color=color, alpha=0.85)
    axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axis.set(xlabel="Hotspot longitude (degrees east)", ylabel="Posterior draws")
    axis.set_title(title)
    figure.savefig(output / f"{stem}.png", dpi=250)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def _condition_positive_rendered_maps(
    contrast: np.ndarray,
    *,
    tolerance: float = _POSITIVITY_CONDITIONING_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select posterior maps that are non-negative on the full display grid.

    This is a reporting-only posterior rejection step.  It does not alter the
    fit, likelihood, sampler, evidence, or the unconstrained posterior
    products.  ``tolerance`` is in the same dimensionless contrast units as
    ``contrast`` and allows tiny negative values caused by floating-point
    round-off.

    Returns
    -------
    selected, accepted, minimum
        ``selected`` contains the accepted maps, ``accepted`` is a boolean
        mask over the input draws, and ``minimum`` contains each draw's
        minimum full-grid contrast.
    """

    maps = np.asarray(contrast, dtype=float)
    if maps.ndim != 3:
        raise ValueError("contrast must have shape (draw, latitude, longitude)")
    if maps.shape[0] == 0 or maps.shape[1] == 0 or maps.shape[2] == 0:
        raise ValueError("contrast must contain at least one draw and one grid cell")
    if not np.all(np.isfinite(maps)):
        raise ValueError("contrast must contain only finite values")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    minimum = np.min(maps, axis=(-2, -1))
    accepted = minimum >= -float(tolerance)
    return np.asarray(maps[accepted], dtype=float), accepted, np.asarray(minimum, dtype=float)


def _save_systematics_plot(
    time_hours: np.ndarray,
    nuisance_samples: np.ndarray,
    coefficient_samples: np.ndarray,
    coefficient_names: list[str],
    output: Path,
    color: str,
) -> None:
    """Plot the inferred fractional/additive nuisance signal and coefficients."""

    import matplotlib.pyplot as plt

    q16, median, q84 = np.quantile(nuisance_samples, [0.16, 0.50, 0.84], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), constrained_layout=True)
    axes[0].fill_between(time_hours, q16 * 1.0e6, q84 * 1.0e6, color=color, alpha=0.25)
    axes[0].plot(time_hours, median * 1.0e6, color=color, linewidth=1.4)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Time from median epoch (hours)", ylabel="Nuisance model (ppm)")
    axes[0].set_title("Jointly fitted light-curve systematics")
    positions = np.arange(coefficient_samples.shape[1])
    coefficient_q16, coefficient_median, coefficient_q84 = np.quantile(
        coefficient_samples, [0.16, 0.50, 0.84], axis=0
    )
    axes[1].errorbar(
        positions,
        coefficient_median,
        yerr=(coefficient_median - coefficient_q16, coefficient_q84 - coefficient_median),
        color=color,
        marker="o",
        linestyle="none",
        capsize=3,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, coefficient_names, rotation=25, ha="right")
    axes[1].set_ylabel("Coefficient")
    figure.savefig(output / "systematics_model.png", dpi=220)
    figure.savefig(output / "systematics_model.pdf")
    plt.close(figure)


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    """Return the standard 16th, 50th, and 84th percentile summary."""

    q16, median, q84 = np.quantile(np.asarray(values, dtype=float), [0.16, 0.50, 0.84])
    return {"q16": float(q16), "median": float(median), "q84": float(q84)}


def _map_information_diagnostics(
    contrast: np.ndarray,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
) -> dict[str, Any]:
    """Calculate two-dimensional peak and north--south information metrics.

    The latitude-averaged longitude profile is calculated separately in
    :func:`make_production_report` and remains the primary hotspot result.
    These diagnostics describe the additional, usually weaker, information in
    the full two-dimensional map.  Peak searches are restricted to the
    dayside (longitude -90 to +90 degrees), matching the Hammond convention.
    """

    maps = np.asarray(contrast, dtype=float)
    longitude = np.asarray(longitude_deg, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    if maps.ndim != 3:
        raise ValueError("contrast must have shape (draw, latitude, longitude)")
    if maps.shape[1:] != (latitude.size, longitude.size):
        raise ValueError("contrast grid shape does not match longitude and latitude")
    if not np.all(np.isfinite(maps)):
        raise ValueError("contrast must contain only finite values")
    if not np.all(np.isfinite(longitude)) or not np.all(np.isfinite(latitude)):
        raise ValueError("longitude and latitude must contain only finite values")
    if latitude.size < 2 or np.any(np.diff(latitude) <= 0.0):
        raise ValueError("latitude must contain at least two strictly increasing values")
    dayside = (longitude >= -90.0) & (longitude <= 90.0)
    north = latitude > 0.0
    south = latitude < 0.0
    if not np.any(dayside) or not np.any(north) or not np.any(south):
        raise ValueError("map grid must include dayside, northern, and southern points")

    dayside_maps = maps[:, :, dayside]
    flat_index = np.argmax(dayside_maps.reshape(maps.shape[0], -1), axis=1)
    peak_latitude_index, peak_longitude_index = np.unravel_index(
        flat_index, (latitude.size, int(np.count_nonzero(dayside)))
    )
    peak_longitude = longitude[dayside][peak_longitude_index]
    peak_latitude = latitude[peak_latitude_index]
    peak_contrast = dayside_maps.reshape(maps.shape[0], -1)[
        np.arange(maps.shape[0]), flat_index
    ]
    latitude_grid_step = float(np.median(np.diff(latitude)))
    pole_peak_fraction = float(
        np.mean(np.abs(np.abs(peak_latitude) - 90.0) <= latitude_grid_step + 1.0e-12)
    )

    # The cosine latitude factor supplies the equal-area weight for this
    # regular latitude--longitude display grid.  Longitude cells are uniform.
    latitude_weights = np.cos(np.deg2rad(latitude))
    north_mean = np.sum(
        np.sum(maps[:, north][:, :, dayside], axis=2) * latitude_weights[north][None, :],
        axis=1,
    ) / (np.count_nonzero(dayside) * np.sum(latitude_weights[north]))
    south_mean = np.sum(
        np.sum(maps[:, south][:, :, dayside], axis=2) * latitude_weights[south][None, :],
        axis=1,
    ) / (np.count_nonzero(dayside) * np.sum(latitude_weights[south]))
    denominator = north_mean + south_mean
    asymmetry = np.full(denominator.shape, np.nan, dtype=float)
    valid = np.abs(denominator) > np.finfo(float).eps
    asymmetry[valid] = (north_mean[valid] - south_mean[valid]) / denominator[valid]
    if not np.all(np.isfinite(asymmetry)):
        raise ValueError("north--south asymmetry is undefined for some posterior draws")

    peak_latitude_summary = _quantile_summary(peak_latitude)
    asymmetry_summary = _quantile_summary(asymmetry)
    latitude_width = peak_latitude_summary["q84"] - peak_latitude_summary["q16"]
    north_peak_fraction = float(np.mean(peak_latitude > 0.0))
    south_peak_fraction = float(np.mean(peak_latitude < 0.0))
    asymmetry_interval_contains_zero = bool(
        asymmetry_summary["q16"] <= 0.0 <= asymmetry_summary["q84"]
    )
    reasons: list[str] = []
    if latitude_width >= 30.0:
        reasons.append(f"the 68% peak-latitude interval spans {latitude_width:.0f} degrees")
    if asymmetry_interval_contains_zero:
        reasons.append("the north--south asymmetry interval includes zero")
    if 0.20 < north_peak_fraction < 0.80:
        reasons.append("posterior map maxima occur in both hemispheres")
    if pole_peak_fraction >= _POLE_PEAK_FRACTION_THRESHOLD:
        status = "prior_dominated"
        warning = (
            "Warning: latitude is boundary-pinned: "
            f"{pole_peak_fraction:.0%} of two-dimensional peaks lie within one "
            "latitude grid step of a pole. This is not evidence for latitude."
        )
    elif latitude_width >= 120.0 and asymmetry_interval_contains_zero:
        status = "prior_dominated"
        warning = (
            "Warning: latitude is broad and may be prior-dominated. "
            "Treat the two-dimensional peak latitude as weak evidence."
        )
    elif reasons:
        status = "weak"
        warning = (
            "Warning: latitude structure is weakly constrained; "
            + "; ".join(reasons)
            + "."
        )
    else:
        status = "informative"
        warning = None

    return {
        "peak_longitude_degrees_east": peak_longitude,
        "peak_latitude_degrees": peak_latitude,
        "peak_contrast": peak_contrast,
        "north_mean_contrast": north_mean,
        "south_mean_contrast": south_mean,
        "north_south_asymmetry": asymmetry,
        "latitude_information_status": status,
        "latitude_information_warning": warning,
        "pole_peak_fraction": pole_peak_fraction,
        "north_peak_fraction": north_peak_fraction,
        "south_peak_fraction": south_peak_fraction,
        "asymmetry_interval_contains_zero": asymmetry_interval_contains_zero,
    }


def make_production_report(config: MappingConfig) -> dict[str, Any]:
    """Create white-light, map, temperature, profile, and trace products."""

    output = Path(config.output.directory).expanduser().resolve()
    summary_path = output / "fit_summary.json"
    samples_path = output / "samples.npz"
    if not summary_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError("Run the configured fit before making its report.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    curve = load_light_curve(config)
    model = np.load(output / "model_flux.npy", allow_pickle=False)
    residual = np.load(output / "residuals.npy", allow_pickle=False)
    with np.load(samples_path, allow_pickle=False) as archive:
        harmonic = np.asarray(archive["harmonic_coefficients"], dtype=float)
        harmonic_by_chain = np.asarray(
            archive["harmonic_coefficients_by_chain"], dtype=float
        )
        nuisance_samples = (
            np.asarray(archive["systematics_model"], dtype=float)
            if "systematics_model" in archive
            else None
        )
        nuisance_coefficient_samples = (
            np.asarray(archive["systematics_coefficients"], dtype=float)
            if "systematics_coefficients" in archive
            else None
        )

    color = config.output.best_fit_color
    target = config.project.name
    time_hours = (np.asarray(curve.time) - np.median(curve.time)) * 24.0
    plot_white_light_curve(
        time_hours,
        curve.flux,
        model,
        curve.flux_err,
        residual,
        metadata={
            "title": f"{target}: white-light curve",
            "time_label": "Time from median epoch (hours)",
            "residual_unit": "ppm",
        },
        output=output / "white_light_curve",
        color=color,
    )

    longitude, latitude, rendered = render_map(harmonic, nlon=121, nlat=61)
    longitude_deg = np.rad2deg(np.asarray(longitude, dtype=float))
    latitude_deg = np.rad2deg(np.asarray(latitude, dtype=float))
    # A uniform coefficient equal to the eclipse depth renders as depth/pi.
    # Multiplying by pi gives the local disk-equivalent contrast convention
    # used by the Hammond temperature conversion.
    contrast = np.pi * np.asarray(rendered, dtype=float)
    rendered_minimum = np.min(contrast, axis=(-2, -1))
    rendered_nonnegative_fraction = float(np.mean(rendered_minimum >= -1.0e-12))
    rendered_minimum_q16, rendered_minimum_median, rendered_minimum_q84 = (
        np.quantile(rendered_minimum, [0.16, 0.50, 0.84])
    )
    conditioned_contrast, conditioned_mask, conditioned_minimum = (
        _condition_positive_rendered_maps(
            contrast,
            tolerance=_POSITIVITY_CONDITIONING_TOLERANCE,
        )
    )
    conditioned_count = int(conditioned_contrast.shape[0])
    total_draws = int(contrast.shape[0])
    conditioned_fraction = float(conditioned_count / total_draws)
    if conditioned_count == 0:
        conditioned_status = "no_accepted_draws"
    elif conditioned_count < _MIN_CONDITIONED_DRAWS:
        conditioned_status = "insufficient_draws"
    else:
        conditioned_status = "available"
    map_q16, map_median, map_q84 = np.quantile(contrast, [0.16, 0.50, 0.84], axis=0)
    for label, values, title in (
        ("brightness_map_q16", map_q16, "Brightness map: 16th percentile"),
        ("brightness_map_median", map_median, "Brightness map: posterior median"),
        ("brightness_map_q84", map_q84, "Brightness map: 84th percentile"),
    ):
        plot_brightness_map(
            longitude_deg,
            latitude_deg,
            values * 1.0e6,
            metadata={"title": f"{target}: {title}", "colorbar_label": "Contrast (ppm)"},
            output=output / label,
            color=color,
        )

    map_information = _map_information_diagnostics(contrast, longitude_deg, latitude_deg)
    peak_longitude = np.asarray(map_information["peak_longitude_degrees_east"], dtype=float)
    peak_latitude = np.asarray(map_information["peak_latitude_degrees"], dtype=float)
    peak_contrast = np.asarray(map_information["peak_contrast"], dtype=float)
    north_mean_contrast = np.asarray(map_information["north_mean_contrast"], dtype=float)
    south_mean_contrast = np.asarray(map_information["south_mean_contrast"], dtype=float)
    north_south_asymmetry = np.asarray(
        map_information["north_south_asymmetry"], dtype=float
    )
    latitude_warning = map_information["latitude_information_warning"]
    plot_map_peak_posterior(
        peak_longitude,
        peak_latitude,
        metadata={
            "title": f"{target}: two-dimensional map-peak posterior",
            "warning": latitude_warning,
        },
        output=output / "map_peak_posterior",
        color=color,
    )
    plot_north_south_asymmetry(
        north_south_asymmetry,
        metadata={
            "title": f"{target}: north--south map asymmetry",
            "warning": latitude_warning,
        },
        output=output / "north_south_asymmetry",
        color=color,
    )

    latitude_weights = np.cos(np.deg2rad(latitude_deg))
    profiles = np.average(contrast, axis=1, weights=latitude_weights)
    profile_q16, profile_median, profile_q84 = np.quantile(
        profiles, [0.16, 0.50, 0.84], axis=0
    )
    dayside = (longitude_deg >= -90.0) & (longitude_deg <= 90.0)
    offsets = longitude_deg[dayside][
        np.argmax(profiles[:, dayside], axis=1)
    ]
    plot_longitude_profile(
        longitude_deg,
        profile_median * 1.0e6,
        profile_q16 * 1.0e6,
        profile_q84 * 1.0e6,
        metadata={
            "title": f"{target}: longitude profile",
            "profile_label_axis": "Contrast (ppm)",
        },
        output=output / "longitude_profile",
        color=color,
    )
    _save_offset_histogram(offsets, output, color)
    _save_trace_plot(harmonic_by_chain, output, color)
    if nuisance_samples is not None and nuisance_coefficient_samples is not None:
        _save_systematics_plot(
            time_hours,
            nuisance_samples,
            nuisance_coefficient_samples,
            list(summary.get("systematics_coefficient_names", [])),
            output,
            color,
        )

    converter, temperature_assumption = _temperature_converter(config)
    positive_floor = float(converter.contrast_grid[0])
    clipped_map = np.maximum(map_median, positive_floor)
    temperature_map = np.asarray(converter.temperature(clipped_map), dtype=float)
    plot_brightness_map(
        longitude_deg,
        latitude_deg,
        temperature_map,
        metadata={
            "title": f"{target}: brightness-temperature map",
            "colorbar_label": "Brightness temperature (K)",
        },
        output=output / "temperature_map",
        color=color,
    )

    conditioning_warning = (
        "This is a posterior-rejection/conditioning summary only. It does not "
        "rerun sampling, impose positivity in the likelihood, or provide "
        "independent mapping evidence. The original posterior report remains "
        "the primary result."
    )
    conditioned_report: dict[str, Any] = {
        "status": conditioned_status,
        "accepted_draws": conditioned_count,
        "total_draws": total_draws,
        "accepted_fraction": conditioned_fraction,
        "tolerance_contrast": _POSITIVITY_CONDITIONING_TOLERANCE,
        "minimum_draws_for_summary": _MIN_CONDITIONED_DRAWS,
        "warning": conditioning_warning,
        "hotspot_longitude_degrees_east": None,
        "map": None,
        "temperature": None,
        "files": {},
    }
    if conditioned_status == "available":
        conditioned_map_q16, conditioned_map_median, conditioned_map_q84 = np.quantile(
            conditioned_contrast, [0.16, 0.50, 0.84], axis=0
        )
        conditioned_info = _map_information_diagnostics(
            conditioned_contrast, longitude_deg, latitude_deg
        )
        conditioned_peak_longitude = np.asarray(
            conditioned_info["peak_longitude_degrees_east"], dtype=float
        )
        conditioned_peak_latitude = np.asarray(
            conditioned_info["peak_latitude_degrees"], dtype=float
        )
        conditioned_peak_contrast = np.asarray(
            conditioned_info["peak_contrast"], dtype=float
        )
        conditioned_north_mean = np.asarray(
            conditioned_info["north_mean_contrast"], dtype=float
        )
        conditioned_south_mean = np.asarray(
            conditioned_info["south_mean_contrast"], dtype=float
        )
        conditioned_asymmetry = np.asarray(
            conditioned_info["north_south_asymmetry"], dtype=float
        )
        conditioned_profiles = np.average(
            conditioned_contrast, axis=1, weights=latitude_weights
        )
        conditioned_profile_q16, conditioned_profile_median, conditioned_profile_q84 = (
            np.quantile(conditioned_profiles, [0.16, 0.50, 0.84], axis=0)
        )
        conditioned_offsets = longitude_deg[dayside][
            np.argmax(conditioned_profiles[:, dayside], axis=1)
        ]
        conditioned_temperature_draws = np.asarray(
            converter.temperature(
                np.maximum(conditioned_contrast, positive_floor)
            ),
            dtype=float,
        )
        conditioned_temperature_q16, conditioned_temperature_median, (
            conditioned_temperature_q84
        ) = np.quantile(conditioned_temperature_draws, [0.16, 0.50, 0.84], axis=0)
        conditioned_peak_temperature = np.max(
            conditioned_temperature_draws, axis=(-2, -1)
        )
        conditioned_minimum_temperature = np.min(
            conditioned_temperature_draws, axis=(-2, -1)
        )

        for label, values, title in (
            (
                "brightness_map_positive_conditioned_q16",
                conditioned_map_q16,
                "Brightness map: 16th percentile (positivity-conditioned)",
            ),
            (
                "brightness_map_positive_conditioned",
                conditioned_map_median,
                "Brightness map: posterior median (positivity-conditioned)",
            ),
            (
                "brightness_map_positive_conditioned_q84",
                conditioned_map_q84,
                "Brightness map: 84th percentile (positivity-conditioned)",
            ),
        ):
            plot_brightness_map(
                longitude_deg,
                latitude_deg,
                values * 1.0e6,
                metadata={
                    "title": f"{target}: {title}",
                    "colorbar_label": "Contrast (ppm)",
                    "warning": conditioning_warning,
                },
                output=output / label,
                color=color,
            )
        plot_brightness_map(
            longitude_deg,
            latitude_deg,
            conditioned_temperature_median,
            metadata={
                "title": f"{target}: brightness-temperature map (positivity-conditioned)",
                "colorbar_label": "Brightness temperature (K)",
                "warning": conditioning_warning,
            },
            output=output / "temperature_map_positive_conditioned",
            color=color,
        )
        plot_longitude_profile(
            longitude_deg,
            conditioned_profile_median * 1.0e6,
            conditioned_profile_q16 * 1.0e6,
            conditioned_profile_q84 * 1.0e6,
            metadata={
                "title": f"{target}: longitude profile (positivity-conditioned)",
                "profile_label_axis": "Contrast (ppm)",
                "warning": conditioning_warning,
            },
            output=output / "longitude_profile_positive_conditioned",
            color=color,
        )
        _save_offset_histogram(
            conditioned_offsets,
            output,
            color,
            stem="hotspot_longitude_positive_conditioned",
            title=f"{target}: hotspot longitude (positivity-conditioned)",
        )
        plot_map_peak_posterior(
            conditioned_peak_longitude,
            conditioned_peak_latitude,
            metadata={
                "title": (
                    f"{target}: map peak "
                    "(positivity-conditioned diagnostic)"
                ),
            },
            output=output / "map_peak_posterior_positive_conditioned",
            color=color,
        )
        plot_north_south_asymmetry(
            conditioned_asymmetry,
            metadata={
                "title": (
                    f"{target}: north--south asymmetry "
                    "(positivity-conditioned diagnostic)"
                ),
            },
            output=output / "north_south_asymmetry_positive_conditioned",
            color=color,
        )
        np.savez_compressed(
            output / "posterior_maps_positive_conditioned.npz",
            longitude_degrees=longitude_deg,
            latitude_degrees=latitude_deg,
            accepted_draw_indices=np.flatnonzero(conditioned_mask),
            minimum_contrast=conditioned_minimum[conditioned_mask],
            contrast_q16=conditioned_map_q16,
            contrast_median=conditioned_map_median,
            contrast_q84=conditioned_map_q84,
            temperature_q16_kelvin=conditioned_temperature_q16,
            temperature_median_kelvin=conditioned_temperature_median,
            temperature_q84_kelvin=conditioned_temperature_q84,
            longitude_profile_q16=conditioned_profile_q16,
            longitude_profile_median=conditioned_profile_median,
            longitude_profile_q84=conditioned_profile_q84,
            hotspot_longitude_degrees=conditioned_offsets,
            map_peak_longitude_degrees_east=conditioned_peak_longitude,
            map_peak_latitude_degrees=conditioned_peak_latitude,
            map_peak_contrast=conditioned_peak_contrast,
            north_mean_contrast=conditioned_north_mean,
            south_mean_contrast=conditioned_south_mean,
            north_south_asymmetry=conditioned_asymmetry,
        )
        conditioned_report.update(
            {
                "hotspot_longitude_degrees_east": _quantile_summary(
                    conditioned_offsets
                ),
                "map": {
                    "contrast_minimum_ppm": _quantile_summary(
                        conditioned_minimum[conditioned_mask] * 1.0e6
                    ),
                    "map_peak_2d": {
                        "longitude_degrees_east": _quantile_summary(
                            conditioned_peak_longitude
                        ),
                        "latitude_degrees": _quantile_summary(conditioned_peak_latitude),
                        "contrast_ppm": {
                            key: value * 1.0e6
                            for key, value in _quantile_summary(
                                conditioned_peak_contrast
                            ).items()
                        },
                    },
                    "latitude_information_status": conditioned_info[
                        "latitude_information_status"
                    ],
                },
                "temperature": {
                    "assumption": temperature_assumption,
                    "median_map_peak_kelvin": float(
                        np.max(conditioned_temperature_median)
                    ),
                    "map_peak_kelvin": _quantile_summary(conditioned_peak_temperature),
                    "map_minimum_kelvin": _quantile_summary(
                        conditioned_minimum_temperature
                    ),
                },
                "files": {
                    "brightness_q16": "brightness_map_positive_conditioned_q16.png",
                    "brightness_median": "brightness_map_positive_conditioned.png",
                    "brightness_q84": "brightness_map_positive_conditioned_q84.png",
                    "temperature": "temperature_map_positive_conditioned.png",
                    "longitude_profile": "longitude_profile_positive_conditioned.png",
                    "hotspot_posterior": "hotspot_longitude_positive_conditioned.png",
                    "map_peak_posterior": "map_peak_posterior_positive_conditioned.png",
                    "north_south_asymmetry": "north_south_asymmetry_positive_conditioned.png",
                    "map_arrays": "posterior_maps_positive_conditioned.npz",
                },
            }
        )
    nonpositive_fraction = float(np.mean(map_median <= 0.0))
    hotspot_q16, hotspot_median, hotspot_q84 = np.quantile(offsets, [0.16, 0.50, 0.84])
    report = {
        "status": "complete",
        "target": target,
        "posterior_draws": int(harmonic.shape[0]),
        "chains": int(harmonic_by_chain.shape[0]),
        "residual_rms_ppm": float(np.sqrt(np.mean(residual**2)) * 1.0e6),
        "divergences": int(summary["divergences"]),
        "maximum_rhat": summary["maximum_rhat"],
        "minimum_effective_sample_size": summary["minimum_effective_sample_size"],
        "systematics": {
            "mode": summary.get("systematics", {}).get("mode", "corrected"),
            "coefficient_names": summary.get("systematics_coefficient_names", []),
            "coefficient_mean": summary.get("systematics_coefficient_mean", []),
            "maximum_rhat": summary.get("maximum_systematics_rhat"),
            "minimum_effective_sample_size": summary.get(
                "minimum_systematics_effective_sample_size"
            ),
        },
        "hotspot_longitude_degrees_east": {
            "q16": float(hotspot_q16),
            "median": float(hotspot_median),
            "q84": float(hotspot_q84),
        },
        "map_peak_2d": {
            "longitude_degrees_east": _quantile_summary(peak_longitude),
            "latitude_degrees": _quantile_summary(peak_latitude),
            "contrast_ppm": {
                key: value * 1.0e6
                for key, value in _quantile_summary(peak_contrast).items()
            },
        },
        "latitude_information": {
            "status": map_information["latitude_information_status"],
            "warning": latitude_warning,
            "peak_latitude_68_percent_width_degrees": float(
                np.quantile(peak_latitude, 0.84) - np.quantile(peak_latitude, 0.16)
            ),
            "pole_peak_fraction": map_information["pole_peak_fraction"],
            "north_peak_fraction": map_information["north_peak_fraction"],
            "south_peak_fraction": map_information["south_peak_fraction"],
            "north_south_asymmetry": _quantile_summary(north_south_asymmetry),
            "north_south_asymmetry_interval_contains_zero": map_information[
                "asymmetry_interval_contains_zero"
            ],
            "north_mean_contrast_ppm": {
                key: value * 1.0e6
                for key, value in _quantile_summary(north_mean_contrast).items()
            },
            "south_mean_contrast_ppm": {
                key: value * 1.0e6
                for key, value in _quantile_summary(south_mean_contrast).items()
            },
        },
        "temperature": {
            "assumption": temperature_assumption,
            "median_map_peak_kelvin": float(np.max(temperature_map)),
            "median_map_nonpositive_fraction_before_clipping": nonpositive_fraction,
            "posterior_draw_nonnegative_fraction": rendered_nonnegative_fraction,
            "posterior_draw_minimum_contrast_q16_q50_q84": [
                float(rendered_minimum_q16),
                float(rendered_minimum_median),
                float(rendered_minimum_q84),
            ],
            "clipping_note": (
                "Only non-positive median harmonic pixels are clipped to the "
                "100 K converter floor for plotting. Direct harmonic coefficients "
                "do not impose positivity; use the dense-grid diagnostic."
                if summary.get("parameterization") == "direct_harmonics"
                else "Only non-positive median harmonic pixels are clipped to the "
                "100 K converter floor for plotting. Positive fitted anchors do not "
                "guarantee positivity between anchors."
            ),
        },
        "positivity_conditioned": conditioned_report,
        "files": {
            "white_light": "white_light_curve.png",
            "brightness_median": "brightness_map_median.png",
            "temperature": "temperature_map.png",
            "longitude_profile": "longitude_profile.png",
            "hotspot_posterior": "hotspot_longitude.png",
            "map_peak_posterior": "map_peak_posterior.png",
            "north_south_asymmetry": "north_south_asymmetry.png",
            "trace": "posterior_trace.png",
            "map_arrays": "posterior_maps.npz",
            **(
                {"systematics": "systematics_model.png"}
                if nuisance_samples is not None
                and nuisance_coefficient_samples is not None
                else {}
            ),
        },
    }
    np.savez_compressed(
        output / "posterior_maps.npz",
        longitude_degrees=longitude_deg,
        latitude_degrees=latitude_deg,
        contrast_q16=map_q16,
        contrast_median=map_median,
        contrast_q84=map_q84,
        temperature_median_kelvin=temperature_map,
        longitude_profile_q16=profile_q16,
        longitude_profile_median=profile_median,
        longitude_profile_q84=profile_q84,
        hotspot_longitude_degrees=offsets,
        map_peak_longitude_degrees_east=peak_longitude,
        map_peak_latitude_degrees=peak_latitude,
        map_peak_contrast=peak_contrast,
        north_mean_contrast=north_mean_contrast,
        south_mean_contrast=south_mean_contrast,
        north_south_asymmetry=north_south_asymmetry,
    )
    (output / "production_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["make_production_report"]
