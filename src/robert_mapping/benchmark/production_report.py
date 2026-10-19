"""Create complete plot and summary products from a sampled map fit."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from robert_mapping.config import MappingConfig
from robert_mapping.data import load_light_curve
from robert_mapping.physics import render_map
from robert_mapping.physics.longitude import evaluate_longitude_map

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
_HAMMOND_HOTSPOT_LONGITUDE_DEGREES = 7.75
_HAMMOND_HOTSPOT_SIGMA_DEGREES = 0.36
_WASP121_NRS1_HOTSPOT_DEGREES = 3.36
_WASP121_NRS1_HOTSPOT_SIGMA_DEGREES = 0.11
_WASP121_NRS2_HOTSPOT_DEGREES = 2.66
_WASP121_NRS2_HOTSPOT_SIGMA_DEGREES = 0.12
_WASP121_MIRI_HOTSPOT_DEGREES = 4.8
_WASP121_MIRI_HOTSPOT_LOWER_DEGREES = 2.0
_WASP121_MIRI_HOTSPOT_UPPER_DEGREES = 7.5
_SIGMA_INTERVAL_QUANTILES = {
    "1sigma": (0.158655253931457, 0.841344746068543),
    "2sigma": (0.022750131948179, 0.977249868051821),
}


def _wasp18_input_path(
    config: MappingConfig,
    filename: str,
    *,
    environment_variable: str,
) -> Path | None:
    """Return one local WASP-18 stellar or instrument input file."""

    configured = os.environ.get(environment_variable)
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(
        config.base_dir.parent
        / "literature_data"
        / "WASP-18b"
        / "JWST-NIRISS-SOSS"
        / "source"
        / "WASP-18b 3D Mapping Archive"
        / "theresa"
        / "inputs"
        / filename
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _latitude_profile_weights(target: str, latitude_degrees: np.ndarray) -> np.ndarray:
    """Return the latitude weighting used for a published profile comparison."""

    cosine = np.cos(np.deg2rad(np.asarray(latitude_degrees, dtype=float)))
    name = target.lower()
    if "wasp18" in name or ("wasp121" in name and "miri" in name):
        # Coulombe et al. (2025) report the longitudinal profile with cos^2
        # latitude weighting. Match that definition for the validation.
        return cosine**2
    return cosine


def _temperature_converter(config: MappingConfig) -> tuple[BandpassTemperatureConverter, str]:
    """Return the benchmark bandpass and its plain-language assumption."""

    name = config.project.name.lower()
    radius_ratio = float(config.system.radius_ratio)
    if "wasp121" in name:
        if "nrs1" in name:
            lower, upper, channel = 2.70, 3.72, "NIRSpec NRS1"
        elif "nrs2" in name:
            lower, upper, channel = 3.82, 5.15, "NIRSpec NRS2"
        elif "miri" in name:
            lower, upper, channel = 5.00, 12.00, "MIRI LRS"
        elif "tess" in name:
            lower, upper, channel = 0.60, 1.00, "TESS"
        else:
            lower, upper, channel = 2.70, 5.15, "NIRSpec G395H"
        wavelength = np.linspace(lower, upper, 501)
        stellar = blackbody_stellar_radiance(
            wavelength, 6460.0, wavelength_unit="micron"
        )
        return (
            BandpassTemperatureConverter(
                wavelength,
                stellar,
                np.ones_like(wavelength),
                radius_ratio,
                wavelength_unit="micron",
            ),
            (
                f"{lower:.2f}-{upper:.2f} micron {channel} band; "
                "portable 6460 K blackbody star; top-hat response"
            ),
        )
    if "wasp18" in name:
        if config.data.file is None:
            raise ValueError("WASP-18 temperature conversion needs a combined NPZ file")
        with np.load(config.data.file, allow_pickle=False) as archive:
            centre = float(np.asarray(archive["wavelength_micron"]))
            width = float(np.asarray(archive["wavelength_width_micron"]))
        lower = centre - 0.5 * width
        upper = centre + 0.5 * width
        wavelength = np.linspace(lower, upper, 1_000)
        phoenix_path = _wasp18_input_path(
            config,
            "phoenix-spectrum.txt",
            environment_variable="ROBERT_MAPPING_WASP18_PHOENIX",
        )
        throughput_path = _wasp18_input_path(
            config,
            "niriss-firstorder.txt",
            environment_variable="ROBERT_MAPPING_WASP18_THROUGHPUT",
        )
        if phoenix_path is not None:
            phoenix = np.loadtxt(phoenix_path)
            stellar = np.interp(wavelength, phoenix[:, 0], phoenix[:, 1])
            stellar_source = "published 6435 K PHOENIX stellar radiance"
        else:
            stellar = blackbody_stellar_radiance(
                wavelength, 6435.0, wavelength_unit="micron"
            )
            stellar_source = "portable 6435 K blackbody fallback"
        if throughput_path is not None:
            throughput = np.loadtxt(throughput_path)
            weights = np.interp(wavelength, throughput[:, 0], throughput[:, 1])
            response_source = "published NIRISS first-order throughput"
        else:
            weights = np.ones_like(wavelength)
            response_source = "uniform within-bin response fallback"
        assumption = (
            f"{lower:.4f}-{upper:.4f} micron NIRISS bin; {stellar_source}; "
            f"{response_source}"
        )
        return (
            BandpassTemperatureConverter(
                wavelength,
                stellar,
                weights,
                radius_ratio,
                wavelength_unit="micron",
            ),
            assumption,
        )
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
    # Hammond et al. (2024) use the 5-10.5 micron MIRI/LRS white channel and
    # a 4300 K stellar effective temperature for the brightness conversion.
    wavelength = np.linspace(5.0, 10.5, 151)
    stellar = blackbody_stellar_radiance(
        wavelength, 4300.0, wavelength_unit="micron"
    )
    return (
        BandpassTemperatureConverter(
            wavelength,
            stellar,
            np.ones_like(wavelength),
            radius_ratio,
            wavelength_unit="micron",
        ),
        "5.0-10.5 micron MIRI/LRS band; 4300 K blackbody star",
    )


def _noise_report(summary: dict[str, Any]) -> dict[str, Any]:
    """Collect configured OU/GP priors and sampled noise values."""

    posterior: dict[str, Any] = {}
    for name in (
        "ou_amplitude",
        "ou_timescale",
        "gp_amplitude",
        "gp_timescale",
        "jitter",
        "white_jitter",
    ):
        posterior[name] = {
            "mean": summary.get(f"{name}_mean"),
            "standard_deviation": summary.get(f"{name}_standard_deviation"),
        }
    return {
        "model": summary.get("noise_model", "white"),
        "ou_amplitude_prior_scale_ppm": summary.get(
            "ou_amplitude_prior_scale_ppm"
        ),
        "ou_timescale_prior_median_seconds": summary.get(
            "ou_timescale_prior_median_seconds"
        ),
        "ou_timescale_prior_sigma_ln": summary.get(
            "ou_timescale_prior_sigma_ln"
        ),
        "gp_kernel": summary.get("gp_kernel"),
        "gp_amplitude_prior_scale_ppm": summary.get(
            "gp_amplitude_prior_scale_ppm"
        ),
        "gp_timescale_prior_median_seconds": summary.get(
            "gp_timescale_prior_median_seconds"
        ),
        "gp_timescale_prior_sigma_ln": summary.get(
            "gp_timescale_prior_sigma_ln"
        ),
        "gp_max_points": summary.get("gp_max_points"),
        "jitter_prior_scale_ppm": summary.get("jitter_prior_scale_ppm"),
        "posterior": posterior,
        "maximum_rhat": summary.get("maximum_noise_rhat"),
        "minimum_effective_sample_size": summary.get(
            "minimum_noise_effective_sample_size"
        ),
    }


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


def _style_longitude_figure() -> Any:
    """Apply the local WASP-15b style and return its axis formatter."""

    import matplotlib

    style_path = Path.home() / ".codex" / "plotting" / "wasp15b_plot_style.py"
    if style_path.is_file():
        spec = importlib.util.spec_from_file_location(
            "_robert_mapping_wasp15b_plot_style", style_path
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.apply_style()
            return module.style_axis

    # Keep the report portable when the optional user-level helper is absent.
    from matplotlib.ticker import AutoMinorLocator

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    def style_axis(axis: Any) -> None:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.7)
        axis.tick_params(
            which="both", direction="in", top=True, right=True, pad=2.0
        )
        axis.tick_params(which="major", length=3.0, width=0.7)
        axis.tick_params(which="minor", length=1.5, width=0.5)
        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

    return style_axis


def _save_longitude_light_curve(
    time_hours: np.ndarray,
    observed: np.ndarray,
    error: np.ndarray,
    median_model: np.ndarray,
    output: Path,
) -> None:
    """Save a WASP-15b-style light-curve and residual figure."""

    import matplotlib.pyplot as plt

    style_axis = _style_longitude_figure()
    order = np.argsort(np.asarray(time_hours, dtype=float))
    time = np.asarray(time_hours, dtype=float)
    data = np.asarray(observed, dtype=float)
    sigma = np.asarray(error, dtype=float)
    model = np.asarray(median_model, dtype=float)
    figure, axes = plt.subplots(
        2, 1, figsize=(6.7, 4.5), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (2.3, 1.0)},
    )
    axes[0].errorbar(
        time,
        data,
        yerr=sigma,
        fmt="o",
        markersize=2.0,
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        ecolor="#111111",
        elinewidth=0.45,
        capsize=0.0,
        linestyle="none",
        label="Observed data",
        zorder=3,
    )
    axes[0].plot(
        time[order],
        model[order],
        color="#6F3CC3",
        linewidth=1.2,
        label="Posterior median model",
        zorder=2,
    )
    axes[0].set_ylabel("Relative flux")
    axes[0].legend(frameon=False, loc="best")
    axes[1].axhline(0.0, color="#777777", linewidth=0.65)
    axes[1].errorbar(
        time,
        (data - model) * 1.0e6,
        yerr=sigma * 1.0e6,
        fmt="o",
        markersize=1.8,
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        ecolor="#111111",
        elinewidth=0.4,
        capsize=0.0,
        linestyle="none",
    )
    axes[1].set_ylabel("Residual (ppm)")
    axes[1].set_xlabel("Time from median epoch (hours)")
    for axis in axes:
        style_axis(axis)
    figure.savefig(output / "white_light_curve.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "white_light_curve.pdf", bbox_inches="tight")
    plt.close(figure)


def _save_longitude_map(
    longitude_degrees: np.ndarray,
    latitude_degrees: np.ndarray,
    contrast_ppm: np.ndarray,
    output: Path,
    *,
    stem: str,
    colorbar_label: str,
) -> None:
    """Save one latitude-flat map with the report's scientific style."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    style_axis = _style_longitude_figure()
    cmap = LinearSegmentedColormap.from_list(
        "robert_mapping_longitude",
        ["#20102c", "#542b73", "#6F3CC3", "#d8bde9", "#f7f3fb"],
    )
    figure, axis = plt.subplots(
        figsize=(6.6, 3.5), constrained_layout=True
    )
    image = axis.pcolormesh(
        longitude_degrees,
        latitude_degrees,
        np.asarray(contrast_ppm, dtype=float),
        shading="auto",
        cmap=cmap,
    )
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude (fixed by model)")
    axis.set_xlim(float(longitude_degrees[0]), float(longitude_degrees[-1]))
    axis.set_ylim(float(latitude_degrees[0]), float(latitude_degrees[-1]))
    style_axis(axis)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _save_longitude_profile(
    longitude_degrees: np.ndarray,
    median: np.ndarray,
    lower_1sigma: np.ndarray,
    upper_1sigma: np.ndarray,
    lower_2sigma: np.ndarray,
    upper_2sigma: np.ndarray,
    output: Path,
    *,
    stem: str = "longitude_profile",
) -> None:
    """Save the median Fourier profile and exact central 1σ and 2σ ranges."""

    import matplotlib.pyplot as plt

    style_axis = _style_longitude_figure()
    figure, axis = plt.subplots(
        figsize=(3.3, 3.3), constrained_layout=True
    )
    axis.fill_between(
        longitude_degrees,
        lower_2sigma * 1.0e6,
        upper_2sigma * 1.0e6,
        color="mediumpurple",
        alpha=0.12,
        linewidth=0.0,
        label="2σ range",
        zorder=1,
    )
    axis.fill_between(
        longitude_degrees,
        lower_1sigma * 1.0e6,
        upper_1sigma * 1.0e6,
        color="mediumpurple",
        alpha=0.25,
        linewidth=0.0,
        label="1σ range",
        zorder=1,
    )
    axis.plot(
        longitude_degrees,
        median * 1.0e6,
        color="#6F3CC3",
        linewidth=1.3,
        label="Posterior median model",
        zorder=2,
    )
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Planet-to-star contrast (ppm)")
    axis.set_xlim(-180.0, 180.0)
    axis.legend(frameon=False, loc="best")
    style_axis(axis)
    figure.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def _save_longitude_trace(
    coefficients_by_chain: np.ndarray,
    output: Path,
) -> None:
    """Save trace panels with labels for constant/cosine/sine coefficients."""

    import matplotlib.pyplot as plt

    style_axis = _style_longitude_figure()
    values = np.asarray(coefficients_by_chain, dtype=float)
    if values.ndim != 3 or values.shape[-1] % 2 != 1:
        raise ValueError(
            "longitude_coefficients_by_chain must have shape (chain, draw, 1 + 2 * degree)"
        )
    count = int(values.shape[-1])
    names = ["Constant coefficient"]
    for order in range(1, (count + 1) // 2):
        names.extend((f"Cosine coefficient, m={order}", f"Sine coefficient, m={order}"))
    ncols = min(3, count)
    nrows = int(np.ceil(count / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8, max(2.0, 1.75 * nrows)),
        squeeze=False,
        constrained_layout=True,
    )
    chain_colors = ("#6F3CC3", "#111111", "mediumpurple")
    chain_styles = ("-", "--", ":")
    for index, axis in enumerate(axes.ravel()):
        if index >= count:
            axis.set_visible(False)
            continue
        for chain in range(values.shape[0]):
            axis.plot(
                values[chain, :, index] * 1.0e6,
                color=chain_colors[chain % len(chain_colors)],
                linestyle=chain_styles[chain % len(chain_styles)],
                linewidth=0.55,
                alpha=0.85,
                label=f"Chain {chain + 1}",
            )
        axis.set_ylabel(names[index] + " (ppm)")
        axis.set_xlabel("Draw")
        style_axis(axis)
    if values.shape[0] > 1:
        axes[0, 0].legend(frameon=False, loc="best")
    figure.savefig(output / "posterior_trace.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "posterior_trace.pdf", bbox_inches="tight")
    plt.close(figure)


def _save_longitude_systematics_plot(
    time_hours: np.ndarray,
    nuisance_samples: np.ndarray,
    coefficient_samples: np.ndarray,
    coefficient_names: list[str],
    output: Path,
) -> None:
    """Save systematics diagnostics with the longitude report style."""

    import matplotlib.pyplot as plt

    style_axis = _style_longitude_figure()
    lower_1sigma, upper_1sigma = _SIGMA_INTERVAL_QUANTILES["1sigma"]
    lower_2sigma, upper_2sigma = _SIGMA_INTERVAL_QUANTILES["2sigma"]
    q2_lower, median, q2_upper = np.quantile(
        nuisance_samples, [lower_2sigma, 0.50, upper_2sigma], axis=0
    )
    q1_lower, _, q1_upper = np.quantile(
        nuisance_samples, [lower_1sigma, 0.50, upper_1sigma], axis=0
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(6.7, 4.5), constrained_layout=True
    )
    axes[0].fill_between(
        time_hours,
        q2_lower * 1.0e6,
        q2_upper * 1.0e6,
        color="mediumpurple",
        alpha=0.12,
        linewidth=0.0,
        label="2σ range",
    )
    axes[0].fill_between(
        time_hours,
        q1_lower * 1.0e6,
        q1_upper * 1.0e6,
        color="mediumpurple",
        alpha=0.25,
        linewidth=0.0,
        label="1σ range",
    )
    axes[0].plot(
        time_hours,
        median * 1.0e6,
        color="#6F3CC3",
        linewidth=1.2,
        label="Posterior median model",
    )
    axes[0].axhline(0.0, color="#111111", linewidth=0.65)
    axes[0].set_ylabel("Systematics (ppm)")
    axes[0].set_xlabel("Time from median epoch (hours)")
    axes[0].legend(frameon=False, loc="best")
    positions = np.arange(coefficient_samples.shape[1])
    coefficient_lower, coefficient_median, coefficient_upper = np.quantile(
        coefficient_samples,
        [lower_1sigma, 0.50, upper_1sigma],
        axis=0,
    )
    axes[1].errorbar(
        positions,
        coefficient_median,
        yerr=(
            coefficient_median - coefficient_lower,
            coefficient_upper - coefficient_median,
        ),
        color="#111111",
        marker="o",
        markerfacecolor="#111111",
        markeredgecolor="#111111",
        linestyle="none",
        capsize=2,
        linewidth=0.65,
    )
    axes[1].axhline(0.0, color="#111111", linewidth=0.65)
    axes[1].set_xticks(positions, coefficient_names, rotation=25, ha="right")
    axes[1].set_ylabel("Coefficient")
    for axis in axes:
        style_axis(axis)
    figure.savefig(output / "systematics_model.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "systematics_model.pdf", bbox_inches="tight")
    plt.close(figure)


def _make_longitude_production_report(
    config: MappingConfig,
    output: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Build report products for the latitude-flat Fourier representation."""

    samples_path = output / "samples.npz"
    if not samples_path.is_file():
        raise FileNotFoundError(
            "The longitude production report needs posterior samples from a NUTS fit."
        )
    curve = load_light_curve(config)
    with np.load(samples_path, allow_pickle=False) as archive:
        if "longitude_coefficients" not in archive:
            raise ValueError(
                "The longitude fit samples must use the `longitude_coefficients` key."
            )
        coefficients = np.asarray(archive["longitude_coefficients"], dtype=float)
        if "longitude_coefficients_by_chain" not in archive:
            raise ValueError(
                "The longitude fit samples must include `longitude_coefficients_by_chain`."
            )
        coefficients_by_chain = np.asarray(
            archive["longitude_coefficients_by_chain"], dtype=float
        )
        flux_samples = (
            np.asarray(archive["flux"], dtype=float)
            if "flux" in archive
            else None
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

    if flux_samples is None:
        raise ValueError(
            "The longitude production report requires posterior `flux` samples."
        )
    if coefficients.ndim != 2 or coefficients.shape[0] == 0:
        raise ValueError("longitude_coefficients must have shape (draw, coefficient)")
    if coefficients.shape[-1] % 2 != 1:
        raise ValueError("longitude_coefficients must contain 1 + 2 * degree values")
    if coefficients_by_chain.ndim != 3 or coefficients_by_chain.shape[-1] != coefficients.shape[-1]:
        raise ValueError(
            "longitude_coefficients_by_chain must have shape (chain, draw, coefficient)"
        )
    if flux_samples is not None and flux_samples.shape != (
        coefficients.shape[0], curve.n_observations
    ):
        raise ValueError("The longitude fit flux samples have an invalid shape")
    if coefficients_by_chain.shape[0] == 0 or coefficients_by_chain.shape[1] == 0:
        raise ValueError("The longitude coefficient trace must contain draws")
    if coefficients_by_chain.shape[0] * coefficients_by_chain.shape[1] != coefficients.shape[0]:
        raise ValueError(
            "The longitude coefficient chain and flat sample arrays do not match."
        )

    degree = (int(coefficients.shape[-1]) - 1) // 2
    summary_degree = summary.get("longitude_degree", summary.get("map_degree"))
    if summary_degree is not None and int(summary_degree) != degree:
        raise ValueError(
            "The longitude coefficient count does not match the fit summary degree."
        )
    if summary.get("parameterization") != "longitude_fourier":
        raise ValueError(
            "Longitude samples require fit_summary.json parameterization: longitude_fourier."
        )

    map_longitude_deg = np.linspace(-180.0, 180.0, 121)
    map_latitude_deg = np.linspace(-90.0, 90.0, 61)
    longitude_rad = np.deg2rad(map_longitude_deg)
    profile_longitude_deg = np.linspace(-180.0, 180.0, 721)
    profile_longitude_rad = np.deg2rad(profile_longitude_deg)
    one_sigma_lower, one_sigma_upper = _SIGMA_INTERVAL_QUANTILES["1sigma"]
    two_sigma_lower, two_sigma_upper = _SIGMA_INTERVAL_QUANTILES["2sigma"]
    profile_draws = np.pi * np.asarray(
        evaluate_longitude_map(coefficients, profile_longitude_rad), dtype=float
    )
    map_profile_draws = np.pi * np.asarray(
        evaluate_longitude_map(coefficients, longitude_rad), dtype=float
    )
    profile_1sigma_lower, profile_median, profile_1sigma_upper = np.quantile(
        profile_draws, [one_sigma_lower, 0.50, one_sigma_upper], axis=0
    )
    profile_2sigma_lower, _, profile_2sigma_upper = np.quantile(
        profile_draws, [two_sigma_lower, 0.50, two_sigma_upper], axis=0
    )
    map_1sigma_lower, _, map_1sigma_upper = np.quantile(
        map_profile_draws, [one_sigma_lower, 0.50, one_sigma_upper], axis=0
    )
    map_2sigma_lower, _, map_2sigma_upper = np.quantile(
        map_profile_draws, [two_sigma_lower, 0.50, two_sigma_upper], axis=0
    )
    map_median_profile = np.quantile(map_profile_draws, 0.50, axis=0)

    contrast_draws = map_profile_draws
    minimum_contrast = np.min(contrast_draws, axis=-1)
    nonnegative_draw_fraction = float(np.mean(minimum_contrast >= -1.0e-12))
    negative_draw_fraction = float(np.mean(minimum_contrast < -1.0e-12))
    min_1sigma_lower, min_median, min_1sigma_upper = np.quantile(
        minimum_contrast, [one_sigma_lower, 0.50, one_sigma_upper]
    )
    min_2sigma_lower, _, min_2sigma_upper = np.quantile(
        minimum_contrast, [two_sigma_lower, 0.50, two_sigma_upper]
    )
    dayside = (profile_longitude_deg >= -90.0) & (profile_longitude_deg <= 90.0)
    hotspot_offsets = _profile_peak_longitudes(
        profile_draws, profile_longitude_deg, dayside
    )
    _, longitude_comparison = _longitude_comparison_references(
        hotspot_offsets,
        include_hammond=(
            "hammond" in config.project.name.lower()
            or "wasp43" in config.project.name.lower()
        ),
        published_reference=_published_longitude_reference(config.project.name),
        lower_quantile=one_sigma_lower,
        upper_quantile=one_sigma_upper,
    )

    map_1sigma_lower_grid = np.broadcast_to(
        map_1sigma_lower[None, :] * 1.0e6, (61, 121)
    )
    map_median_grid = np.broadcast_to(
        map_median_profile[None, :] * 1.0e6, (61, 121)
    )
    map_1sigma_upper_grid = np.broadcast_to(
        map_1sigma_upper[None, :] * 1.0e6, (61, 121)
    )
    map_2sigma_lower_grid = np.broadcast_to(
        map_2sigma_lower[None, :] * 1.0e6, (61, 121)
    )
    map_2sigma_upper_grid = np.broadcast_to(
        map_2sigma_upper[None, :] * 1.0e6, (61, 121)
    )
    target = config.project.name
    for stem, values in (
        ("brightness_map_1sigma_lower", map_1sigma_lower_grid),
        ("brightness_map_median", map_median_grid),
        ("brightness_map_1sigma_upper", map_1sigma_upper_grid),
        ("brightness_map_2sigma_lower", map_2sigma_lower_grid),
        ("brightness_map_2sigma_upper", map_2sigma_upper_grid),
    ):
        _save_longitude_map(
            map_longitude_deg,
            map_latitude_deg,
            values,
            output,
            stem=stem,
            colorbar_label="Planet-to-star contrast (ppm)",
        )
    _save_longitude_profile(
        profile_longitude_deg,
        profile_median,
        profile_1sigma_lower,
        profile_1sigma_upper,
        profile_2sigma_lower,
        profile_2sigma_upper,
        output,
    )
    _save_longitude_trace(coefficients_by_chain, output)
    median_model = np.quantile(flux_samples, 0.50, axis=0)
    time_unit = str(curve.time_unit).lower()
    time_hours_scale = {
        "day": 24.0,
        "days": 24.0,
        "hour": 1.0,
        "hours": 1.0,
        "second": 1.0 / 3600.0,
        "seconds": 1.0 / 3600.0,
    }.get(time_unit)
    if time_hours_scale is None:
        raise ValueError(f"Unsupported light-curve time unit {curve.time_unit!r}.")
    time_hours = (np.asarray(curve.time, dtype=float) - np.median(curve.time)) * time_hours_scale
    _save_longitude_light_curve(
        time_hours,
        np.asarray(curve.flux, dtype=float),
        np.asarray(curve.flux_err, dtype=float),
        median_model,
        output,
    )
    if nuisance_samples is not None and nuisance_coefficient_samples is not None:
        _save_longitude_systematics_plot(
            time_hours,
            nuisance_samples,
            nuisance_coefficient_samples,
            list(summary.get("systematics_coefficient_names", [])),
            output,
        )

    peak_longitude = _profile_peak_longitudes(
        profile_draws, profile_longitude_deg, dayside
    )
    hotspot_1sigma_lower, hotspot_median, hotspot_1sigma_upper = np.quantile(
        peak_longitude, [one_sigma_lower, 0.50, one_sigma_upper]
    )
    hotspot_2sigma_lower, _, hotspot_2sigma_upper = np.quantile(
        peak_longitude, [two_sigma_lower, 0.50, two_sigma_upper]
    )
    chains = int(coefficients_by_chain.shape[0])
    prior_mean = summary.get("coefficient_prior_mean")
    prior_sigma = summary.get("coefficient_prior_sigma")
    coefficient_prior = {
        "distribution": "independent Gaussian in the real Fourier coefficients",
        "mean": prior_mean,
        "standard_deviation": prior_sigma,
        "positivity": "unconstrained signed coefficients; positivity is not imposed",
    }
    basis_metadata = {
        "name": "latitude-flat real Fourier basis",
        "coefficient_order": "constant, cos(m longitude), sin(m longitude), m=1..degree",
        "normalization": "each map-intensity column is divided by pi; a unit constant coefficient has unit disk flux",
        "degree": degree,
    }
    model_flux_residual = np.asarray(curve.flux, dtype=float) - median_model
    report: dict[str, Any] = {
        "status": "complete",
        "target": target,
        "map_representation": "longitude",
        "parameterization": "longitude_fourier",
        "map_basis": basis_metadata,
        "coefficient_prior": coefficient_prior,
        "posterior_interval_bounds": {
            "1sigma": {
                "central_probability": 0.682689492137086,
                "lower_quantile": one_sigma_lower,
                "upper_quantile": one_sigma_upper,
            },
            "2sigma": {
                "central_probability": 0.954499736103642,
                "lower_quantile": two_sigma_lower,
                "upper_quantile": two_sigma_upper,
            },
        },
        "posterior_draws": int(coefficients.shape[0]),
        "chains": chains,
        "residual_rms_ppm": float(np.sqrt(np.mean(model_flux_residual**2)) * 1.0e6),
        "divergences": int(summary.get("divergences", 0)),
        "maximum_rhat": summary.get("maximum_rhat"),
        "minimum_effective_sample_size": summary.get(
            "minimum_effective_sample_size"
        ),
        "systematics": {
            "mode": summary.get("systematics", {}).get("mode", "corrected"),
            "coefficient_names": summary.get("systematics_coefficient_names", []),
            "coefficient_mean": summary.get("systematics_coefficient_mean", []),
            "maximum_rhat": summary.get("maximum_systematics_rhat"),
            "minimum_effective_sample_size": summary.get(
                "minimum_systematics_effective_sample_size"
            ),
        },
        "noise": _noise_report(summary),
        "hotspot_longitude_degrees_east": {
            "lower_1sigma": float(hotspot_1sigma_lower),
            "median": float(hotspot_median),
            "upper_1sigma": float(hotspot_1sigma_upper),
            "lower_2sigma": float(hotspot_2sigma_lower),
            "upper_2sigma": float(hotspot_2sigma_upper),
        },
        "longitude_comparison": longitude_comparison,
        "map_peak_2d": None,
        "latitude_information": {
            "status": "fixed_by_model",
            "warning": (
                "The map is constant with latitude by definition. This fit gives no evidence about latitude structure."
            ),
            "north_south_asymmetry": None,
            "peak_latitude_degrees": None,
        },
        "contrast": {
            "minimum_ppm": {
                "lower_1sigma": float(min_1sigma_lower * 1.0e6),
                "median": float(min_median * 1.0e6),
                "upper_1sigma": float(min_1sigma_upper * 1.0e6),
                "lower_2sigma": float(min_2sigma_lower * 1.0e6),
                "upper_2sigma": float(min_2sigma_upper * 1.0e6),
            },
            "posterior_draw_nonnegative_fraction": nonnegative_draw_fraction,
            "posterior_draw_with_negative_region_fraction": negative_draw_fraction,
        },
        "temperature": {
            "status": "unavailable",
            "reason": (
                "The longitude fit does not include a configured stellar spectrum and instrumental bandpass. No brightness-temperature product was calculated."
            ),
        },
        "positivity_conditioned": {
            "status": "not_applied",
            "warning": "The longitude Fourier coefficients are signed and have no positivity constraint.",
        },
        "files": {
            "white_light": "white_light_curve.png",
            "brightness_1sigma_lower": "brightness_map_1sigma_lower.png",
            "brightness_median": "brightness_map_median.png",
            "brightness_1sigma_upper": "brightness_map_1sigma_upper.png",
            "brightness_2sigma_lower": "brightness_map_2sigma_lower.png",
            "brightness_2sigma_upper": "brightness_map_2sigma_upper.png",
            "longitude_profile": "longitude_profile.png",
            "trace": "posterior_trace.png",
            "map_arrays": "posterior_maps.npz",
            **(
                {"systematics": "systematics_model.png"}
                if nuisance_samples is not None
                and nuisance_coefficient_samples is not None
                else {}
            ),
        },
        "figure_captions": {
            "brightness_map_1sigma_lower": "Lower bound of the central 1σ posterior contrast interval; the map is flat in latitude by construction.",
            "brightness_map_median": "Pointwise posterior-median contrast map; the map is flat in latitude by construction.",
            "brightness_map_1sigma_upper": "Upper bound of the central 1σ posterior contrast interval; the map is flat in latitude by construction.",
            "brightness_map_2sigma_lower": "Lower bound of the central 2σ posterior contrast interval; the map is flat in latitude by construction.",
            "brightness_map_2sigma_upper": "Upper bound of the central 2σ posterior contrast interval; the map is flat in latitude by construction.",
            "longitude_profile": "Posterior median Fourier model with pointwise central 1σ and 2σ intervals.",
            "white_light": "Observed light curve with the pointwise posterior median model and residuals.",
            "systematics": "Posterior median systematics model with exact central 1σ and 2σ ranges.",
            "trace": "Posterior draws for each signed longitude Fourier coefficient.",
        },
    }
    map_arrays: dict[str, np.ndarray] = dict(
        longitude_degrees=map_longitude_deg,
        latitude_degrees=map_latitude_deg,
        contrast_1sigma_lower=map_1sigma_lower_grid * 1.0e-6,
        contrast_median=map_median_grid * 1.0e-6,
        contrast_1sigma_upper=map_1sigma_upper_grid * 1.0e-6,
        contrast_2sigma_lower=map_2sigma_lower_grid * 1.0e-6,
        contrast_2sigma_upper=map_2sigma_upper_grid * 1.0e-6,
        longitude_profile_longitude_degrees=profile_longitude_deg,
        longitude_profile_1sigma_lower=profile_1sigma_lower,
        longitude_profile_1sigma_upper=profile_1sigma_upper,
        longitude_profile_2sigma_lower=profile_2sigma_lower,
        longitude_profile_2sigma_upper=profile_2sigma_upper,
        longitude_profile_median=profile_median,
        hotspot_longitude_degrees=peak_longitude,
        map_peak_longitude_degrees_east=peak_longitude,
    )
    np.savez_compressed(output / "posterior_maps.npz", **map_arrays)
    (output / "production_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    """Return the standard 16th, 50th, and 84th percentile summary."""

    q16, median, q84 = np.quantile(np.asarray(values, dtype=float), [0.16, 0.50, 0.84])
    return {"q16": float(q16), "median": float(median), "q84": float(q84)}


def _sigma_quantile_summary(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, float]:
    """Return an exact central interval with 1σ or 2σ percentile bounds."""

    lower, median, upper = np.quantile(
        np.asarray(values, dtype=float),
        [lower_quantile, 0.5, upper_quantile],
    )
    return {
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
    }


def _longitude_comparison_references(
    offsets: np.ndarray,
    *,
    include_hammond: bool,
    published_reference: dict[str, Any] | None = None,
    lower_quantile: float = 0.16,
    upper_quantile: float = 0.84,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build shared longitude markers and their machine-readable summary."""

    lower, median, upper = np.quantile(
        np.asarray(offsets, dtype=float), [lower_quantile, 0.50, upper_quantile]
    )
    lower_error = float(median - lower)
    upper_error = float(upper - median)
    references: list[dict[str, Any]] = [
        {
            "kind": "substellar",
            "longitude": 0.0,
            "label": "Sub-stellar point: 0°",
        },
        {
            "kind": "robert",
            "longitude": float(median),
            "lower": float(lower),
            "upper": float(upper),
            "label": (
                f"robert-mapping: {median:+.2f}° "
                f"(−{lower_error:.2f}°/+{upper_error:.2f}°)"
            ),
        },
    ]
    robert_interval = (
        {
            "lower_1sigma": float(lower),
            "median": float(median),
            "upper_1sigma": float(upper),
        }
        if lower_quantile == _SIGMA_INTERVAL_QUANTILES["1sigma"][0]
        and upper_quantile == _SIGMA_INTERVAL_QUANTILES["1sigma"][1]
        else {"q16": float(lower), "median": float(median), "q84": float(upper)}
    )
    summary: dict[str, Any] = {
        "substellar_longitude_degrees": 0.0,
        "robert_mapping": robert_interval,
        "hammond_et_al_2024": None,
        "published_reference": None,
    }
    if include_hammond:
        hammond = _HAMMOND_HOTSPOT_LONGITUDE_DEGREES
        sigma = _HAMMOND_HOTSPOT_SIGMA_DEGREES
        references.append(
            {
                "kind": "hammond",
                "longitude": hammond,
                "lower": hammond - sigma,
                "upper": hammond + sigma,
                "label": f"Hammond et al. (2024): +{hammond:.2f}° ± {sigma:.2f}°",
            }
        )
        summary["hammond_et_al_2024"] = {
            "median": hammond,
            "sigma": sigma,
            "lower": hammond - sigma,
            "upper": hammond + sigma,
        }
    if published_reference is not None:
        longitude = float(published_reference["longitude"])
        sigma = float(published_reference.get("sigma", 0.0))
        lower = float(published_reference.get("lower", longitude - sigma))
        upper = float(published_reference.get("upper", longitude + sigma))
        label = str(published_reference["label"])
        references.append(
            {
                "kind": "literature",
                "longitude": longitude,
                "lower": lower,
                "upper": upper,
                "label": label,
            }
        )
        summary["published_reference"] = {
            "label": label,
            "median": longitude,
            "sigma": sigma,
            "lower": lower,
            "upper": upper,
        }
    return references, summary


def _published_longitude_reference(target: str) -> dict[str, Any] | None:
    """Return the white-light literature offset for a validation target."""

    name = target.lower()
    if "wasp121" in name and "nrs1" in name:
        return {
            "longitude": _WASP121_NRS1_HOTSPOT_DEGREES,
            "sigma": _WASP121_NRS1_HOTSPOT_SIGMA_DEGREES,
            "label": "Mikal-Evans et al. (2023), NRS1: +3.36° ± 0.11°",
        }
    if "wasp121" in name and "nrs2" in name:
        return {
            "longitude": _WASP121_NRS2_HOTSPOT_DEGREES,
            "sigma": _WASP121_NRS2_HOTSPOT_SIGMA_DEGREES,
            "label": "Mikal-Evans et al. (2023), NRS2: +2.66° ± 0.12°",
        }
    if "wasp121" in name and "miri" in name:
        return {
            "longitude": _WASP121_MIRI_HOTSPOT_DEGREES,
            "lower": _WASP121_MIRI_HOTSPOT_LOWER_DEGREES,
            "upper": _WASP121_MIRI_HOTSPOT_UPPER_DEGREES,
            "label": "Kahle et al. (2026), MIRI: +4.8° -2.8°/+2.7°",
        }
    return None


def _profile_peak_longitudes(
    profiles: np.ndarray,
    longitude_deg: np.ndarray,
    dayside: np.ndarray,
) -> np.ndarray:
    """Return sub-grid dayside peaks from equally spaced longitude profiles."""

    values = np.asarray(profiles, dtype=float)[:, np.asarray(dayside, dtype=bool)]
    longitude = np.asarray(longitude_deg, dtype=float)[np.asarray(dayside, dtype=bool)]
    indices = np.argmax(values, axis=1)
    peaks = longitude[indices].astype(float, copy=True)
    if longitude.size < 3:
        return peaks
    step = float(np.median(np.diff(longitude)))
    interior = (indices > 0) & (indices < longitude.size - 1)
    rows = np.flatnonzero(interior)
    centres = indices[interior]
    lower = values[rows, centres - 1]
    middle = values[rows, centres]
    upper = values[rows, centres + 1]
    denominator = lower - 2.0 * middle + upper
    valid = np.abs(denominator) > np.finfo(float).eps
    delta = np.zeros(rows.size, dtype=float)
    delta[valid] = 0.5 * (lower[valid] - upper[valid]) / denominator[valid]
    peaks[rows] = longitude[centres] + np.clip(delta, -1.0, 1.0) * step
    return peaks


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
    representation = str(config.map.representation).lower()
    if representation == "longitude":
        if summary.get("parameterization") != "longitude_fourier":
            raise ValueError(
                "The configured longitude fit does not have longitude Fourier samples."
            )
        return _make_longitude_production_report(config, output, summary)
    if summary.get("parameterization") == "longitude_fourier":
        raise ValueError(
            "The fit summary uses longitude Fourier samples, but the configuration does not select map.representation: longitude."
        )
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
    latitude_weights = _latitude_profile_weights(target, latitude_deg)
    profiles = np.average(contrast, axis=1, weights=latitude_weights)
    profile_q16, profile_median, profile_q84 = np.quantile(
        profiles, [0.16, 0.50, 0.84], axis=0
    )
    dayside = (longitude_deg >= -90.0) & (longitude_deg <= 90.0)
    offsets = _profile_peak_longitudes(profiles, longitude_deg, dayside)
    longitude_references, longitude_comparison = _longitude_comparison_references(
        offsets,
        include_hammond=("hammond" in target.lower() or "wasp43" in target.lower()),
        published_reference=_published_longitude_reference(target),
    )
    for label, values, title in (
        ("brightness_map_q16", map_q16, "Brightness map: 16th percentile"),
        ("brightness_map_median", map_median, "Brightness map: posterior median"),
        ("brightness_map_q84", map_q84, "Brightness map: 84th percentile"),
    ):
        plot_brightness_map(
            longitude_deg,
            latitude_deg,
            values * 1.0e6,
            metadata={
                "title": f"{target}: {title}",
                "colorbar_label": "Contrast (ppm)",
                "longitude_references": longitude_references,
            },
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

    plot_longitude_profile(
        longitude_deg,
        profile_median * 1.0e6,
        profile_q16 * 1.0e6,
        profile_q84 * 1.0e6,
        metadata={
            "title": f"{target}: longitude profile",
            "profile_label_axis": "Contrast (ppm)",
            "longitude_references": longitude_references,
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
            "longitude_references": longitude_references,
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
        conditioned_offsets = _profile_peak_longitudes(
            conditioned_profiles, longitude_deg, dayside
        )
        conditioned_longitude_references, _ = _longitude_comparison_references(
            conditioned_offsets,
            include_hammond=("hammond" in target.lower() or "wasp43" in target.lower()),
            published_reference=_published_longitude_reference(target),
        )
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
                    "longitude_references": conditioned_longitude_references,
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
                "longitude_references": conditioned_longitude_references,
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
                "longitude_references": conditioned_longitude_references,
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
        "noise": _noise_report(summary),
        "hotspot_longitude_degrees_east": {
            "q16": float(hotspot_q16),
            "median": float(hotspot_median),
            "q84": float(hotspot_q84),
        },
        "longitude_comparison": longitude_comparison,
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
