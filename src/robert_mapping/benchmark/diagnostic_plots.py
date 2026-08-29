"""Publication-ready diagnostic plots for eclipse-mapping analyses.

The plotting functions in this module are deliberately case agnostic.  They
accept arrays that have already been prepared by a fit or a recovery run.  No
data files are opened here.  Every function returns a Matplotlib
``~matplotlib.figure.Figure`` and, when ``output`` is supplied, writes a PNG
and a PDF with the same stem.

The default accent is ``mediumpurple``.  Arial is used when it is installed;
otherwise Matplotlib's normal sans-serif fallback is used.  Plotting is kept
in a separate module so the numerical model can run without importing
Matplotlib until a figure is requested.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


BEST_FIT_COLOR = "mediumpurple"
"""Accent colour used for fitted models and recovered results."""

_BLACK = "#222222"
_GREY = "#777777"
_LIGHT_GREY = "#d9d9d9"
_HAMMOND_COLOR = "#00838f"


def _mpl():
    """Import Matplotlib lazily and apply the project plotting defaults."""

    import matplotlib
    from matplotlib import font_manager

    # Do not require Arial on a headless Linux or HPC installation.  The
    # explicit check avoids the repeated "Arial not found" warning emitted by
    # Matplotlib when the font is not available.
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except (ValueError, OSError):
        family = "DejaVu Sans"
    else:
        family = "Arial"
    matplotlib.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
        }
    )
    return matplotlib


def _metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a mutable metadata copy and reject non-mapping values."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    return dict(metadata)


def _finite_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=float)


def _same_length(*arrays: tuple[str, NDArray[np.float64]]) -> None:
    lengths = {array.size for _, array in arrays}
    if len(lengths) != 1:
        names = ", ".join(name for name, _ in arrays)
        raise ValueError(f"{names} must have the same length")


def _output_stem(output: str | Path | None) -> Path | None:
    """Normalise an output path to a stem without a PNG/PDF suffix."""

    if output is None:
        return None
    path = Path(output)
    if path.suffix.lower() in {".png", ".pdf"}:
        path = path.with_suffix("")
    if not path.name:
        raise ValueError("output must include a file stem")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(figure: Any, output: str | Path | None, *, dpi: int = 300) -> None:
    """Save a figure to ``<output>.png`` and ``<output>.pdf``."""

    stem = _output_stem(output)
    if stem is None:
        return
    figure.savefig(stem.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _colour_map():
    """Return a purple map with light colours for high values."""

    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "robert_mapping_purple",
        ["#20102c", "#542b73", BEST_FIT_COLOR, "#d8bde9", "#f7f3fb"],
    )


def _axis_text(axis: Any, metadata: Mapping[str, Any], key: str, default: str) -> str:
    value = metadata.get(key, default)
    return str(value)


def _finish(figure: Any, output: str | Path | None, *, dpi: int = 300) -> Any:
    _save_figure(figure, output, dpi=dpi)
    return figure


def _draw_longitude_references(
    axis: Any,
    metadata: Mapping[str, Any],
    *,
    best_fit_color: str,
    map_mode: bool,
) -> bool:
    """Draw sub-stellar, fitted-hotspot, and literature longitude markers."""

    references = metadata.get("longitude_references")
    if references is None:
        return False
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise TypeError("longitude_references must be a sequence of mappings")
    if not references:
        return False

    style = {
        "substellar": (_BLACK, ":"),
        "robert": (best_fit_color, "-"),
        "hammond": (_HAMMOND_COLOR, "--"),
        "literature": (_HAMMOND_COLOR, "--"),
    }
    errorbar_level = (
        {"robert": 0.965, "hammond": 0.900, "literature": 0.900}
        if map_mode
        else {"robert": 0.960, "hammond": 0.885, "literature": 0.885}
    )
    for reference in references:
        if not isinstance(reference, Mapping):
            raise TypeError("each longitude reference must be a mapping")
        kind = str(reference.get("kind", "reference")).lower()
        longitude = float(reference["longitude"])
        if not np.isfinite(longitude):
            raise ValueError("reference longitude must be finite")
        color, linestyle = style.get(kind, (_GREY, "--"))
        label = str(reference.get("label", f"Reference: {longitude:+.2f}°"))
        lower = reference.get("lower")
        upper = reference.get("upper")
        axis.axvline(
            longitude,
            color=color,
            linestyle=linestyle,
            linewidth=1.35 if kind != "substellar" else 1.0,
            alpha=0.95,
            label=label,
            zorder=5,
        )
        if kind == "substellar" and map_mode:
            axis.scatter(
                [longitude],
                [0.0],
                marker="+",
                s=64,
                linewidths=1.5,
                color=color,
                zorder=7,
            )
        if lower is None and upper is None:
            continue
        if lower is None or upper is None:
            raise ValueError("reference lower and upper bounds must be supplied together")
        lower_value = float(lower)
        upper_value = float(upper)
        if not (
            np.isfinite(lower_value)
            and np.isfinite(upper_value)
            and lower_value <= longitude <= upper_value
        ):
            raise ValueError("reference bounds must be finite and contain longitude")
        axis.axvspan(
            lower_value,
            upper_value,
            color=color,
            alpha=0.13,
            linewidth=0.0,
            zorder=4,
        )
        level = errorbar_level.get(kind, 1.025)
        axis.errorbar(
            longitude,
            level,
            xerr=np.asarray(
                [[longitude - lower_value], [upper_value - longitude]], dtype=float
            ),
            fmt="o",
            markersize=4.5,
            capsize=3.5,
            elinewidth=1.35,
            capthick=1.35,
            color=color,
            transform=axis.get_xaxis_transform(),
            clip_on=False,
            zorder=8,
        )
    return True


def _validate_map_grid(
    longitude: ArrayLike, latitude: ArrayLike, values: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lon = _finite_vector(longitude, "longitude")
    lat = _finite_vector(latitude, "latitude")
    image = np.asarray(values, dtype=float)
    if image.ndim != 2 or image.shape != (lat.size, lon.size):
        raise ValueError("values must have shape (latitude.size, longitude.size)")
    if not np.all(np.isfinite(image)):
        raise ValueError("values must contain only finite values")
    if np.any(np.diff(lon) <= 0.0) or np.any(np.diff(lat) <= 0.0):
        raise ValueError("longitude and latitude must be strictly increasing")
    return lon, lat, image


def _angle_arrays(
    longitude: ArrayLike,
    latitude: ArrayLike,
    values: ArrayLike,
    angle_unit: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lon, lat, image = _validate_map_grid(longitude, latitude, values)
    unit = str(angle_unit).lower().replace("°", "deg")
    if unit in {"rad", "radian", "radians"}:
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)
    elif unit not in {"deg", "degree", "degrees"}:
        raise ValueError("angle_unit must be degrees or radians")
    return lon, lat, image


def plot_white_light_curve(
    time: ArrayLike,
    flux: ArrayLike,
    model: ArrayLike | None = None,
    flux_err: ArrayLike | None = None,
    residuals: ArrayLike | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
) -> Any:
    """Plot a white-light curve, model, and residuals.

    ``time``, ``flux``, and every supplied vector must have the same length.
    If ``residuals`` is omitted and ``model`` is supplied, residuals are
    calculated as ``flux - model``.  Residuals are shown in ppm when the
    metadata value ``residual_unit`` is ``"ppm"``; the default is the same
    flux unit as the input arrays.
    """

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    x = _finite_vector(time, "time")
    y = _finite_vector(flux, "flux")
    arrays = [("time", x), ("flux", y)]
    fitted = None if model is None else _finite_vector(model, "model")
    error = None if flux_err is None else _finite_vector(flux_err, "flux_err")
    residual = None if residuals is None else _finite_vector(residuals, "residuals")
    if fitted is not None:
        arrays.append(("model", fitted))
    if error is not None:
        arrays.append(("flux_err", error))
        if np.any(error <= 0.0):
            raise ValueError("flux_err must be strictly positive")
    if residual is not None:
        arrays.append(("residuals", residual))
    _same_length(*arrays)
    if residual is None and fitted is not None:
        residual = y - fitted

    figure, axes = plt.subplots(
        2 if residual is not None else 1,
        1,
        figsize=tuple(info.get("figsize", (7.4, 5.4 if residual is not None else 3.8))),
        sharex=residual is not None,
        gridspec_kw={"height_ratios": (3.0, 1.0)} if residual is not None else None,
        constrained_layout=True,
    )
    if residual is None:
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes))
    top = axes[0]
    if error is None:
        top.plot(x, y, color=_BLACK, linewidth=0.8, marker="o", markersize=2.4,
                 linestyle="none", alpha=0.8, label=info.get("data_label", "Data"))
    else:
        top.errorbar(x, y, yerr=error, color=_BLACK, linewidth=0.5, marker="o",
                     markersize=2.0, linestyle="none", alpha=0.65,
                     label=info.get("data_label", "Data"))
    if fitted is not None:
        order = np.argsort(x)
        sorted_x = x[order]
        sorted_fitted = fitted[order]
        positive_steps = np.diff(sorted_x)
        positive_steps = positive_steps[positive_steps > 0.0]
        gap_factor = float(info.get("model_gap_factor", 5.0))
        if positive_steps.size:
            split_at = np.flatnonzero(
                np.diff(sorted_x) > gap_factor * np.median(positive_steps)
            ) + 1
        else:
            split_at = np.array([], dtype=int)
        x_segments = np.split(sorted_x, split_at)
        fitted_segments = np.split(sorted_fitted, split_at)
        for segment_index, (segment_x, segment_fitted) in enumerate(
            zip(x_segments, fitted_segments, strict=True)
        ):
            label = info.get("model_label", "Best-fitting model") if segment_index == 0 else None
            top.plot(
                segment_x,
                segment_fitted,
                color=color,
                linewidth=1.7,
                label=label,
            )
    top.set_ylabel(_axis_text(top, info, "flux_label", "Relative flux"))
    top.set_title(str(info.get("title", "White-light eclipse")))
    top.grid(alpha=0.25)
    if fitted is not None or error is not None:
        top.legend(loc=str(info.get("legend_loc", "best")), frameon=False)
    if residual is not None:
        bottom = axes[1]
        scale = 1.0e6 if str(info.get("residual_unit", "")).lower() == "ppm" else 1.0
        bottom.axhline(0.0, color=_GREY, linewidth=0.9)
        bottom.plot(x, residual * scale, color=color, marker="o", markersize=2.0,
                    linewidth=0.7, linestyle="none", alpha=0.85)
        bottom.set_ylabel(
            str(info.get("residual_label", "Residual" if scale == 1.0 else "Residual (ppm)"))
        )
        bottom.set_xlabel(_axis_text(bottom, info, "time_label", "Time"))
        bottom.grid(alpha=0.25)
    else:
        top.set_xlabel(_axis_text(top, info, "time_label", "Time"))
    return _finish(figure, output, dpi=dpi)


def plot_brightness_map(
    longitude: ArrayLike,
    latitude: ArrayLike,
    brightness: ArrayLike,
    *,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
    angle_unit: str = "degrees",
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """Plot a longitude--latitude brightness map.

    The brightness array must use the common ``(latitude, longitude)`` shape.
    ``angle_unit="radians"`` converts the axes to degrees for display.
    """

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    lon, lat, image = _angle_arrays(longitude, latitude, brightness, angle_unit)
    figure, axis = plt.subplots(figsize=tuple(info.get("figsize", (7.2, 4.2))),
                                constrained_layout=True)
    image_artist = axis.pcolormesh(
        lon, lat, image, shading="auto", cmap=cmap or _colour_map(), vmin=vmin, vmax=vmax
    )
    axis.set_xlabel(_axis_text(axis, info, "longitude_label", "Longitude (degrees east)"))
    axis.set_ylabel(_axis_text(axis, info, "latitude_label", "Latitude (degrees)"))
    axis.set_title(str(info.get("title", "Brightness map")))
    if info.get("xlim") is not None:
        axis.set_xlim(*info["xlim"])
    axis.set_ylim(float(lat[0]), float(lat[-1]))
    has_references = _draw_longitude_references(
        axis,
        info,
        best_fit_color=color,
        map_mode=True,
    )
    if has_references:
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=1,
            frameon=False,
        )
    cbar = figure.colorbar(image_artist, ax=axis, pad=0.02)
    cbar.set_label(str(info.get("colorbar_label", "Relative brightness")))
    return _finish(figure, output, dpi=dpi)


# Alias with the spelling used in some downstream notebooks.
plot_longitude_latitude_map = plot_brightness_map
plot_map = plot_brightness_map


def planck_brightness_temperature(
    wavelength: ArrayLike,
    specific_intensity: ArrayLike,
    *,
    wavelength_unit: str = "m",
    intensity_unit: str = "si",
) -> NDArray[np.float64] | np.float64:
    """Invert Planck's law for monochromatic brightness temperature.

    ``specific_intensity`` is per metre by default in SI units.  Set
    ``intensity_unit="per_micron"`` (or ``"per_um"``) for intensity per
    micrometre.  The implementation works in log space, so very faint maps do
    not overflow the exponential in the inversion.
    """

    wavelength_array = np.asarray(wavelength, dtype=float)
    intensity_array = np.asarray(specific_intensity, dtype=float)
    try:
        wavelength_array, intensity_array = np.broadcast_arrays(
            wavelength_array, intensity_array
        )
    except ValueError as error:
        raise ValueError("wavelength and specific_intensity cannot be broadcast") from error
    if wavelength_array.size == 0:
        raise ValueError("wavelength and specific_intensity must not be empty")
    if not np.all(np.isfinite(wavelength_array)) or np.any(wavelength_array <= 0.0):
        raise ValueError("wavelength must be finite and strictly positive")
    if not np.all(np.isfinite(intensity_array)) or np.any(intensity_array <= 0.0):
        raise ValueError("specific_intensity must be finite and strictly positive")
    unit = str(wavelength_unit).lower().replace("µ", "u").replace("μ", "u")
    if unit in {"m", "metre", "meter", "metres", "meters"}:
        wavelength_m = wavelength_array
    elif unit in {"um", "micron", "microns", "micrometre", "micrometer"}:
        wavelength_m = wavelength_array * 1.0e-6
    elif unit in {"nm", "nanometre", "nanometer"}:
        wavelength_m = wavelength_array * 1.0e-9
    elif unit in {"angstrom", "angstroms", "a"}:
        wavelength_m = wavelength_array * 1.0e-10
    else:
        raise ValueError("wavelength_unit must be m, micron, nm, or angstrom")
    iunit = str(intensity_unit).lower().replace(" ", "").replace("-", "_")
    if iunit in {"si", "per_m", "w_m2_sr_m", "w/m2/sr/m"}:
        intensity_m = intensity_array
    elif iunit in {"per_um", "per_micron", "w_m2_sr_um", "w/m2/sr/um"}:
        intensity_m = intensity_array * 1.0e6
    else:
        raise ValueError("intensity_unit must be SI per metre or per micron")

    # CODATA exact constants.  Work with log(A) and softplus(log(A)) where
    # A = 2 h c^2 / (lambda^5 I), avoiding overflow for very small I.
    h = 6.62607015e-34
    c = 299_792_458.0
    k = 1.380649e-23
    log_a = np.log(2.0 * h * c**2) - 5.0 * np.log(wavelength_m) - np.log(intensity_m)
    # ``logaddexp`` is the stable soft-plus implementation.  Unlike a
    # hand-written ``where(exp(log_a), ...)`` expression it does not evaluate
    # the overflowing branch for very faint intensities.
    log_denominator = np.logaddexp(0.0, log_a)
    temperature = (h * c / k) / (wavelength_m * log_denominator)
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("the Planck inversion returned a non-finite temperature")
    if np.ndim(temperature) == 0:
        return np.float64(temperature)
    return np.asarray(temperature, dtype=float)


def plot_temperature_map(
    longitude: ArrayLike,
    latitude: ArrayLike,
    specific_intensity: ArrayLike,
    wavelength: ArrayLike | float,
    *,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    wavelength_unit: str = "m",
    intensity_unit: str = "si",
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
    angle_unit: str = "degrees",
    cmap: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Any:
    """Plot a monochromatic brightness-temperature map in kelvin."""

    info = _metadata(metadata)
    temperature = planck_brightness_temperature(
        wavelength, specific_intensity, wavelength_unit=wavelength_unit,
        intensity_unit=intensity_unit,
    )
    temperature = np.asarray(temperature, dtype=float)
    # A scalar wavelength broadcasts over the map.  A wavelength array is
    # allowed only when it already matches the map, in which case the caller
    # receives a clear shape error rather than a silently wrong image.
    if temperature.shape != np.asarray(specific_intensity).shape:
        try:
            temperature = np.broadcast_to(temperature, np.asarray(specific_intensity).shape)
        except ValueError as error:
            raise ValueError("temperature map must match specific_intensity shape") from error
    info.setdefault("title", "Brightness-temperature map")
    info.setdefault("colorbar_label", "Brightness temperature (K)")
    return plot_brightness_map(
        longitude, latitude, temperature, metadata=info, output=output, color=color,
        dpi=dpi, angle_unit=angle_unit, cmap=cmap, vmin=vmin, vmax=vmax,
    )


plot_brightness_temperature_map = plot_temperature_map
plot_temperature = plot_temperature_map


def _profile_interval(
    longitude: NDArray[np.float64],
    lower: ArrayLike | None,
    upper: ArrayLike | None,
    interval: Sequence[float] | None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    if interval is not None:
        if len(interval) != 2:
            raise ValueError("interval must contain (lower, upper)")
        if lower is not None or upper is not None:
            raise ValueError("use interval or lower/upper, not both")
        lower, upper = interval
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must be supplied together")
    if lower is None:
        return None, None
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if low.ndim == 0:
        low = np.full(longitude.size, float(low))
    if high.ndim == 0:
        high = np.full(longitude.size, float(high))
    low = _finite_vector(low, "lower")
    high = _finite_vector(high, "upper")
    _same_length(("longitude", longitude), ("lower", low), ("upper", high))
    if np.any(high < low):
        raise ValueError("upper interval values must not be below lower values")
    return low, high


def plot_longitude_profile(
    longitude: ArrayLike,
    profile: ArrayLike,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    *,
    interval: Sequence[float] | None = None,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
) -> Any:
    """Plot a longitude profile with optional pointwise uncertainty bands."""

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    x = _finite_vector(longitude, "longitude")
    y = _finite_vector(profile, "profile")
    _same_length(("longitude", x), ("profile", y))
    low, high = _profile_interval(x, lower, upper, interval)
    order = np.argsort(x)
    figure, axis = plt.subplots(figsize=tuple(info.get("figsize", (7.2, 3.8))),
                                 constrained_layout=True)
    if low is not None and high is not None:
        axis.fill_between(x[order], low[order], high[order], color=color, alpha=0.20,
                          linewidth=0.0, label=str(info.get("interval_label", "68% interval")))
    axis.plot(x[order], y[order], color=color, linewidth=1.8,
              label=str(info.get("profile_label", "Profile")))
    has_comparison_references = _draw_longitude_references(
        axis,
        info,
        best_fit_color=color,
        map_mode=False,
    )
    if "reference_longitude" in info:
        axis.axvline(float(info["reference_longitude"]), color=_BLACK, linestyle="--",
                     linewidth=1.0, label=str(info.get("reference_label", "Reference")))
    if "injected_longitude" in info:
        axis.axvline(float(info["injected_longitude"]), color=_GREY, linestyle=":",
                     linewidth=1.0, label=str(info.get("injected_label", "Injected")))
    if not has_comparison_references:
        axis.axvline(0.0, color=_LIGHT_GREY, linewidth=0.8)
    axis.set_xlabel(str(info.get("longitude_label", "Longitude (degrees east)")))
    axis.set_ylabel(str(info.get("profile_label_axis", "Relative brightness")))
    axis.set_title(str(info.get("title", "Longitude profile")))
    axis.grid(alpha=0.25)
    if (
        low is not None
        or has_comparison_references
        or "reference_longitude" in info
        or "injected_longitude" in info
    ):
        if has_comparison_references:
            axis.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.24),
                ncol=1,
                frameon=False,
            )
        else:
            axis.legend(loc=str(info.get("legend_loc", "best")), frameon=False)
    return _finish(figure, output, dpi=dpi)


def plot_map_peak_posterior(
    peak_longitude: ArrayLike,
    peak_latitude: ArrayLike,
    *,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
) -> Any:
    """Plot the posterior location of the two-dimensional map maximum.

    The two coordinates are plotted separately.  This is intentional: the
    latitude posterior is often much broader than the longitude posterior,
    and a single scatter plot can hide that difference.  ``peak_longitude``
    and ``peak_latitude`` must contain one value per posterior draw.
    """

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    longitude = _finite_vector(peak_longitude, "peak_longitude")
    latitude = _finite_vector(peak_latitude, "peak_latitude")
    _same_length(("peak_longitude", longitude), ("peak_latitude", latitude))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=tuple(info.get("figsize", (8.2, 3.8))),
        constrained_layout=True,
    )
    for axis, values, label, title in (
        (
            axes[0],
            longitude,
            "2-D peak longitude (degrees east)",
            "Longitude of map maximum",
        ),
        (axes[1], latitude, "2-D peak latitude (degrees)", "Latitude of map maximum"),
    ):
        q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
        axis.hist(values, bins=24, color=color, alpha=0.82)
        axis.axvspan(q16, q84, color=color, alpha=0.15, linewidth=0.0)
        axis.axvline(median, color=color, linewidth=1.7, label="Posterior median")
        axis.axvline(0.0, color=_GREY, linestyle="--", linewidth=0.9)
        axis.set_xlabel(label)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", frameon=False)
    axes[0].set_ylabel("Posterior draws")
    warning = info.get("warning")
    if warning:
        axes[1].text(
            0.98,
            0.97,
            str(warning),
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            wrap=True,
            fontsize=8.5,
            color=_BLACK,
            bbox={"facecolor": "white", "edgecolor": _LIGHT_GREY, "alpha": 0.85, "pad": 3.0},
        )
    figure.suptitle(str(info.get("title", "Two-dimensional map-peak posterior")))
    return _finish(figure, output, dpi=dpi)


def plot_north_south_asymmetry(
    asymmetry: ArrayLike,
    *,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
) -> Any:
    """Plot the posterior of the normalised north--south map asymmetry."""

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    values = _finite_vector(asymmetry, "asymmetry")
    q16, median, q84 = np.quantile(values, [0.16, 0.50, 0.84])
    figure, axis = plt.subplots(
        figsize=tuple(info.get("figsize", (7.2, 3.8))), constrained_layout=True
    )
    axis.hist(values, bins=24, color=color, alpha=0.82)
    axis.axvspan(q16, q84, color=color, alpha=0.15, linewidth=0.0)
    axis.axvline(median, color=color, linewidth=1.7, label="Posterior median")
    axis.axvline(0.0, color=_GREY, linestyle="--", linewidth=0.9, label="Symmetric map")
    axis.set_xlabel(
        str(
            info.get(
                "xlabel",
                "North--south asymmetry: (north - south) / (north + south)",
            )
        )
    )
    axis.set_ylabel("Posterior draws")
    axis.set_title(str(info.get("title", "North--south map asymmetry")))
    axis.grid(alpha=0.25)
    axis.legend(loc="best", frameon=False)
    warning = info.get("warning")
    if warning:
        axis.text(
            0.98,
            0.97,
            str(warning),
            transform=axis.transAxes,
            ha="right",
            va="top",
            wrap=True,
            fontsize=8.5,
            color=_BLACK,
            bbox={"facecolor": "white", "edgecolor": _LIGHT_GREY, "alpha": 0.85, "pad": 3.0},
        )
    return _finish(figure, output, dpi=dpi)


def _recovery_arrays(
    injected_longitude: ArrayLike | None,
    recovered_longitude: ArrayLike,
    lower: ArrayLike | None,
    upper: ArrayLike | None,
    intervals: ArrayLike | None,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64],
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
]:
    recovered = _finite_vector(recovered_longitude, "recovered_longitude")
    injected = None if injected_longitude is None else np.asarray(injected_longitude, dtype=float)
    if injected is not None:
        if injected.ndim == 0:
            injected = np.full(recovered.size, float(injected))
        injected = _finite_vector(injected, "injected_longitude")
        _same_length(("injected_longitude", injected), ("recovered_longitude", recovered))
    if intervals is not None:
        if lower is not None or upper is not None:
            raise ValueError("use intervals or lower/upper, not both")
        pair = np.asarray(intervals, dtype=float)
        if pair.ndim != 2 or pair.shape != (recovered.size, 2):
            raise ValueError("intervals must have shape (n, 2)")
        lower, upper = pair[:, 0], pair[:, 1]
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must be supplied together")
    low = high = None
    if lower is not None:
        low = np.asarray(lower, dtype=float)
        high = np.asarray(upper, dtype=float)
        if low.ndim == 0:
            low = np.full(recovered.size, float(low))
        if high.ndim == 0:
            high = np.full(recovered.size, float(high))
        low = _finite_vector(low, "lower")
        high = _finite_vector(high, "upper")
        _same_length(("recovered_longitude", recovered), ("lower", low), ("upper", high))
        if np.any(high < low):
            raise ValueError("upper interval values must not be below lower values")
    return injected, recovered, low, high


def plot_recovery_summary(
    injected_longitude: ArrayLike | None,
    recovered_longitude: ArrayLike,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    delta_bic: ArrayLike | None = None,
    *,
    intervals: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    detection_threshold: float | None = -6.0,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
) -> Any:
    """Plot longitude recovery intervals and optional evidence statistics."""

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    injected, recovered, low, high = _recovery_arrays(
        injected_longitude, recovered_longitude, lower, upper, intervals
    )
    n = recovered.size
    if labels is None:
        labels = [str(index + 1) for index in range(n)]
    labels = list(labels)
    if len(labels) != n:
        raise ValueError("labels must have one value per recovery trial")
    evidence = None if delta_bic is None else _finite_vector(delta_bic, "delta_bic")
    if evidence is not None and evidence.size != n:
        raise ValueError("delta_bic must have one value per recovery trial")
    rows = 2 if evidence is not None else 1
    figure, axes = plt.subplots(
        rows,
        1,
        figsize=tuple(info.get("figsize", (8.0, 5.8 if rows == 2 else 3.8))),
        sharex=True if rows == 2 else False,
        gridspec_kw={"height_ratios": (2.0, 1.0)} if rows == 2 else None,
        constrained_layout=True,
    )
    axes = list(np.atleast_1d(axes))
    axis = axes[0]
    x = np.arange(n)
    if low is not None and high is not None:
        axis.errorbar(x, recovered, yerr=np.vstack((recovered - low, high - recovered)),
                      fmt="o", color=color, ecolor=color, capsize=3, linewidth=1.0,
                      label=str(info.get("recovery_label", "Recovered longitude")))
    else:
        axis.plot(x, recovered, "o", color=color,
                  label=str(info.get("recovery_label", "Recovered longitude")))
    if injected is not None:
        axis.scatter(x, injected, marker="x", color=_BLACK, s=38,
                     label=str(info.get("injection_label", "Injected longitude")), zorder=3)
    axis.axhline(0.0, color=_LIGHT_GREY, linewidth=0.8)
    axis.set_ylabel(str(info.get("longitude_label", "Longitude (degrees east)")))
    axis.set_title(str(info.get("title", "Recovery summary")))
    axis.set_xticks(x, labels)
    axis.grid(alpha=0.25)
    if low is not None or injected is not None:
        axis.legend(loc=str(info.get("legend_loc", "best")), frameon=False)
    if evidence is not None:
        bottom = axes[1]
        bottom.bar(x, evidence, color=color, alpha=0.85)
        bottom.axhline(0.0, color=_GREY, linewidth=0.8)
        if detection_threshold is not None:
            bottom.axhline(
                float(detection_threshold),
                color=_BLACK,
                linestyle="--",
                linewidth=1.0,
                label=str(info.get("threshold_label", "Detection threshold")),
            )
            bottom.legend(loc="best", frameon=False)
        bottom.set_ylabel(str(info.get("evidence_label", "Delta BIC (map - uniform)")))
        bottom.set_xlabel(str(info.get("trial_label", "Trial")))
        bottom.grid(alpha=0.25)
    else:
        axis.set_xlabel(str(info.get("trial_label", "Trial")))
    return _finish(figure, output, dpi=dpi)


plot_recovery_and_evidence = plot_recovery_summary


def plot_model_comparison(
    time: ArrayLike,
    reference_flux: ArrayLike,
    new_flux: ArrayLike,
    *,
    observed_flux: ArrayLike | None = None,
    flux_err: ArrayLike | None = None,
    metadata: Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    color: str = BEST_FIT_COLOR,
    dpi: int = 300,
) -> Any:
    """Compare reference and new model fluxes with a ppm residual panel."""

    _mpl()
    import matplotlib.pyplot as plt

    info = _metadata(metadata)
    x = _finite_vector(time, "time")
    reference = _finite_vector(reference_flux, "reference_flux")
    new = _finite_vector(new_flux, "new_flux")
    _same_length(("time", x), ("reference_flux", reference), ("new_flux", new))
    observed = None if observed_flux is None else _finite_vector(observed_flux, "observed_flux")
    error = None if flux_err is None else _finite_vector(flux_err, "flux_err")
    if observed is not None:
        _same_length(("time", x), ("observed_flux", observed))
    if error is not None:
        _same_length(("time", x), ("flux_err", error))
        if np.any(error <= 0.0):
            raise ValueError("flux_err must be strictly positive")
    residual_ppm = (new - reference) * 1.0e6
    rms = float(np.sqrt(np.mean(residual_ppm**2)))
    maximum = float(np.max(np.abs(residual_ppm)))
    metrics = dict(info.get("metrics", {})) if isinstance(info.get("metrics", {}), Mapping) else {}
    metrics.setdefault("RMS", f"{rms:.2f} ppm")
    metrics.setdefault("max |residual|", f"{maximum:.2f} ppm")

    figure, axes = plt.subplots(2, 1, figsize=tuple(info.get("figsize", (7.4, 5.4))),
                                sharex=True, gridspec_kw={"height_ratios": (3.0, 1.0)},
                                constrained_layout=True)
    top, bottom = axes
    order = np.argsort(x)
    if observed is not None:
        if error is None:
            top.plot(
                x,
                observed,
                color=_BLACK,
                marker="o",
                markersize=2.0,
                linestyle="none",
                alpha=0.50,
                label=str(info.get("observed_label", "Observed")),
            )
        else:
            top.errorbar(x, observed, yerr=error, color=_BLACK, marker="o", markersize=1.8,
                         linestyle="none", linewidth=0.5, alpha=0.45,
                         label=str(info.get("observed_label", "Observed")))
    top.plot(x[order], reference[order], color=_GREY, linewidth=1.3,
             label=str(info.get("reference_label", "Reference model")))
    top.plot(x[order], new[order], color=color, linewidth=1.7,
             label=str(info.get("new_label", "New model")))
    top.set_ylabel(str(info.get("flux_label", "Relative flux")))
    top.set_title(str(info.get("title", "Model comparison")))
    top.legend(loc=str(info.get("legend_loc", "best")), frameon=False)
    top.grid(alpha=0.25)
    bottom.axhline(0.0, color=_GREY, linewidth=0.8)
    bottom.plot(x[order], residual_ppm[order], color=color, linewidth=0.8)
    bottom.set_xlabel(str(info.get("time_label", "Time")))
    bottom.set_ylabel(str(info.get("residual_label", "New - reference (ppm)")))
    bottom.grid(alpha=0.25)
    metrics_text = "\n".join(f"{key}: {value}" for key, value in metrics.items())
    bottom.text(0.99, 0.96, metrics_text, transform=bottom.transAxes, ha="right", va="top",
                fontsize=10, color=_BLACK,
                bbox={"facecolor": "white", "edgecolor": _LIGHT_GREY, "alpha": 0.85, "pad": 3.0})
    return _finish(figure, output, dpi=dpi)


plot_two_model_comparison = plot_model_comparison
plot_white_lightcurve = plot_white_light_curve
plot_light_curve = plot_white_light_curve


__all__ = [
    "BEST_FIT_COLOR",
    "planck_brightness_temperature",
    "plot_brightness_map",
    "plot_brightness_temperature_map",
    "plot_longitude_latitude_map",
    "plot_map_peak_posterior",
    "plot_north_south_asymmetry",
    "plot_longitude_profile",
    "plot_light_curve",
    "plot_map",
    "plot_model_comparison",
    "plot_recovery_and_evidence",
    "plot_recovery_summary",
    "plot_temperature_map",
    "plot_temperature",
    "plot_two_model_comparison",
    "plot_white_light_curve",
    "plot_white_lightcurve",
]
