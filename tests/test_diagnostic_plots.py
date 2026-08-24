"""Focused tests for the generic diagnostic plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from robert_mapping.benchmark.diagnostic_plots import (  # noqa: E402
    BEST_FIT_COLOR,
    planck_brightness_temperature,
    plot_brightness_map,
    plot_longitude_profile,
    plot_map_peak_posterior,
    plot_model_comparison,
    plot_north_south_asymmetry,
    plot_recovery_summary,
    plot_temperature_map,
    plot_white_light_curve,
)


def _longitude_references():
    return [
        {"kind": "substellar", "longitude": 0.0, "label": "Sub-stellar point: 0°"},
        {
            "kind": "robert",
            "longitude": 7.49,
            "lower": 6.86,
            "upper": 8.14,
            "label": "robert-mapping: +7.49° (−0.63°/+0.65°)",
        },
        {
            "kind": "hammond",
            "longitude": 7.75,
            "lower": 7.39,
            "upper": 8.11,
            "label": "Hammond et al. (2024): +7.75° ± 0.36°",
        },
    ]


def _planck(wavelength_m: float, temperature: float) -> float:
    h = 6.62607015e-34
    c = 299_792_458.0
    k = 1.380649e-23
    exponent = h * c / (wavelength_m * k * temperature)
    return 2.0 * h * c**2 / wavelength_m**5 / np.expm1(exponent)


def test_planck_inversion_recovers_known_temperature() -> None:
    wavelength = 4.0e-6
    intensity = _planck(wavelength, 2_800.0)
    recovered = planck_brightness_temperature(wavelength, intensity)
    assert np.isclose(recovered, 2_800.0, rtol=1.0e-12)


def test_planck_inversion_handles_faint_array_and_units() -> None:
    wavelength = np.array([2.0, 4.0, 5.0])
    temperatures = np.array([1_500.0, 2_500.0, 3_000.0])
    intensity = np.array(
        [_planck(lam * 1.0e-6, temp) / 1.0e6 for lam, temp in zip(wavelength, temperatures)]
    )
    recovered = planck_brightness_temperature(
        wavelength, intensity, wavelength_unit="micron", intensity_unit="per_micron"
    )
    np.testing.assert_allclose(recovered, temperatures, rtol=1.0e-11)
    with pytest.raises(ValueError, match="strictly positive"):
        planck_brightness_temperature(4.0e-6, 0.0)


def test_white_light_curve_saves_png_and_pdf(tmp_path) -> None:
    time = np.linspace(-0.1, 0.1, 21)
    model = 1.0 - 4.0e-4 * np.exp(-0.5 * (time / 0.03) ** 2)
    observed = model + 2.0e-5 * np.sin(30.0 * time)
    figure = plot_white_light_curve(
        time,
        observed,
        model=model,
        flux_err=np.full(time.size, 2.0e-5),
        metadata={"title": "Test white light", "residual_unit": "ppm"},
        output=tmp_path / "white_light",
    )
    assert figure.axes
    assert (tmp_path / "white_light.png").exists()
    assert (tmp_path / "white_light.pdf").exists()
    assert BEST_FIT_COLOR in str(figure.axes[0].lines[-1].get_color())


def test_brightness_and_temperature_maps_save_files(tmp_path) -> None:
    longitude = np.linspace(-180.0, 180.0, 17)
    latitude = np.linspace(-90.0, 90.0, 9)
    brightness = 1.0 + 0.4 * np.cos(np.deg2rad(longitude))[None, :] * np.cos(
        np.deg2rad(latitude)
    )[:, None]
    brightness_figure = plot_brightness_map(
        longitude,
        latitude,
        brightness,
        metadata={"title": "Brightness"},
        output=tmp_path / "brightness",
    )
    temperature_figure = plot_temperature_map(
        longitude,
        latitude,
        np.full_like(brightness, _planck(4.0e-6, 2_700.0)),
        4.0e-6,
        metadata={"title": "Temperature"},
        output=tmp_path / "temperature",
    )
    assert len(brightness_figure.axes) >= 2
    assert len(temperature_figure.axes) >= 2
    assert temperature_figure.axes[-1].get_ylabel() == "Brightness temperature (K)"
    colormap = brightness_figure.axes[0].collections[0].cmap
    dark_rgb = np.asarray(colormap(0.0)[:3])
    bright_rgb = np.asarray(colormap(1.0)[:3])
    assert np.mean(bright_rgb) > np.mean(dark_rgb)
    for stem in ("brightness", "temperature"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()


def test_maps_and_longitude_profile_label_hotspot_comparison() -> None:
    longitude = np.linspace(-180.0, 180.0, 121)
    latitude = np.linspace(-90.0, 90.0, 61)
    brightness = 1.0 + np.cos(np.deg2rad(longitude - 7.49))[None, :] * np.cos(
        np.deg2rad(latitude)
    )[:, None]
    metadata = {"longitude_references": _longitude_references()}

    map_figure = plot_brightness_map(
        longitude,
        latitude,
        brightness,
        metadata=metadata,
    )
    profile_figure = plot_longitude_profile(
        longitude,
        np.mean(brightness, axis=0),
        metadata=metadata,
    )

    expected_labels = {reference["label"] for reference in _longitude_references()}
    for figure in (map_figure, profile_figure):
        axis = figure.axes[0]
        labels = {text.get_text() for text in axis.get_legend().get_texts()}
        assert expected_labels <= labels
        vertical_lines = {
            float(np.asarray(line.get_xdata(), dtype=float)[0])
            for line in axis.lines
            if np.asarray(line.get_xdata()).size >= 2
            and np.allclose(line.get_xdata(), np.asarray(line.get_xdata())[0])
        }
        assert {0.0, 7.49, 7.75} <= vertical_lines
        assert len(axis.containers) >= 2


def test_profile_recovery_and_model_comparison_save_files(tmp_path) -> None:
    longitude = np.linspace(-90.0, 90.0, 19)
    profile = np.exp(-0.5 * (longitude / 35.0) ** 2)
    profile_figure = plot_recovery_summary(
        np.array([10.0, -27.0]),
        np.array([12.0, -4.0]),
        intervals=np.array([[-15.0, 42.0], [-50.0, 40.0]]),
        delta_bic=np.array([-1.5, 4.0]),
        labels=["A", "B"],
        output=tmp_path / "recovery",
    )
    from robert_mapping.benchmark.diagnostic_plots import plot_longitude_profile

    plot_longitude_profile(
        longitude,
        profile,
        interval=(profile * 0.9, profile * 1.1),
        output=tmp_path / "profile",
    )
    model_figure = plot_model_comparison(
        longitude,
        1.0 - 1.0e-4 * profile,
        1.0 - 1.1e-4 * profile,
        metadata={"metrics": {"chi2": "1.02"}},
        output=tmp_path / "comparison",
    )
    assert len(profile_figure.axes) == 2
    assert len(model_figure.axes) == 2
    for stem in ("profile", "recovery", "comparison"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()


def test_latitude_diagnostic_plots_save_files(tmp_path) -> None:
    peak_longitude = np.array([24.0, 30.0, 36.0, 42.0])
    peak_latitude = np.array([-60.0, -20.0, 25.0, 65.0])
    peak_figure = plot_map_peak_posterior(
        peak_longitude,
        peak_latitude,
        metadata={"warning": "Warning: latitude is weakly constrained."},
        output=tmp_path / "map_peak",
    )
    asymmetry_figure = plot_north_south_asymmetry(
        np.array([-0.1, -0.02, 0.01, 0.08]),
        output=tmp_path / "asymmetry",
    )
    assert len(peak_figure.axes) == 2
    assert len(asymmetry_figure.axes) == 1
    for stem in ("map_peak", "asymmetry"):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
