"""Fit and report tests for latitude-flat longitude Fourier maps."""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from robert_mapping.benchmark.production_report import make_production_report
from robert_mapping.config import mapping_config_from_dict
from robert_mapping.data import load_light_curve
from robert_mapping.inference.numpyro_backend import NumpyroRun
from robert_mapping.inference.run import _secondary_design, run_fit


def _longitude_config(tmp_path: Path, *, sampler: str = "nuts"):
    time = 55934.292283 + 0.813474 / 2.0 + np.linspace(-0.04, 0.04, 24)
    np.save(tmp_path / "time.npy", time)
    np.save(tmp_path / "flux.npy", np.ones(time.size))
    np.save(tmp_path / "error.npy", np.full(time.size, 2.0e-3))
    return mapping_config_from_dict(
        {
            "project": {"name": "longitude-fit-test", "seed": 13},
            "data": {
                "time": str(tmp_path / "time.npy"),
                "flux": str(tmp_path / "flux.npy"),
                "flux_err": str(tmp_path / "error.npy"),
            },
            "map": {
                "representation": "longitude",
                "harmonic_degree": 1,
                "positive": False,
                "regularization": "none",
            },
            "model": {"integrate_exposure": False},
            "inference": {
                "sampler": sampler,
                "chains": 1,
                "warmup": 2,
                "draws": 6,
                "progress_bar": False,
            },
            "compute": {
                "max_cpus": 1,
                "threads": 1,
                "quadrature_radial": 8,
                "quadrature_azimuth": 32,
                "jit": False,
            },
            "output": {"directory": str(tmp_path / "results")},
        }
    )


def _fake_longitude_sampler(design, observed, sigma, **kwargs):
    del observed, sigma
    draws = 6
    coefficients = np.tile(np.array([0.005, 0.010, 0.001]), (draws, 1))
    coefficients[:, 0] += np.linspace(-2.0e-4, 2.0e-4, draws)
    coefficients[:, 1] += np.linspace(-3.0e-4, 3.0e-4, draws)
    stellar_flux = np.asarray(kwargs["stellar_flux"], dtype=float)
    flux = stellar_flux[None, :] + coefficients @ np.asarray(design).T
    return NumpyroRun(
        samples={"coefficients": coefficients, "flux": flux},
        extra_fields={"diverging": np.zeros(draws, dtype=bool)},
        sampler=None,
        grouped_samples={
            "coefficients": coefficients[None, :, :],
            "flux": flux[None, :, :],
        },
    )


def test_longitude_fit_saves_true_basis_keys_and_latitude_flat_report(
    tmp_path: Path, monkeypatch
) -> None:
    engine = importlib.import_module("robert_mapping.inference.run")
    monkeypatch.setattr(engine, "sample_harmonic_map", _fake_longitude_sampler)
    config = _longitude_config(tmp_path)

    result = run_fit(config)

    assert result.coefficients_path.name == "coefficients.npy"
    assert (result.output_directory / "longitude_coefficients.npy").exists()
    assert not (result.output_directory / "harmonic_coefficients.npy").exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["map_representation"] == "longitude"
    assert summary["parameterization"] == "longitude_fourier"
    assert summary["longitude_degree"] == 1
    assert summary["map_basis"]["name"] == "latitude-flat real Fourier basis"
    assert summary["coefficient_prior"]["distribution"].startswith(
        "independent Gaussian"
    )
    assert "harmonic_coefficient_mean" not in summary
    assert "active_harmonic_indices" not in summary

    with np.load(result.samples_path, allow_pickle=False) as samples:
        assert "longitude_coefficients" in samples
        assert "longitude_coefficients_by_chain" in samples
        assert "harmonic_coefficients" not in samples
        assert samples["longitude_coefficients"].shape == (6, 3)
        assert samples["longitude_coefficients_by_chain"].shape == (1, 6, 3)

    report = make_production_report(config)
    assert report["map_representation"] == "longitude"
    assert report["latitude_information"]["status"] == "fixed_by_model"
    assert report["latitude_information"]["peak_latitude_degrees"] is None
    assert report["map_peak_2d"] is None
    assert report["contrast"]["posterior_draw_with_negative_region_fraction"] == 1.0
    assert report["temperature"]["status"] == "unavailable"
    assert "bandpass" in report["temperature"]["reason"]
    assert (result.output_directory / "longitude_profile.pdf").is_file()
    assert (result.output_directory / "brightness_map_median.pdf").is_file()
    assert (result.output_directory / "posterior_trace.pdf").is_file()
    with np.load(result.output_directory / "posterior_maps.npz") as maps:
        np.testing.assert_allclose(
            maps["contrast_median"][0], maps["contrast_median"][-1]
        )
        assert "temperature_median_kelvin" not in maps


def test_exact_direct_harmonic_fit_uses_only_selected_coefficients(
    tmp_path: Path,
) -> None:
    config = _longitude_config(tmp_path, sampler="none")
    config = replace(
        config,
        map=replace(
            config.map,
            representation="direct_harmonics",
            harmonic_degree=1,
            active_harmonic_indices=(0, 1, 3),
        ),
    )

    result = run_fit(config)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    coefficients = np.load(result.coefficients_path)
    covariance = np.load(result.output_directory / "coefficient_covariance.npy")
    assert summary["active_harmonic_indices"] == [0, 1, 3]
    assert summary["n_parameters"] == 3
    assert summary["design_shape"] == [24, 4]
    assert coefficients.shape == (4,)
    assert coefficients[2] == 0.0
    assert covariance.shape == (4, 4)
    assert np.all(covariance[2, :] == 0.0)
    assert np.all(covariance[:, 2] == 0.0)


def test_longitude_gaussian_fit_recovers_a_known_hotspot_offset(
    tmp_path: Path,
) -> None:
    config = _longitude_config(tmp_path, sampler="none")
    time = 55934.292283 + np.linspace(0.0, 0.813474, 72, endpoint=False)
    np.save(tmp_path / "time.npy", time)
    np.save(tmp_path / "error.npy", np.full(time.size, 2.0e-5))
    np.save(tmp_path / "flux.npy", np.ones(time.size))

    curve = load_light_curve(config)
    bundle = _secondary_design(config, curve)
    injected_coefficients = np.array([0.005, 0.0025, 0.00125])
    np.save(
        tmp_path / "flux.npy",
        bundle.stellar_flux + bundle.map_design @ injected_coefficients,
    )

    result = run_fit(config)

    recovered = np.asarray(result.summary["coefficient_mean"], dtype=float)
    recovered_offset = np.rad2deg(np.arctan2(recovered[2], recovered[1]))
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["parameterization"] == "longitude_fourier"
    np.testing.assert_allclose(recovered, injected_coefficients, atol=2.0e-5)
    assert recovered_offset == pytest.approx(26.565, abs=0.25)


def test_longitude_fit_forwards_gp_priors_and_saves_noise_draws(
    tmp_path: Path, monkeypatch
) -> None:
    engine = importlib.import_module("robert_mapping.inference.run")
    captured: dict[str, object] = {}
    config = _longitude_config(tmp_path)
    config = replace(
        config,
        model=replace(
            config.model,
            noise_model="gp",
            gp_kernel="matern32",
            gp_amplitude_prior_scale_ppm=250.0,
            gp_timescale_prior_median_seconds=600.0,
            gp_timescale_prior_sigma_ln=0.7,
            gp_max_points=24,
        ),
    )

    def fake_gp_sampler(design, observed, sigma, **kwargs):
        captured.update(kwargs)
        run = _fake_longitude_sampler(design, observed, sigma, **kwargs)
        n_draws = run.samples["coefficients"].shape[0]
        gp_amplitude = np.full(n_draws, 80.0e-6)
        gp_timescale = np.full(n_draws, 700.0)
        samples = dict(run.samples)
        samples.update(
            {"gp_amplitude": gp_amplitude, "gp_timescale": gp_timescale}
        )
        grouped = dict(run.grouped_samples)
        grouped.update(
            {
                "gp_amplitude": gp_amplitude[None, :],
                "gp_timescale": gp_timescale[None, :],
            }
        )
        return NumpyroRun(
            samples=samples,
            extra_fields=run.extra_fields,
            sampler=None,
            grouped_samples=grouped,
        )

    monkeypatch.setattr(engine, "sample_harmonic_map", fake_gp_sampler)

    result = run_fit(config)

    assert captured["gp_kernel"] == "matern32"
    assert captured["gp_amplitude_prior_scale"] == pytest.approx(250.0e-6)
    assert captured["gp_timescale_prior_median"] == 600.0
    assert captured["gp_timescale_prior_sigma_ln"] == 0.7
    assert captured["gp_max_points"] == 24
    with np.load(result.samples_path, allow_pickle=False) as samples:
        assert samples["gp_amplitude"].shape == (6,)
        assert samples["gp_timescale"].shape == (6,)
        assert samples["gp_amplitude_by_chain"].shape == (1, 6)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["gp_kernel"] == "matern32"
    assert summary["gp_amplitude_mean"] == pytest.approx(80.0e-6)
    report = make_production_report(config)
    assert report["noise"]["model"] == "gp"
    assert report["noise"]["gp_kernel"] == "matern32"
    assert report["noise"]["posterior"]["gp_amplitude"]["mean"] == pytest.approx(
        80.0e-6
    )


@pytest.mark.skipif(
    importlib.util.find_spec("numpyro") is None,
    reason="NumPyro is not installed in this Python environment.",
)
def test_longitude_nuts_smoke_runs_the_real_sampler(tmp_path: Path) -> None:
    config = _longitude_config(tmp_path)
    config = replace(
        config,
        inference=replace(config.inference, warmup=2, draws=2, chains=1),
    )

    result = run_fit(config)

    assert result.samples_path is not None
    with np.load(result.samples_path, allow_pickle=False) as samples:
        assert samples["longitude_coefficients"].shape == (2, 3)
        assert "harmonic_coefficients" not in samples
