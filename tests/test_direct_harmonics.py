"""Focused fit-engine tests for direct harmonic coefficient sampling."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robert_mapping.config import mapping_config_from_dict
from robert_mapping.inference.numpyro_backend import NumpyroRun
from robert_mapping.inference.run import run_fit


def _direct_config(tmp_path: Path):
    time = np.linspace(55934.45, 55935.05, 24)
    np.save(tmp_path / "time.npy", time)
    np.save(tmp_path / "flux.npy", np.ones(time.size))
    np.save(tmp_path / "error.npy", np.full(time.size, 2.0e-3))
    config = mapping_config_from_dict(
        {
            "project": {"name": "direct-harmonic-test", "seed": 13},
            "data": {
                "time": str(tmp_path / "time.npy"),
                "flux": str(tmp_path / "flux.npy"),
                "flux_err": str(tmp_path / "error.npy"),
            },
            "map": {"harmonic_degree": 2, "n_pixels": 16},
            "model": {"integrate_exposure": False},
            "inference": {
                "sampler": "nuts",
                "chains": 2,
                "warmup": 2,
                "draws": 4,
                "progress_bar": False,
            },
            "compute": {"max_cpus": 2, "threads": 2},
            "output": {"directory": str(tmp_path / "results")},
        }
    )
    return replace(config, map=replace(config.map, representation="direct_harmonics"))


def test_direct_harmonic_fit_writes_harmonic_only_samples(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    engine = importlib.import_module("robert_mapping.inference.run")
    captured: dict[str, object] = {}

    def fake_sampler(design, observed, sigma, **kwargs):
        del observed, sigma
        captured["design_shape"] = tuple(design.shape)
        captured["prior_mean"] = np.asarray(kwargs["coefficient_prior_mean"])
        captured["prior_sigma"] = np.asarray(kwargs["coefficient_prior_sigma"])
        captured["systematics_design"] = kwargs["systematics_design"]
        n_samples = 8
        n_chains = 2
        draws = 4
        coefficients = np.tile(
            np.array([0.005, 0.001, -0.001, 0.0005, 0.0, 0.0, 0.0, 0.0, 0.0]),
            (n_samples, 1),
        ) + np.arange(n_samples, dtype=float)[:, None] * 1.0e-6
        stellar_flux = np.asarray(kwargs["stellar_flux"])
        flux = stellar_flux[None, :] + coefficients @ np.asarray(design).T
        return NumpyroRun(
            samples={"coefficients": coefficients, "flux": flux},
            extra_fields={"diverging": np.zeros(n_samples, dtype=bool)},
            sampler=None,
            grouped_samples={
                "coefficients": coefficients.reshape(n_chains, draws, -1),
                "flux": flux.reshape(n_chains, draws, -1),
            },
        )

    monkeypatch.setattr(engine, "sample_harmonic_map", fake_sampler)
    result = run_fit(_direct_config(tmp_path))

    assert captured["design_shape"] == (24, 9)
    assert captured["systematics_design"] is None
    assert np.allclose(
        captured["prior_mean"], np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert np.isclose(captured["prior_sigma"][0], 0.0025)
    assert np.allclose(captured["prior_sigma"][1:], 0.005 * 0.75)

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["parameterization"] == "direct_harmonics"
    assert summary["positivity_policy"] == "diagnostic_only"
    assert summary["n_parameters"] == 9
    assert summary["design_shape"] == [24, 9]
    assert summary["maximum_rhat"] is not None
    assert summary["minimum_effective_sample_size"] is not None

    archive = np.load(result.samples_path)
    assert "pixels" not in archive.files
    assert archive["harmonic_coefficients"].shape == (8, 9)
    assert archive["harmonic_coefficients_by_chain"].shape == (2, 4, 9)
    assert archive["flux"].shape == (8, 24)
    assert archive["flux_by_chain"].shape == (2, 4, 24)
    assert np.load(result.coefficients_path).shape == (9,)
    assert not (result.output_directory / "pixel_coefficients.npy").exists()


def test_direct_harmonic_fit_passes_centered_times_for_ou_noise(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    engine = importlib.import_module("robert_mapping.inference.run")
    config = _direct_config(tmp_path)
    config = replace(
        config,
        model=replace(config.model, noise_model="ou"),
    )
    captured: dict[str, object] = {}

    def fake_sampler(design, observed, sigma, **kwargs):
        del observed, sigma
        captured.update(kwargs)
        n_samples = 4
        coefficients = np.zeros((n_samples, design.shape[1]))
        flux = np.asarray(kwargs["stellar_flux"])[None, :] + coefficients @ np.asarray(design).T
        return NumpyroRun(
            samples={
                "coefficients": coefficients,
                "flux": flux,
                "ou_amplitude": np.full(n_samples, 50.0e-6),
                "ou_timescale": np.full(n_samples, 900.0),
                "jitter": np.full(n_samples, 20.0e-6),
            },
            extra_fields={"diverging": np.zeros(n_samples, dtype=bool)},
            sampler=None,
        )

    monkeypatch.setattr(engine, "sample_harmonic_map", fake_sampler)
    result = run_fit(config)

    times = np.asarray(captured["time_seconds"])
    assert times.shape == (24,)
    assert np.isclose(times[0], 0.0)
    assert np.all(np.diff(times) >= 0.0)
    assert captured["noise_model"] == "ou"
    assert np.isclose(captured["ou_amplitude_prior_scale"], 100.0e-6)
    assert np.isclose(captured["ou_timescale_prior_median"], 900.0)
    assert np.isclose(captured["jitter_prior_scale"], 100.0e-6)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["noise_model"] == "ou"
    assert np.isclose(summary["ou_amplitude_mean"], 50.0e-6)
    archive = np.load(result.samples_path)
    assert archive["ou_timescale"].shape == (4,)
