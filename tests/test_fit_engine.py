"""Small integration tests for the configuration-driven fit engine."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np

from robert_mapping.config import SystematicsConfig, mapping_config_from_dict
from robert_mapping.inference import run_fit
from robert_mapping.inference.numpyro_backend import NumpyroRun


def _config(tmp_path: Path, *, sampler: str, n_pixels: int = 6):
    time = np.linspace(55934.45, 55935.05, 24)
    flux = np.ones(time.size, dtype=float)
    error = np.full(time.size, 2.0e-3, dtype=float)
    np.save(tmp_path / "time.npy", time)
    np.save(tmp_path / "flux.npy", flux)
    np.save(tmp_path / "error.npy", error)
    return mapping_config_from_dict(
        {
            "project": {"name": "fit-engine-test", "seed": 7},
            "data": {
                "time": str(tmp_path / "time.npy"),
                "flux": str(tmp_path / "flux.npy"),
                "flux_err": str(tmp_path / "error.npy"),
            },
            "map": {
                "representation": "pixels",
                "harmonic_degree": 1,
                "n_pixels": n_pixels,
            },
            "model": {"integrate_exposure": False},
            "inference": {
                "sampler": sampler,
                "chains": 3,
                "warmup": 3,
                "draws": 5,
                "progress_bar": False,
            },
            "compute": {"max_cpus": 3, "threads": 3},
            "output": {"directory": str(tmp_path / "results")},
        }
    )


def test_exact_harmonic_fit_writes_machine_readable_outputs(tmp_path: Path) -> None:
    result = run_fit(_config(tmp_path, sampler="none"))

    assert result.status == "complete"
    assert result.sampler == "none"
    assert result.samples_path is None
    assert result.summary_path.name == "fit_summary.json"
    assert result.coefficients_path.exists()
    assert (result.output_directory / "coefficient_covariance.npy").exists()
    assert (result.output_directory / "model_flux.npy").exists()
    assert (result.output_directory / "residuals.npy").exists()
    assert np.load(result.coefficients_path).shape == (4,)
    summary = result.summary_path.read_text(encoding="utf-8")
    assert '"theta0_radians": 3.141592653589793' in summary
    assert '"sampler": "none"' in summary


def test_nuts_fit_uses_configured_small_run_and_saves_samples(tmp_path: Path, monkeypatch) -> None:
    # The physics operator is exercised by run_fit.  Replace only the
    # sampler so this unit test stays fast and deterministic on all CI CPUs.
    import importlib

    engine = importlib.import_module("robert_mapping.inference.run")
    config = _config(tmp_path, sampler="nuts")
    calls: dict[str, object] = {}

    def fake_sampler(design, observed, sigma, **kwargs):
        calls.update({key: int(kwargs[key]) for key in ("chains", "warmup", "draws")})
        calls["likelihood"] = kwargs["likelihood"]
        nsamples = 3
        npixels = design.shape[1]
        ntime = design.shape[0]
        pixels = np.full((nsamples, npixels), 1.0e-3)
        return NumpyroRun(
            samples={
                "pixels": pixels,
                "flux": np.ones((nsamples, ntime)),
                "entropy": np.zeros(nsamples),
            },
            extra_fields={"diverging": np.zeros(nsamples, dtype=bool)},
            sampler=None,
        )

    monkeypatch.setattr(engine, "sample_positive_map", fake_sampler)
    result = run_fit(config)

    assert calls == {
        "chains": 3,
        "warmup": 3,
        "draws": 5,
        "likelihood": "gaussian",
    }
    assert result.status == "complete"
    assert result.sampler == "nuts"
    assert result.samples_path is not None and result.samples_path.exists()
    samples = np.load(result.samples_path)
    assert samples["pixels"].shape == (3, 6)
    assert samples["harmonic_coefficients"].shape == (3, 4)
    assert np.load(result.coefficients_path).shape == (6,)


def test_nuts_fit_passes_multiplicative_ramp_systematics(tmp_path: Path, monkeypatch) -> None:
    import importlib

    engine = importlib.import_module("robert_mapping.inference.run")
    config = _config(tmp_path, sampler="nuts")
    config = replace(
        config,
        model=replace(config.model, likelihood="student_t", student_t_nu=5.0),
        systematics=SystematicsConfig(
            mode="multiplicative",
            polynomial_order=1,
            exponential_ramp=True,
            ramp_timescale_hours=2.0,
        ),
    )
    captured: dict[str, object] = {}

    def fake_sampler(design, observed, sigma, **kwargs):
        captured.update(kwargs)
        nsamples = 3
        npixels = design.shape[1]
        ntime = design.shape[0]
        nuisance = np.asarray(kwargs["systematics_design"])
        coefficients = np.zeros((nsamples, nuisance.shape[1]))
        return NumpyroRun(
            samples={
                "pixels": np.full((nsamples, npixels), 1.0e-3),
                "flux": np.ones((nsamples, ntime)),
                "entropy": np.zeros(nsamples),
                "systematics_coefficients": coefficients,
                "systematics_model": coefficients @ nuisance.T,
            },
            extra_fields={"diverging": np.zeros(nsamples, dtype=bool)},
            sampler=None,
        )

    monkeypatch.setattr(engine, "sample_positive_map", fake_sampler)
    result = run_fit(config)

    assert captured["systematics_mode"] == "multiplicative"
    assert captured["likelihood"] == "student_t"
    assert captured["student_t_nu"] == 5.0
    nuisance = np.asarray(captured["systematics_design"])
    assert nuisance.shape == (24, 3)
    saved = np.load(result.samples_path)
    assert saved["systematics_coefficients"].shape == (3, 3)


def test_nuts_fit_passes_offset_setting_to_systematics_design(tmp_path: Path, monkeypatch) -> None:
    import importlib

    engine = importlib.import_module("robert_mapping.inference.run")
    config = _config(tmp_path, sampler="nuts")
    config = replace(
        config,
        systematics=SystematicsConfig(
            mode="additive",
            fit_offset=False,
            polynomial_order=1,
        ),
    )
    captured: dict[str, object] = {}

    def fake_sampler(design, observed, sigma, **kwargs):
        nuisance = np.asarray(kwargs["systematics_design"])
        captured["nuisance"] = nuisance
        nsamples = 2
        ntime = design.shape[0]
        return NumpyroRun(
            samples={
                "pixels": np.full((nsamples, design.shape[1]), 1.0e-3),
                "flux": np.ones((nsamples, ntime)),
                "entropy": np.zeros(nsamples),
                "systematics_coefficients": np.zeros((nsamples, nuisance.shape[1])),
                "systematics_model": np.zeros((nsamples, ntime)),
            },
            extra_fields={"diverging": np.zeros(nsamples, dtype=bool)},
            sampler=None,
        )

    monkeypatch.setattr(engine, "sample_positive_map", fake_sampler)
    run_fit(config)

    nuisance = np.asarray(captured["nuisance"])
    assert nuisance.shape == (24, 1)
    assert np.allclose(nuisance[:, 0], (np.arange(24) - 11.5) / 11.5)
