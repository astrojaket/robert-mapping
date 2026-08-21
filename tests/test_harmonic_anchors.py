"""Focused tests for the rank-revealing harmonic-anchor parameterization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robert_mapping.config import mapping_config_from_dict
from robert_mapping.inference.numpyro_backend import NumpyroRun
from robert_mapping.inference.run import run_fit
from robert_mapping.physics import rank_revealing_anchor_transform


def test_anchor_transform_is_deterministic_and_round_trips_harmonics() -> None:
    first = rank_revealing_anchor_transform(2)
    second = rank_revealing_anchor_transform(2)

    assert first.rank == 9
    assert first.anchor_indices.shape == (9,)
    assert np.array_equal(first.anchor_indices, second.anchor_indices)
    assert np.allclose(first.anchor_to_harmonics, second.anchor_to_harmonics)
    assert first.condition_number < 10.0
    assert np.all(np.isfinite(first.anchor_longitude_degrees))
    assert np.all(np.isfinite(first.anchor_latitude_degrees))

    coefficients = np.linspace(-0.2, 0.2, 9)
    anchors = first.evaluation_matrix[first.anchor_indices] @ coefficients
    recovered = first.anchor_to_harmonics @ anchors
    assert np.allclose(recovered, coefficients, rtol=0.0, atol=1.0e-12)

    degree_three = rank_revealing_anchor_transform(3)
    assert degree_three.rank == 16
    assert degree_three.anchor_indices.shape == (16,)
    assert degree_three.anchor_to_harmonics.shape == (16, 16)


def _config(tmp_path: Path, representation: str):
    time = np.linspace(55934.45, 55935.05, 24)
    np.save(tmp_path / "time.npy", time)
    np.save(tmp_path / "flux.npy", np.ones(time.size))
    np.save(tmp_path / "error.npy", np.full(time.size, 2.0e-3))
    return mapping_config_from_dict(
        {
            "project": {"name": "harmonic-anchor-test", "seed": 11},
            "data": {
                "time": str(tmp_path / "time.npy"),
                "flux": str(tmp_path / "flux.npy"),
                "flux_err": str(tmp_path / "error.npy"),
            },
            "map": {
                "representation": representation,
                "harmonic_degree": 2,
                "n_pixels": 16,
            },
            "model": {"integrate_exposure": False},
            "inference": {
                "sampler": "nuts",
                "chains": 1,
                "warmup": 2,
                "draws": 3,
                "progress_bar": False,
            },
            "compute": {"max_cpus": 2, "threads": 2},
            "output": {"directory": str(tmp_path / f"results_{representation}")},
        }
    )


def test_pixel_fit_selects_anchors_only_for_harmonic_representation(
    tmp_path: Path, monkeypatch
) -> None:
    import importlib

    engine = importlib.import_module("robert_mapping.inference.run")
    captured: dict[str, object] = {}

    def fake_sampler(design, observed, sigma, **kwargs):
        del observed, sigma
        captured.setdefault("design_shapes", []).append(tuple(design.shape))
        n_samples = 3
        n_parameters = design.shape[1]
        n_observations = design.shape[0]
        return NumpyroRun(
            samples={
                "pixels": np.full((n_samples, n_parameters), 1.0e-3),
                "flux": np.ones((n_samples, n_observations)),
                "entropy": np.zeros(n_samples),
            },
            extra_fields={"diverging": np.zeros(n_samples, dtype=bool)},
            sampler=None,
        )

    monkeypatch.setattr(engine, "sample_positive_map", fake_sampler)

    harmonic_result = run_fit(_config(tmp_path, "harmonics"))
    harmonic_summary = json.loads(
        harmonic_result.summary_path.read_text(encoding="utf-8")
    )
    assert harmonic_summary["parameterization"] == "harmonic_anchors"
    assert harmonic_summary["design_shape"][1] == 9
    assert harmonic_summary["n_pixels"] == 9
    assert len(harmonic_summary["anchor_indices"]) == 9
    assert len(harmonic_summary["anchor_longitude_degrees"]) == 9
    assert len(harmonic_summary["anchor_latitude_degrees"]) == 9
    assert np.asarray(harmonic_summary["anchor_coordinates_degrees"]).shape == (9, 2)
    assert harmonic_summary["anchor_rank"] == 9
    assert harmonic_summary["anchor_condition_number"] < 10.0
    harmonic_samples = np.load(harmonic_result.samples_path)
    assert harmonic_samples["pixels"].shape == (3, 9)
    assert harmonic_samples["harmonic_coefficients"].shape == (3, 9)

    pixel_result = run_fit(_config(tmp_path, "pixels"))
    pixel_summary = json.loads(pixel_result.summary_path.read_text(encoding="utf-8"))
    assert pixel_summary["parameterization"] == "pixels"
    assert pixel_summary["design_shape"][1] == 16
    assert pixel_summary["n_pixels"] == 16
    assert pixel_summary["anchor_indices"] == []
    assert pixel_summary["anchor_coordinates_degrees"] == []
    assert pixel_summary["anchor_rank"] is None
    assert pixel_summary["anchor_condition_number"] is None
    pixel_samples = np.load(pixel_result.samples_path)
    assert pixel_samples["pixels"].shape == (3, 16)
    assert captured["design_shapes"] == [(24, 9), (24, 16)]
