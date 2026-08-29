"""Focused tests for the WASP-121b literature-input preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prepare_literature_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_literature_inputs", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wasp121_nirspec_preserves_models_and_metadata(tmp_path: Path) -> None:
    directory = tmp_path / "JWST-NIRSpec-G395H"
    source = directory / "source"
    source.mkdir(parents=True)
    for detector in ("nrs1", "nrs2"):
        data = np.array(
            [
                [1.0, 0.1, -0.1, 1.001, 0.0001, 0.0],
                [2.0, 0.2, -0.2, 1.002, 0.0002, 1.0],
                [3.0, 0.3, -0.3, 1.003, 0.0003, 1.0],
            ]
        )
        model = np.array([[2.0, 0.999, 1.002], [3.0, 0.998, 1.003]])
        np.savetxt(source / f"whitelc_data_{detector}.txt", data)
        np.savetxt(source / f"whitelc_model_{detector}.txt", model)

    report = MODULE._w121(directory)

    assert report["nrs1"]["kept_rows"] == 2
    with np.load(directory / "prepared" / "white_nrs1.npz") as archive:
        np.testing.assert_allclose(archive["time"], [2.0, 3.0])
        np.testing.assert_allclose(
            archive["published_systematics_model"], [0.999, 0.998]
        )
        assert float(archive["wavelength_min_micron"]) == 2.70
        assert float(archive["wavelength_max_micron"]) == 3.72


def test_wasp121_tess_converts_relative_hours_to_days(tmp_path: Path) -> None:
    directory = tmp_path / "TESS"
    source = directory / "source"
    source.mkdir(parents=True)
    np.savetxt(
        source / "lccurve.dat",
        np.array([[-0.2, 1.0, 0.001], [0.0, 0.99, 0.001], [0.2, 1.0, 0.001]]),
    )

    report = MODULE._wasp121_tess(directory)

    assert report["source_rows"] == 3
    with np.load(directory / "prepared" / "white_light_curve.npz") as archive:
        np.testing.assert_allclose(archive["time"], [-0.2 / 24.0, 0.0, 0.2 / 24.0])
        assert float(archive["exposure_seconds"]) == 720.0
