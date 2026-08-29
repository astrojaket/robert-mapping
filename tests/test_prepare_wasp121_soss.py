from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_wasp121_soss import _bjd_tdb, prepare_wasp121_soss  # noqa: E402


def _write_inputs(source: Path) -> None:
    source.mkdir()
    time = np.array([60243.0, 60243.0004, 60243.0008])
    error1 = np.full(3, 4.0e-5)
    error2 = np.full(3, 7.0e-5)
    np.savetxt(
        source / "W-121b-exoTEDRF-WLC-o1.csv",
        np.column_stack((time, [1.0, 1.001, 0.999], error1)),
        delimiter=",",
        header="time [BJD],Normalized WLC,Error",
        comments="",
    )
    np.savetxt(
        source / "W-121b-exoTEDRF-WLC-o2.csv",
        np.column_stack((time, [1.0, 1.002, 0.998], error2)),
        delimiter=",",
        header="time [BJD],Normalized WLC,Error",
        comments="",
    )
    np.save(source / "detrended_flux.npy", np.vstack(([1.0, 1.0009, 0.999], error1)))
    np.save(
        source / "detrended_flux_o2-new.npy",
        np.vstack(([1.0, 1.0018, 0.998], error2)),
    )


def test_bmjd_is_converted_to_bjd() -> None:
    converted, label = _bjd_tdb(np.array([60243.0]))
    assert converted[0] == pytest.approx(2460243.5)
    assert "mislabeled" in label


def test_prepare_keeps_alternative_as_comparison(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    _write_inputs(source)
    audit = prepare_wasp121_soss(source, output)
    assert audit["state"] == "exploratory_processed_input"
    assert audit["products"]["order1"]["rows"] == 3
    with np.load(output / "white_order1.npz", allow_pickle=False) as archive:
        assert archive["time"][0] == pytest.approx(2460243.5)
        np.testing.assert_allclose(archive["alternative_detrended_flux"], [1.0, 1.0009, 0.999])


def test_unrecognized_time_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="not recognized"):
        _bjd_tdb(np.array([123.0]))
