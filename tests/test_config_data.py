from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robert_mapping.config import ConfigError, load_config
from robert_mapping.data import DataError, load_light_curve


def test_load_separate_numpy_arrays(tmp_path: Path) -> None:
    np.save(tmp_path / "time.npy", np.arange(4, dtype=float))
    np.save(tmp_path / "flux.npy", np.ones(4))
    np.save(tmp_path / "err.npy", np.full(4, 0.01))
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "data:\n  time: time.npy\n  flux: flux.npy\n  flux_err: err.npy\n",
        encoding="utf-8",
    )
    curve = load_light_curve(load_config(config_path))
    assert curve.n_observations == 4
    assert curve.source == tmp_path / "time.npy"


def test_load_rejects_nonpositive_errors(tmp_path: Path) -> None:
    np.save(tmp_path / "time.npy", np.arange(2, dtype=float))
    np.save(tmp_path / "flux.npy", np.ones(2))
    np.save(tmp_path / "err.npy", np.array([0.01, 0.0]))
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "data:\n  time: time.npy\n  flux: flux.npy\n  flux_err: err.npy\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="positive"):
        load_light_curve(load_config(config_path))


def test_mixed_single_and_separate_inputs_are_rejected() -> None:
    with pytest.raises(ConfigError, match="file or separate"):
        load_config_from_dict_for_test()


def load_config_from_dict_for_test():
    # Keep this helper separate so the test shows the exact friendly failure.
    from robert_mapping.config import mapping_config_from_dict

    return mapping_config_from_dict(
        {
            "data": {
                "file": "lightcurve.csv",
                "time": "time.npy",
                "flux": "flux.npy",
                "flux_err": "err.npy",
            }
        }
    )


def test_load_raw_table_with_systematics_columns(tmp_path: Path) -> None:
    table = tmp_path / "raw.csv"
    table.write_text(
        "time,flux,flux_err,airmass,trace_x,visit\n"
        "0.0,1.001,0.0001,1.1,-0.2,0\n"
        "0.1,1.000,0.0001,1.2,0.0,0\n"
        "1.0,0.999,0.0001,1.3,0.2,1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "data:\n"
        "  file: raw.csv\n"
        "  format: csv\n"
        "systematics:\n"
        "  mode: multiplicative\n"
        "  polynomial_order: 2\n"
        "  exponential_ramp: true\n"
        "  ramp_timescale_hours: 1.5\n"
        "  regressor_columns: [airmass, trace_x]\n"
        "  segment_column: visit\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    curve = load_light_curve(config)
    assert config.systematics.mode == "multiplicative"
    assert config.systematics.polynomial_order == 2
    assert curve.regressor_names == ("airmass", "trace_x")
    assert curve.regressors is not None and curve.regressors.shape == (3, 2)
    assert curve.segments is not None
    assert np.array_equal(curve.segments, np.array([0.0, 0.0, 1.0]))


def test_load_raw_table_with_text_segment_labels(tmp_path: Path) -> None:
    """CSV systematics segments may use readable non-numeric visit labels."""

    table = tmp_path / "labelled.csv"
    table.write_text(
        "time,flux,flux_err,visit\n"
        "0.0,1.001,0.0001,visit_A\n"
        "0.1,1.000,0.0001,visit_A\n"
        "1.0,0.999,0.0001,visit_B\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "data:\n"
        "  file: labelled.csv\n"
        "  format: csv\n"
        "systematics:\n"
        "  mode: additive\n"
        "  segment_column: visit\n",
        encoding="utf-8",
    )

    curve = load_light_curve(load_config(config_path))
    assert curve.segments is not None
    assert curve.segments.tolist() == ["visit_A", "visit_A", "visit_B"]


def test_systematics_fit_offset_defaults_true_and_can_be_disabled() -> None:
    from robert_mapping.config import default_config, mapping_config_from_dict

    assert default_config().systematics.fit_offset is True
    config = mapping_config_from_dict(
        {
            "systematics": {
                "mode": "additive",
                "fit_offset": False,
                "polynomial_order": 1,
            }
        }
    )
    assert config.systematics.fit_offset is False


def test_npz_channel_index_selects_one_spectral_light_curve(tmp_path: Path) -> None:
    source = tmp_path / "spectroscopic.npz"
    np.savez(
        source,
        time=np.array([1.0, 2.0, 3.0]),
        flux=np.array([[1.0, 1.1], [0.9, 1.0], [1.1, 1.2]]),
        flux_err=np.array([[0.01, 0.02], [0.01, 0.02], [0.01, 0.02]]),
    )
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "data:\n"
        "  file: spectroscopic.npz\n"
        "  format: npz\n"
        "  channel_index: 1\n",
        encoding="utf-8",
    )
    curve = load_light_curve(load_config(config_path))
    np.testing.assert_allclose(curve.flux, [1.1, 1.0, 1.2])
    np.testing.assert_allclose(curve.flux_err, [0.02, 0.02, 0.02])


def test_npz_channel_index_rejects_out_of_range_column(tmp_path: Path) -> None:
    source = tmp_path / "spectroscopic.npz"
    np.savez(
        source,
        time=np.array([1.0, 2.0]),
        flux=np.ones((2, 2)),
        flux_err=np.full((2, 2), 0.01),
    )
    config_path = tmp_path / "run.yml"
    config_path.write_text(
        "data:\n"
        "  file: spectroscopic.npz\n"
        "  format: npz\n"
        "  channel_index: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(DataError, match="outside"):
        load_light_curve(load_config(config_path))
