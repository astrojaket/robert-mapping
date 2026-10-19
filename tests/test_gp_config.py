"""GP configuration must select a real kernel and reject invalid runs early."""

import numpy as np
import pytest

from robert_mapping.cli import main
from robert_mapping.config import ConfigError, mapping_config_from_dict


def test_gp_settings_roundtrip():
    model = {"noise_model": "gp", "gp_kernel": "matern32",
             "gp_amplitude_prior_scale_ppm": 80.0,
             "gp_timescale_prior_median_seconds": 300.0,
             "gp_timescale_prior_sigma_ln": 0.5, "gp_max_points": 300}
    config = mapping_config_from_dict({"model": model})
    assert all(config.to_dict()["model"][key] == value for key, value in model.items())


@pytest.mark.parametrize("settings", [{"gp_kernel": "unused"}, {"gp_max_points": 0},
                                        {"gp_amplitude_prior_scale_ppm": 0},
                                        {"gp_timescale_prior_median_seconds": -1},
                                        {"fit_white_jitter": True}, {"likelihood": "student_t"}])
def test_gp_rejects_unsupported_or_inactive_settings(settings):
    with pytest.raises(ConfigError):
        mapping_config_from_dict({"model": {"noise_model": "gp", **settings}})


def test_gp_size_limit_is_checked_by_dry_run(tmp_path, capsys):
    for name, data in (("time", np.linspace(0, 1, 8)), ("flux", np.ones(8)),
                       ("error", np.full(8, 0.001))):
        np.save(tmp_path / f"{name}.npy", data)
    path = tmp_path / "gp.yml"
    path.write_text("data:\n  time: time.npy\n  flux: flux.npy\n  flux_err: error.npy\n"
                    "model:\n  noise_model: gp\n  gp_max_points: 4\n")
    assert main(["fit", str(path), "--dry-run"]) == 2
    assert "above model.gp_max_points=4" in capsys.readouterr().err
