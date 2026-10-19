"""The new scientific modes have explicit, strict YAML controls."""

import pytest

from robert_mapping.config import ConfigError, mapping_config_from_dict


def test_longitude_mode_has_explicit_signed_basis():
    config = mapping_config_from_dict({"map": {"representation": "longitude", "positive": False}})
    assert config.map.representation == "longitude"
    with pytest.raises(ConfigError, match="signed Gaussian Fourier"):
        mapping_config_from_dict({"map": {"representation": "longitude"}})


@pytest.mark.parametrize("values", [{"harmonic_degree": 11}, {"active_harmonic_indices": [0, 1]},
                                     {"entropy_penalty": 0.5},
                                     {"pixel_prior_mean_ppm": 5000, "pixel_prior_sd_ppm": 1000}])
def test_longitude_mode_rejects_settings_that_would_be_inactive(values):
    with pytest.raises(ConfigError):
        mapping_config_from_dict({"map": {"representation": "longitude", "positive": False, **values}})


def test_cross_validation_roundtrips_explicit_settings():
    values = {"n_folds": 7, "strategy": "contiguous", "null_model": "uniform", "prior_scale": 0.02}
    config = mapping_config_from_dict({"cross_validation": values})
    assert config.to_dict()["cross_validation"] == values


@pytest.mark.parametrize("values", [{"n_folds": 1}, {"n_folds": True}, {"strategy": "random"},
                                     {"null_model": "typo"}, {"prior_scale": 0}, {"fold": 3}])
def test_cross_validation_rejects_invalid_or_unknown_settings(values):
    with pytest.raises(ConfigError):
        mapping_config_from_dict({"cross_validation": values})
