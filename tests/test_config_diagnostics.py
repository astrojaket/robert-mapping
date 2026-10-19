from __future__ import annotations

import pytest

from robert_mapping.config import ConfigError, default_config, mapping_config_from_dict
from robert_mapping.config.diagnostics import (
    configuration_diagnostics,
    validate_operation,
)


def test_default_cross_validate_setting_is_reported_as_inactive() -> None:
    messages = configuration_diagnostics(default_config())

    assert any(
        "map.regularization: cross_validate is not active" in message
        for message in messages
    )


def test_entropy_penalty_is_not_controlled_by_regularization_mode() -> None:
    config = mapping_config_from_dict(
        {"map": {"regularization": "none", "entropy_penalty": 0.5}}
    )

    messages = configuration_diagnostics(config)

    assert any(
        "map.entropy_penalty is applied even when map.regularization" in message
        for message in messages
    )


@pytest.mark.parametrize(
    ("representation", "positive", "message"),
    [
        (
            "pixels",
            False,
            "map.positive: false does not make pixel or harmonic-anchor maps signed",
        ),
        (
            "direct_harmonics",
            True,
            "map.positive: true does not constrain direct_harmonics coefficients",
        ),
    ],
)
def test_positive_setting_is_checked_against_representation(
    representation: str, positive: bool, message: str
) -> None:
    config = mapping_config_from_dict(
        {"map": {"representation": representation, "positive": positive}}
    )

    assert any(message in item for item in configuration_diagnostics(config))


def test_stale_model_fields_report_only_nondefault_choices() -> None:
    defaults = configuration_diagnostics(default_config())
    config = mapping_config_from_dict(
        {
            "model": {
                "null_model": "uniform_disk",
                "mapped_model": "eclipse_map",
                "fourier_degree": 4,
                "fit_baseline": True,
                "fit_ramp": True,
            },
            "system": {"argument_of_periastron_degrees": 45.0},
        }
    )
    messages = configuration_diagnostics(config)

    assert not any("model.null_model" in item for item in defaults)
    assert not any("model.mapped_model" in item for item in defaults)
    assert any("model.null_model is accepted but does not select" in item for item in messages)
    assert any("model.mapped_model is accepted but does not select" in item for item in messages)
    assert any("model.fourier_degree does not affect fit" in item for item in messages)
    assert any("model.fit_baseline does not add a baseline" in item for item in messages)
    assert any("model.fit_ramp does not add a ramp in fit" in item for item in messages)
    assert any("argument_of_periastron_degrees has no effect" in item for item in messages)


def test_recovery_reports_ramp_scope_and_ignores_sampler_settings() -> None:
    config = mapping_config_from_dict(
        {
            "model": {"fit_ramp": True},
            "inference": {"chains": 1},
            "recovery": {"enabled": True},
        }
    )

    messages = configuration_diagnostics(config, "recover")

    assert any("adds a ramp to the recovery profile grid only" in item for item in messages)
    assert any("NUTS settings are ignored by recovery" in item for item in messages)


def test_map_sampler_reports_nuts_only_settings_and_map_priors() -> None:
    config = mapping_config_from_dict(
        {
            "inference": {"sampler": "map"},
            "model": {"fit_error_scale": True, "fit_white_jitter": False},
        }
    )

    messages = configuration_diagnostics(config)

    assert any("does not use NUTS settings" in item for item in messages)
    assert any("map.positive and pixel priors do not apply" in item for item in messages)
    assert any("model.fit_error_scale is ignored by the map sampler" in item for item in messages)


def test_recovery_reports_ignored_error_scaling_without_duplicate_messages() -> None:
    config = mapping_config_from_dict(
        {
            "model": {"fit_error_scale": True},
            "recovery": {"enabled": True},
        }
    )

    messages = configuration_diagnostics(config, "recover")

    assert sum("error-scale and white-jitter settings are ignored by recovery" in item for item in messages) == 1


def test_fit_rejects_synthetic_input_before_loading_files() -> None:
    config = mapping_config_from_dict({"data": {"kind": "synthetic"}})

    with pytest.raises(ConfigError, match="synthetic input is only supported by recovery"):
        validate_operation(config, "fit")


@pytest.mark.parametrize(
    "raw, message",
    [
        (
            {"inference": {"sampler": "map"}, "model": {"likelihood": "student_t"}},
            "map supports only model.likelihood: gaussian",
        ),
        (
            {"inference": {"sampler": "none"}, "model": {"noise_model": "ou"}},
            "map does not support correlated noise",
        ),
        (
            {
                "inference": {"sampler": "map"},
                "systematics": {"mode": "additive"},
            },
            "map does not support joint systematics",
        ),
        (
            {"inference": {"sampler": "nuts", "warmup": 0}},
            "nuts requires at least one warmup step",
        ),
    ],
)
def test_fit_rejects_backend_combinations_that_fail_at_runtime(
    raw: dict[str, object], message: str
) -> None:
    config = mapping_config_from_dict(raw)

    with pytest.raises(ConfigError, match=message):
        validate_operation(config, "fit")


def test_recovery_rejects_disabled_and_incompatible_cases() -> None:
    with pytest.raises(ConfigError, match="recovery.enabled: true"):
        validate_operation(default_config(), "recover")

    config = mapping_config_from_dict(
        {
            "data": {"kind": "synthetic"},
            "recovery": {"enabled": True, "case": "wasp178b"},
        }
    )
    with pytest.raises(ConfigError, match="wasp178b loads a real light curve"):
        validate_operation(config, "recover")


def test_systematics_selection_rejects_synthetic_input() -> None:
    config = mapping_config_from_dict({"data": {"kind": "synthetic"}})

    with pytest.raises(ConfigError, match="not supported by systematics selection"):
        validate_operation(config, "select-systematics")


def test_synthetic_recovery_case_accepts_synthetic_input() -> None:
    config = mapping_config_from_dict(
        {
            "data": {"kind": "synthetic"},
            "recovery": {"enabled": True, "case": "synthetic_matrix"},
        }
    )

    validate_operation(config, "recover")
