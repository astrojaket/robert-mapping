from __future__ import annotations

from pathlib import Path

import pytest

from robert_mapping.config import ConfigError, default_config, load_config, mapping_config_from_dict, write_resolved_config
from robert_mapping.cli import main


def test_default_config_has_safe_quick_profile() -> None:
    config = default_config()
    assert config.inference.chains == 2
    assert config.inference.warmup == 150
    assert config.inference.draws == 150
    assert config.compute.max_cpus == 3
    assert config.compute.threads == 2


def test_unknown_keys_have_an_actionable_message() -> None:
    with pytest.raises(ConfigError, match="unknown setting.*Allowed settings"):
        mapping_config_from_dict({"project": {"naem": "typo"}})


def test_nonzero_eccentricity_is_rejected_until_physics_is_implemented() -> None:
    with pytest.raises(ConfigError, match="eccentric-orbit physics is not implemented"):
        mapping_config_from_dict({"system": {"eccentricity": 0.1}})


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yml"
    path.write_text("project:\n  name: first\n  name: second\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Duplicate setting"):
        load_config(path)


def test_relative_paths_resolve_from_yaml_location(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "run.yml"
    config_path.parent.mkdir()
    config_path.write_text(
        "data:\n  time: ../time.npy\n  flux: ../flux.npy\n  flux_err: ../err.npy\n"
        "output:\n  directory: ../results\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.data.time == (config_path.parent / "../time.npy").resolve()
    assert config.output.directory == (config_path.parent / "../results").resolve()


def test_resolved_config_contains_absolute_paths(tmp_path: Path) -> None:
    path = tmp_path / "run.yml"
    path.write_text("data:\n  time: t.npy\n  flux: f.npy\n  flux_err: e.npy\n", encoding="utf-8")
    config = load_config(path)
    output = write_resolved_config(config, tmp_path / "out" / "resolved.yml")
    text = output.read_text(encoding="utf-8")
    assert str(tmp_path) in text
    assert "resolved" not in text.split("output:", 1)[1].splitlines()[0]


def test_init_and_validate_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.yml"
    assert main(["init", str(path), "--template", "minimal"]) == 0
    assert path.exists()
    assert main(["validate", str(path)]) == 0
    output = capsys.readouterr().out
    assert "Configuration is valid" in output
