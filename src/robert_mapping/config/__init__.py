"""Configuration models and YAML helpers for ``robert-mapping``.

The configuration layer has three goals:

* make the common workflow readable without Python knowledge;
* fail early with a useful message when a setting is misspelled; and
* produce a resolved copy of every run for reproducibility.

Paths in a loaded configuration are resolved relative to the YAML file.  The
``to_dict`` method converts them back to strings so that YAML and JSON writers
do not need special handling.
"""

from .models import (
    ConfigError,
    ComputeConfig,
    DataConfig,
    InferenceConfig,
    MapConfig,
    MappingConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
    RecoveryConfig,
    SystemConfig,
    SystematicsCandidateConfig,
    SystematicsConfig,
    SystematicsSelectionConfig,
    default_config,
    load_config,
    mapping_config_from_dict,
    write_config,
    write_json_summary,
    write_resolved_config,
)

__all__ = [
    "ConfigError",
    "ComputeConfig",
    "DataConfig",
    "InferenceConfig",
    "MapConfig",
    "MappingConfig",
    "ModelConfig",
    "OutputConfig",
    "ProjectConfig",
    "RecoveryConfig",
    "SystemConfig",
    "SystematicsCandidateConfig",
    "SystematicsConfig",
    "SystematicsSelectionConfig",
    "default_config",
    "load_config",
    "mapping_config_from_dict",
    "write_config",
    "write_json_summary",
    "write_resolved_config",
]
