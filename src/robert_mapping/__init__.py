"""A small, standalone eclipse-mapping interface.

The public API is deliberately light.  Configuration files are the primary
user interface, while the physics and inference modules can be used directly
by Python users when needed.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "ConfigError",
    "MappingConfig",
    "load_config",
    "write_resolved_config",
]

__version__ = "0.1.0"

from .config import ConfigError, MappingConfig, load_config, write_resolved_config

