"""Small backend helpers shared by the physics modules.

The production package uses JAX when callers pass JAX arrays.  Importing this
module does not require JAX, which keeps the reference NumPy implementation
usable in minimal environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np


try:  # pragma: no cover - the fallback is exercised on NumPy-only installs
    import jax
    import jax.numpy as jnp

    _JAX_ARRAY_TYPES = (jax.Array,)
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    _JAX_ARRAY_TYPES = ()


def is_jax_array(value: Any) -> bool:
    """Return ``True`` for a JAX array or tracer."""

    if jnp is None:
        return False
    if isinstance(value, _JAX_ARRAY_TYPES):
        return True
    # Tracers do not always inherit from jax.Array.  This check is intentionally
    # conservative and does not inspect values, so it is safe under jit.
    return type(value).__module__.startswith("jax.")


def xp_for(*values: Any):
    """Choose NumPy or JAX based on the first JAX-valued argument."""

    if any(is_jax_array(value) for value in values):
        if jnp is None:  # pragma: no cover
            raise RuntimeError("A JAX array was supplied but JAX is unavailable")
        return jnp
    return np


def asarray(value: Any, xp):
    """Convert an input using the selected backend without forcing a copy."""

    return xp.asarray(value)


def maybe_float(value: Any) -> float:
    """Convert a scalar to a Python float outside traced JAX code."""

    return float(np.asarray(value))

