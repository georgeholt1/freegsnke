"""
Configuration and device management for the FreeGSNKE JAX backend.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.
"""

import logging
import os

logger = logging.getLogger("freegsnke.jax")

# Ensure 64-bit precision is enabled for physics calculations
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

_GLOBAL_BACKEND = "numpy"


def is_jax_available() -> bool:
    """Return True if JAX is installed and available."""
    return _JAX_AVAILABLE


def get_jax_devices():
    """Return available JAX devices."""
    if not _JAX_AVAILABLE:
        return []
    import jax
    return jax.devices()


def get_default_device():
    """Return default JAX device (GPU if available, else CPU)."""
    if not _JAX_AVAILABLE:
        return None
    import jax
    devices = jax.devices()
    for d in devices:
        if d.platform == "gpu" or d.platform == "cuda":
            return d
    return devices[0] if devices else None


def set_backend(backend: str):
    """
    Set the global default backend for FreeGSNKE solvers.

    Parameters
    ----------
    backend : str
        Backend to use: 'numpy' or 'jax'.
    """
    global _GLOBAL_BACKEND
    backend = backend.lower()
    if backend not in ("numpy", "jax"):
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'numpy', 'jax'")
    if backend == "jax" and not _JAX_AVAILABLE:
        raise ImportError("JAX is not installed. Please install jax and jaxlib to use backend='jax'.")
    _GLOBAL_BACKEND = backend


def get_backend() -> str:
    """
    Get the global default backend for FreeGSNKE solvers.

    Returns
    -------
    str
        Currently active global backend ('numpy' or 'jax').
    """
    return _GLOBAL_BACKEND
