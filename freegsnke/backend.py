"""
Backend configuration and detection for FreeGSNKE.

Provides seamless switching between the native high-performance Rust backend
and the reference pure-Python implementation.
"""

import os
import logging

logger = logging.getLogger("freegsnke.backend")

_RUST_AVAILABLE = False
_RUST_MODULE = None

try:
    from freegsnke import _freegsnke_rs
    _RUST_AVAILABLE = True
    _RUST_MODULE = _freegsnke_rs
except ImportError as e:
    logger.debug(f"Rust backend not available: {e}")
    _RUST_AVAILABLE = False
    _RUST_MODULE = None

# Configure default backend via environment variable or availability
_env_backend = os.environ.get("FREEGSNKE_BACKEND", "auto").strip().lower()

if _env_backend == "rust":
    if not _RUST_AVAILABLE:
        raise RuntimeError("FREEGSNKE_BACKEND=rust was requested, but the Rust backend (_freegsnke_rs) is not available.")
    _ACTIVE_BACKEND = "rust"
elif _env_backend == "python":
    _ACTIVE_BACKEND = "python"
else:  # "auto" or anything else
    _ACTIVE_BACKEND = "rust" if _RUST_AVAILABLE else "python"


def is_rust_available() -> bool:
    """Return True if the Rust accelerated backend is compiled and available."""
    return _RUST_AVAILABLE


def get_backend() -> str:
    """Return current active backend: 'rust' or 'python'."""
    return _ACTIVE_BACKEND


def set_backend(backend: str):
    """
    Set active backend for FreeGSNKE ('rust' or 'python').
    
    Parameters
    ----------
    backend : str
        Either 'rust' or 'python'.
    """
    global _ACTIVE_BACKEND
    b = backend.strip().lower()
    if b == "rust":
        if not _RUST_AVAILABLE:
            raise RuntimeError("Cannot select 'rust' backend: _freegsnke_rs extension is not built or importable.")
        _ACTIVE_BACKEND = "rust"
    elif b == "python":
        _ACTIVE_BACKEND = "python"
    elif b == "auto":
        _ACTIVE_BACKEND = "rust" if _RUST_AVAILABLE else "python"
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'rust', 'python', 'auto'.")


def get_rust_module():
    """Return the raw _freegsnke_rs C-extension module, or None if unavailable."""
    return _RUST_MODULE
