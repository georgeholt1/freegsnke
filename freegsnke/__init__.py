"""
FreeGSNKE package.

Provides tools for solving Grad-Shafranov equilibria and related
numerical methods for plasma physics modelling.
"""

import importlib.metadata

from .jax.config import get_backend, set_backend

__version__ = importlib.metadata.version("freegsnke")
__author__ = "The FreeGSNKE Developers"

__all__ = [
    "__version__",
    "__author__",
    "get_backend",
    "set_backend",
]
