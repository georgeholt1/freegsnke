"""
FreeGSNKE package.

Provides tools for solving Grad-Shafranov equilibria and related
numerical methods for plasma physics modelling.
"""

import importlib.metadata

__version__ = importlib.metadata.version("freegsnke")
__author__ = "The FreeGSNKE Developers"

from .backend import get_backend, set_backend, is_rust_available
