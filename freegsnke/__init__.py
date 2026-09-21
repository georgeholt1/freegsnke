"""
FreeGSNKE package.

Provides tools for solving Grad-Shafranov equilibria and related
numerical methods for plasma physics modelling.
"""

import importlib.metadata

from .surrogate import SurrogateInitialGuess

__version__ = importlib.metadata.version("freegsnke")
__author__ = "The FreeGSNKE Developers"

__all__ = ["SurrogateInitialGuess"]
