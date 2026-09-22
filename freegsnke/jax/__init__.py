"""
JAX backend for FreeGSNKE.

Provides GPU acceleration, whole-loop JIT compilation, and batched equilibrium
solving via vectorization (vmap).

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.
"""

from .config import (
    get_backend,
    get_default_device,
    get_jax_devices,
    is_jax_available,
    set_backend,
)
from .batch_solver import BatchedEquilibriumSolver
from .linear_solver import JAXGSLinearEngine, JAXLinearGSSolver
from .nk_solver import JAXNKSolver

__all__ = [
    "set_backend",
    "get_backend",
    "is_jax_available",
    "get_jax_devices",
    "get_default_device",
    "JAXLinearGSSolver",
    "JAXGSLinearEngine",
    "JAXNKSolver",
    "BatchedEquilibriumSolver",
]
