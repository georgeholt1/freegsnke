"""
FreeGSNKE package.

Provides tools for solving Grad-Shafranov equilibria and related
numerical methods for plasma physics modelling.
"""

import importlib.metadata

__version__ = importlib.metadata.version("freegsnke")
__author__ = "The FreeGSNKE Developers"

from . import (
    boundary,
    coil,
    control,
    critical,
    dump,
    equilibrium,
    gradshafranov,
    jtor,
    machine,
    multi_coil,
    multigrid,
    optimise,
    optimiser,
    picard,
    plotting,
    polygons,
    quadrature,
    shaped_coil,
)
from .dump import OutputFile
from .equilibrium import Equilibrium
from .picard import solve
