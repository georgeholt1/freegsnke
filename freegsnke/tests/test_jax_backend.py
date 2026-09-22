"""
Tests for FreeGSNKE JAX backend.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.
"""

from pathlib import Path
import numpy as np
import pytest

from freegs4e.critical import find_critical
import freegs4e.gradshafranov
from freegsnke import GSstaticsolver, build_machine, equilibrium_update, get_backend, set_backend
from freegsnke.jax import is_jax_available
from freegsnke.jax.linear_solver import JAXGSLinearEngine, JAXLinearGSSolver
from freegsnke.jtor_update import ConstrainPaxisIp
from freegsnke.tests.test_inverse_static_solver import (
    INVERSE_CURRENT_BASELINE,
    INVERSE_PSI_BASELINE,
    _build_diverted_inverse_case,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = Path(__file__).resolve().parent / "baselines"
MACHINE_CONFIG_DIR = REPO_ROOT / "machine_configs" / "test"
STATIC_CURRENT_BASELINE = TEST_DATA_DIR / "test_controlCurrents.npy"
STATIC_PSI_BASELINE = TEST_DATA_DIR / "test_psi.npy"


def test_jax_availability_and_config():
    """Verify JAX backend configuration and availability."""
    assert is_jax_available()
    orig = get_backend()
    try:
        set_backend("jax")
        assert get_backend() == "jax"
        set_backend("numpy")
        assert get_backend() == "numpy"
    finally:
        set_backend(orig)


def test_jax_linear_solver_accuracy():
    """Verify JAX inverted GS operator matches SciPy sparse LU to numerical precision."""
    Rmin, Rmax = 0.1, 2.0
    Zmin, Zmax = -2.2, 2.2
    nx, ny = 65, 129
    op = freegs4e.gradshafranov.GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)
    A_sparse = op(nx, ny)

    solver_jax = JAXLinearGSSolver(A_sparse, nx, ny)

    rng = np.random.default_rng(42)
    b = rng.standard_normal(nx * ny)

    import scipy.sparse.linalg as spla
    lu = spla.splu(A_sparse.tocsc())
    x_scipy = lu.solve(b)
    x_jax = np.asarray(solver_jax.solve_jax(b))

    rel_diff = np.max(np.abs(x_scipy - x_jax)) / np.max(np.abs(x_scipy))
    assert rel_diff < 1e-10, f"Relative difference {rel_diff} exceeds 1e-10"


def test_jax_forward_static_solve():
    """Verify forward static equilibrium solve with backend='jax' matches reference baseline."""
    tokamak = build_machine.tokamak(
        active_coils_path=str(MACHINE_CONFIG_DIR / "active_coils.pickle"),
        passive_coils_path=str(MACHINE_CONFIG_DIR / "passive_coils.pickle"),
        limiter_path=str(MACHINE_CONFIG_DIR / "limiter.pickle"),
        wall_path=str(MACHINE_CONFIG_DIR / "wall.pickle"),
        magnetic_probe_path=str(MACHINE_CONFIG_DIR / "magnetic_probes.pickle"),
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.2,
        Zmax=2.2,
        nx=65,
        ny=129,
    )
    profiles = ConstrainPaxisIp(
        eq, 8.1e3, 6.2e5, 0.5, alpha_m=1.8, alpha_n=1.2
    )

    solver = GSstaticsolver.NKGSsolver(eq, backend="jax")
    assert solver.backend == "jax"

    eq.tokamak.set_coil_current("P6", 0)
    eq.tokamak["P6"].control = False
    eq.tokamak["Solenoid"].control = False
    eq.tokamak.set_coil_current("Solenoid", 15000)

    controlCurrents = np.load(STATIC_CURRENT_BASELINE)
    eq.tokamak.setControlCurrents(controlCurrents)

    solver.forward_solve(eq, profiles, 1e-8)
    assert eq.solved

    test_psi = np.load(STATIC_PSI_BASELINE)
    psi_tolerance = (np.max(test_psi) - np.min(test_psi)) * 0.003
    assert np.allclose(eq.psi(), test_psi, atol=psi_tolerance)


def test_jax_inverse_static_solve():
    """Verify inverse static equilibrium solve with backend='jax' matches reference baseline."""
    eq, profiles, constrain, _ = _build_diverted_inverse_case()

    solver = GSstaticsolver.NKGSsolver(eq=eq, backend="jax")
    assert solver.backend == "jax"

    solver.solve(
        eq=eq,
        profiles=profiles,
        constrain=constrain,
        target_relative_tolerance=1e-6,
        target_relative_psit_update=1e-3,
        verbose=False,
        l2_reg=np.array([1e-12] * 10 + [1e-6]),
    )
    assert eq.solved

    solved_currents = np.asarray(eq.tokamak.getCurrentsVec())[:12]
    reference_currents = np.load(INVERSE_CURRENT_BASELINE)
    reference_psi = np.load(INVERSE_PSI_BASELINE)

    assert np.allclose(
        solved_currents, reference_currents, atol=2e-2
    ), "Inverse-solve control currents differ from baseline"

    psi_tolerance = (np.max(reference_psi) - np.min(reference_psi)) * 0.003
    assert np.allclose(
        eq.psi(), reference_psi, atol=psi_tolerance
    ), "Inverse-solve psi map differs from baseline"

    opt, xpt = find_critical(
        eq.R,
        eq.Z,
        eq.psi(),
        eq.mask_inside_limiter.astype(bool),
        None,
    )
    assert len(opt) == 1
    assert len(xpt) >= 2
