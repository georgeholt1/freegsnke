"""
Tests for linear Grad-Shafranov equation solvers, including direct and ILU preconditioned solvers.
"""

from pathlib import Path

import freegs4e
import numpy as np
import pytest
import scipy.sparse as sp

from freegsnke.linear_gs_solver import ILULinearGSSolver, create_linear_gs_solver

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def sample_gs_operators():
    """Constructs 2nd- and 4th-order Grad-Shafranov sparse matrices on a test grid."""
    Rmin, Rmax = 0.1, 2.0
    Zmin, Zmax = -2.2, 2.2
    nx, ny = 33, 65

    A_order2 = freegs4e.gradshafranov.GSsparse(Rmin, Rmax, Zmin, Zmax)(nx, ny)
    A_order4 = freegs4e.gradshafranov.GSsparse4thOrder(Rmin, Rmax, Zmin, Zmax)(nx, ny)

    return (nx, ny), A_order2, A_order4


def test_linear_gs_solver_dirichlet_boundary(sample_gs_operators):
    """Verifies that the ILU linear solver preserves Dirichlet boundary conditions."""
    (nx, ny), _, A_order4 = sample_gs_operators

    rng = np.random.default_rng(42)
    psi_boundary = rng.standard_normal((nx, ny))
    rhs = rng.standard_normal((nx, ny))

    # In FreeGS, boundary entries in rhs are set to boundary psi
    rhs[0, :] = psi_boundary[0, :]
    rhs[-1, :] = psi_boundary[-1, :]
    rhs[:, 0] = psi_boundary[:, 0]
    rhs[:, -1] = psi_boundary[:, -1]

    solver_direct = create_linear_gs_solver(A_order4, (nx, ny), solver_type="direct")
    solver_ilu = create_linear_gs_solver(A_order4, (nx, ny), solver_type="ilu")

    sol_direct = solver_direct(psi_boundary, rhs)
    sol_ilu = solver_ilu(psi_boundary, rhs)

    # Check boundaries match exactly on boundary points
    for sol in [sol_direct, sol_ilu]:
        np.testing.assert_allclose(
            sol[0, :], psi_boundary[0, :], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            sol[-1, :], psi_boundary[-1, :], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            sol[:, 0], psi_boundary[:, 0], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            sol[:, -1], psi_boundary[:, -1], rtol=1e-12, atol=1e-12
        )


def test_linear_gs_solver_accuracy_order4(sample_gs_operators):
    """ILU solver matches direct LU solve to high accuracy on 4th-order operator."""
    (nx, ny), _, A_order4 = sample_gs_operators

    rng = np.random.default_rng(101)
    psi_boundary = np.zeros((nx, ny))
    rhs = np.zeros((nx, ny))
    interior_x = slice(nx // 4, 3 * nx // 4)
    interior_y = slice(ny // 4, 3 * ny // 4)
    rhs[interior_x, interior_y] = rng.standard_normal(rhs[interior_x, interior_y].shape)

    solver_direct = create_linear_gs_solver(A_order4, (nx, ny), solver_type="direct")
    sol_direct = solver_direct(psi_boundary, rhs)

    # Test defect correction (fast default)
    solver_ilu_dc = create_linear_gs_solver(
        A_order4,
        (nx, ny),
        solver_type="ilu",
        options={"method": "defect_correction", "n_refine": 3},
    )
    sol_ilu_dc = solver_ilu_dc(psi_boundary, rhs)

    rel_diff_dc = np.max(np.abs(sol_ilu_dc - sol_direct)) / np.ptp(sol_direct)
    assert rel_diff_dc < 1e-5, f"Defect correction difference too large: {rel_diff_dc}"

    # Test BiCGSTAB
    solver_ilu_bicg = create_linear_gs_solver(
        A_order4,
        (nx, ny),
        solver_type="ilu",
        options={"method": "bicgstab", "rtol": 1e-8},
    )
    sol_ilu_bicg = solver_ilu_bicg(psi_boundary, rhs)

    rel_diff_bicg = np.max(np.abs(sol_ilu_bicg - sol_direct)) / np.ptp(sol_direct)
    assert rel_diff_bicg < 1e-5, f"BiCGSTAB difference too large: {rel_diff_bicg}"


def test_linear_gs_solver_accuracy_order2(sample_gs_operators):
    """ILU solver matches direct LU solve to high accuracy on 2nd-order operator."""
    (nx, ny), A_order2, _ = sample_gs_operators

    rng = np.random.default_rng(202)
    psi_boundary = np.zeros((nx, ny))
    rhs = np.zeros((nx, ny))
    interior_x = slice(nx // 4, 3 * nx // 4)
    interior_y = slice(ny // 4, 3 * ny // 4)
    rhs[interior_x, interior_y] = rng.standard_normal(rhs[interior_x, interior_y].shape)

    solver_direct = create_linear_gs_solver(A_order2, (nx, ny), solver_type="direct")
    sol_direct = solver_direct(psi_boundary, rhs)

    solver_ilu = create_linear_gs_solver(
        A_order2,
        (nx, ny),
        solver_type="ilu",
        options={"method": "defect_correction", "n_refine": 2},
    )
    sol_ilu = solver_ilu(psi_boundary, rhs)

    rel_diff = np.max(np.abs(sol_ilu - sol_direct)) / np.ptp(sol_direct)
    assert rel_diff < 1e-5, f"Order 2 ILU difference too large: {rel_diff}"


def test_linear_gs_solver_methods(sample_gs_operators):
    """Verifies that all supported iterative methods converge properly."""
    (nx, ny), _, A_order4 = sample_gs_operators

    rng = np.random.default_rng(303)
    psi_boundary = np.zeros((nx, ny))
    rhs = np.zeros((nx, ny))
    interior_x = slice(nx // 4, 3 * nx // 4)
    interior_y = slice(ny // 4, 3 * ny // 4)
    rhs[interior_x, interior_y] = rng.standard_normal(rhs[interior_x, interior_y].shape)

    solver_direct = create_linear_gs_solver(A_order4, (nx, ny), solver_type="direct")
    sol_direct = solver_direct(psi_boundary, rhs)

    for method in ["defect_correction", "bicgstab", "gmres"]:
        solver = create_linear_gs_solver(
            A_order4, (nx, ny), solver_type="ilu", options={"method": method}
        )
        sol = solver(psi_boundary, rhs)
        rel_diff = np.max(np.abs(sol - sol_direct)) / np.ptp(sol_direct)
        assert rel_diff < 1e-4, f"Method {method} difference too large: {rel_diff}"


def test_linear_gs_solver_invalid_method(sample_gs_operators):
    """ILULinearGSSolver raises ValueError on invalid method."""
    (nx, ny), _, A = sample_gs_operators
    with pytest.raises(ValueError, match="Unknown method"):
        ILULinearGSSolver(A, (nx, ny), method="invalid_method")


def test_create_linear_gs_solver_invalid_type(sample_gs_operators):
    """create_linear_gs_solver raises ValueError on invalid solver_type."""
    (nx, ny), _, A = sample_gs_operators
    with pytest.raises(ValueError, match="Unknown linear solver type"):
        create_linear_gs_solver(A, (nx, ny), solver_type="unsupported_solver")
