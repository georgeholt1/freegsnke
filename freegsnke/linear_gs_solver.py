"""
Module providing linear Grad-Shafranov equation solvers, including direct LU
and Incomplete LU (ILU) preconditioned iterative solvers.

Copyright 2025-2026 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

from typing import Any, Dict, Optional, Tuple

import freegs4e
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class ILULinearGSSolver:
    """
    Incomplete LU (ILU) preconditioned linear solver for the Grad-Shafranov equation.

    Solves the linearised Grad-Shafranov system:
        A * psi = rhs
    where A includes Dirichlet boundary conditions, using row-equilibrated ILU
    preconditioning combined with iterative refinement (defect correction) or
    Krylov subspace methods (BiCGSTAB, GMRES).

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse discretized Grad-Shafranov operator matrix.
    shape : tuple of int
        Shape of the 2D poloidal flux grid (nx, ny).
    method : {"defect_correction", "bicgstab", "gmres"}, optional
        Solution algorithm used with the ILU preconditioner (default "defect_correction").
    n_refine : int, optional
        Number of defect-correction refinement iterations (default 2).
    fill_factor : float, optional
        Maximum fill factor for ILU factorization (default 10.0).
    drop_tol : float, optional
        Threshold for dropping small entries during ILU factorization (default 1e-4).
    permc_spec : str, optional
        Column permutation strategy for SuperLU ILU, e.g. "MMD_AT_PLUS_A" (default "MMD_AT_PLUS_A").
    rtol : float, optional
        Relative convergence tolerance for iterative Krylov solvers (default 1e-8).
    atol : float, optional
        Absolute convergence tolerance for iterative Krylov solvers (default 1e-12).
    maxiter : int, optional
        Maximum number of iterations for iterative Krylov solvers (default 50).
    """

    def __init__(
        self,
        A: sp.spmatrix,
        shape: Tuple[int, int],
        method: str = "defect_correction",
        n_refine: int = 2,
        fill_factor: float = 10.0,
        drop_tol: float = 1e-4,
        permc_spec: str = "MMD_AT_PLUS_A",
        rtol: float = 1e-8,
        atol: float = 1e-12,
        maxiter: int = 50,
    ):
        """Initialise the ILU preconditioned linear solver."""
        if method not in ("defect_correction", "bicgstab", "gmres"):
            raise ValueError(
                f"Unknown method '{method}'. Allowed methods: 'defect_correction', 'bicgstab', 'gmres'."
            )

        self.shape = shape
        self.method = method
        self.n_refine = int(n_refine)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.maxiter = int(maxiter)
        self.fill_factor = float(fill_factor)
        self.drop_tol = float(drop_tol)
        self.permc_spec = permc_spec

        # Ensure CSC format for SuperLU ILU
        self.A = A.tocsc() if not sp.isspmatrix_csc(A) else A

        # Row equilibration: scale each row by 1 / max(|A_ij|) to stabilize ILU
        row_norms = np.abs(self.A).max(axis=1).toarray().flatten()
        row_norms[row_norms == 0] = 1.0
        self.D_inv = sp.diags(1.0 / row_norms)
        self.A_scaled = (self.D_inv @ self.A).tocsc()

        # Incomplete LU factorization
        self.ilu = spla.spilu(
            self.A_scaled,
            permc_spec=self.permc_spec,
            fill_factor=self.fill_factor,
            drop_tol=self.drop_tol,
        )
        self.M = spla.LinearOperator(self.A_scaled.shape, self.ilu.solve)

    def __call__(self, psi_boundary: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        Solve the linear GS system for the given boundary flux and RHS.

        Parameters
        ----------
        psi_boundary : ndarray
            Boundary poloidal flux values or field.
        rhs : ndarray
            Right-hand side of the linearised GS equation.

        Returns
        -------
        ndarray
            Calculated poloidal flux field of shape `self.shape`.
        """
        b = rhs.reshape(-1)
        b_scaled = self.D_inv @ b

        if self.method == "defect_correction":
            x = self.ilu.solve(b_scaled)
            for _ in range(self.n_refine):
                r = b_scaled - self.A_scaled @ x
                x = x + self.ilu.solve(r)
        elif self.method == "bicgstab":
            x0 = psi_boundary.reshape(-1) if psi_boundary.shape == self.shape else None
            x, info = spla.bicgstab(
                self.A_scaled,
                b_scaled,
                x0=x0,
                M=self.M,
                rtol=self.rtol,
                atol=self.atol,
                maxiter=self.maxiter,
            )
            if info != 0:
                # If Krylov fails to converge, fallback to defect correction
                x = self.ilu.solve(b_scaled)
                for _ in range(self.n_refine):
                    r = b_scaled - self.A_scaled @ x
                    x = x + self.ilu.solve(r)
        elif self.method == "gmres":
            x0 = psi_boundary.reshape(-1) if psi_boundary.shape == self.shape else None
            x, info = spla.gmres(
                self.A_scaled,
                b_scaled,
                x0=x0,
                M=self.M,
                rtol=self.rtol,
                atol=self.atol,
                maxiter=self.maxiter,
            )
            if info != 0:
                # Fallback to defect correction
                x = self.ilu.solve(b_scaled)
                for _ in range(self.n_refine):
                    r = b_scaled - self.A_scaled @ x
                    x = x + self.ilu.solve(r)

        return x.reshape(self.shape)


def create_linear_gs_solver(
    A: sp.spmatrix,
    shape: Tuple[int, int],
    solver_type: str = "direct",
    options: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Factory function to instantiate a linear Grad-Shafranov equation solver.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Discretized linear Grad-Shafranov operator matrix.
    shape : tuple of int
        Grid shape (nx, ny).
    solver_type : {"direct", "ilu"}, optional
        Type of linear solver to construct (default "direct").
    options : dict, optional
        Additional configuration options passed to the solver constructor.

    Returns
    -------
    callable
        Linear solver callable accepting `(psi_boundary, rhs)` and returning `psi`.
    """
    options = options or {}
    if solver_type == "direct":
        return freegs4e.multigrid.MGDirect(A)
    if solver_type == "ilu":
        return ILULinearGSSolver(A, shape, **options)

    raise ValueError(
        f"Unknown linear solver type '{solver_type}'. Must be 'direct' or 'ilu'."
    )
