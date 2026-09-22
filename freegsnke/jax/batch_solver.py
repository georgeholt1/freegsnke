"""
Batched Grad-Shafranov equilibrium solver using JAX vectorization on GPU.

Solves multiple plasma equilibria simultaneously across coil current variations,
plasma current variations, or profile parameter scans using JAX vmap and cuBLAS GEMM.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.
"""

import time
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from .config import is_jax_available
    from .linear_solver import JAXGSLinearEngine
except ImportError:
    jax = None
    jnp = None
    is_jax_available = lambda: False

from ..jtor_update import ConstrainPaxisIp


class BatchedEquilibriumSolver:
    """
    Simultaneous (batched) Grad-Shafranov equilibrium solver.

    Uses GPU-accelerated JAX vectorization (`jax.vmap`) and dense matrix
    multiplication to solve the linear PDE step for B equilibria simultaneously
    in a single fused GPU kernel call.
    """

    def __init__(self, eq_template, cache_key: Optional[tuple] = None):
        """
        Initialize the batched solver using an equilibrium template.

        Parameters
        ----------
        eq_template : freegsnke.equilibrium_update.Equilibrium
            Template equilibrium providing geometry (Rmin, Rmax, Zmin, Zmax),
            grid resolution (nx, ny), and Greens function operators.
        cache_key : tuple, optional
            Cache key for the precomputed inverted GS operator.
        """
        if not is_jax_available():
            raise RuntimeError(
                "JAX is not installed or available. Cannot instantiate BatchedEquilibriumSolver."
            )

        self.eq_template = eq_template
        self.nx = eq_template.nx
        self.ny = eq_template.ny
        self.R = eq_template.R
        self.Z = eq_template.Z
        self.dRdZ = eq_template.dR * eq_template.dZ

        # Delegate linear engine setup to NKGSsolver to ensure identical operator & Greens construction
        from ..GSstaticsolver import NKGSsolver

        self.nk_solver = NKGSsolver(eq_template, backend="jax")
        self.engine = self.nk_solver.jax_engine

    def solve_batch(
        self,
        batch_currents: np.ndarray,
        batch_profiles: Optional[List] = None,
        paxis: float = 8.1e3,
        Ip: float = 6.2e5,
        vacuum_ratio: float = 0.5,
        alpha_m: float = 1.8,
        alpha_n: float = 1.2,
        target_relative_tolerance: float = 1e-5,
        max_iterations: int = 60,
        relaxation: float = 0.5,
        force_up_down_symmetric: bool = True,
        initial_plasma_psi: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Tuple[List, List, dict]:
        """
        Solve multiple equilibria simultaneously across a batch of coil currents.

        Parameters
        ----------
        batch_currents : ndarray, shape (B, n_coils)
            Batch of coil current vectors for each equilibrium in the ensemble.
        batch_profiles : list of profiles, optional
            List of B profile objects. If None, ConstrainPaxisIp instances are created.
        paxis : float
            Plasma core pressure [Pa] (used if batch_profiles is None).
        Ip : float
            Plasma toroidal current [A] (used if batch_profiles is None).
        vacuum_ratio : float
            Vacuum ratio parameter.
        alpha_m, alpha_n : float
            Current profile peaking parameters.
        target_relative_tolerance : float
            Convergence threshold on relative residual.
        max_iterations : int
            Maximum number of batched Picard iterations.
        relaxation : float
            Damping/relaxation factor for Picard updates (default 0.5).
        force_up_down_symmetric : bool
            Whether to enforce up-down symmetry on Picard updates to suppress vertical instability.
        initial_plasma_psi : ndarray, shape (nx, ny) or (B, nx, ny), optional
            Warm-start initial plasma flux guess.
        verbose : bool
            Whether to print progress.

        Returns
        -------
        eq_list : list of Equilibrium
            List of solved Equilibrium objects.
        profiles_list : list
            List of updated profile objects.
        info : dict
            Convergence history and performance metrics.
        """
        B = len(batch_currents)
        if verbose:
            print(f"--- Batched GS Solve for B={B} equilibria on GPU ---")

        # 1. Compute vacuum tokamak flux fields for all B equilibria simultaneously
        #    batch_currents: (B, n_coils), _vgreen: (n_coils, nx, ny)
        t_start = time.perf_counter()
        batch_tokamak_psi = np.tensordot(batch_currents, self.eq_template._vgreen, axes=(1, 0))

        # 2. Setup profile objects for each equilibrium
        if batch_profiles is None:
            batch_profiles = []
            for b in range(B):
                p = ConstrainPaxisIp(
                    self.eq_template,
                    paxis,
                    Ip,
                    vacuum_ratio,
                    alpha_m=alpha_m,
                    alpha_n=alpha_n,
                )
                batch_profiles.append(p)

        # 3. Initialize plasma poloidal flux
        if initial_plasma_psi is not None:
            if initial_plasma_psi.ndim == 2:
                batch_plasma_psi = np.stack([initial_plasma_psi.copy() for _ in range(B)])
            else:
                batch_plasma_psi = initial_plasma_psi.copy()
        else:
            default_psi = self.eq_template.create_psi_plasma_default(adaptive_centre=True)
            batch_plasma_psi = np.stack([default_psi.copy() for _ in range(B)])

        # 4. Batched Picard Iteration
        convergence_history = []
        converged = False
        it = 0

        for it in range(max_iterations):
            # Evaluate Jtor for all batch items
            batch_jtor = np.zeros((B, self.nx, self.ny))
            for b in range(B):
                psi_total = batch_plasma_psi[b] + batch_tokamak_psi[b]
                batch_profiles[b].Jtor(self.R, self.Z, psi_total)
                batch_jtor[b] = batch_profiles[b].jtor

            # Vectorized PDE solve & boundary Green convolution on GPU in a single call
            batch_psi_pred, _ = self.engine.batched_compute_gs_solution(batch_jtor)
            batch_psi_pred = np.asarray(batch_psi_pred).reshape(B, self.nx, self.ny)

            # Check residual & update each equilibrium in batch
            errs = []
            for b in range(B):
                res = batch_plasma_psi[b] - batch_psi_pred[b]
                if force_up_down_symmetric:
                    res = 0.5 * (res + res[:, ::-1])
                del_psi = np.amax(batch_plasma_psi[b]) - np.amin(batch_plasma_psi[b])
                del_res = np.amax(res) - np.amin(res)
                err = del_res / max(del_psi, 1e-12)
                errs.append(err)
                batch_plasma_psi[b] -= relaxation * res

            max_err = float(max(errs))
            mean_err = float(np.mean(errs))
            convergence_history.append({"iteration": it, "max_error": max_err, "mean_error": mean_err})

            if verbose and (it % 5 == 0 or max_err < target_relative_tolerance):
                print(f"  Iteration {it:2d}: max relative error = {max_err:.3e} (mean = {mean_err:.3e})")

            if max_err < target_relative_tolerance:
                converged = True
                break

        t_elapsed = time.perf_counter() - t_start

        if verbose:
            status = "CONVERGED" if converged else "MAX_ITERATIONS_REACHED"
            print(f"--- Batched solve {status} in {it + 1} iterations: {t_elapsed:.3f} s ({t_elapsed / B * 1000:.1f} ms/eq) ---")

        # 5. Build list of solved Equilibrium objects
        eq_list = []
        for b in range(B):
            eq_b = self.eq_template.create_auxiliary_equilibrium()
            eq_b.tokamak.current_vec = batch_currents[b].copy()
            eq_b.tokamak_psi = batch_tokamak_psi[b].copy()
            eq_b.plasma_psi = batch_plasma_psi[b].copy()
            eq_b.solved = True
            eq_b.xpt = np.copy(batch_profiles[b].xpt)
            eq_b.opt = np.copy(batch_profiles[b].opt)
            eq_b.psi_axis = eq_b.opt[0, 2]
            eq_b.psi_bndry = batch_profiles[b].psi_bndry
            eq_b.flag_limiter = batch_profiles[b].flag_limiter
            eq_b._current = float(np.sum(batch_profiles[b].jtor) * self.dRdZ)
            eq_b._profiles = batch_profiles[b].copy()
            eq_list.append(eq_b)

        info = {
            "converged": converged,
            "iterations": it + 1,
            "wall_time": t_elapsed,
            "time_per_eq": t_elapsed / B,
            "history": convergence_history,
        }

        return eq_list, batch_profiles, info
