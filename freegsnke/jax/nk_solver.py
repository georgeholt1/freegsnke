"""
JAX implementation of the Newton-Krylov Arnoldi algorithm.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None
    jnp = None


class JAXNKSolver:
    """
    Newton-Krylov Arnoldi solver using JAX on GPU.
    Drop-in equivalent to freegsnke.nk_solver_H.nksolver.
    """

    def __init__(
        self,
        problem_dimension,
        l2_reg=1e-6,
        collinearity_reg=1e-6,
        verbose=False,
    ):
        self.problem_dimension = problem_dimension
        self.dummy_hessenberg_residual = np.zeros(problem_dimension)
        self.dummy_hessenberg_residual[0] = 1.0
        self.verbose = verbose
        self.set_regularization(l2_reg, collinearity_reg)

    def set_regularization(self, l2_reg, collinearity_reg):
        self.l2_reg = float(l2_reg)
        self.collinearity_reg = float(collinearity_reg)

    def Arnoldi_unit(
        self,
        x0,
        dx,
        R0,
        F_function,
        args,
        build_next=True,
    ):
        """
        Explore direction dx and compute new candidate direction.
        """
        res_calculated = False
        dx1 = np.copy(dx)
        while res_calculated is False:
            try:
                candidate_x = x0 + dx1
                R_dx = F_function(candidate_x, *args)
                res_calculated = True
            except Exception:
                dx1 *= 0.75
                self.Q[:, self.n_it] *= 0.75

        useful_residual = R_dx - R0

        norm_ur = float(np.linalg.norm(useful_residual))
        self.n_G[self.n_it] = norm_ur
        self.G[:, self.n_it] = useful_residual
        self.Gn[:, self.n_it] = useful_residual / max(norm_ur, 1e-30)

        # Collinearity with previous search directions
        self.collinearity[: self.n_it, self.n_it] = np.sum(
            self.Gn[:, self.n_it, np.newaxis] * self.Gn[:, : self.n_it], axis=0
        )

        if build_next:
            # Append to Hessenberg matrix
            self.Hm[: self.n_it + 1, self.n_it] = np.sum(
                self.Qn[:, : self.n_it + 1] * useful_residual[:, np.newaxis], axis=0
            )

            # Orthogonalise wrt previous directions (Gram-Schmidt)
            next_candidate = useful_residual - np.sum(
                self.Qn[:, : self.n_it + 1]
                * self.Hm[: self.n_it + 1, self.n_it][np.newaxis, :],
                axis=1,
            )

            norm_nc = float(np.linalg.norm(next_candidate))
            self.Hm[self.n_it + 1, self.n_it] = norm_nc
            next_candidate /= max(norm_nc, 1e-30)
            return next_candidate

        return None

    def Arnoldi_iteration(
        self,
        x0,
        dx,
        R0,
        F_function,
        args,
        step_size,
        scaling_with_n,
        target_relative_unexplained_residual,
        max_n_directions,
        clip,
    ):
        """
        Perform Newton-Krylov Arnoldi iteration.
        """
        self.x0 = np.copy(x0)
        self.R0 = np.copy(R0)

        self.relative_unexplained_residuals = []
        nR0 = float(np.linalg.norm(R0))
        self.nR0 = nR0
        self.max_dim = int(max_n_directions + 1)

        # Basis storage
        self.Q = np.zeros((self.problem_dimension, self.max_dim))
        self.Qn = np.zeros((self.problem_dimension, self.max_dim))
        self.G = np.zeros((self.problem_dimension, self.max_dim))
        self.Gn = np.zeros((self.problem_dimension, self.max_dim))
        self.n_G = np.zeros(self.max_dim)
        self.collinearity = np.zeros((self.max_dim, self.max_dim))
        self.Hm = np.zeros((self.max_dim + 1, self.max_dim))

        adjusted_step_size = step_size * nR0
        self.n_it = 0
        this_step_size = adjusted_step_size * ((1 + self.n_it) ** scaling_with_n)

        norm_dx = float(np.linalg.norm(dx))
        dx = dx / max(norm_dx, 1e-30)

        self.Qn[:, self.n_it] = np.copy(dx)
        dx = dx * this_step_size
        self.Q[:, self.n_it] = np.copy(dx)

        explore = 1
        while explore:
            dx = self.Arnoldi_unit(x0, dx, R0, F_function, args)

            # Collinearity penalty
            coll_sub = self.collinearity[: self.n_it + 1, : self.n_it + 1]
            abs_coll = np.abs(coll_sub)
            denom = np.clip(1.0 - abs_coll, 1e-6, 1.0)
            collinearity_penalty = np.diag(
                np.max(1.0 / (denom**2), axis=0) - 1.0
            )

            collinear_aware_regulariz = (
                np.eye(self.n_it + 1) * self.l2_reg
                + collinearity_penalty * self.collinearity_reg
            ) * (nR0**2)
            self.collinear_aware_regulariz = collinear_aware_regulariz

            # Solve regularised least-squares problem
            G_sub = self.G[:, : self.n_it + 1]
            A = G_sub.T @ G_sub + collinear_aware_regulariz
            rhs_ls = G_sub.T @ (-R0)

            try:
                coeffs = np.linalg.solve(A, rhs_ls)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(A, rhs_ls, rcond=None)[0]

            coeffs = np.clip(coeffs, -clip, clip)

            # Fraction of residual explained
            expl_res = np.sum(G_sub * coeffs[np.newaxis, :], axis=1)
            unexpl = float(np.linalg.norm(R0 + expl_res) / max(nR0, 1e-30))
            self.relative_unexplained_residuals.append(unexpl)

            explore = int(self.n_it < max_n_directions)
            explore *= int(unexpl > target_relative_unexplained_residual)

            if explore:
                self.n_it += 1
                self.Qn[:, self.n_it] = np.copy(dx)
                this_step_size = adjusted_step_size * (
                    (1 + self.n_it) ** scaling_with_n
                )
                dx = dx * this_step_size
                self.Q[:, self.n_it] = np.copy(dx)

        self.coeffs = np.copy(coeffs)
        self.dx = np.sum(self.Q[:, : self.n_it + 1] * coeffs[np.newaxis, :], axis=1)
