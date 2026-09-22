"""
JAX-accelerated linear Grad-Shafranov PDE solver and boundary Green's evaluator.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.
"""

from functools import partial
import numpy as np
import scipy.sparse.linalg as spla

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    jax = None
    jnp = None

# Global cache for inverted GS operators: key = (nx, ny, Rmin, Rmax, Zmin, Zmax, order)
_A_INV_CACHE = {}


@partial(jax.jit, static_argnames=("nx", "ny"))
def _assemble_rhs_and_solve_jit(
    jtor,
    rhs_before_jtor,
    greenfunc,
    mask_indices,
    A_inv,
    nx,
    ny,
):
    """
    JIT-compiled fused computation of:
    1. Boundary flux from toroidal current density: psi_bnd = greenfunc @ jtor[mask]
    2. RHS source term: rhs = rhs_before_jtor * jtor
    3. Dirichlet boundary condition injection into rhs
    4. Direct linear GS solve: psi = A_inv @ rhs
    """
    jtor_1d = jtor.ravel()
    jtor_masked = jtor_1d[mask_indices]
    psi_bnd = greenfunc @ jtor_masked

    base_rhs = rhs_before_jtor * jtor
    rhs = base_rhs.at[:, 0].set(psi_bnd[:nx])
    rhs = rhs.at[:, -1].set(psi_bnd[nx : 2 * nx])
    rhs = rhs.at[0, 1 : ny - 1].set(psi_bnd[2 * nx : 2 * nx + ny - 2])
    rhs = rhs.at[-1, 1 : ny - 1].set(psi_bnd[2 * nx + ny - 2 :])

    psi_predicted = A_inv @ rhs.ravel()
    return psi_predicted, psi_bnd


@partial(jax.jit, static_argnames=("nx", "ny"))
def _batched_solve_jit(
    batch_jtor,
    rhs_before_jtor,
    greenfunc,
    mask_indices,
    A_inv,
    nx,
    ny,
):
    """
    Batched solve over leading dimension of batch_jtor using vmap.
    """
    def solve_single(jtor):
        return _assemble_rhs_and_solve_jit(
            jtor,
            rhs_before_jtor,
            greenfunc,
            mask_indices,
            A_inv,
            nx,
            ny,
        )

    return jax.vmap(solve_single)(batch_jtor)


class JAXLinearGSSolver:
    """
    Precomputed direct linear Grad-Shafranov PDE operator on GPU using JAX.
    """

    def __init__(self, A_sparse, nx, ny, cache_key=None):
        """
        Initialize the linear solver with the finite-difference operator.

        Parameters
        ----------
        A_sparse : scipy.sparse matrix
            CSR or CSC sparse representation of the GS elliptic operator with BCs.
        nx : int
            Number of radial grid points.
        ny : int
            Number of vertical grid points.
        cache_key : tuple, optional
            Cache key to reuse inverted matrix across solver instances.
        """
        self.nx = nx
        self.ny = ny
        self.N = nx * ny

        if cache_key is not None and cache_key in _A_INV_CACHE:
            self.A_inv = _A_INV_CACHE[cache_key]
        else:
            # Invert the dense matrix in float64
            A_dense = A_sparse.toarray()
            A_inv_np = np.linalg.inv(A_dense)
            self.A_inv = jnp.array(A_inv_np, dtype=jnp.float64)
            if cache_key is not None:
                _A_INV_CACHE[cache_key] = self.A_inv

    def solve(self, psi_boundary, rhs):
        """
        Solve linear GS system A * psi = rhs with Dirichlet boundary conditions.

        Matches the signature of freegs4e MGDirect / createVcycle:
        solver(psi_boundary, rhs) -> 2D flux array of shape (nx, ny).
        """
        rhs_jax = jnp.asarray(rhs, dtype=jnp.float64)
        psi_1d = self.A_inv @ rhs_jax.ravel()
        return np.asarray(psi_1d.reshape(self.nx, self.ny))

    def solve_jax(self, rhs):
        """Pure JAX linear solve: A_inv @ rhs.ravel()."""
        return self.A_inv @ rhs.ravel()


class JAXGSLinearEngine:
    """
    Engine combining boundary Green's function convolution, RHS assembly,
    and linear GS PDE solve on GPU.
    """

    def __init__(
        self,
        A_sparse,
        greenfunc,
        plasma_source_mask,
        rhs_before_jtor,
        nx,
        ny,
        cache_key=None,
    ):
        self.nx = nx
        self.ny = ny
        self.N = nx * ny

        self.linear_solver = JAXLinearGSSolver(A_sparse, nx, ny, cache_key=cache_key)
        self.A_inv = self.linear_solver.A_inv

        self.greenfunc = jnp.array(greenfunc, dtype=jnp.float64)
        self.plasma_source_mask = np.asarray(plasma_source_mask, dtype=bool)
        self.mask_indices = jnp.array(
            np.where(self.plasma_source_mask.reshape(-1))[0], dtype=jnp.int32
        )
        self.rhs_before_jtor = jnp.array(rhs_before_jtor, dtype=jnp.float64)

    def compute_gs_solution(self, jtor):
        """
        Compute predicted plasma flux and boundary flux from toroidal current density.

        Parameters
        ----------
        jtor : ndarray or jnp.ndarray, shape (nx, ny)
            Toroidal current density.

        Returns
        -------
        psi_pred_1d : jnp.ndarray, shape (N,)
            Predicted plasma poloidal flux (flattened).
        psi_bnd_1d : jnp.ndarray
            Dirichlet boundary flux values.
        """
        jtor_jax = jnp.asarray(jtor, dtype=jnp.float64)
        return _assemble_rhs_and_solve_jit(
            jtor_jax,
            self.rhs_before_jtor,
            self.greenfunc,
            self.mask_indices,
            self.A_inv,
            self.nx,
            self.ny,
        )

    def batched_compute_gs_solution(self, batch_jtor):
        """
        Batched computation across a batch of jtor profiles using vmap.
        """
        batch_jtor_jax = jnp.asarray(batch_jtor, dtype=jnp.float64)
        return _batched_solve_jit(
            batch_jtor_jax,
            self.rhs_before_jtor,
            self.greenfunc,
            self.mask_indices,
            self.A_inv,
            self.nx,
            self.ny,
        )
