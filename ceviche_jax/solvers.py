""" This file stores the various linear system solvers you can use for FDFD 
"""

from functools import partial

from jax import jit, vmap
import jax.numpy as npj
from jaxopt.linear_solve import (
    solve_bicgstab,
    solve_cg,
    solve_cholesky,
    solve_gmres,
    solve_lu,
)

# default iterative method to use
DEFAULT_ITERATIVE_METHOD = "bicgstab"

# dict of iterative methods supported (name: function)
ITERATIVE_METHODS = {
    "bicgstab": solve_bicgstab,
    "cg": solve_cg,
    "gmres": solve_gmres,
}

# convergence tolerance for iterative solvers.
ATOL = 1e-8

""" ========================== SOLVER FUNCTIONS ========================== """


def solve_linear(
    A, b, iterative=False, iterative_method=DEFAULT_ITERATIVE_METHOD
):
    """Top-level function to call the selected solver.

    Default behavior is to use the direct solver. If iterative=True, then the
    specified iterative method will be used. If no iterative method is
    specified, the default iterative method is Bi-Conjugate Gradient Stable
    (see: https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.
    linalg.bicgstab.html)
    """

    if iterative:
        return _solve_iterative(A, b, iterative_method=iterative_method)
    else:
        return _solve_direct(A, b)


@jit
def _solve_direct(A, b):
    """Direct solver."""
    return solve_cholesky(lambda b: npj.dot(A, b), b)


@partial(jit, static_argnums=2)
def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD):
    """Iterative solver"""
    solver_fn = ITERATIVE_METHODS[iterative_method]
    x = solver_fn(lambda b: npj.dot(A, b), b, atol=ATOL)
    return x


if __name__ == "__main__":
    """========================= SPEED TESTS ============================="""

    # to run speed tests use `python -W ignore ceviche/solvers.py` to suppress
    # warnings

    from time import time

    import numpy as np
    import scipy.sparse as sp

    N = 100  # dimension of the x, and b vectors
    density = 0.2  # sparsity of the dense matrix
    A_sp = sp.random(N, N, density=density)
    A = A_sp.A
    b = np.random.random(N) - 0.5

    print("\nWITH RANDOM MATRICES:\n")
    print("\tfor N = {} and density = {}\n".format(N, density))

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print(f"\tdirect solver:\n\t\ttook {t1 - t0} seconds\n")

    # ITERATIVE SOLVES
    for method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=method)
        t1 = time()
        print(f"\titerative solver ({method}):\n\t\ttook {t1 - t0} seconds")

    # Test the solvers in the FDFD simulation
    from jax import random

    from ceviche_jax.constants import *
    from ceviche_jax.fdfd import FDFD_Ez

    print("\n")
    print("WITH FDFD MATRICES:\n")

    key = random.PRNGKey(0)

    m, n = 50, 40
    print("\tfor dimensions = {}\n".format((m, n)))
    eps_r = random.uniform(key, (m, n), dtype=np.float32) + 1
    b = random.uniform(key, (m * n,), dtype=np.float32) - 0.5
    b = b.astype(np.complex64)

    npml = 3
    dl = 2e-8
    lambda0 = 1550e-9
    omega0 = 2 * np.pi * C_0 / lambda0

    F = FDFD_Ez(omega0, dl, eps_r, (npml, npml))
    A = F._make_A(eps_r.flatten())

    # DIRECT SOLVE
    t0 = time()
    x = _solve_direct(A, b)
    t1 = time()
    print("\tdirect solver:\n\t\ttook {} seconds\n".format(t1 - t0))

    # ITERATIVE SOLVES
    for iterative_method in ITERATIVE_METHODS.keys():
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=iterative_method)
        t1 = time()
        print(
            "\titerative solver ({}):\n\t\ttook {} seconds".format(
                iterative_method, t1 - t0
            )
        )
