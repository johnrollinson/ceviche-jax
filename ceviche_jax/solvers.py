""" This file stores the various sparse linear system solvers you can use for
FDFD """

from functools import partial

import jax.experimental.sparse.linalg as spjle
import jax.scipy.sparse.linalg as spjl
from jax import jit

# default iterative method to use
DEFAULT_ITERATIVE_METHOD = "bicgstab"

# dict of iterative methods supported (name: function)
# NOTE: Currently, JAX only supports the bicgstab, cg, and gmres methods
ITERATIVE_METHODS = {
    "bicgstab": spjl.bicgstab,
    "cg": spjl.cg,
    "gmres": spjl.gmres,
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
    """Direct solver.

    NOTE: Currently spsolve only supports CSR array types and the CUDA GPU
    backend
    """

    return spjle.spsolve(A.data, A.indices, A.indptr, b, tol=ATOL)


@partial(jit, static_argnums=2)
def _solve_iterative(A, b, iterative_method=DEFAULT_ITERATIVE_METHOD):
    """Iterative solver"""
    solver_fn = ITERATIVE_METHODS[iterative_method]
    x, info = solver_fn(A, b, atol=ATOL)  # Note that info is just a placeholder
    return x


if __name__ == "__main__":
    """========================= SPEED TESTS ============================="""

    # to run speed tests use `python -W ignore ceviche/solvers.py` to suppress
    # warnings

    from time import time

    import jax.experimental.sparse as spj
    import numpy as np
    import scipy.sparse as sp

    N = 100  # dimension of the x, and b vectors
    density = 0.3  # sparsity of the dense matrix
    A = spj.BCSR.from_scipy_sparse(sp.random(N, N, density=density))
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

    m, n = 200, 200
    print("\tfor dimensions = {}\n".format((m, n)))
    eps_r = random.uniform(key, (m, n), dtype=np.float32) + 1
    b = random.uniform(key, (m * n,), dtype=np.float32) - 0.5
    b = b.astype(np.complex64)

    npml = 20
    dl = 2e-8
    lambda0 = 1550e-9
    omega0 = 2 * np.pi * C_0 / lambda0

    F = FDFD_Ez(omega0, dl, eps_r, [10, 0])
    A = F._make_A(eps_r.flatten())

    # DIRECT SOLVE
    # t0 = time()
    # A_bcsr = spj.BCSR.from_bcoo(A)
    # x = _solve_direct(A_bcsr, b)
    # t1 = time()
    # print("\tdirect solver:\n\t\ttook {} seconds\n".format(t1 - t0))

    # ITERATIVE SOLVES
    for iterative_method in ITERATIVE_METHODS.keys()[-1]:
        t0 = time()
        x = _solve_iterative(A, b, iterative_method=iterative_method)
        t1 = time()
        print(
            "\titerative solver ({}):\n\t\ttook {} seconds".format(
                iterative_method, t1 - t0
            )
        )
