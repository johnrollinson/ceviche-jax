"""This file defines the very lowest level sparse matrix primitives that allow
autograd to be compatible with FDFD.  One needs to define the derivatives of Ax
= b and x = A^-1 b for sparse A.

TODO: Update the docstring for this module

This is done using the entries and indices of A, instead of the sparse matrix 
objects, since autograd doesn't know how to handle those as arguments to 
functions.

GUIDE TO THE PRIMITIVES DEFINED BELOW:
    naming convention for gradient functions:
        "def grad_{function_name}_{argument_name}_{mode}"
    defines the derivative of `function_name` with respect to 
    `argument_name` using `mode`-mode differentiation    
    where 'mode' is one of 'reverse' or 'forward'

    These functions define the basic operations needed for FDFD and also their 
    derivatives in a form that autograd can understand. This allows you to use 
    fdfd classes in autograd functions. The code is organized so that autograd 
    never sees sparse matrices in arguments, since it doesn't know how to 
    handle them. Look but don't touch!

    NOTES for the curious (since this information isn't in autograd 
    documentation...)

        To define a function as being trackable by autograd, need to add the 
        @primitive decorator

    REVERSE MODE
        'vjp' defines the vector-jacobian product for reverse mode (adjoint)
        a vjp_maker function takes as arguments
            1. the output of the @primitive
            2. the rest of the original arguments in the @primitive
        and returns
            a *function* of the backprop vector (v) that defines the operation
            (d{function} / d{argument_i})^T @ v

    FORWARD MODE:
        'jvp' defines the jacobian-vector product for forward mode (FMD)
        a jvp_maker function takes as arguments
            1. the forward propagating vector (g)
            2. the output of the @primitive
            3. the rest of the original arguments in the @primitive
        and returns
            (d{function} / d{argument_i}) @ g

    After this, you need to link the @primitive to its vjp/jvp using
    defvjp(function, arg1's vjp, arg2's vjp, ...)
    defjvp(function, arg1's jvp, arg2's jvp, ...)
"""

import autograd as ag
import autograd.numpy as npa
import jax.experimental.sparse as spj
from jax import jit

from ceviche_jax.solvers import solve_linear
from ceviche_jax.utils import (
    der_num,
    get_entries_indices,
    grad_num,
    make_IO_matrices,
    make_rand,
    make_rand_complex,
    make_rand_indices,
    make_rand_sparse,
    make_sparse,
    transpose_indices,
)

""" ========== Sparse Matrix-Vector Multiplication ===================== """


@spj.sparsify
def sp_mult(A, x):
    """Multiply a sparse matrix (A) by a dense vector (x)

    TODO: Do we still need this function? Can we just use the JAX methods?

    NOTE: The bcoo_multiply_dense does not seem to produce the correct output,
    so we're just using the '@' here. Not sure what the implication for speed
    it.

    Args:
        A: A sparse JAX array
        x: 1d numpy array specifying the vector to multiply by.
    Returns:
        A dense 1d numpy array corresponding to the result (b) of A * x = b.
    """
    return A @ x


def grad_sp_mult_entries_reverse(ans, entries, indices, x):
    # x^T @ dA/de^T @ v => the outer product of x and v using the indices of A
    ia, ja = indices

    def vjp(v):
        return v[ia] * x[ja]

    return vjp


def grad_sp_mult_x_reverse(b, entries, indices, x):
    # dx/de^T @ A^T @ v => multiplying A^T by v
    indices_T = transpose_indices(indices)

    def vjp(v):
        return sp_mult(entries, indices_T, v)

    return vjp


ag.extend.defvjp(
    sp_mult, grad_sp_mult_entries_reverse, None, grad_sp_mult_x_reverse
)


def grad_sp_mult_entries_forward(g, b, entries, indices, x):
    # dA/de @ x @ g => use `g` as the entries into A and multiply by x
    return sp_mult(g, indices, x)


def grad_sp_mult_x_forward(g, b, entries, indices, x):
    # A @ dx/de @ g -> simply multiply A @ g
    return sp_mult(entries, indices, g)


ag.extend.defjvp(
    sp_mult, grad_sp_mult_entries_forward, None, grad_sp_mult_x_forward
)


""" ================== Sparse Matrix-Vector Solve ========================= """


@jit
def sp_solve(A, b):
    """Solve a sparse matrix (A) with source (b)

    TODO: Wrapper for solve_linear, consider removing in future

    Args:
      A: A sparse JAX array
      b: 1d jax.numpy array specifying the source.
    Returns:
      1d numpy array corresponding to the solution of A * x = b.
    """
    return solve_linear(A, b)


def grad_sp_solve_entries_reverse(x, entries, indices, b):
    # x^T @ dA/de^T @ A_inv^T @ -v => do the solve on the RHS, then take outer
    # product with x using indices of A
    indices_T = transpose_indices(indices)
    i, j = indices

    def vjp(v):
        adj = sp_solve(entries, indices_T, -v)
        return adj[i] * x[j]

    return vjp


def grad_sp_solve_b_reverse(ans, entries, indices, b):
    # dx/de^T @ A_inv^T @ v => do the solve on the RHS and you're done.
    indices_T = transpose_indices(indices)

    def vjp(v):
        return sp_solve(entries, indices_T, v)

    return vjp


ag.extend.defvjp(
    sp_solve, grad_sp_solve_entries_reverse, None, grad_sp_solve_b_reverse
)


def grad_sp_solve_entries_forward(g, x, entries, indices, b):
    # -A_inv @ dA/de @ A_inv @ b @ g => insert x = A_inv @ b and multiply with g using A indices.  Then solve as source for A_inv.
    forward = sp_mult(g, indices, x)
    return sp_solve(entries, indices, -forward)


def grad_sp_solve_b_forward(g, x, entries, indices, b):
    # A_inv @ db/de @ g => simply solve A_inv @ g
    return sp_solve(entries, indices, g)


ag.extend.defjvp(
    sp_solve, grad_sp_solve_entries_forward, None, grad_sp_solve_b_forward
)


""" ========= Sparse Matrix-Sparse Matrix Multiplication =================== """

@spj.sparsify
def spsp_mult(A, B):
    """Multiply a sparse matrix (A) by a sparse matrix (X) A @ X = B

    TODO: Wrapper function for the JAX sparse-sparse mult., consider removing

    NOTE: The bcoo_multiply_sparse does not seem to produce the correct output,
    so we're just using the '@' here. Not sure what the implication for speed
    it.

    Args:
        A: First term, A, sparse JAX array
        A: Second term, B, sparse JAX array
    Returns:
        Matrix-Matrix product of A and B in a sparse representation
    """
    return A @ B


def grad_spsp_mult_entries_a_reverse(
    b_out, entries_a, indices_a, entries_x, indices_x, N
):
    """For AX=B, we want to relate the entries of A to the entries of B.
    The goal is to compute the gradient of the output entries with respect to the input.
    For this, though, we need to convert into matrix form, do our computation, and convert back to the entries.
    If you write out the matrix elements and do the calculation, you can derive the code below, but it's a hairy derivation.
    """

    # make the indices matrices for A
    _, indices_b = b_out
    Ia, Oa = make_IO_matrices(indices_a, N)

    def vjp(v):
        # multiply the v_entries with X^T using the indices of B
        entries_v, _ = v
        indices_xT = transpose_indices(indices_x)
        entries_vxt, indices_vxt = spsp_mult(
            entries_v, indices_b, entries_x, indices_xT, N
        )

        # rutn this into a sparse matrix and convert to the basis of A's indices
        VXT = make_sparse(entries_vxt, indices_vxt, shape=(N, N))
        M = (Ia.T).dot(VXT).dot(Oa.T)

        # return the diagonal elements, which contain the entries
        return M.diagonal()

    return vjp


def grad_spsp_mult_entries_x_reverse(
    b_out, entries_a, indices_a, entries_x, indices_x, N
):
    """Now we wish to do the gradient with respect to the X matrix in AX=B
    Instead of doing it all out again, we just use the previous grad function on the transpose equation X^T A^T = B^T
    """

    # get the transposes of the original problem
    entries_b, indices_b = b_out
    indices_aT = transpose_indices(indices_a)
    indices_xT = transpose_indices(indices_x)
    indices_bT = transpose_indices(indices_b)
    b_T_out = entries_b, indices_bT

    # call the vjp maker for AX=B using the substitution A=>X^T, X=>A^T, B=>B^T
    vjp_XT_AT = grad_spsp_mult_entries_a_reverse(
        b_T_out, entries_x, indices_xT, entries_a, indices_aT, N
    )

    # return the function of the transpose vjp maker being called on the backprop vector
    return lambda v: vjp_XT_AT(v)


ag.extend.defvjp(
    spsp_mult,
    grad_spsp_mult_entries_a_reverse,
    None,
    grad_spsp_mult_entries_x_reverse,
    None,
    None,
)


def grad_spsp_mult_entries_a_forward(
    g, b_out, entries_a, indices_a, entries_x, indices_x, N
):
    """Forward mode is not much better than reverse mode, but the same general logic aoplies:
    Convert to matrix form, do the calculation, convert back to entries.
        dA/de @ x @ g
    """

    # get the IO indices matrices for B
    _, indices_b = b_out
    Mb = indices_b.shape[1]
    Ib, Ob = make_IO_matrices(indices_b, N)

    # multiply g by X using a's index
    entries_gX, indices_gX = spsp_mult(g, indices_a, entries_x, indices_x, N)
    gX = make_sparse(entries_gX, indices_gX, shape=(N, N))

    # convert these entries and indides into the basis of the indices of B
    M = (Ib.T).dot(gX).dot(Ob.T)

    # return the diagonal (resulting entries) and indices of 0 (because indices are not affected by entries)
    return M.diagonal(), npa.zeros(Mb)


def grad_spsp_mult_entries_x_forward(
    g, b_out, entries_a, indices_a, entries_x, indices_x, N
):
    """Same trick as before: Reuse the previous VJP but for the transpose system"""

    # Transpose A, X, and B
    indices_aT = transpose_indices(indices_a)
    indices_xT = transpose_indices(indices_x)
    entries_b, indices_b = b_out
    indices_bT = transpose_indices(indices_b)
    b_T_out = entries_b, indices_bT

    # return the jvp of B^T = X^T A^T
    return grad_spsp_mult_entries_a_forward(
        g, b_T_out, entries_x, indices_xT, entries_a, indices_aT, N
    )


ag.extend.defjvp(
    spsp_mult,
    grad_spsp_mult_entries_a_forward,
    None,
    grad_spsp_mult_entries_x_forward,
    None,
    None,
)


""" ========================== Nonlinear Solve ========================== """

# this is just a sketch of how to do problems involving sparse matrix solves
# with nonlinear elements...  WIP.


def sp_solve_nl(parameters, a_indices, b, fn_nl):
    """
    parameters: entries into matrix A are function of parameters and solution x
    a_indices: indices into sparse A matrix
    b: source vector for A(xx = b
    fn_nl: describes how the entries of a depend on the solution of A(x,p) @ x = b and the parameters  `a_entries = fn_nl(params, x)`
    """

    # do the actual nonlinear solve in `_solve_nl_problem` (using newton, picard, whatever)
    # this tells you the final entries into A given the parameters and the nonlinear function.
    a_entries = ceviche.solvers._solve_nl_problem(
        parameters, a_indices, fn_nl, a_entries0=None
    )  # optinally, give starting a_entries
    x = sp_solve(a_entries, a_indices, b)  # the final solution to A(x) x = b
    return x


def grad_sp_solve_nl_parameters(x, parameters, a_indices, b, fn_nl):
    """
    We are finding the solution (x) to the nonlinear function:

        f = A(x, p) @ x - b = 0

    And need to define the vjp of the solution (x) with respect to the parameters (p)

        vjp(v) = (dx / dp)^T @ v

    To do this (see Eq. 5 of https://pubs-acs-org.stanford.idm.oclc.org/doi/pdf/10.1021/acsphotonics.8b01522)
    we need to solve the following linear system:

        [ df  / dx,  df  / dx*] [ dx  / dp ] = -[ df  / dp]
        [ df* / dx,  df* / dx*] [ dx* / dp ]    [ df* / dp]

    Note that we need to explicitly make A a function of x and x* for complex x

    In our case:

        (df / dx)  = (dA / dx) @ x + A
        (df / dx*) = (dA / dx*) @ x
        (df / dp)  = (dA / dp) @ x

    How do we put this into code?  Let

        A(x, p) @ x -> Ax = sp_mult(entries_a(x, p), indices_a, x)

    Since we already defined the primitive of sp_mult, we can just do:

        (dA / dx) @ x -> ag.jacobian(Ax, 0)

    Now how about the source term?

        (dA / dp) @ x -> ag.jacobian(Ax, 1)

    Note that this is a matrix, not a vector.
    We'll have to handle dA/dx* but this can probably be done, maybe with autograd directly.

    Other than this, assuming entries_a(x, p) is fully autograd compatible, we can get these terms no problem!

    Coming back to our problem, we actually need to compute:

        (dx / dp)^T @ v

    Because

        (dx / dp) = -(df / dx)^{-1} @ (df / dp)

    (ignoring the complex conjugate terms).  We can write this vjp as

        (df / dp)^T @ (df / dx)^{-T} @ v

    Since df / dp is a matrix, not a vector, its more efficient to do the mat_mul on the right first.
    So we first solve

        adjoint(v) = -(df / dx)^{-T} @ v
                   => sp_solve(entries_a_big, transpose(indices_a_big), -v)

    and then it's a simple matter of doing the matrix multiplication

        vjp(v) = (df / dp)^T @ adjoint(v)
               => sp_mult(entries_dfdp, transpose(indices_dfdp), adjoint)

    and then return the result, making sure to strip the complex conjugate.

        return vjp[:N]
    """

    def vjp(v):
        raise NotImplementedError

    return vjp


def grad_sp_solve_nl_b(x, parameters, a_indices, b, fn_nl):
    """
    Computing the derivative w.r.t b is simpler

        f = A(x) @ x - b(p) = 0

    And now the terms we need are

        df / dx  = (dA / dx) @ x + A
        df / dx* = (dA / dx*) @ x
        df / dp  = -(db / dp)

    So it's basically the same problem with a differenct source term now.
    """

    def vjp(v):
        raise NotImplementedError

    return vjp


ag.extend.defvjp(
    sp_solve_nl, grad_sp_solve_nl_parameters, None, grad_sp_solve_nl_b, None
)


if __name__ == "__main__":
    """Testing ground for sparse-sparse matmul primitives."""
    import jax.numpy as npj
    import numpy as np
    import scipy.sparse as sp

    n = 200

    M = sp.random(n, n, density=0.2)
    x = np.random.random(n)

    M_sp = spj.BCOO.fromdense(M.A)

    # prod_ref = M @ x
    # prod_cpj = jit(sp_mult)(M_sp, x)

    # print(npj.allclose(prod_ref, prod_cpj))

    B = sp.random(n, n, density=0.2)
    B_sp = spj.BCOO.fromdense(B.A)

    matmat_ref = M @ B
    matmat_cpj = jit(spsp_mult)(M_sp, B_sp)

    print(npj.allclose(matmat_ref.todense(), matmat_cpj.todense()))
    print(npj.allclose(matmat_ref.todense(), matmat_cpj.todense()))
