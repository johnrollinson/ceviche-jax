"""
Functions related to performing derivative operations used in the simulation 
tools.

    - The FDTD method requires curl operations, which are  performed using 
    numpy.roll
    - The FDFD method requires sparse derivative matrices, with PML added, 
    which are constructed here.

TODO: I feel like there is some inefficiency somewhere in this module - do the
derivate matrices really need to be (Nx*Ny),(Nx*Ny)? Seems like we should be
able to reduce this somehow but I don't quite understand yet

TODO: The functions in this module should be modified to not require string
inputs, as this is not a supported JAX type

TODO: Conditional control should be implemented using something like lax.switch
or similar (related to removing strings)
"""

from functools import partial

import jax.scipy as spj
import jax.numpy as npj
import scipy.sparse as sp
from jax import jit

from ceviche_jax.constants import COMPLEX, EPSILON_0, ETA_0, FLOAT
from ceviche_jax.primitives import sp_mult, spsp_kron, spsp_mult

""" =========================== CURLS FOR FDTD =========================== """


@jit
def curl_E(axis, Ex, Ey, Ez, dL):
    if axis == 0:
        t1 = (npj.roll(Ez, shift=-1, axis=1) - Ez) / dL
        t2 = (npj.roll(Ey, shift=-1, axis=2) - Ey) / dL
        return t1 - t2
    elif axis == 1:
        t1 = (npj.roll(Ex, shift=-1, axis=2) - Ex) / dL
        t2 = (npj.roll(Ez, shift=-1, axis=0) - Ez) / dL
        return t1 - t2
    elif axis == 2:
        t1 = (npj.roll(Ey, shift=-1, axis=0) - Ey) / dL
        t2 = (npj.roll(Ex, shift=-1, axis=1) - Ex) / dL
        return t1 - t2


@jit
def curl_H(axis, Hx, Hy, Hz, dL):
    if axis == 0:
        t1 = (Hz - npj.roll(Hz, shift=1, axis=1)) / dL
        t2 = (Hy - npj.roll(Hy, shift=1, axis=2)) / dL
        return t1 - t2
    elif axis == 1:
        t1 = (Hx - npj.roll(Hx, shift=1, axis=2)) / dL
        t2 = (Hz - npj.roll(Hz, shift=1, axis=0)) / dL
        return t1 - t2
    elif axis == 2:
        t1 = (Hy - npj.roll(Hy, shift=1, axis=0)) / dL
        t2 = (Hx - npj.roll(Hx, shift=1, axis=1)) / dL
        return t1 - t2


""" ========= STUFF THAT CONSTRUCTS THE DERIVATIVE MATRIX ================= """


def compute_derivative_matrices(
    omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0
):
    """Returns sparse derivative matrices.  Currently works for 2D and 1D

    omega: angular frequency (rad/sec)
    shape: shape of the FDFD grid
    npml: list of number of PML cells in x and y.
    dL: spatial grid size (m)
    block_x: bloch phase (phase across periodic boundary) in x
    block_y: bloch phase (phase across periodic boundary) in y
    """

    # Construct derivate matrices without PML
    Dxf = createDws("x", "f", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dxb = createDws("x", "b", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyf = createDws("y", "f", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyb = createDws("y", "b", shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)

    # Make the S-matrices for PML
    Sxf, Sxb, Syf, Syb = create_S_matrices(omega, shape, npml, dL)

    # Apply PML to derivative matrices
    Dxf = spsp_mult(Sxf, Dxf)
    Dxb = spsp_mult(Sxb, Dxb)
    Dyf = spsp_mult(Syf, Dyf)
    Dyb = spsp_mult(Syb, Dyb)

    return Dxf, Dxb, Dyf, Dyb


""" Derivative Matrices (no PML) """


@partial(jit, static_argnums=(0, 1, 2))
def createDws(component, direc, shape, dL, bloch_x=0.0, bloch_y=0.0):
    """Creates the derivative matrices.

    TODO: Convert string args to JAX

    component: one of 'x' or 'y' for derivative in x or y direction
    dir: one of 'f' or 'b', whether to take forward or backward finite
        difference
    shape: shape of the FDFD grid
    dL: spatial grid size (m)
    block_x: bloch phase (phase across periodic boundary) in x
    block_y: bloch phase (phase across periodic boundary) in y
    """

    Nx, Ny = shape

    # special case, a 1D problem
    if component == "x" and Nx == 1:
        return npj.eye(Ny)
    if component == "y" and Ny == 1:
        return npj.eye(Nx)

    # select a `make_DXX` function based on the component and direction
    component_dir = component + direc
    if component_dir == "xf":
        return make_Dxf(dL, shape, bloch_x=bloch_x)
    elif component_dir == "xb":
        return make_Dxb(dL, shape, bloch_x=bloch_x)
    elif component_dir == "yf":
        return make_Dyf(dL, shape, bloch_y=bloch_y)
    elif component_dir == "yb":
        return make_Dyb(dL, shape, bloch_y=bloch_y)
    else:
        raise ValueError(
            f"component and direction {component} and {direc} not recognized"
        )


@partial(jit, static_argnums=1)
def make_Dxf(dL, shape, bloch_x=0.0):
    """Forward derivative in x.

    Returns a sparse representation of Dxf.
    """
    Nx, Ny = shape
    phasor_x = npj.exp(1j * bloch_x)
    d1 = -npj.eye(Nx, k=0)
    d2 = npj.eye(Nx, k=-1)
    d3 = sp_mult(npj.eye(Nx, k=Nx - 1), phasor_x)
    Dxf = d1 + d2 + d3
    Dxf = sp_mult(spsp_kron(Dxf, npj.eye(Ny)), 1 / dL)

    return Dxf.reshape(Nx * Ny, Nx * Ny)


@partial(jit, static_argnums=1)
def make_Dxb(dL, shape, bloch_x=0.0):
    """Backward derivative in x.

    Returns the sparse representation of Dxb.
    """
    Nx, Ny = shape
    phasor_x = npj.exp(1j * bloch_x)
    d1 = npj.eye(Nx, k=0)
    d2 = -npj.eye(Nx, k=1)
    d3 = sp_mult(npj.eye(Nx, k=-Nx + 1), -npj.conj(phasor_x))
    Dxb = d1 + d2 + d3
    Dxb = sp_mult(spsp_kron(Dxb, npj.eye(Ny)), 1 / dL)

    return Dxb.reshape(Nx * Ny, Nx * Ny)


@partial(jit, static_argnums=1)
def make_Dyf(dL, shape, bloch_y=0.0):
    """Forward derivative in y"""
    Nx, Ny = shape
    phasor_y = npj.exp(1j * bloch_y)
    d1 = -npj.eye(Ny, k=0)
    d2 = npj.eye(Ny, k=-1)
    d3 = sp_mult(npj.eye(Ny, k=Ny - 1), phasor_y)
    Dyf = d1 + d2 + d3
    Dyf = sp_mult(spsp_kron(npj.eye(Nx), Dyf), 1 / dL)

    return Dyf.reshape(Nx * Ny, Nx * Ny)


@partial(jit, static_argnums=1)
def make_Dyb(dL, shape, bloch_y=0.0):
    """Backward derivative in y"""
    Nx, Ny = shape
    phasor_y = npj.exp(1j * bloch_y)
    d1 = npj.eye(Ny, k=0)
    d2 = -npj.eye(Ny, k=1)
    d3 = sp_mult(npj.eye(Ny, k=-Ny + 1), -npj.conj(phasor_y))
    Dyb = d1 + d2 + d3
    Dyb = sp_mult(spsp_kron(npj.eye(Nx), Dyb), 1 / dL)

    return Dyb.reshape(Nx * Ny, Nx * Ny)


""" PML Functions """


@partial(jit, static_argnums=(1, 2))
def create_S_matrices(omega, shape, npml, dL):
    """Makes the 'S-matrices'.

    When dotted with derivative matrices, the S-matrices add the PML.

    TODO: If this is just producing a diagonal matrix, do we actually need to
    output a BCOO matrix?
    """

    # strip out some information needed
    Nx, Ny = shape
    Nx_pml, Ny_pml = npml

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor("f", omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor("b", omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor("f", omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor("b", omega, dL, Ny, Ny_pml)

    # insert the cross sections into the S-grids
    Sx_f_vec = npj.repeat(1 / s_vector_x_f, Ny)
    Sx_b_vec = npj.repeat(1 / s_vector_x_b, Ny)
    Sy_f_vec = npj.tile(1 / s_vector_y_f, Nx)
    Sy_b_vec = npj.tile(1 / s_vector_y_b, Nx)

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = npj.eye(Nx * Ny) * Sx_f_vec
    Sx_b = npj.eye(Nx * Ny) * Sx_b_vec
    Sy_f = npj.eye(Nx * Ny) * Sy_f_vec
    Sy_b = npj.eye(Nx * Ny) * Sy_b_vec

    return Sx_f, Sx_b, Sy_f, Sy_b


@partial(jit, static_argnums=(0, 3, 4))
def create_sfactor(direc, omega, dL, N, N_pml):
    """creates the S-factor cross section needed in the S-matrices"""

    # for no PML, this should just be ones
    if N_pml == 0:
        return npj.ones(N, dtype=COMPLEX)

    # otherwise, get different profiles for forward and reverse derivative
    # matrices
    d_w = N_pml * dL
    if direc == "f":
        return create_sfactor_f(omega, dL, N, N_pml, d_w)
    elif direc == "b":
        return create_sfactor_b(omega, dL, N, N_pml, d_w)
    else:
        raise ValueError(f"Direction value {direc} not recognized")


@partial(jit, static_argnums=(2, 3))
def create_sfactor_f(omega, dL, N, N_pml, dw):
    """S-factor profile for forward derivative matrix"""
    idx_lower = npj.arange(N_pml + 1)
    idx_upper = npj.arange(N - N_pml + 1, N)
    sf_lower = s_value(dL * (N_pml - idx_lower + 0.5), dw, omega)
    sf_upper = s_value(dL * (idx_upper - (N - N_pml) - 0.5), dw, omega)

    sfactor_vec = npj.concatenate(
        [sf_lower, npj.ones(N - 2 * N_pml, dtype=COMPLEX), sf_upper]
    )

    return sfactor_vec


@partial(jit, static_argnums=(2, 3))
def create_sfactor_b(omega, dL, N, N_pml, dw):
    """S-factor profile for backward derivative matrix"""
    idx_lower = npj.arange(N_pml + 1)
    idx_upper = npj.arange(N - N_pml + 1, N)
    sf_lower = s_value(dL * (N_pml - idx_lower + 1), dw, omega)
    sf_upper = s_value(dL * (idx_upper - (N - N_pml) - 1), dw, omega)

    sfactor_vec = npj.concatenate(
        [sf_lower, npj.ones(N - 2 * N_pml, dtype=COMPLEX), sf_upper]
    )

    return sfactor_vec


@jit
def sig_w(l, dw, m=3, lnR=-30):
    """Fictional conductivity, note that these values might need tuning"""
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw) ** m


@jit
def s_value(l, dw, omega):
    """S-value to use in the S-matrices"""
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)


if __name__ == "__main__":
    import sys
    from time import time
    from timeit import timeit

    import numpy as np

    np.set_printoptions(linewidth=np.inf)

    import ceviche.derivatives as cd

    COMPARE_D = False
    COMPARE_S = False

    n = 4
    m = 4
    shape = (n, m)
    n_pml = 0
    omega = 1
    d_l = 0.1
    d_w = d_l * n_pml

    if COMPARE_D:
        print(
            npj.allclose(
                cd.make_Dxf(d_l, shape).A, make_Dxf(d_l, shape).todense().T
            )
        )
        print(
            npj.allclose(
                cd.make_Dxb(d_l, shape).A, make_Dxb(d_l, shape).todense().T
            )
        )
        print(
            npj.allclose(
                cd.make_Dyf(d_l, shape).A, make_Dyf(d_l, shape).todense().T
            )
        )
        print(
            npj.allclose(
                cd.make_Dyb(d_l, shape).A, make_Dyb(d_l, shape).todense().T
            )
        )

        t0 = time()
        cd.make_Dxf(d_l, shape)
        t1 = time()
        t2 = time()
        make_Dxf(d_l, shape)
        t3 = time()
        print(f"make_Dxf: Ceviche time: {t1-t0};\t Ceviche-jax time: {t3-t2}")

    if COMPARE_S:
        Sxf, Sxb, Syf, Syb = cd.create_S_matrices(
            omega, shape, (n_pml, n_pml), d_l
        )
        Sxfj, Sxbj, Syfj, Sybj = create_S_matrices(
            omega, shape, (n_pml, n_pml), d_l
        )

        print(npj.allclose(Sxf.A, Sxfj.todense()))
        print(npj.allclose(Sxb.A, Sxbj.todense()))
        print(npj.allclose(Syf.A, Syfj.todense()))
        print(npj.allclose(Syb.A, Sybj.todense()))

    t0 = time()
    dm = cd.compute_derivative_matrices(omega, (n, m), (n_pml, n_pml), d_l)
    t1 = time()
    print(f"Ceviche derivative time: {t1-t0}")

    dm_jax = compute_derivative_matrices(
        omega,
        (n, m),
        (n_pml, n_pml),
        d_l,
    )
    t0 = time()
    dm_jax = compute_derivative_matrices(
        omega,
        (n, m),
        (n_pml, n_pml),
        d_l,
    )
    t1 = time()
    print(f"Ceviche-jax derivative time: {t1-t0}")

    for d, d_jax in zip(dm, dm_jax):
        print(npj.allclose(d.todense(), d_jax))
