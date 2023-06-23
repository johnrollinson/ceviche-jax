"""
Functions related to performing derivative operations used in the simulation 
tools.

    - The FDTD method requires curl operations, which are  performed using 
    numpy.roll
    - The FDFD method requires sparse derivative matrices, with PML added, 
    which are constructed here.

TODO: The functions in this module should be modified to not require string
inputs, as this is not a supported JAX type
TODO: Conditional control 
"""

from functools import partial

import jax.experimental.sparse as spj
import jax.numpy as npj
from jax import jit

from .constants import EPSILON_0, ETA_0

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
    (Sxf, Sxb, Syf, Syb) = create_S_matrices(omega, shape, npml, dL)

    # Apply PML to derivative matrices
    Dxf = npj.dot(Sxf, Dxf)
    Dxb = npj.dot(Sxb, Dxb)
    Dyf = npj.dot(Syf, Dyf)
    Dyb = npj.dot(Syb, Dyb)

    return Dxf, Dxb, Dyf, Dyb


""" Derivative Matrices (no PML) """


def createDws(component, direc, shape, dL, bloch_x=0.0, bloch_y=0.0):
    """Creates the derivative matrices.

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


def make_Dxf(dL, shape, bloch_x=0.0):
    """Forward derivative in x.

    Returns a sparse representation of Dxf.

    TODO: In converting to jax, I had to remove the sparse computation, not
    sure if it can be re-implemented using current jax methods
    """
    Nx, Ny = shape
    phasor_x = npj.exp(1j * bloch_x)

    x1 = npj.diag(npj.full(Nx, -1, dtype=npj.complex128), 0)
    x2 = npj.diag(npj.full(Nx - 1, 1, dtype=npj.complex128), 1)
    x3 = npj.diag(npj.full(1, phasor_x, dtype=npj.complex128), -Nx + 1)

    Dxf = x1 + x2 + x3
    Dxf = 1 / dL * npj.kron(Dxf, npj.eye(Ny, dtype=npj.complex128))

    Dxf = spj.BCOO.fromdense(Dxf)

    return Dxf


def make_Dxb(dL, shape, bloch_x=0.0):
    """Backward derivative in x.

    Returns the sparse representation of Dxb.

    TODO: Can we implement this using only sparse methods?
    """
    Nx, Ny = shape
    phasor_x = npj.exp(1j * bloch_x)

    x1 = npj.diag(npj.full(Nx, 1, dtype=npj.complex128), 0)
    x2 = npj.diag(npj.full(Nx - 1, -1, dtype=npj.complex128), -1)
    x3 = npj.diag(
        npj.full(1, -npj.conj(phasor_x), dtype=npj.complex128), Nx - 1
    )

    Dxb = x1 + x2 + x3
    Dxb = 1 / dL * npj.kron(Dxb, npj.eye(Ny, dtype=npj.complex128))

    Dxb = spj.BCOO.fromdense(Dxb)

    return Dxb


def make_Dyf(dL, shape, bloch_y=0.0):
    """Forward derivative in y"""
    Nx, Ny = shape
    phasor_y = npj.exp(1j * bloch_y)

    y1 = npj.diag(npj.full(Ny, -1, dtype=npj.complex128), 0)
    y2 = npj.diag(npj.full(Ny - 1, 1, dtype=npj.complex128), 1)
    y3 = npj.diag(npj.full(1, phasor_y, dtype=npj.complex128), -Ny + 1)

    Dyf = y1 + y2 + y3
    Dyf = 1 / dL * npj.kron(npj.eye(Nx, dtype=npj.complex128), Dyf)

    Dyf = spj.BCOO.fromdense(Dyf)

    return Dyf


def make_Dyb(dL, shape, bloch_y=0.0):
    """Backward derivative in y"""
    Nx, Ny = shape
    phasor_y = npj.exp(1j * bloch_y)

    y1 = npj.diag(npj.full(Ny, 1, dtype=npj.complex128), 0)
    y2 = npj.diag(npj.full(Ny - 1, -1, dtype=npj.complex128), 1)
    y3 = npj.diag(
        npj.full(1, -npj.conj(phasor_y), dtype=npj.complex128), Ny - 1
    )

    Dyb = y1 + y2 + y3
    Dyb = 1 / dL * npj.kron(npj.eye(Nx, dtype=npj.complex128), Dyb)

    Dyb = spj.BCOO.fromdense(Dyb)

    return Dyb


""" PML Functions """


@jit
def create_S_matrices(omega, shape, npml, dL):
    """Makes the 'S-matrices'.

    When dotted with derivative matrices, the S-matrices add the PML.
    """

    # strip out some information needed
    Nx, Ny = shape
    Nx_pml, Ny_pml = npml

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor("f", omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor("b", omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor("f", omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor("b", omega, dL, Ny, Ny_pml)

    # TODO: Do we actually need this initialization? Could we just create the 2D
    # array from npj.full or similar?
    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = npj.zeros(shape, dtype=npj.complex128)
    Sx_b_2D = npj.zeros(shape, dtype=npj.complex128)
    Sy_f_2D = npj.zeros(shape, dtype=npj.complex128)
    Sy_b_2D = npj.zeros(shape, dtype=npj.complex128)

    # insert the cross sections into the S-grids
    Sx_f_2D = Sx_f_2D.at[:, :Ny].set(1 / s_vector_x_f)
    Sx_b_2D = Sx_b_2D.at[:, :Ny].set(1 / s_vector_x_b)
    Sy_f_2D = Sy_f_2D.at[:Nx, :].set(1 / s_vector_y_f)
    Sy_b_2D = Sy_b_2D.at[:Nx, :].set(1 / s_vector_y_b)

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.flatten()
    Sx_b_vec = Sx_b_2D.flatten()
    Sy_f_vec = Sy_f_2D.flatten()
    Sy_b_vec = Sy_b_2D.flatten()

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = spj.BCOO.fromdense(npj.diag(Sx_f_vec, 0))
    Sx_b = spj.BCOO.fromdense(npj.diag(Sx_b_vec, 0))
    Sy_f = spj.BCOO.fromdense(npj.diag(Sy_f_vec, 0))
    Sy_b = spj.BCOO.fromdense(npj.diag(Sy_b_vec, 0))

    return Sx_f, Sx_b, Sy_f, Sy_b


@partial(jit, static_argnums=(0, 3, 4))
def create_sfactor(direc, omega, dL, N, N_pml):
    """creates the S-factor cross section needed in the S-matrices"""

    # for no PML, this should just be ones
    # if N_pml == 0:
    #     return npj.ones(N, dtype=npj.complex128)

    # otherwise, get different profiles for forward and reverse derivative
    # matrices
    d_w = N_pml * dL
    if direc == "f":
        return create_sfactor_f(omega, dL, N, N_pml, d_w)
    elif direc == "b":
        return create_sfactor_b(omega, dL, N, N_pml, d_w)
    else:
        raise ValueError(f"Direction value {direc} not recognized")


# @partial(jit, static_argnums=(2, 3))
def create_sfactor_f(omega, dL, N, N_pml, dw):
    """S-factor profile for forward derivative matrix"""
    idx_lower = npj.arange(N_pml + 1)
    idx_upper = npj.arange(N - N_pml + 1, N)
    sf_lower = s_value(dL * (N_pml - idx_lower + 0.5), dw, omega)
    sf_upper = s_value(dL * (idx_upper - (N - N_pml) - 0.5), dw, omega)

    sfactor_array = npj.concatenate(
        [sf_lower, npj.ones(N - 2 * N_pml, dtype=npj.complex128), sf_upper]
    )

    return sfactor_array


# @partial(jit, static_argnums=(2, 3))
def create_sfactor_b(omega, dL, N, N_pml, dw):
    """S-factor profile for backward derivative matrix"""
    idx_lower = npj.arange(N_pml + 1)
    idx_upper = npj.arange(N - N_pml + 1, N)
    sf_lower = s_value(dL * (N_pml - idx_lower + 1), dw, omega)
    sf_upper = s_value(dL * (idx_upper - (N - N_pml) - 1), dw, omega)

    sfactor_array = npj.concatenate(
        [sf_lower, npj.ones(N - 2 * N_pml, dtype=npj.complex128), sf_upper]
    )

    return sfactor_array


@jit
def sig_w(l, dw, m=3, lnR=-30):
    """Fictional conductivity, note that these values might need tuning"""
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw) ** m


@jit
def s_value(l, dw, omega):
    """S-value to use in the S-matrices"""
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)
