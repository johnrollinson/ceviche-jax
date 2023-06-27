from functools import partial

import jax.numpy as npj
import jax.scipy as spj
from jax import jit

from ceviche_jax.constants import *
from ceviche_jax.derivatives import compute_derivative_matrices


def get_modes(eps_cross, omega, dL, npml, m=1, filtering=True):
    """Solve for the modes of a waveguide cross section

    ARGUMENTS
        eps_cross: the permittivity profile of the waveguide
        omega:     angular frequency of the modes
        dL:        grid size of the cross section
        npml:      number of PML points on each side of the cross section
        m:         number of modes to solve for
        filtering: whether to filter out evanescent modes
    RETURNS
        vals:      array of effective indices of the modes
        vectors:   array containing the corresponding mode profiles
    """

    k0 = omega / C_0

    N = eps_cross.size

    Dxf, Dxb, Dyf, Dyb = compute_derivative_matrices(
        omega, (N, 1), (npml, 0), dL=dL
    )

    diag_eps_r = npj.diag(eps_cross.flatten(), 0)
    A = diag_eps_r + Dxf.dot(Dxb) * (1 / k0) ** 2

    n_max = npj.sqrt(npj.max(eps_cross))
    vals, vecs = solver_eigs(A)

    if filtering:
        filter_re = lambda vals: npj.real(vals) > 0.0
        # filter_im = lambda vals: npj.abs(npj.imag(vals)) <= 1e-12
        filters = [filter_re]
        vals, vecs = filter_modes(vals, vecs, filters=filters)

    if vals.size == 0:
        raise BaseException("Could not find any eigenmodes for this waveguide")

    vecs = normalize_modes(vecs)

    return vals, vecs


def insert_mode(
    omega, dx, x, y, epsr, target=None, npml=0, m=1, filtering=False
):
    """Solve for the modes in a cross section of epsr at the location defined by 'x' and 'y'

    The mode is inserted into the 'target' array if it is supplied, if the
    target array is not supplied, then a target array is created with the same
    shape as epsr, and the mode is inserted into it.
    """
    if target is None:
        target = npj.zeros(epsr.shape, dtype=complex)

    epsr_cross = epsr[x, y]
    _, mode_field = get_modes(
        epsr_cross, omega, dx, npml, m=m, filtering=filtering
    )
    target = target.at[x, y].set(npj.atleast_2d(mode_field)[:, m - 1].squeeze())

    return target


@partial(jit, backend="cpu")
def solver_eigs(A):
    """solves for eigenmodes of A

    TODO: JAX currently does not support solving for only the lowest N modes so
    we are stuck solving for all modes
    NOTE: JAX currently does not support eig on GPU, so we cannot jit this
    function
    """
    # return spj.linalg.eigh(A)
    return npj.linalg.eig(A)


def filter_modes(values, vectors, filters=None):
    """Generic Filtering Function

    TODO: Right now it seems like this filtering does not properly filter out
    the non-physical modes, need to fix
    ARGUMENTS
        values: array of effective index values
        vectors: array of mode profiles
        filters: list of functions of `values` that return True for modes satisfying the desired filter condition
    RETURNS
        vals:      array of filtered effective indices of the modes
        vectors:   array containing the corresponding, filtered mode profiles
    """

    # if no filters, just return
    if filters is None:
        return values, vectors

    # elements to keep, all for starts
    keep_elements = npj.ones(values.shape)

    for f in filters:
        keep_f = f(values)
        keep_elements = npj.logical_and(keep_elements, keep_f)

    # get the indices you want to keep
    keep_indices = npj.where(keep_elements)[0]

    # filter and return arrays
    return values[keep_indices], vectors[:, keep_indices]


@jit
def normalize_modes(vectors):
    """Normalize each `vec` in `vectors` such that `sum(|vec|^2)=1`
        vectors: array with shape (n_points, n_vectors)

    TODO: Does npj.linalg.eig normalize the eigenvectors? If so, remove this
    function
    """
    powers = npj.sum(npj.square(npj.abs(vectors)), axis=0)
    return vectors / npj.sqrt(powers)


def Ez_to_H(Ez, omega, dL, npml):
    """Converts the Ez output of mode solver to Hx and Hy components

    TODO: Do we need this function? A version of it is implemented in fdfd.py
    """

    N = Ez.size
    matrices = compute_derivative_matrices(omega, (N, 1), [npml, 0], dL=dL)
    Dxf, Dxb, Dyf, Dyb = matrices

    # save to a dictionary for convenience passing to primitives
    info_dict = {}
    info_dict["Dxf"] = Dxf
    info_dict["Dxb"] = Dxb
    info_dict["Dyf"] = Dyf
    info_dict["Dyb"] = Dyb

    Hx, Hy = Ez_to_Hx_Hy(Ez)

    return Hx, Hy


if __name__ == "__main__":
    """Test on a simple ridge waveguide"""

    import matplotlib.pylab as plt

    from ceviche_jax.fdfd import FDFD_Ez

    lambda0 = 1.550e-6  # free space wavelength (m)
    dL = lambda0 / 100
    npml = int(lambda0 / dL)  # number of grid points in PML
    omega_0 = 2 * npj.pi * C_0 / lambda0  # angular frequency (rad/s)

    Lx = lambda0 * 10  # length in horizontal direction (m)

    Nx = int(Lx / dL)

    wg_perm = 4

    wg_width = lambda0
    wg_points = npj.arange(
        Nx // 2 - int(wg_width / dL / 2), Nx // 2 + int(wg_width / dL / 2)
    )

    eps_wg = npj.ones((Nx,))
    eps_wg = eps_wg.at[wg_points].set(wg_perm)

    vals, vecs = get_modes(eps_wg, omega_0, dL, npml=10, m=10)

    plt.plot(npj.linspace(-Lx, Lx, Nx) / 2 / lambda0, npj.abs(vecs[:, :3]))
    plt.xlabel("x position ($\lambda_0$)")
    plt.ylabel("mode profile (normalized)")
    plt.show()
