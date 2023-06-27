# notation is similar to that used in:
# https://www.jpier.org/ac_api/download.php?id=11092006
# TODO: Probably need to remove these classes and make it more 'functional' to
# be JAX-friendly

from functools import partial

import jax.numpy as npj
from jax import jit

from ceviche_jax.constants import *
from ceviche_jax.derivatives import compute_derivative_matrices
from ceviche_jax.primitives import sp_mult, sp_solve, spsp_mult
from ceviche_jax.solvers import _solve_iterative


class FDFD:
    """Base class for FDFD simulation"""

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        """initialize with a given structure and source
        omega: angular frequency (rad/s)
        dL: grid cell size (m)
        eps_r: array containing relative permittivity
        npml: list of number of PML grid cells in [x, y]
        bloch_{x,y}: phase difference across {x,y} boundaries for bloch
            periodic boundary conditions (default = 0 = periodic)
        """

        self.omega = omega
        self.dL = dL
        self.eps_r = eps_r
        self.npml = npml

        self._init_bloch_phases(bloch_phases)
        self._init_derivatives()

    """ Getter and setter for the permittivity of the fdfd simulation region """

    @property
    def eps_r(self):
        """Returns the relative permittivity grid"""
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """Defines some attributes when eps_r is set."""
        self._save_shape(new_eps)
        self._eps_r = new_eps

    """ classes inherited from fdfd() must implement their own versions of these
    functions for `fdfd.solve()` to work 
    """

    def _make_A(self, eps_r):
        """Construct the entries and indices into the system matrix"""
        raise NotImplementedError("need to make a _make_A() method")

    def _solve_fn(self, eps_vec, A, source_vec):
        """Returns the x, y, and z field components from the system matrix and
        source"""
        raise NotImplementedError(
            "need to implement function to solve for field components"
        )

    """ You call this to function to solve for the electromagnetic fields """

    """ Utility functions for FDFD object """

    def _init_derivatives(self):
        """Initialize the sparse derivative matrices and does some processing for
        ease of use"""

        # Creates all of the operators needed for later
        derivs = compute_derivative_matrices(
            self.omega,
            self.shape,
            self.npml,
            self.dL,
            bloch_x=self.bloch_x,
            bloch_y=self.bloch_y,
        )

        # Store derivative arrays (convert to JAX CSR sparse type)
        self.Dxf = derivs[0]
        self.Dxb = derivs[1]
        self.Dyf = derivs[2]
        self.Dyb = derivs[3]

    def _init_bloch_phases(self, bloch_phases):
        """Saves the x y and z bloch phases based on list of them 'bloch_phases'"""

        self.bloch_x = 0.0
        self.bloch_y = 0.0
        self.bloch_z = 0.0

        if bloch_phases is not None:
            self.bloch_x = bloch_phases[0]
            if len(bloch_phases) > 1:
                self.bloch_y = bloch_phases[1]
            if len(bloch_phases) > 2:
                self.bloch_z = bloch_phases[2]

    def _vec_to_grid(self, vec):
        """converts a vector quantity into an array of the shape of the FDFD
        simulation"""
        return npj.reshape(vec, self.shape)

    def _grid_to_vec(self, grid):
        """converts a grid of the shape of the FDFD simulation to a flat vector"""
        return grid.flatten()

    def _save_shape(self, grid):
        """Stores the shape and size of `grid` array to the FDFD object"""
        self.shape = grid.shape
        self.Nx, self.Ny = self.shape
        self.N = self.Nx * self.Ny

    """ Field conversion functions for 2D """

    def _Ex_Ey_to_Hz(self, Ex_vec, Ey_vec):
        return (
            1
            / 1j
            / self.omega
            / MU_0
            * (sp_mult(self.Dxb, Ey_vec) - sp_mult(self.Dyb, Ex_vec))
        )

    def _Hz_to_Ex(self, Hz_vec, eps_vec_xx):
        # addition of 1e-5 is for numerical stability when tracking
        # gradients of eps_xx, and eps_yy -> 0
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_xx + 1e-5)
            * sp_mult(self.Dyf, Hz_vec)
        )

    def _Hz_to_Ey(self, Hz_vec, eps_vec_yy):
        return (
            -1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_yy + 1e-5)
            * sp_mult(self.Dxf, Hz_vec)
        )

    def _Hx_Hy_to_Ez(self, Hx_vec, Hy_vec, eps_vec_zz):
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_zz + 1e-5)
            * (sp_mult(self.Dxf, Hy_vec) - sp_mult(self.Dyf, Hx_vec))
        )

    def _Hz_to_Ex_Ey(self, Hz_vec, eps_vec_xx, eps_vec_yy):
        Ex_vec = self._Hz_to_Ex(Hz_vec, eps_vec_xx)
        Ey_vec = self._Hz_to_Ey(Hz_vec, eps_vec_yy)
        return Ex_vec, Ey_vec


@jit
def _Ez_to_Hx(omega, Ez_vec, Dyb):
    return -1 / 1j / omega / MU_0 * sp_mult(Dyb, Ez_vec)


@jit
def _Ez_to_Hy(omega, Ez_vec, Dxb):
    return 1 / 1j / omega / MU_0 * sp_mult(Dxb, Ez_vec)


@jit
def _Ez_to_Hx_Hy(omega, Ez_vec, Dxb, Dyb):
    Hx_vec = _Ez_to_Hx(omega, Ez_vec, Dxb)
    Hy_vec = _Ez_to_Hy(omega, Ez_vec, Dyb)
    return Hx_vec, Hy_vec


@partial(jit, static_argnums=(1))
def _make_A(omega, shape, eps_vec, Dxf, Dxb, Dyf, Dyb):
    C = -1 / MU_0 * spsp_mult(Dxf, Dxb) - 1 / MU_0 * spsp_mult(Dyf, Dyb)

    # Diagonal of A matrix
    diag = npj.eye(shape[0] * shape[1]) * (-EPSILON_0 * omega**2 * eps_vec)

    A = C + diag

    return A


@partial(jit, static_argnums=(1))
def _solve_fn(omega, shape, eps_vec, A, Jz_vec, Dxb, Dyb):
    b_vec = 1j * omega * Jz_vec
    Ez_vec = _solve_iterative(A, b_vec)
    Hx_vec, Hy_vec = _Ez_to_Hx_Hy(omega, Ez_vec, Dxb, Dyb)
    return Hx_vec, Hy_vec, Ez_vec


@partial(jit, static_argnums=(1, 2))
def solve(omega, shape, pml_shape, dl, eps_r, source_z):
    """Outward facing function (what gets called by user) that takes a
    source grid and returns the field components"""

    # TODO: Why is it necessary to flatten the inputs and then reshape the
    # outputs? Could we modify this to remove the need for flattening?

    # Initialize bloch phases and derivatives
    Dxf, Dxb, Dyf, Dyb = compute_derivative_matrices(
        omega,
        shape,
        pml_shape,
        dl,
        bloch_x=0.0,
        bloch_y=0.0,
    )

    # Flatten the source and permittivity matrices to vectors
    source_vec = source_z.flatten()
    eps_vec = eps_r.flatten()

    # create the A matrix for this polarization
    A = _make_A(omega, shape, eps_vec, Dxf, Dxb, Dyf, Dyb)

    # solve field components using A and the source
    Fx_vec, Fy_vec, Fz_vec = _solve_fn(
        omega, shape, eps_vec, A, source_vec, Dxb, Dyb
    )

    # put all field components into a tuple, convert to grid shape and
    # return them all
    Fx = npj.reshape(Fx_vec, shape)
    Fy = npj.reshape(Fy_vec, shape)
    Fz = npj.reshape(Fz_vec, shape)

    return Fx, Fy, Fz


if __name__ == "__main__":
    import numpy as np

    import ceviche_jax

    np.set_printoptions(precision=0, linewidth=np.inf)

    Nx = 200
    Ny = 80
    Npml = 20
    dl = 50e-9
    lambda0 = 1550e-9
    omega0 = 2 * npj.pi * C_0 / lambda0

    # Define permittivity for a straight waveguide
    epsr = npj.ones((Nx, Ny))
    epsr = epsr.at[:, 35:45].set(12)

    # Source position and amplitude
    src_y = npj.arange(0, 80)
    src_x = 30 * npj.ones(src_y.shape, dtype=int)

    source = ceviche_jax.modes.insert_mode(
        omega0, dl, src_x, src_y, epsr, m=1, filtering=True
    )

    Hx, Hy, Ez = solve(omega0, (Nx, Ny), (Npml, Npml), dl, epsr, source)
