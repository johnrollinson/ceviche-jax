# notation is similar to that used in:
# https://www.jpier.org/ac_api/download.php?id=11092006

import jax.experimental.sparse as spj
import jax.numpy as npj
import scipy.sparse as sp
from jax import jit

from .constants import *
from .derivatives import compute_derivative_matrices
from .primitives import sp_mult, sp_solve, spsp_mult

# from .utils import get_entries_indices


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

    def solve(self, source_z):
        """Outward facing function (what gets called by user) that takes a
        source grid and returns the field components"""

        # TODO: Why is it necessary to flatten the inputs and then reshape the
        # outputs? Could we modify this to remove the need for flattening?

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        A = self._make_A(eps_vec)

        # solve field components using A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(eps_vec, A, source_vec)

        # put all field components into a tuple, convert to grid shape and
        # return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

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
        # TODO: Convert the derivative functions to return JAX CSR type
        self.Dxf = spj.BCOO.from_scipy_sparse(derivs[0])
        self.Dxb = spj.BCOO.from_scipy_sparse(derivs[1])
        self.Dyf = spj.BCOO.from_scipy_sparse(derivs[2])
        self.Dyb = spj.BCOO.from_scipy_sparse(derivs[3])

        # stores some convenience functions for multiplying derivative matrices
        # by a vector `vec`
        # TODO: Remove this convenience functions (no longer needed)
        self.sp_mult_Dxf = lambda vec: sp_mult(self.Dxf, vec)
        self.sp_mult_Dxb = lambda vec: sp_mult(self.Dxb, vec)
        self.sp_mult_Dyf = lambda vec: sp_mult(self.Dyf, vec)
        self.sp_mult_Dyb = lambda vec: sp_mult(self.Dyb, vec)

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

    @jit
    def _Ex_Ey_to_Hz(self, Ex_vec, Ey_vec):
        return (
            1
            / 1j
            / self.omega
            / MU_0
            * (sp_mult(self.Dxb, Ey_vec) - sp_mult(self.Dyb, Ex_vec))
        )

    @jit
    def _Ez_to_Hx(self, Ez_vec):
        return -1 / 1j / self.omega / MU_0 * sp_mult(self.Dyb, Ez_vec)

    @jit
    def _Ez_to_Hy(self, Ez_vec):
        return 1 / 1j / self.omega / MU_0 * sp_mult(self.Dxb, Ez_vec)

    @jit
    def _Ez_to_Hx_Hy(self, Ez_vec):
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    def _Hz_to_Ex(self, Hz_vec, eps_vec_xx):
        # addition of 1e-5 is for numerical stability when tracking
        # gradients of
        # eps_xx, and eps_yy -> 0
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_xx + 1e-5)
            * self.sp_mult_Dyf(Hz_vec)
        )

    def _Hz_to_Ey(self, Hz_vec, eps_vec_yy):
        return (
            -1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_yy + 1e-5)
            * self.sp_mult_Dxf(Hz_vec)
        )

    def _Hx_Hy_to_Ez(self, Hx_vec, Hy_vec, eps_vec_zz):
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_zz + 1e-5)
            * (self.sp_mult_Dxf(Hy_vec) - self.sp_mult_Dyf(Hx_vec))
        )

    def _Hz_to_Ex_Ey(self, Hz_vec, eps_vec_xx, eps_vec_yy):
        Ex_vec = self._Hz_to_Ex(Hz_vec, eps_vec_xx)
        Ey_vec = self._Hz_to_Ey(Hz_vec, eps_vec_yy)
        return Ex_vec, Ey_vec


""" These are the fdfd classes that you'll actually want to use """


class FDFD_Ez(FDFD):
    """FDFD class for linear Ez polarization"""

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    def _make_A(self, eps_vec):
        C = -1 / MU_0 * spsp_mult(self.Dxf, self.Dxb) - 1 / MU_0 * spsp_mult(
            self.Dyf, self.Dyb
        )

        # Diagonal of A matrix
        # TODO: Is this the most efficient way of creating a sparse diag?
        # Consider creating an efficient sparse diagonal function since there is
        # not a built-in one in JAX
        diag = spj.eye(self.N, sparse_format="bcoo") * (
            -EPSILON_0 * self.omega**2 * eps_vec
        )

        A = C + diag

        return A

    @jit
    def _solve_fn(self, eps_vec, A, Jz_vec):
        b_vec = 1j * self.omega * Jz_vec
        A = self._make_A(eps_vec)
        Ez_vec = sp_solve(A, b_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec


class FDFD_Hz(FDFD):
    """FDFD class for linear Hz polarization"""

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    def _grid_average_2d(self, eps_vec):
        eps_grid = self._vec_to_grid(eps_vec)
        eps_grid_xx = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=1, shift=1))
        eps_grid_yy = 1 / 2 * (eps_grid + npa.roll(eps_grid, axis=0, shift=1))
        eps_vec_xx = self._grid_to_vec(eps_grid_xx)
        eps_vec_yy = self._grid_to_vec(eps_grid_yy)
        eps_vec_xx = eps_vec_xx
        eps_vec_yy = eps_vec_yy
        return eps_vec_xx, eps_vec_yy

    def _make_A(self, eps_vec):
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)
        eps_vec_xx_inv = 1 / (
            eps_vec_xx + 1e-5
        )  # the 1e-5 is for numerical stability
        eps_vec_yy_inv = 1 / (
            eps_vec_yy + 1e-5
        )  # autograd throws 'divide by zero' errors.

        indices_diag = npa.vstack((npa.arange(self.N), npa.arange(self.N)))

        entries_DxEpsy, indices_DxEpsy = spsp_mult(
            self.entries_Dxb,
            self.indices_Dxb,
            eps_vec_yy_inv,
            indices_diag,
            self.N,
        )
        entires_DxEpsyDx, indices_DxEpsyDx = spsp_mult(
            entries_DxEpsy,
            indices_DxEpsy,
            self.entries_Dxf,
            self.indices_Dxf,
            self.N,
        )

        entries_DyEpsx, indices_DyEpsx = spsp_mult(
            self.entries_Dyb,
            self.indices_Dyb,
            eps_vec_xx_inv,
            indices_diag,
            self.N,
        )
        entires_DyEpsxDy, indices_DyEpsxDy = spsp_mult(
            entries_DyEpsx,
            indices_DyEpsx,
            self.entries_Dyf,
            self.indices_Dyf,
            self.N,
        )

        entries_d = (
            1 / EPSILON_0 * npa.hstack((entires_DxEpsyDx, entires_DyEpsxDy))
        )
        indices_d = npa.hstack((indices_DxEpsyDx, indices_DyEpsxDy))

        entries_diag = MU_0 * self.omega**2 * npa.ones(self.N)

        entries_a = npa.hstack((entries_d, entries_diag))
        indices_a = npa.hstack((indices_d, indices_diag))

        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec):
        b_vec = 1j * self.omega * Mz_vec  # needed so fields are SI units
        Hz_vec = sp_solve(entries_a, indices_a, b_vec)
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)

        # strip out the x and y components of E and find the Hz component
        Ex_vec, Ey_vec = self._Hz_to_Ex_Ey(Hz_vec, eps_vec_xx, eps_vec_yy)

        return Ex_vec, Ey_vec, Hz_vec


class fdfd_mf_ez(FDFD):
    """FDFD class for multi-frequency linear Ez polarization. New variables:
    omega_mod: angular frequency of modulation (rad/s)
    delta: array of shape (Nfreq, Nx, Ny) containing point-wise modulation depth for each modulation harmonic (1,...,Nfreq)
    phi: array of same shape as delta containing point-wise modulation phase for each modulation harmonic
    Nsb: number of numerical sidebands to consider when solving for fields.
    This is not the same as the number of modulation frequencies Nfreq. For physically meaningful results, Nsb >= Nfreq.
    """

    def __init__(
        self,
        omega,
        dL,
        eps_r,
        omega_mod,
        delta,
        phi,
        Nsb,
        npml,
        bloch_phases=None,
    ):
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)
        self.omega_mod = omega_mod
        self.delta = delta
        self.phi = phi
        self.Nsb = Nsb

    def solve(self, source_z):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""
        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)
        Nfreq = npa.shape(self.delta)[0]
        delta_matrix = self.delta.reshape([Nfreq, npa.prod(self.shape)])
        phi_matrix = self.phi.reshape([Nfreq, npa.prod(self.shape)])
        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec, delta_matrix, phi_matrix)

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
            eps_vec, entries_a, indices_a, source_vec
        )

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

    def _make_A(self, eps_vec, delta_matrix, phi_matrix):
        """Builds the multi-frequency electromagnetic operator A in Ax = b"""
        M = 2 * self.Nsb + 1
        N = self.Nx * self.Ny
        W = self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod

        C = sp.kron(
            sp.eye(M),
            -1 / MU_0 * self.Dxf.dot(self.Dxb)
            - 1 / MU_0 * self.Dyf.dot(self.Dyb),
        )
        entries_c, indices_c = get_entries_indices(C)

        # diagonal entries representing static refractive index
        # this part is just a block diagonal version of the single frequency fdfd_ez
        entries_diag = -EPSILON_0 * npa.kron(W**2, eps_vec)
        indices_diag = npa.vstack((npa.arange(M * N), npa.arange(M * N)))

        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))

        # off-diagonal entries representing dynamic modulation
        # this part couples different frequencies due to modulation
        # for a derivation of these entries, see Y. Shi, W. Shin, and S. Fan. Optica 3(11), 2016.
        Nfreq = npa.shape(delta_matrix)[0]
        for k in npa.arange(Nfreq):
            # super-diagonal entries (note the +1j phase)
            mod_p = (
                -0.5
                * EPSILON_0
                * delta_matrix[k, :]
                * npa.exp(1j * phi_matrix[k, :])
            )
            entries_p = npa.kron(W[: -k - 1] ** 2, mod_p)
            indices_p = npa.vstack(
                (npa.arange((M - k - 1) * N), npa.arange((k + 1) * N, M * N))
            )
            entries_a = npa.hstack((entries_p, entries_a))
            indices_a = npa.hstack((indices_p, indices_a))
            # sub-diagonal entries (note the -1j phase)
            mod_m = (
                -0.5
                * EPSILON_0
                * delta_matrix[k, :]
                * npa.exp(-1j * phi_matrix[k, :])
            )
            entries_m = npa.kron(W[k + 1 :] ** 2, mod_m)
            indices_m = npa.vstack(
                (npa.arange((k + 1) * N, M * N), npa.arange((M - k - 1) * N))
            )
            entries_a = npa.hstack((entries_m, entries_a))
            indices_a = npa.hstack((indices_m, indices_a))

        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec):
        """Multi-frequency version of _solve_fn() defined in fdfd_ez"""
        M = 2 * self.Nsb + 1
        N = self.Nx * self.Ny
        W = self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod
        P = sp.kron(sp.spdiags(W, [0], M, M), sp.eye(N))
        entries_p, indices_p = get_entries_indices(P)
        b_vec = 1j * sp_mult(entries_p, indices_p, Jz_vec)
        Ez_vec = sp_solve(entries_a, indices_a, b_vec)
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

    def _Ez_to_Hx(self, Ez_vec):
        """Multi-frequency version of _Ez_to_Hx() defined in fdfd"""
        M = 2 * self.Nsb + 1
        Winv = 1 / (
            self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod
        )
        Dyb_mf = sp.kron(sp.spdiags(Winv, [0], M, M), self.Dyb)
        entries_Dyb_mf, indices_Dyb_mf = get_entries_indices(Dyb_mf)
        return -1 / 1j / MU_0 * sp_mult(entries_Dyb_mf, indices_Dyb_mf, Ez_vec)

    def _Ez_to_Hy(self, Ez_vec):
        """Multi-frequency version of _Ez_to_Hy() defined in fdfd"""
        M = 2 * self.Nsb + 1
        Winv = 1 / (
            self.omega + npa.arange(-self.Nsb, self.Nsb + 1) * self.omega_mod
        )
        Dxb_mf = sp.kron(sp.spdiags(Winv, [0], M, M), self.Dxb)
        entries_Dxb_mf, indices_Dxb_mf = get_entries_indices(Dxb_mf)
        return 1 / 1j / MU_0 * sp_mult(entries_Dxb_mf, indices_Dxb_mf, Ez_vec)

    def _Ez_to_Hx_Hy(self, Ez_vec):
        """Multi-frequency version of _Ez_to_Hx_Hy() defined in fdfd"""
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    def _vec_to_grid(self, vec):
        """Multi-frequency version of _vec_to_grid() defined in fdfd"""
        # grid shape has Nx*Ny cells per frequency sideband
        grid_shape = (2 * self.Nsb + 1, self.Nx, self.Ny)
        return npa.reshape(vec, grid_shape)


class fdfd_3d(FDFD):
    """3D FDFD class (work in progress)"""

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None):
        raise NotImplementedError

    def _grid_average_3d(self, eps_vec):
        raise NotImplementedError

    def _make_A(self, eps_vec):
        # notation: C = [[C11, C12], [C21, C22]]
        C11 = -1 / MU_0 * self.Dyf.dot(self.Dyb)
        C22 = -1 / MU_0 * self.Dxf.dot(self.Dxb)
        C12 = 1 / MU_0 * self.Dyf.dot(self.Dxb)
        C21 = 1 / MU_0 * self.Dxf.dot(self.Dyb)

        # get entries and indices
        entries_c11, indices_c11 = get_entries_indices(C11)
        entries_c22, indices_c22 = get_entries_indices(C22)
        entries_c12, indices_c12 = get_entries_indices(C12)
        entries_c21, indices_c21 = get_entries_indices(C21)

        # shift the indices into each of the 4 quadrants
        indices_c22 += self.N  # shift into bottom right quadrant
        indices_c12[1, :] += self.N  # shift into top right quadrant
        indices_c21[0, :] += self.N  # shift into bottom left quadrant

        # get full matrix entries and indices
        entries_c = npa.hstack(
            (entries_c11, entries_c12, entries_c21, entries_c22)
        )
        indices_c = npa.hstack(
            (indices_c11, indices_c12, indices_c21, indices_c22)
        )

        # indices into the diagonal of a sparse matrix
        eps_vec_xx, eps_vec_yy, eps_vec_zz = self._grid_average_3d(eps_vec)
        entries_diag = (
            -EPSILON_0 * self.omega**2 * npa.hstack((eps_vec_xx, eps_vec_yy))
        )
        indices_diag = npa.vstack(
            (npa.arange(2 * self.N), npa.arange(2 * self.N))
        )

        # put together the big A and return entries and indices
        entries_a = npa.hstack((entries_diag, entries_c))
        indices_a = npa.hstack((indices_diag, indices_c))
        return entries_a, indices_a

    def _solve_fn(self, eps_vec, entries_a, indices_a, Mz_vec):
        # convert the Mz current into Jx, Jy
        eps_vec_xx, eps_vec_yy = self._grid_average_2d(eps_vec)
        Jx_vec, Jy_vec = self._Hz_to_Ex_Ey(Mz_vec, eps_vec_xx, eps_vec_yy)

        # lump the current sources together and solve for electric field
        source_J_vec = npa.hstack((Jx_vec, Jy_vec))
        E_vec = sp_solve(entries_a, indices_a, source_J_vec)

        # strip out the x and y components of E and find the Hz component
        Ex_vec = E_vec[: self.N]
        Ey_vec = E_vec[self.N :]
        Hz_vec = self._Ex_Ey_to_Hz(Ex_vec, Ey_vec)

        return Ex_vec, Ey_vec, Hz_vec
