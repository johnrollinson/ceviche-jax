"""
This file contains constants that are used throughout the codebase

TODO: Convert this submodule to pull constants from scipy.constants instead
"""
from numpy import sqrt

import jax.numpy as npj

EPSILON_0 = 8.85418782e-12  # vacuum permittivity
MU_0 = 1.25663706e-6  # vacuum permeability
C_0 = 1 / sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = sqrt(MU_0 / EPSILON_0)  # vacuum impedance
Q_e = 1.602176634e-19  # fundamental charge

COMPLEX = npj.complex64
FLOAT = npj.float32