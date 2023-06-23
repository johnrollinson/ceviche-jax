# used for setup.py
name = "ceviche_jax"

__version__ = "0.1.3"

from . import modes, utils, viz
from .fdfd import FDFD_Ez, FDFD_Hz, fdfd_mf_ez
from .fdtd import fdtd
from .jacobians import jacobian
