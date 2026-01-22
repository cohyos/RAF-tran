"""
RAF-tran: Open Source Atmospheric Radiative Transfer Simulation

A modular Python-based atmospheric radiative transfer tool inspired by MODTRAN,
implementing correlated-k gas absorption, Mie/Rayleigh scattering, and
plane-parallel RTE solvers.
"""

__version__ = "0.1.0"
__author__ = "RAF-tran Development Team"

from .core.atmosphere import Atmosphere, AtmosphericProfile
from .core.simulation import RTSimulation
from .physics.gas_absorption import GasAbsorption
from .physics.scattering import RayleighScattering, MieScattering
from .solvers.plane_parallel import PlaneParallelSolver

__all__ = [
    "Atmosphere",
    "AtmosphericProfile",
    "RTSimulation",
    "GasAbsorption",
    "RayleighScattering",
    "MieScattering",
    "PlaneParallelSolver",
]
