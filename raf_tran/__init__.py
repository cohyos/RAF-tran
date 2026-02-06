"""
RAF-tran: Open source atmospheric radiative transfer library.

A Python-native, high-performance atmospheric radiative transfer library
based on scientific literature and modern computational techniques.

Modules
-------
atmosphere
    Atmospheric profile models (temperature, pressure, gas concentrations)
gas_optics
    Gas absorption using correlated-k distribution method
scattering
    Rayleigh (molecular) and Mie (aerosol/particle) scattering
rte_solver
    Radiative transfer equation solvers (two-stream, discrete ordinates)
turbulence
    Atmospheric optical turbulence (Cn2 profiles, scintillation, beam propagation)
detectors
    IR detector models (FPA detectors: InSb, MCT)
targets
    Target signature models (aircraft IR signatures)
detection
    Detection range calculations (IR range equation)
utils
    Utility functions and constants (including air mass calculations)
"""

__version__ = "0.1.0"
__author__ = "RAF-tran Contributors"

from raf_tran.atmosphere import StandardAtmosphere, AtmosphereProfile
from raf_tran.rte_solver import TwoStreamSolver

__all__ = [
    "__version__",
    "StandardAtmosphere",
    "AtmosphereProfile",
    "TwoStreamSolver",
]
