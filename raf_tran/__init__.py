"""
RAF-tran: Open source atmospheric radiative transfer library.

A Python-native, high-performance atmospheric radiative transfer library
based on scientific literature and modern computational techniques.

OFFLINE OPERATION
-----------------
RAF-tran works fully offline using built-in data and models.
Optional online features (HITRAN, weather APIs) enhance accuracy
but are not required.

Modules
-------
atmosphere
    Atmospheric profile models (temperature, pressure, gas concentrations)
gas_optics
    Gas absorption using correlated-k distribution method
    Optional: HITRAN line-by-line via HAPI
scattering
    Rayleigh (molecular) and Mie (aerosol/particle) scattering
rte_solver
    Radiative transfer equation solvers (two-stream, discrete ordinates)
turbulence
    Atmospheric optical turbulence (Cn2 profiles, scintillation, beam propagation)
    Includes: Adaptive optics simulation, real Cn2 data integration
detectors
    IR detector models (FPA detectors: InSb, MCT, Digital LWIR)
targets
    Target signature models (aircraft IR signatures)
detection
    Detection range calculations (IR range equation)
geometry
    3D spherical Earth geometry, path integration, ray tracing
weather
    Atmospheric profiles (US Standard, CIRA-86, AFGL)
    Optional: ECMWF, GFS, MERRA-2 online data
utils
    Utility functions and constants (including air mass calculations)
"""

__version__ = "0.2.0"
__author__ = "RAF-tran Contributors"

from raf_tran.atmosphere import StandardAtmosphere, AtmosphereProfile
from raf_tran.rte_solver import TwoStreamSolver

# Check offline capability
def can_run_offline() -> bool:
    """
    Check if RAF-tran can run fully offline.

    Returns
    -------
    bool
        Always True - RAF-tran supports full offline operation
    """
    return True

__all__ = [
    "__version__",
    "StandardAtmosphere",
    "AtmosphereProfile",
    "TwoStreamSolver",
    "can_run_offline",
]
