"""
RAF-Tran: Open-source Atmospheric Radiative Transfer Simulation System.

A modular Python library for spectral simulation and radiative transfer calculations,
designed to provide MODTRAN-like functionality with Line-by-Line (LBL) accuracy.

Features:
- Line-by-Line molecular absorption calculations using HITRAN database
- Mie scattering for aerosols (Rural, Urban, Maritime, Desert)
- Standard atmosphere models (US Standard 1976, Mid-Latitude, Sub-Arctic, Tropical)
- Multiple path geometries (Horizontal, Slant Path)
- Full offline operation capability
- GPU acceleration support (optional)

Example:
    >>> from raf_tran import Simulation
    >>> config = {
    ...     "atmosphere": {"model": "US_STANDARD_1976"},
    ...     "spectral": {"min_wavenumber": 2000, "max_wavenumber": 2500}
    ... }
    >>> sim = Simulation(config)
    >>> result = sim.run()
"""

__version__ = "0.1.0"
__author__ = "RAF-Tran Contributors"

from raf_tran.core.simulation import Simulation
from raf_tran.config.manager import ConfigurationManager
from raf_tran.config.settings import SimulationConfig

__all__ = [
    "Simulation",
    "ConfigurationManager",
    "SimulationConfig",
    "__version__",
]
