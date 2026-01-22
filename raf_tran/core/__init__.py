"""
Core computational modules for RAF-Tran radiative transfer calculations.

This module contains the physics engines and solvers:
- GasEngine: Molecular absorption calculations (Line-by-Line)
- ScatteringEngine: Mie and Rayleigh scattering for aerosols
- RTESolver: Radiative Transfer Equation integration
"""

from raf_tran.core.gas_engine import GasEngine
from raf_tran.core.scattering_engine import ScatteringEngine
from raf_tran.core.rte_solver import RTESolver
from raf_tran.core.simulation import Simulation

__all__ = [
    "GasEngine",
    "ScatteringEngine",
    "RTESolver",
    "Simulation",
]
