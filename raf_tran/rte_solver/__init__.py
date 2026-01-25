"""
Radiative Transfer Equation (RTE) solvers.

This module provides numerical solvers for the radiative transfer equation
in plane-parallel atmospheres.

Classes
-------
TwoStreamSolver
    Two-stream approximation solver for plane-parallel atmospheres
DiscreteOrdinatesSolver
    Discrete ordinates method solver (higher accuracy)

Functions
---------
solve_rte
    General RTE solver interface
"""

from raf_tran.rte_solver.two_stream import TwoStreamSolver, TwoStreamMethod
from raf_tran.rte_solver.disort import DiscreteOrdinatesSolver

__all__ = [
    "TwoStreamSolver",
    "TwoStreamMethod",
    "DiscreteOrdinatesSolver",
]
