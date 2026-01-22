"""
Gas absorption optics module.

This module implements gas absorption calculations using the correlated-k
distribution method for efficient spectral integration.

Classes
-------
CKDTable
    Correlated-k distribution lookup table
GasOptics
    Main gas optics calculator combining multiple absorbing species

Functions
---------
compute_optical_depth
    Calculate optical depth for a gas layer
"""

from raf_tran.gas_optics.ckd import CKDTable, GasOptics, compute_optical_depth

__all__ = [
    "CKDTable",
    "GasOptics",
    "compute_optical_depth",
]
