"""
Gas absorption optics module.

This module implements gas absorption calculations using the correlated-k
distribution method for efficient spectral integration.

OFFLINE OPERATION
-----------------
RAF-tran works fully offline using the built-in correlated-k (CKD) method.
For high-fidelity line-by-line calculations, the optional HITRAN/HAPI
integration can be enabled by installing hitran-api.

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

Optional HITRAN Integration
---------------------------
The hitran submodule provides optional line-by-line calculations:
    >>> from raf_tran.gas_optics import hitran
    >>> if hitran.HAPI_AVAILABLE:
    ...     # Use HITRAN for ~1% accuracy
    ... else:
    ...     # Use CKD for ~5% accuracy (fully offline)
"""

from raf_tran.gas_optics.ckd import CKDTable, GasOptics, compute_optical_depth

# Import HITRAN module (always available, but HAPI is optional)
from raf_tran.gas_optics import hitran
from raf_tran.gas_optics.hitran import (
    HAPI_AVAILABLE,
    HITRANGasOptics,
    can_run_offline,
)

__all__ = [
    # Core CKD classes (always available, offline)
    "CKDTable",
    "GasOptics",
    "compute_optical_depth",
    # HITRAN integration (optional)
    "hitran",
    "HAPI_AVAILABLE",
    "HITRANGasOptics",
    "can_run_offline",
]
