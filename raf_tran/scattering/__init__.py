"""
Scattering module for Rayleigh and Mie scattering calculations.

This module provides implementations for:
- Rayleigh scattering by atmospheric molecules
- Mie scattering by spherical aerosol particles

Classes
-------
RayleighScattering
    Rayleigh scattering calculations for molecular atmosphere
MieScattering
    Mie scattering calculations for spherical particles

Functions
---------
rayleigh_cross_section
    Calculate Rayleigh scattering cross section
rayleigh_phase_function
    Rayleigh phase function
mie_coefficients
    Calculate Mie scattering coefficients
"""

from raf_tran.scattering.rayleigh import (
    RayleighScattering,
    rayleigh_cross_section,
    rayleigh_phase_function,
)
from raf_tran.scattering.mie import MieScattering, mie_coefficients

__all__ = [
    "RayleighScattering",
    "rayleigh_cross_section",
    "rayleigh_phase_function",
    "MieScattering",
    "mie_coefficients",
]
