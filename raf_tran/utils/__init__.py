"""
Utility functions and physical constants.

This module provides common utilities and physical constants used throughout
the RAF-tran library.

Constants
---------
SPEED_OF_LIGHT : float
    Speed of light in vacuum (m/s)
PLANCK_CONSTANT : float
    Planck's constant (J·s)
BOLTZMANN_CONSTANT : float
    Boltzmann constant (J/K)
STEFAN_BOLTZMANN : float
    Stefan-Boltzmann constant (W/m²/K⁴)
AVOGADRO : float
    Avogadro's number (mol⁻¹)
GAS_CONSTANT : float
    Universal gas constant (J/mol/K)
EARTH_RADIUS : float
    Mean Earth radius (m)
SOLAR_CONSTANT : float
    Solar constant at TOA (W/m²)

Functions
---------
planck_function
    Planck blackbody radiance
wavenumber_to_wavelength
    Convert wavenumber to wavelength
wavelength_to_wavenumber
    Convert wavelength to wavenumber
"""

from raf_tran.utils.constants import (
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT,
    STEFAN_BOLTZMANN,
    AVOGADRO,
    GAS_CONSTANT,
    EARTH_RADIUS,
    SOLAR_CONSTANT,
)
from raf_tran.utils.spectral import (
    planck_function,
    wavenumber_to_wavelength,
    wavelength_to_wavenumber,
)

__all__ = [
    "SPEED_OF_LIGHT",
    "PLANCK_CONSTANT",
    "BOLTZMANN_CONSTANT",
    "STEFAN_BOLTZMANN",
    "AVOGADRO",
    "GAS_CONSTANT",
    "EARTH_RADIUS",
    "SOLAR_CONSTANT",
    "planck_function",
    "wavenumber_to_wavelength",
    "wavelength_to_wavenumber",
]
