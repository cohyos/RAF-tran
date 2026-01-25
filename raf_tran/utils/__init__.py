"""
Utility functions and physical constants.

This module provides common utilities and physical constants used throughout
the RAF-tran library.

Constants
---------
SPEED_OF_LIGHT : float
    Speed of light in vacuum (m/s)
PLANCK_CONSTANT : float
    Planck's constant (J-s)
BOLTZMANN_CONSTANT : float
    Boltzmann constant (J/K)
STEFAN_BOLTZMANN : float
    Stefan-Boltzmann constant (W/m^2/K^4)
AVOGADRO : float
    Avogadro's number (1/mol)
GAS_CONSTANT : float
    Universal gas constant (J/mol/K)
EARTH_RADIUS : float
    Mean Earth radius (m)
SOLAR_CONSTANT : float
    Solar constant at TOA (W/m^2)

Functions
---------
planck_function
    Planck blackbody radiance
wavenumber_to_wavelength
    Convert wavenumber to wavelength
wavelength_to_wavenumber
    Convert wavelength to wavenumber
optical_air_mass
    Calculate optical air mass with automatic method selection
plane_parallel_air_mass
    Simple 1/cos(SZA) air mass
kasten_young_air_mass
    Empirical air mass formula accurate to SZA=90 deg
chapman_function
    Chapman function for spherical atmosphere
validate_solar_geometry
    Validate solar zenith angle and return cosine
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
from raf_tran.utils.air_mass import (
    optical_air_mass,
    plane_parallel_air_mass,
    kasten_young_air_mass,
    chapman_function,
    validate_solar_geometry,
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
    "optical_air_mass",
    "plane_parallel_air_mass",
    "kasten_young_air_mass",
    "chapman_function",
    "validate_solar_geometry",
]
