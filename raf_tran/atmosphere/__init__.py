"""
Atmospheric profile models.

This module provides standard atmosphere models and custom profile definitions
for temperature, pressure, density, and gas concentrations as functions of altitude.

Classes
-------
AtmosphereProfile
    Base class for atmospheric profiles
StandardAtmosphere
    US Standard Atmosphere 1976 implementation
MidlatitudeSummer
    Midlatitude summer atmospheric profile
MidlatitudeWinter
    Midlatitude winter atmospheric profile
TropicalAtmosphere
    Tropical atmospheric profile
SubarcticSummer
    Subarctic summer atmospheric profile
SubarcticWinter
    Subarctic winter atmospheric profile
"""

from raf_tran.atmosphere.profiles import (
    AtmosphereProfile,
    StandardAtmosphere,
    MidlatitudeSummer,
    MidlatitudeWinter,
    TropicalAtmosphere,
    SubarcticSummer,
    SubarcticWinter,
)

__all__ = [
    "AtmosphereProfile",
    "StandardAtmosphere",
    "MidlatitudeSummer",
    "MidlatitudeWinter",
    "TropicalAtmosphere",
    "SubarcticSummer",
    "SubarcticWinter",
]
