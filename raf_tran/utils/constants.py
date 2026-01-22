"""Physical constants for atmospheric radiative transfer calculations."""

import numpy as np

# Fundamental constants (SI units)
SPEED_OF_LIGHT = 299792458.0  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
STEFAN_BOLTZMANN = 5.670374419e-8  # W/m²/K⁴
AVOGADRO = 6.02214076e23  # mol⁻¹

# Derived constants
GAS_CONSTANT = AVOGADRO * BOLTZMANN_CONSTANT  # J/mol/K (≈ 8.314)

# Earth and solar constants
EARTH_RADIUS = 6.371e6  # m (mean radius)
SOLAR_CONSTANT = 1361.0  # W/m² (at TOA, mean value)
EARTH_SURFACE_GRAVITY = 9.80665  # m/s²

# Atmospheric constants
DRY_AIR_MOLAR_MASS = 28.9647e-3  # kg/mol
WATER_VAPOR_MOLAR_MASS = 18.01528e-3  # kg/mol
CO2_MOLAR_MASS = 44.01e-3  # kg/mol
O3_MOLAR_MASS = 48.0e-3  # kg/mol

# Standard conditions
STANDARD_PRESSURE = 101325.0  # Pa
STANDARD_TEMPERATURE = 288.15  # K (15°C)

# Spectral constants
# First radiation constant: c1 = 2 * h * c^2
FIRST_RADIATION_CONSTANT = 2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2  # W·m²
# Second radiation constant: c2 = h * c / k
SECOND_RADIATION_CONSTANT = PLANCK_CONSTANT * SPEED_OF_LIGHT / BOLTZMANN_CONSTANT  # m·K

# Wavelength conversions
MICROMETERS_TO_METERS = 1e-6
CM_TO_M = 1e-2
