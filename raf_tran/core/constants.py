"""
Physical constants and standard values for radiative transfer calculations.

All units are in SI unless otherwise noted.
Spectroscopic constants follow HITRAN conventions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

# =============================================================================
# Fundamental Physical Constants
# =============================================================================

# Speed of light in vacuum [m/s]
SPEED_OF_LIGHT = 2.99792458e8

# Planck constant [J·s]
PLANCK_CONSTANT = 6.62607015e-34

# Boltzmann constant [J/K]
BOLTZMANN_CONSTANT = 1.380649e-23

# Avogadro number [mol^-1]
AVOGADRO_NUMBER = 6.02214076e23

# Universal gas constant [J/(mol·K)]
GAS_CONSTANT = 8.314462618

# Stefan-Boltzmann constant [W/(m²·K⁴)]
STEFAN_BOLTZMANN = 5.670374419e-8

# Second radiation constant c2 = hc/k [cm·K]
# Used in Planck function with wavenumber in cm^-1
C2_RADIATION = 1.4387769

# First radiation constant for spectral radiance c1 = 2hc² [W·m²/sr]
C1_RADIATION = 1.191042953e-16

# =============================================================================
# Earth and Atmosphere Constants
# =============================================================================

# Earth radius [km]
EARTH_RADIUS_KM = 6371.0

# Standard atmosphere pressure at sea level [Pa]
STANDARD_PRESSURE = 101325.0

# Standard atmosphere temperature at sea level [K]
STANDARD_TEMPERATURE = 288.15

# Reference pressure for HITRAN [atm]
HITRAN_REFERENCE_PRESSURE_ATM = 1.0

# Reference temperature for HITRAN [K]
HITRAN_REFERENCE_TEMPERATURE = 296.0

# Loschmidt constant (number density at STP) [molecules/cm³]
LOSCHMIDT_CONSTANT = 2.6867774e19

# =============================================================================
# Spectroscopic Constants
# =============================================================================

# Conversion: wavenumber [cm^-1] to wavelength [µm]
def wavenumber_to_wavelength(wavenumber_cm1: np.ndarray) -> np.ndarray:
    """Convert wavenumber [cm^-1] to wavelength [µm]."""
    return 1e4 / wavenumber_cm1


def wavelength_to_wavenumber(wavelength_um: np.ndarray) -> np.ndarray:
    """Convert wavelength [µm] to wavenumber [cm^-1]."""
    return 1e4 / wavelength_um


# Line shape cutoff distance [cm^-1]
LINE_CUTOFF_CM1 = 25.0

# Minimum line intensity threshold [cm^-1/(molecule·cm^-2)]
MIN_LINE_INTENSITY = 1e-30

# =============================================================================
# Molecule Data (HITRAN molecule IDs)
# =============================================================================

MOLECULE_IDS: Dict[str, int] = {
    "H2O": 1,
    "CO2": 2,
    "O3": 3,
    "N2O": 4,
    "CO": 5,
    "CH4": 6,
    "O2": 7,
    "NO": 8,
    "SO2": 9,
    "NO2": 10,
    "NH3": 11,
    "HNO3": 12,
    "OH": 13,
    "HF": 14,
    "HCl": 15,
    "HBr": 16,
    "HI": 17,
    "ClO": 18,
    "OCS": 19,
    "H2CO": 20,
    "HOCl": 21,
    "N2": 22,
    "HCN": 23,
    "CH3Cl": 24,
    "H2O2": 25,
    "C2H2": 26,
    "C2H6": 27,
    "PH3": 28,
    "COF2": 29,
    "SF6": 30,
}

# Molecule names (reverse mapping)
MOLECULE_NAMES: Dict[int, str] = {v: k for k, v in MOLECULE_IDS.items()}

# Molecular masses [g/mol]
MOLECULAR_MASSES: Dict[str, float] = {
    "H2O": 18.015,
    "CO2": 44.010,
    "O3": 47.998,
    "N2O": 44.013,
    "CO": 28.010,
    "CH4": 16.043,
    "O2": 31.999,
    "NO": 30.006,
    "SO2": 64.066,
    "NO2": 46.006,
    "NH3": 17.031,
    "N2": 28.014,
    "AIR": 28.964,
}

# =============================================================================
# Standard Atmosphere Model Names
# =============================================================================

@dataclass(frozen=True)
class AtmosphereModelNames:
    """Standard atmosphere model identifiers."""
    US_STANDARD_1976 = "US_STANDARD_1976"
    TROPICAL = "TROPICAL"
    MID_LATITUDE_SUMMER = "MID_LATITUDE_SUMMER"
    MID_LATITUDE_WINTER = "MID_LATITUDE_WINTER"
    SUB_ARCTIC_SUMMER = "SUB_ARCTIC_SUMMER"
    SUB_ARCTIC_WINTER = "SUB_ARCTIC_WINTER"


ATMOSPHERE_MODELS = AtmosphereModelNames()

# =============================================================================
# Aerosol Types
# =============================================================================

@dataclass(frozen=True)
class AerosolTypes:
    """Standard aerosol model identifiers."""
    NONE = "NONE"
    RURAL = "RURAL"
    URBAN = "URBAN"
    MARITIME = "MARITIME"
    DESERT = "DESERT"


AEROSOL_TYPES = AerosolTypes()

# =============================================================================
# Path Geometry Types
# =============================================================================

@dataclass(frozen=True)
class PathTypes:
    """Path geometry identifiers."""
    HORIZONTAL = "HORIZONTAL"
    SLANT = "SLANT"
    VERTICAL = "VERTICAL"


PATH_TYPES = PathTypes()

# =============================================================================
# Default Spectral Parameters
# =============================================================================

# Default spectral resolution [cm^-1]
DEFAULT_SPECTRAL_RESOLUTION = 0.01

# Default spectral range for MWIR [cm^-1]
DEFAULT_MIN_WAVENUMBER = 2000.0  # 5 µm
DEFAULT_MAX_WAVENUMBER = 3333.0  # 3 µm

# =============================================================================
# Performance Thresholds
# =============================================================================

# Maximum memory for spectral arrays [bytes]
MAX_SPECTRAL_ARRAY_MEMORY = 4 * 1024 * 1024 * 1024  # 4 GB

# Chunk size for large spectral calculations
SPECTRAL_CHUNK_SIZE = 100000

# =============================================================================
# Validation Tolerances
# =============================================================================

# Maximum allowed RMS error vs MODTRAN reference
MAX_RMS_ERROR_TRANSMITTANCE = 0.01  # 1%

# Numerical precision for comparisons
NUMERICAL_EPSILON = 1e-10
