"""
Physical constants for atmospheric radiative transfer calculations.

All constants are in SI units unless otherwise specified.
"""

import numpy as np

# Fundamental constants
SPEED_OF_LIGHT = 2.99792458e8  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
AVOGADRO_NUMBER = 6.02214076e23  # mol^-1
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2·K^4)

# Earth and atmosphere constants
EARTH_RADIUS = 6.371e6  # m
STANDARD_PRESSURE = 101325.0  # Pa
STANDARD_TEMPERATURE = 288.15  # K (15°C)
GRAVITY = 9.80665  # m/s^2
DRY_AIR_MOLAR_MASS = 0.0289644  # kg/mol
LOSCHMIDT_CONSTANT = 2.6867774e25  # molecules/m^3 at STP

# Gas molecular weights (kg/mol)
MOLECULAR_WEIGHTS = {
    "H2O": 0.01801528,
    "CO2": 0.04401,
    "O3": 0.0479982,
    "N2O": 0.0440128,
    "CO": 0.02801,
    "CH4": 0.01604,
    "O2": 0.032,
    "N2": 0.028014,
}

# Refractive index of air at STP (at 550 nm)
REFRACTIVE_INDEX_AIR = 1.000293

# King correction factors for depolarization
KING_FACTORS = {
    "N2": 1.034,
    "O2": 1.096,
    "Ar": 1.0,
    "CO2": 1.15,
    "air": 1.05,  # Effective value for dry air
}

# Standard atmospheric composition (volume mixing ratios)
STANDARD_VMR = {
    "N2": 0.7808,
    "O2": 0.2095,
    "Ar": 0.00934,
    "CO2": 420e-6,  # 420 ppm (current level)
    "CH4": 1.9e-6,  # 1.9 ppm
    "N2O": 335e-9,  # 335 ppb
}

# Spectral bands (wavenumber ranges in cm^-1)
SPECTRAL_BANDS = {
    "solar": (2000, 50000),  # 0.2-5 μm
    "thermal": (100, 3000),  # 3.3-100 μm
    "visible": (14286, 25000),  # 0.4-0.7 μm
    "near_ir": (4000, 14286),  # 0.7-2.5 μm
    "mid_ir": (400, 4000),  # 2.5-25 μm
    "far_ir": (100, 400),  # 25-100 μm
}


def wavenumber_to_frequency(wavenumber_cm: float) -> float:
    """Convert wavenumber (cm^-1) to frequency (Hz)."""
    return wavenumber_cm * 100 * SPEED_OF_LIGHT


def planck_function(wavenumber_cm: np.ndarray, temperature: float) -> np.ndarray:
    """
    Calculate Planck function (spectral radiance) at given wavenumber and temperature.

    Parameters
    ----------
    wavenumber_cm : np.ndarray
        Wavenumber in cm^-1
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    np.ndarray
        Spectral radiance in W/(m^2·sr·cm^-1)
    """
    # Convert wavenumber from cm^-1 to m^-1
    wavenumber = wavenumber_cm * 100  # m^-1

    c1 = 2 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2  # First radiation constant
    c2 = PLANCK_CONSTANT * SPEED_OF_LIGHT / BOLTZMANN_CONSTANT  # Second radiation constant

    # Planck function in terms of wavenumber
    exponent = c2 * wavenumber / temperature
    # Avoid overflow
    exponent = np.clip(exponent, None, 700)

    radiance = c1 * wavenumber**3 / (np.exp(exponent) - 1)

    # Convert from W/(m^2·sr·m^-1) to W/(m^2·sr·cm^-1)
    return radiance * 100
