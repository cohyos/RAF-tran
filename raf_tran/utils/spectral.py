"""Spectral utility functions for radiative transfer calculations."""

import numpy as np
import jax.numpy as jnp
from jax import jit

from raf_tran.utils.constants import (
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    FIRST_RADIATION_CONSTANT,
    SECOND_RADIATION_CONSTANT,
)


def wavenumber_to_wavelength(wavenumber: np.ndarray) -> np.ndarray:
    """
    Convert wavenumber to wavelength.

    Parameters
    ----------
    wavenumber : array_like
        Wavenumber in cm⁻¹

    Returns
    -------
    wavelength : ndarray
        Wavelength in micrometers
    """
    # wavenumber [cm⁻¹] -> wavelength [μm]
    # 1 cm⁻¹ = 10000 μm⁻¹
    return 1e4 / np.asarray(wavenumber)


def wavelength_to_wavenumber(wavelength: np.ndarray) -> np.ndarray:
    """
    Convert wavelength to wavenumber.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers

    Returns
    -------
    wavenumber : ndarray
        Wavenumber in cm⁻¹
    """
    return 1e4 / np.asarray(wavelength)


def planck_function(wavelength: np.ndarray, temperature: float) -> np.ndarray:
    """
    Calculate Planck blackbody spectral radiance.

    Computes the spectral radiance B(λ, T) using Planck's law:
    B(λ, T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)

    Parameters
    ----------
    wavelength : array_like
        Wavelength in meters
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    radiance : ndarray
        Spectral radiance in W/m²/sr/m
    """
    wavelength = np.asarray(wavelength)

    # Avoid division by zero
    if temperature <= 0:
        return np.zeros_like(wavelength)

    # c1 / λ⁵
    term1 = FIRST_RADIATION_CONSTANT / (wavelength**5)

    # exp(c2 / λT) - 1
    exponent = SECOND_RADIATION_CONSTANT / (wavelength * temperature)

    # Handle numerical overflow for large exponents
    with np.errstate(over='ignore'):
        exp_term = np.exp(exponent)

    # For very large exponents, use asymptotic form
    mask = exponent > 700  # exp(700) ≈ 1e304
    result = np.where(mask, 0.0, term1 / (exp_term - 1))

    return result


@jit
def planck_function_jax(wavelength: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """
    JAX-accelerated Planck blackbody spectral radiance.

    Parameters
    ----------
    wavelength : jnp.ndarray
        Wavelength in meters
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    radiance : jnp.ndarray
        Spectral radiance in W/m²/sr/m
    """
    term1 = FIRST_RADIATION_CONSTANT / (wavelength**5)
    exponent = SECOND_RADIATION_CONSTANT / (wavelength * temperature)

    # Clip exponent to avoid overflow
    exponent = jnp.clip(exponent, 0, 700)

    return term1 / (jnp.exp(exponent) - 1)


def planck_function_wavenumber(wavenumber: np.ndarray, temperature: float) -> np.ndarray:
    """
    Calculate Planck blackbody spectral radiance in wavenumber space.

    Computes B(ν̃, T) where ν̃ is wavenumber in cm⁻¹.

    Parameters
    ----------
    wavenumber : array_like
        Wavenumber in cm⁻¹
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    radiance : ndarray
        Spectral radiance in W/m²/sr/cm⁻¹
    """
    wavenumber = np.asarray(wavenumber)

    if temperature <= 0:
        return np.zeros_like(wavenumber)

    # Convert wavenumber from cm⁻¹ to m⁻¹
    nu_m = wavenumber * 100.0  # cm⁻¹ to m⁻¹

    # c1_nu = 2hc² in appropriate units
    c1_nu = 2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2

    # c2 = hc/k
    c2 = SECOND_RADIATION_CONSTANT

    term1 = c1_nu * nu_m**3
    exponent = c2 * nu_m / temperature

    with np.errstate(over='ignore'):
        exp_term = np.exp(exponent)

    mask = exponent > 700
    result = np.where(mask, 0.0, term1 / (exp_term - 1))

    # Convert from W/m²/sr/m⁻¹ to W/m²/sr/cm⁻¹
    return result * 100.0


def stefan_boltzmann_flux(temperature: float) -> float:
    """
    Calculate total blackbody flux using Stefan-Boltzmann law.

    F = σT⁴

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin

    Returns
    -------
    flux : float
        Total hemispherical flux in W/m²
    """
    from raf_tran.utils.constants import STEFAN_BOLTZMANN
    return STEFAN_BOLTZMANN * temperature**4
