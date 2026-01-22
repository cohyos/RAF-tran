"""
Spectral utilities for wavelength/wavenumber conversions.
"""

import numpy as np


def wavenumber_to_wavelength(wavenumber_cm: np.ndarray) -> np.ndarray:
    """
    Convert wavenumber (cm^-1) to wavelength (μm).

    Parameters
    ----------
    wavenumber_cm : np.ndarray
        Wavenumber in cm^-1

    Returns
    -------
    np.ndarray
        Wavelength in micrometers (μm)
    """
    return 1e4 / wavenumber_cm


def wavelength_to_wavenumber(wavelength_um: np.ndarray) -> np.ndarray:
    """
    Convert wavelength (μm) to wavenumber (cm^-1).

    Parameters
    ----------
    wavelength_um : np.ndarray
        Wavelength in micrometers (μm)

    Returns
    -------
    np.ndarray
        Wavenumber in cm^-1
    """
    return 1e4 / wavelength_um


def create_spectral_grid(start: float, end: float, resolution: float,
                         unit: str = "wavenumber") -> np.ndarray:
    """
    Create a uniform spectral grid.

    Parameters
    ----------
    start : float
        Starting value
    end : float
        Ending value
    resolution : float
        Grid spacing
    unit : str
        Either "wavenumber" (cm^-1) or "wavelength" (μm)

    Returns
    -------
    np.ndarray
        Spectral grid array
    """
    if unit not in ["wavenumber", "wavelength"]:
        raise ValueError("unit must be 'wavenumber' or 'wavelength'")

    return np.arange(start, end + resolution, resolution)


def spectral_response_gaussian(center: float, fwhm: float,
                                grid: np.ndarray) -> np.ndarray:
    """
    Generate a Gaussian spectral response function.

    Parameters
    ----------
    center : float
        Center position
    fwhm : float
        Full width at half maximum
    grid : np.ndarray
        Spectral grid

    Returns
    -------
    np.ndarray
        Normalized Gaussian response
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    response = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
    return response / np.trapz(response, grid)
