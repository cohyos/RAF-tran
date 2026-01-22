"""
Rayleigh scattering module.

Implements Rayleigh scattering calculations for molecular atmosphere.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import jit

from raf_tran.utils.constants import AVOGADRO


def rayleigh_cross_section(
    wavelength: np.ndarray,
    depolarization_factor: float = 0.0279,
) -> np.ndarray:
    """
    Calculate Rayleigh scattering cross section.

    Uses the formula from Bodhaine et al. (1999) for air.

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers
    depolarization_factor : float, optional
        King factor for depolarization (default: 0.0279 for air)

    Returns
    -------
    cross_section : ndarray
        Rayleigh scattering cross section in m²

    References
    ----------
    Bodhaine, B.A., et al., 1999: On Rayleigh optical depth calculations.
    J. Atmos. Oceanic Technol., 16, 1854-1861.
    """
    wavelength = np.asarray(wavelength)

    # Wavelength in micrometers
    wl_um = wavelength

    # Refractive index of air (Peck and Reeder, 1972)
    # n - 1 for standard air at 288.15 K, 1013.25 hPa
    wl_um2 = wl_um**2
    n_minus_1 = (
        8060.51
        + 2480990.0 / (132.274 - 1.0 / wl_um2)
        + 17455.7 / (39.32957 - 1.0 / wl_um2)
    ) * 1e-8

    # Number density at STP (molecules/m³)
    # N_s = P / (k * T) at standard conditions
    N_s = 2.546899e25  # molecules/m³ at STP

    # King factor (accounts for molecular anisotropy)
    F_k = (6 + 3 * depolarization_factor) / (6 - 7 * depolarization_factor)

    # Rayleigh cross section (m²)
    # σ = (24π³/N²λ⁴) * ((n²-1)/(n²+2))² * F_k
    wavelength_m = wl_um * 1e-6  # convert to meters

    n = 1 + n_minus_1
    term1 = 24.0 * np.pi**3 / (N_s**2 * wavelength_m**4)
    term2 = ((n**2 - 1) / (n**2 + 2)) ** 2

    cross_section = term1 * term2 * F_k

    return cross_section


@jit
def rayleigh_cross_section_jax(
    wavelength: jnp.ndarray,
    depolarization_factor: float = 0.0279,
) -> jnp.ndarray:
    """
    JAX-accelerated Rayleigh scattering cross section calculation.

    Parameters
    ----------
    wavelength : jnp.ndarray
        Wavelength in micrometers
    depolarization_factor : float, optional
        King factor for depolarization

    Returns
    -------
    cross_section : jnp.ndarray
        Rayleigh scattering cross section in m²
    """
    wl_um = wavelength
    wl_um2 = wl_um**2

    n_minus_1 = (
        8060.51
        + 2480990.0 / (132.274 - 1.0 / wl_um2)
        + 17455.7 / (39.32957 - 1.0 / wl_um2)
    ) * 1e-8

    N_s = 2.546899e25
    F_k = (6 + 3 * depolarization_factor) / (6 - 7 * depolarization_factor)

    wavelength_m = wl_um * 1e-6
    n = 1 + n_minus_1
    term1 = 24.0 * jnp.pi**3 / (N_s**2 * wavelength_m**4)
    term2 = ((n**2 - 1) / (n**2 + 2)) ** 2

    return term1 * term2 * F_k


def rayleigh_phase_function(cos_theta: np.ndarray) -> np.ndarray:
    """
    Calculate Rayleigh scattering phase function.

    P(θ) = (3/4)(1 + cos²θ)

    Parameters
    ----------
    cos_theta : array_like
        Cosine of scattering angle

    Returns
    -------
    phase : ndarray
        Phase function value (normalized to integrate to 4π over sphere)
    """
    cos_theta = np.asarray(cos_theta)
    return 0.75 * (1 + cos_theta**2)


def rayleigh_optical_depth(
    wavelength: np.ndarray,
    column_density: float,
    depolarization_factor: float = 0.0279,
) -> np.ndarray:
    """
    Calculate Rayleigh optical depth for an atmospheric column.

    τ = σ * N_column

    Parameters
    ----------
    wavelength : array_like
        Wavelength in micrometers
    column_density : float
        Column number density in molecules/m²
    depolarization_factor : float, optional
        King factor for depolarization

    Returns
    -------
    optical_depth : ndarray
        Rayleigh optical depth (dimensionless)
    """
    sigma = rayleigh_cross_section(wavelength, depolarization_factor)
    return sigma * column_density


@dataclass
class RayleighScattering:
    """
    Rayleigh scattering calculator for atmospheric layers.

    Attributes
    ----------
    depolarization_factor : float
        King factor for depolarization (default: 0.0279 for air)
    """

    depolarization_factor: float = 0.0279

    def cross_section(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate Rayleigh scattering cross section.

        Parameters
        ----------
        wavelength : array_like
            Wavelength in micrometers

        Returns
        -------
        cross_section : ndarray
            Scattering cross section in m²
        """
        return rayleigh_cross_section(wavelength, self.depolarization_factor)

    def optical_depth(
        self, wavelength: np.ndarray, number_density: np.ndarray, dz: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Rayleigh optical depth for atmospheric layers.

        Parameters
        ----------
        wavelength : array_like
            Wavelength in micrometers, shape (n_wavelengths,)
        number_density : array_like
            Number density in molecules/m³, shape (n_layers,)
        dz : array_like
            Layer thicknesses in meters, shape (n_layers,)

        Returns
        -------
        tau : ndarray
            Optical depth per layer, shape (n_wavelengths, n_layers)
        """
        wavelength = np.asarray(wavelength)
        number_density = np.asarray(number_density)
        dz = np.asarray(dz)

        sigma = self.cross_section(wavelength)  # (n_wavelengths,)
        column_density = number_density * dz  # (n_layers,)

        # Broadcast: (n_wavelengths, 1) * (1, n_layers) -> (n_wavelengths, n_layers)
        tau = sigma[:, np.newaxis] * column_density[np.newaxis, :]

        return tau

    def phase_function(self, cos_theta: np.ndarray) -> np.ndarray:
        """
        Calculate Rayleigh phase function.

        Parameters
        ----------
        cos_theta : array_like
            Cosine of scattering angle

        Returns
        -------
        phase : ndarray
            Phase function value
        """
        return rayleigh_phase_function(cos_theta)

    def asymmetry_parameter(self) -> float:
        """
        Get asymmetry parameter g for Rayleigh scattering.

        Returns
        -------
        g : float
            Asymmetry parameter (always 0 for Rayleigh scattering)
        """
        return 0.0

    def single_scattering_albedo(self) -> float:
        """
        Get single scattering albedo for Rayleigh scattering.

        Returns
        -------
        omega : float
            Single scattering albedo (always 1.0 for pure Rayleigh)
        """
        return 1.0
