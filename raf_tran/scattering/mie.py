"""
Mie scattering module.

Implements Mie scattering calculations for spherical aerosol particles.
Based on Bohren and Huffman (1983) algorithm.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import jax.numpy as jnp
from jax import jit


def mie_coefficients(
    size_parameter: float,
    refractive_index: complex,
    n_terms: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Mie scattering coefficients a_n and b_n.

    Uses the algorithm from Bohren and Huffman (1983).

    Parameters
    ----------
    size_parameter : float
        Size parameter x = 2πr/λ
    refractive_index : complex
        Complex refractive index m = n + ik
    n_terms : int, optional
        Number of terms in series. If None, estimated automatically.

    Returns
    -------
    a_n : ndarray
        Electric multipole coefficients
    b_n : ndarray
        Magnetic multipole coefficients

    References
    ----------
    Bohren, C.F. and Huffman, D.R., 1983: Absorption and Scattering of
    Light by Small Particles. Wiley.
    """
    x = size_parameter
    m = refractive_index
    mx = m * x

    if n_terms is None:
        # Wiscombe's criterion for number of terms
        n_terms = int(x + 4 * x ** (1 / 3) + 2)

    # Initialize arrays
    a_n = np.zeros(n_terms, dtype=complex)
    b_n = np.zeros(n_terms, dtype=complex)

    # Calculate spherical Bessel functions using logarithmic derivative
    # D_n(mx) = d/d(mx)[ln(ψ_n(mx))]
    # Using downward recurrence for numerical stability

    n_mx = int(abs(mx) + 15)
    D = np.zeros(n_mx + 1, dtype=complex)

    # Downward recurrence for D_n
    for n in range(n_mx - 1, 0, -1):
        D[n] = (n + 1) / mx - 1.0 / (D[n + 1] + (n + 1) / mx)

    # Riccati-Bessel functions using upward recurrence
    # Standard definitions (Bohren & Huffman):
    # ψ_n(x) = x·j_n(x) where j_n is spherical Bessel function
    # χ_n(x) = -x·y_n(x) where y_n is spherical Neumann function
    # ξ_n(x) = ψ_n(x) + i·χ_n(x) = x·h_n^(1)(x) where h_n^(1) is spherical Hankel
    #
    # Initial values:
    # ψ_{-1}(x) = cos(x), ψ_0(x) = sin(x)
    # χ_{-1}(x) = -sin(x), χ_0(x) = cos(x)
    #
    # Recurrence: f_{n+1} = (2n+1)/x · f_n - f_{n-1}

    psi_nm2 = np.cos(x)   # ψ_{-1}
    psi_nm1 = np.sin(x)   # ψ_0
    psi_n = (1.0 / x) * psi_nm1 - psi_nm2  # ψ_1 = (1/x)·ψ_0 - ψ_{-1}

    chi_nm2 = -np.sin(x)  # χ_{-1}
    chi_nm1 = np.cos(x)   # χ_0
    chi_n = (1.0 / x) * chi_nm1 - chi_nm2  # χ_1

    xi_nm1 = psi_nm1 - 1j * chi_nm1  # ξ_0 = ψ_0 - i·χ_0
    xi_n = psi_n - 1j * chi_n        # ξ_1 = ψ_1 - i·χ_1

    for n in range(1, n_terms + 1):
        # At this point:
        # psi_nm1 = ψ_{n-1}, psi_n = ψ_n
        # xi_nm1 = ξ_{n-1}, xi_n = ξ_n

        # Mie coefficients
        D_n = D[n]
        a_n[n - 1] = (
            (D_n / m + n / x) * psi_n - psi_nm1
        ) / ((D_n / m + n / x) * xi_n - xi_nm1)
        b_n[n - 1] = (
            (m * D_n + n / x) * psi_n - psi_nm1
        ) / ((m * D_n + n / x) * xi_n - xi_nm1)

        # Update Riccati-Bessel functions for next iteration
        psi_np1 = (2 * n + 1) / x * psi_n - psi_nm1
        chi_np1 = (2 * n + 1) / x * chi_n - chi_nm1
        xi_np1 = psi_np1 - 1j * chi_np1

        # Shift for next iteration
        psi_nm1 = psi_n
        psi_n = psi_np1
        chi_nm1 = chi_n
        chi_n = chi_np1
        xi_nm1 = xi_n
        xi_n = xi_np1

    return a_n, b_n


def mie_efficiencies(
    size_parameter: float, refractive_index: complex
) -> Tuple[float, float, float, float]:
    """
    Calculate Mie scattering and absorption efficiencies.

    Parameters
    ----------
    size_parameter : float
        Size parameter x = 2πr/λ
    refractive_index : complex
        Complex refractive index m = n + ik

    Returns
    -------
    Q_ext : float
        Extinction efficiency
    Q_sca : float
        Scattering efficiency
    Q_abs : float
        Absorption efficiency
    g : float
        Asymmetry parameter
    """
    x = size_parameter
    a_n, b_n = mie_coefficients(x, refractive_index)

    n = np.arange(1, len(a_n) + 1)
    prefactor = 2 * n + 1

    # Extinction efficiency
    Q_ext = (2 / x**2) * np.sum(prefactor * np.real(a_n + b_n))

    # Scattering efficiency
    Q_sca = (2 / x**2) * np.sum(prefactor * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2))

    # Absorption efficiency
    Q_abs = Q_ext - Q_sca

    # Ensure non-negative efficiencies (numerical precision floor)
    Q_ext = max(0.0, Q_ext)
    Q_sca = max(0.0, Q_sca)
    Q_abs = max(0.0, Q_abs)

    # Asymmetry parameter
    n_max = len(a_n)
    g_sum = 0.0
    for n in range(1, n_max):
        term1 = (n * (n + 2) / (n + 1)) * np.real(
            a_n[n - 1] * np.conj(a_n[n]) + b_n[n - 1] * np.conj(b_n[n])
        )
        term2 = ((2 * n + 1) / (n * (n + 1))) * np.real(
            a_n[n - 1] * np.conj(b_n[n - 1])
        )
        g_sum += term1 + term2

    if Q_sca > 0:
        g = (4 / (x**2 * Q_sca)) * g_sum
    else:
        g = 0.0

    return Q_ext, Q_sca, Q_abs, g


@dataclass
class MieScattering:
    """
    Mie scattering calculator for spherical particles.

    Attributes
    ----------
    refractive_index : complex
        Complex refractive index of particles (n + ik)
    """

    refractive_index: complex

    def efficiencies(
        self, size_parameter: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Mie efficiencies for array of size parameters.

        Parameters
        ----------
        size_parameter : array_like
            Size parameter x = 2πr/λ

        Returns
        -------
        Q_ext : ndarray
            Extinction efficiency
        Q_sca : ndarray
            Scattering efficiency
        Q_abs : ndarray
            Absorption efficiency
        g : ndarray
            Asymmetry parameter
        """
        size_parameter = np.asarray(size_parameter)
        shape = size_parameter.shape
        size_parameter = size_parameter.ravel()

        Q_ext = np.zeros_like(size_parameter)
        Q_sca = np.zeros_like(size_parameter)
        Q_abs = np.zeros_like(size_parameter)
        g = np.zeros_like(size_parameter)

        for i, x in enumerate(size_parameter):
            if x > 0:
                Q_ext[i], Q_sca[i], Q_abs[i], g[i] = mie_efficiencies(
                    x, self.refractive_index
                )

        return (
            Q_ext.reshape(shape),
            Q_sca.reshape(shape),
            Q_abs.reshape(shape),
            g.reshape(shape),
        )

    def cross_sections(
        self, wavelength: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Mie cross sections for given wavelength and particle radius.

        Parameters
        ----------
        wavelength : array_like
            Wavelength in micrometers
        radius : float
            Particle radius in micrometers

        Returns
        -------
        sigma_ext : ndarray
            Extinction cross section in μm^2
        sigma_sca : ndarray
            Scattering cross section in μm^2
        sigma_abs : ndarray
            Absorption cross section in μm^2
        """
        wavelength = np.asarray(wavelength)
        size_parameter = 2 * np.pi * radius / wavelength

        Q_ext, Q_sca, Q_abs, _ = self.efficiencies(size_parameter)

        geometric_cross_section = np.pi * radius**2

        return (
            Q_ext * geometric_cross_section,
            Q_sca * geometric_cross_section,
            Q_abs * geometric_cross_section,
        )

    def optical_properties(
        self,
        wavelength: np.ndarray,
        radius: float,
        number_density: float,
        thickness: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate optical depth, single scattering albedo, and asymmetry parameter.

        Parameters
        ----------
        wavelength : array_like
            Wavelength in micrometers
        radius : float
            Particle radius in micrometers
        number_density : float
            Particle number density in particles/μm³
        thickness : float
            Layer thickness in micrometers

        Returns
        -------
        tau : ndarray
            Optical depth
        omega : ndarray
            Single scattering albedo
        g : ndarray
            Asymmetry parameter
        """
        wavelength = np.asarray(wavelength)
        size_parameter = 2 * np.pi * radius / wavelength

        Q_ext, Q_sca, Q_abs, g = self.efficiencies(size_parameter)

        geometric_cross_section = np.pi * radius**2
        sigma_ext = Q_ext * geometric_cross_section

        tau = sigma_ext * number_density * thickness
        omega = np.where(Q_ext > 0, Q_sca / Q_ext, 1.0)

        return tau, omega, g


def lognormal_size_distribution(
    radius: np.ndarray,
    r_g: float,
    sigma_g: float,
    N_total: float = 1.0,
) -> np.ndarray:
    """
    Calculate lognormal particle size distribution.

    n(r) = N / (sqrt(2π) r ln(sigma_g)) * exp(-(ln(r/r_g))^2 / (2 ln^2(sigma_g)))

    Parameters
    ----------
    radius : array_like
        Particle radii
    r_g : float
        Geometric mean radius
    sigma_g : float
        Geometric standard deviation
    N_total : float, optional
        Total number concentration (default: 1.0)

    Returns
    -------
    n : ndarray
        Number distribution dn/dr
    """
    radius = np.asarray(radius)
    ln_sigma_g = np.log(sigma_g)

    n = (
        N_total
        / (np.sqrt(2 * np.pi) * radius * ln_sigma_g)
        * np.exp(-((np.log(radius / r_g)) ** 2) / (2 * ln_sigma_g**2))
    )

    return n
