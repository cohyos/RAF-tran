"""
Two-stream radiative transfer equation solver.

Implements the two-stream approximation for solving the plane-parallel
radiative transfer equation. This is a fast, approximate method suitable
for broadband flux calculations.

References
----------
Meador, W.E. and Weaver, W.R., 1980: Two-stream approximations to radiative
transfer in planetary atmospheres: A unified description of existing methods
and a new improvement. J. Atmos. Sci., 37, 630-643.

Toon, O.B., et al., 1989: Rapid calculation of radiative heating rates and
photodissociation rates in inhomogeneous multiple scattering atmospheres.
J. Geophys. Res., 94, 16287-16301.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import jax.numpy as jnp
from jax import jit


class TwoStreamMethod(Enum):
    """Two-stream approximation variants."""

    EDDINGTON = "eddington"
    QUADRATURE = "quadrature"
    HEMISPHERIC_MEAN = "hemispheric_mean"
    DELTA_EDDINGTON = "delta_eddington"


@dataclass
class TwoStreamResult:
    """
    Results from two-stream radiative transfer calculation.

    Attributes
    ----------
    flux_up : ndarray
        Upward flux at layer interfaces, shape (n_levels,)
    flux_down : ndarray
        Downward flux at layer interfaces, shape (n_levels,)
    flux_direct : ndarray
        Direct (unscattered) solar flux, shape (n_levels,)
    heating_rate : ndarray
        Heating rate in K/day, shape (n_layers,)
    """

    flux_up: np.ndarray
    flux_down: np.ndarray
    flux_direct: np.ndarray
    heating_rate: Optional[np.ndarray] = None


@dataclass
class TwoStreamSolver:
    """
    Two-stream radiative transfer equation solver.

    Solves the plane-parallel RTE using the two-stream approximation
    for a multi-layer atmosphere with scattering and absorption.

    Attributes
    ----------
    method : TwoStreamMethod
        Two-stream approximation method to use
    """

    method: TwoStreamMethod = TwoStreamMethod.DELTA_EDDINGTON

    def _get_two_stream_coefficients(
        self, omega: float, g: float
    ) -> Tuple[float, float]:
        """
        Get two-stream coefficients gamma1 and gamma2.

        Parameters
        ----------
        omega : float
            Single scattering albedo
        g : float
            Asymmetry parameter

        Returns
        -------
        gamma1 : float
            Two-stream coefficient
        gamma2 : float
            Two-stream coefficient
        """
        if self.method == TwoStreamMethod.EDDINGTON:
            gamma1 = (7 - omega * (4 + 3 * g)) / 4
            gamma2 = -(1 - omega * (4 - 3 * g)) / 4
        elif self.method == TwoStreamMethod.QUADRATURE:
            mu1 = 1 / np.sqrt(3)
            gamma1 = (1 - omega * (1 + g) / 2) / mu1
            gamma2 = omega * (1 - g) / (2 * mu1)
        elif self.method == TwoStreamMethod.HEMISPHERIC_MEAN:
            gamma1 = 2 - omega * (1 + g)
            gamma2 = omega * (1 - g)
        elif self.method == TwoStreamMethod.DELTA_EDDINGTON:
            # Delta-scaling
            f = g * g
            omega_prime = omega * (1 - f) / (1 - omega * f)
            g_prime = (g - f) / (1 - f)
            gamma1 = (7 - omega_prime * (4 + 3 * g_prime)) / 4
            gamma2 = -(1 - omega_prime * (4 - 3 * g_prime)) / 4
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return gamma1, gamma2

    def solve_thermal(
        self,
        tau: np.ndarray,
        omega: np.ndarray,
        g: np.ndarray,
        temperature: np.ndarray,
        surface_temperature: float,
        surface_emissivity: float = 1.0,
    ) -> TwoStreamResult:
        """
        Solve thermal (longwave) radiative transfer.

        Parameters
        ----------
        tau : array_like
            Optical depth per layer, shape (n_layers,)
        omega : array_like
            Single scattering albedo per layer, shape (n_layers,)
        g : array_like
            Asymmetry parameter per layer, shape (n_layers,)
        temperature : array_like
            Layer temperature in K, shape (n_layers,)
        surface_temperature : float
            Surface temperature in K
        surface_emissivity : float
            Surface emissivity (default: 1.0)

        Returns
        -------
        result : TwoStreamResult
            Radiative transfer results
        """
        from raf_tran.utils.constants import STEFAN_BOLTZMANN

        tau = np.asarray(tau)
        omega = np.asarray(omega)
        g = np.asarray(g)
        temperature = np.asarray(temperature)

        n_layers = len(tau)
        n_levels = n_layers + 1

        # Initialize flux arrays
        flux_up = np.zeros(n_levels)
        flux_down = np.zeros(n_levels)

        # Surface boundary condition
        surface_emission = surface_emissivity * STEFAN_BOLTZMANN * surface_temperature**4

        # Layer Planck emissions (blackbody flux = σT⁴)
        B_flux = STEFAN_BOLTZMANN * temperature**4

        # Precompute layer transmission, reflection, and emission coefficients
        t_layer = np.zeros(n_layers)
        r_layer = np.zeros(n_layers)
        emission = np.zeros(n_layers)

        for i in range(n_layers):
            gamma1, gamma2 = self._get_two_stream_coefficients(omega[i], g[i])
            k = np.sqrt(max(gamma1**2 - gamma2**2, 1e-12))

            if tau[i] < 1e-6:
                # Optically thin limit
                t_layer[i] = 1 - tau[i] * gamma1
                r_layer[i] = tau[i] * gamma2
            else:
                exp_k_tau = np.exp(-k * tau[i])
                denom = (k + gamma1) + (k - gamma1) * exp_k_tau**2

                t_layer[i] = 2 * k * exp_k_tau / denom
                r_layer[i] = (k - gamma1) * (1 - exp_k_tau**2) / denom

            # Thermal emission: what's absorbed must be emitted (Kirchhoff's law)
            # For each direction, emission = (1 - t - r) * B_flux
            # This ensures energy conservation: absorbed = emitted
            absorption = 1 - t_layer[i] - r_layer[i]
            emission[i] = absorption * B_flux[i]

        # Iterative solution for coupled up/down fluxes
        flux_up[-1] = surface_emission

        for iteration in range(20):  # Usually converges in <10 iterations
            flux_up_old = flux_up.copy()

            # Upward pass: bottom to top
            for i in range(n_layers - 1, -1, -1):
                flux_up[i] = (
                    t_layer[i] * flux_up[i + 1]
                    + r_layer[i] * flux_down[i]
                    + emission[i]
                )

            # Downward pass: top to bottom
            flux_down[0] = 0  # No incoming thermal radiation at TOA
            for i in range(n_layers):
                flux_down[i + 1] = (
                    t_layer[i] * flux_down[i]
                    + r_layer[i] * flux_up[i + 1]
                    + emission[i]
                )

            # Update surface reflection (for non-unity emissivity)
            if surface_emissivity < 1.0:
                flux_up[-1] = (
                    surface_emission
                    + (1 - surface_emissivity) * flux_down[-1]
                )

            # Check for convergence
            if np.max(np.abs(flux_up - flux_up_old)) < 1e-6:
                break

        return TwoStreamResult(
            flux_up=flux_up,
            flux_down=flux_down,
            flux_direct=np.zeros(n_levels),
        )

    def solve_solar(
        self,
        tau: np.ndarray,
        omega: np.ndarray,
        g: np.ndarray,
        mu0: float,
        flux_toa: float,
        surface_albedo: float = 0.0,
    ) -> TwoStreamResult:
        """
        Solve solar (shortwave) radiative transfer.

        Parameters
        ----------
        tau : array_like
            Optical depth per layer, shape (n_layers,)
        omega : array_like
            Single scattering albedo per layer, shape (n_layers,)
        g : array_like
            Asymmetry parameter per layer, shape (n_layers,)
        mu0 : float
            Cosine of solar zenith angle
        flux_toa : float
            Incoming solar flux at TOA in W/m²
        surface_albedo : float
            Surface albedo (default: 0.0)

        Returns
        -------
        result : TwoStreamResult
            Radiative transfer results
        """
        tau = np.asarray(tau)
        omega = np.asarray(omega)
        g = np.asarray(g)

        n_layers = len(tau)
        n_levels = n_layers + 1

        # Delta-scaling if using delta-Eddington
        if self.method == TwoStreamMethod.DELTA_EDDINGTON:
            f = g * g
            tau_prime = tau * (1 - omega * f)
            omega_prime = omega * (1 - f) / (1 - omega * f)
            g_prime = (g - f) / (1 - f)
        else:
            tau_prime = tau
            omega_prime = omega
            g_prime = g

        # Cumulative optical depth
        tau_cumsum = np.concatenate([[0], np.cumsum(tau_prime)])

        # Direct beam
        flux_direct = flux_toa * mu0 * np.exp(-tau_cumsum / mu0)

        # Initialize diffuse flux arrays
        flux_up = np.zeros(n_levels)
        flux_down = np.zeros(n_levels)

        # Build tridiagonal system for adding-doubling
        # Simplified two-stream solution

        # Single-layer reflection and transmission
        r = np.zeros(n_layers)
        t = np.zeros(n_layers)
        source_up = np.zeros(n_layers)
        source_down = np.zeros(n_layers)

        for i in range(n_layers):
            gamma1, gamma2 = self._get_two_stream_coefficients(omega_prime[i], g_prime[i])
            gamma3 = (2 - 3 * g_prime[i] * mu0) / 4
            gamma4 = 1 - gamma3

            k = np.sqrt(max(gamma1**2 - gamma2**2, 1e-12))

            if tau_prime[i] < 1e-6:
                # Optically thin limit
                t[i] = 1 - tau_prime[i] * gamma1
                r[i] = tau_prime[i] * gamma2

                # Solar source terms
                source_factor = omega_prime[i] * flux_direct[i] * tau_prime[i]
                source_up[i] = source_factor * gamma4
                source_down[i] = source_factor * gamma3
            else:
                exp_k_tau = np.exp(-k * tau_prime[i])
                denom = (k + gamma1) + (k - gamma1) * exp_k_tau**2

                t[i] = 2 * k * exp_k_tau / denom
                r[i] = (k - gamma1) * (1 - exp_k_tau**2) / denom

                # Solar source terms (simplified)
                exp_tau_mu0 = np.exp(-tau_prime[i] / mu0)
                omega_term = omega_prime[i] * flux_direct[i]

                alpha1 = gamma1 * gamma4 + gamma2 * gamma3
                alpha2 = gamma1 * gamma3 + gamma2 * gamma4

                source_up[i] = omega_term * (
                    alpha1 * (1 - exp_tau_mu0 * exp_k_tau) / (k + 1 / mu0)
                    + alpha2 * (exp_tau_mu0 - exp_k_tau) / (k - 1 / mu0 + 1e-10)
                )
                source_down[i] = omega_term * (
                    alpha2 * (1 - exp_tau_mu0 * exp_k_tau) / (k + 1 / mu0)
                    + alpha1 * (exp_tau_mu0 - exp_k_tau) / (k - 1 / mu0 + 1e-10)
                )

        # Adding method: combine layers from top to bottom
        # Then solve for boundary conditions

        # Top boundary: no incoming diffuse radiation
        flux_down[0] = 0

        # Bottom boundary: surface reflection
        flux_up[-1] = surface_albedo * (flux_direct[-1] + flux_down[-1])

        # Simple iterative solution
        for iteration in range(10):
            # Downward pass
            for i in range(n_layers):
                flux_down[i + 1] = (
                    t[i] * flux_down[i] + r[i] * flux_up[i + 1] + source_down[i]
                )

            # Update surface reflection
            flux_up[-1] = surface_albedo * (flux_direct[-1] + flux_down[-1])

            # Upward pass
            for i in range(n_layers - 1, -1, -1):
                flux_up[i] = t[i] * flux_up[i + 1] + r[i] * flux_down[i] + source_up[i]

        return TwoStreamResult(
            flux_up=flux_up,
            flux_down=flux_down,
            flux_direct=flux_direct,
        )

    def compute_heating_rate(
        self,
        flux_up: np.ndarray,
        flux_down: np.ndarray,
        flux_direct: np.ndarray,
        pressure: np.ndarray,
        cp: float = 1004.0,
    ) -> np.ndarray:
        """
        Compute heating rate from flux divergence.

        dT/dt = -g/cp * d(F_net)/dp

        Parameters
        ----------
        flux_up : array_like
            Upward flux at levels, shape (n_levels,)
        flux_down : array_like
            Downward flux at levels, shape (n_levels,)
        flux_direct : array_like
            Direct flux at levels, shape (n_levels,)
        pressure : array_like
            Pressure at levels in Pa, shape (n_levels,)
        cp : float
            Specific heat capacity at constant pressure (J/kg/K)

        Returns
        -------
        heating_rate : ndarray
            Heating rate in K/day, shape (n_layers,)
        """
        from raf_tran.utils.constants import EARTH_SURFACE_GRAVITY

        flux_net = flux_down + flux_direct - flux_up

        # Flux divergence across each layer
        d_flux = np.diff(flux_net)  # F_net(i+1) - F_net(i)
        d_pressure = np.diff(pressure)  # P(i+1) - P(i)

        # Heating rate: dT/dt = -g/cp * dF/dp
        heating_rate = (
            -EARTH_SURFACE_GRAVITY / cp * d_flux / d_pressure
        )

        # Convert from K/s to K/day
        heating_rate *= 86400

        return heating_rate
