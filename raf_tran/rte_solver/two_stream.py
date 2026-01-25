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
        levels_surface_to_toa: bool = True,
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
        levels_surface_to_toa : bool
            If True (default), input arrays are ordered from surface (index 0)
            to TOA (index n). If False, they are ordered from TOA to surface.

        Returns
        -------
        result : TwoStreamResult
            Radiative transfer results (same ordering as input)
        """
        from raf_tran.utils.constants import STEFAN_BOLTZMANN

        tau = np.asarray(tau)
        omega = np.asarray(omega)
        g = np.asarray(g)
        temperature = np.asarray(temperature)

        # Internal calculations assume TOA-to-surface ordering (index 0 = TOA)
        # If input is surface-to-TOA, reverse the arrays
        if levels_surface_to_toa:
            tau = tau[::-1]
            omega = omega[::-1]
            g = g[::-1]
            temperature = temperature[::-1]

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

        # Reverse output to match input ordering if needed
        if levels_surface_to_toa:
            flux_up = flux_up[::-1]
            flux_down = flux_down[::-1]

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
        levels_surface_to_toa: bool = True,
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
        levels_surface_to_toa : bool
            If True (default), input arrays are ordered from surface (index 0)
            to TOA (index n). If False, they are ordered from TOA to surface.

        Returns
        -------
        result : TwoStreamResult
            Radiative transfer results (same ordering as input)
        """
        tau = np.asarray(tau)
        omega = np.asarray(omega)
        g = np.asarray(g)

        # Internal calculations assume TOA-to-surface ordering
        # If input is surface-to-TOA, reverse the arrays
        if levels_surface_to_toa:
            tau = tau[::-1]
            omega = omega[::-1]
            g = g[::-1]

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
                r[i] = tau_prime[i] * abs(gamma2)  # Use abs to ensure positive

                # Solar source terms
                source_factor = omega_prime[i] * flux_direct[i] * tau_prime[i]
                source_up[i] = source_factor * gamma4
                source_down[i] = source_factor * gamma3
            else:
                exp_k_tau = np.exp(-k * tau_prime[i])

                # Handle conservative scattering case (k -> 0)
                # When gamma1^2 - gamma2^2 ~ 0, use asymptotic expansion
                if k * tau_prime[i] < 0.01:
                    # Conservative scattering limit
                    # t ~ 1 / (1 + gamma1*tau), r ~ gamma1*tau / (1 + gamma1*tau)
                    g1_tau = abs(gamma1) * tau_prime[i]
                    t[i] = 1.0 / (1.0 + g1_tau)
                    r[i] = g1_tau / (1.0 + g1_tau)
                else:
                    denom = (k + gamma1) + (k - gamma1) * exp_k_tau**2

                    t[i] = 2 * k * exp_k_tau / denom
                    r[i] = abs(k - gamma1) * (1 - exp_k_tau**2) / abs(denom)

                    # Ensure physical bounds
                    r[i] = max(0, min(r[i], 1 - t[i]))

                # Solar source terms (simplified)
                exp_tau_mu0 = np.exp(-tau_prime[i] / mu0)
                omega_term = omega_prime[i] * flux_direct[i]

                alpha1 = gamma1 * gamma4 + gamma2 * gamma3
                alpha2 = gamma1 * gamma3 + gamma2 * gamma4

                # Handle singularity when k ~ 1/mu0
                k_minus_inv_mu0 = k - 1 / mu0
                if abs(k_minus_inv_mu0) < 1e-4:
                    # Use L'Hopital's rule / Taylor expansion for singular case
                    # (exp_tau_mu0 - exp_k_tau) / (k - 1/mu0) -> tau * exp(-tau/mu0)
                    singular_term = tau_prime[i] * exp_tau_mu0
                    source_up[i] = omega_term * (
                        alpha1 * (1 - exp_tau_mu0 * exp_k_tau) / (k + 1 / mu0)
                        + alpha2 * singular_term
                    )
                    source_down[i] = omega_term * (
                        alpha2 * (1 - exp_tau_mu0 * exp_k_tau) / (k + 1 / mu0)
                        + alpha1 * singular_term
                    )
                else:
                    source_up[i] = omega_term * (
                        alpha1 * (1 - exp_tau_mu0 * exp_k_tau) / (k + 1 / mu0)
                        + alpha2 * (exp_tau_mu0 - exp_k_tau) / k_minus_inv_mu0
                    )
                    source_down[i] = omega_term * (
                        alpha2 * (1 - exp_tau_mu0 * exp_k_tau) / (k + 1 / mu0)
                        + alpha1 * (exp_tau_mu0 - exp_k_tau) / k_minus_inv_mu0
                    )

                # Apply energy conservation constraint: total scattered cannot exceed incident
                max_source = omega_term * (1 - exp_tau_mu0)  # Max scattered energy
                total_source = source_up[i] + source_down[i]
                if total_source > max_source and total_source > 0:
                    scale = max_source / total_source
                    source_up[i] *= scale
                    source_down[i] *= scale

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

        # Reverse output to match input ordering if needed
        if levels_surface_to_toa:
            flux_up = flux_up[::-1]
            flux_down = flux_down[::-1]
            flux_direct = flux_direct[::-1]

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

        dT/dt = (g/cp) * d(F_absorbed)/dp

        where F_absorbed is the flux absorbed by each layer (energy deposited).

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

        # Net flux (positive downward = energy entering from above)
        flux_net = flux_down + flux_direct - flux_up

        n_layers = len(pressure) - 1
        heating_rate = np.zeros(n_layers)

        for i in range(n_layers):
            # Energy absorbed by layer i = flux entering top - flux leaving bottom
            # For layer between level i and i+1:
            # Flux in from above = flux_net[i] (if pressure increases with index)
            # or flux_net[i+1] (if pressure decreases with index)

            # Determine pressure ordering
            if pressure[0] > pressure[-1]:
                # Pressure decreases with index: level 0 is surface, level n is TOA
                # Flux absorbed = (flux entering from above) - (flux leaving below)
                # = (F_down[i+1] + F_direct[i+1] - F_up[i+1]) - (F_down[i] + F_direct[i] - F_up[i])
                flux_absorbed = flux_net[i+1] - flux_net[i]
            else:
                # Pressure increases with index: level 0 is TOA, level n is surface
                flux_absorbed = flux_net[i] - flux_net[i+1]

            # Mass of layer per unit area: dp/g
            dp = abs(pressure[i+1] - pressure[i])
            mass_per_area = dp / EARTH_SURFACE_GRAVITY

            # Heating rate: dT/dt = absorbed_energy / (mass * cp)
            # flux_absorbed is W/m^2 = J/s/m^2
            # mass_per_area is kg/m^2
            # cp is J/kg/K
            # Result is K/s
            if mass_per_area > 0:
                heating_rate[i] = flux_absorbed / (mass_per_area * cp)

        # Convert from K/s to K/day
        heating_rate *= 86400

        return heating_rate
