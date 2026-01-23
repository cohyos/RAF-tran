"""
Discrete Ordinates (DISORT) radiative transfer solver.

This module provides a simplified discrete ordinates method implementation
for higher-accuracy radiative transfer calculations.

Note: This is a simplified implementation. For production use, consider
using the full DISORT algorithm or wrappers like pyDISORT.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class DiscreteOrdinatesResult:
    """
    Results from discrete ordinates calculation.

    Attributes
    ----------
    flux_up : ndarray
        Upward flux at layer interfaces
    flux_down : ndarray
        Downward flux at layer interfaces
    flux_direct : ndarray
        Direct (unscattered) flux
    intensity : ndarray, optional
        Radiance at specified angles
    """

    flux_up: np.ndarray
    flux_down: np.ndarray
    flux_direct: np.ndarray
    intensity: Optional[np.ndarray] = None


@dataclass
class DiscreteOrdinatesSolver:
    """
    Discrete ordinates method solver.

    Uses Gaussian quadrature to discretize the angular integration
    in the radiative transfer equation.

    Attributes
    ----------
    n_streams : int
        Number of streams (quadrature points per hemisphere)
    """

    n_streams: int = 4

    def __post_init__(self):
        """Initialize quadrature points and weights."""
        # Full Gauss-Legendre quadrature for [-1, 1]
        self.mu, self.weights = np.polynomial.legendre.leggauss(2 * self.n_streams)

        # Separate into upward (positive mu) and downward (negative mu) directions
        self.mu_up = self.mu[self.n_streams:]
        self.mu_down = -self.mu[:self.n_streams][::-1]
        self.w_up = self.weights[self.n_streams:]
        self.w_down = self.weights[:self.n_streams][::-1]

    def solve(
        self,
        tau: np.ndarray,
        omega: np.ndarray,
        g: np.ndarray,
        mu0: Optional[float] = None,
        flux_toa: float = 0.0,
        thermal_emission: Optional[np.ndarray] = None,
        surface_albedo: float = 0.0,
        surface_emission: float = 0.0,
    ) -> DiscreteOrdinatesResult:
        """
        Solve radiative transfer using discrete ordinates.

        Parameters
        ----------
        tau : array_like
            Optical depth per layer, shape (n_layers,)
        omega : array_like
            Single scattering albedo per layer, shape (n_layers,)
        g : array_like
            Asymmetry parameter per layer, shape (n_layers,)
        mu0 : float, optional
            Cosine of solar zenith angle (for solar calculations)
        flux_toa : float
            Incoming flux at TOA
        thermal_emission : array_like, optional
            Thermal emission per layer in W/m²
        surface_albedo : float
            Surface albedo
        surface_emission : float
            Surface thermal emission in W/m²

        Returns
        -------
        result : DiscreteOrdinatesResult
            Radiative transfer results
        """
        tau = np.asarray(tau)
        omega = np.asarray(omega)
        g = np.asarray(g)

        n_layers = len(tau)
        n_levels = n_layers + 1
        n_mu = self.n_streams

        # Cumulative optical depth from TOA
        tau_cumsum = np.concatenate([[0], np.cumsum(tau)])

        # Initialize intensity arrays for each direction
        I_up = np.zeros((n_levels, n_mu))
        I_down = np.zeros((n_levels, n_mu))

        # Direct beam (if solar)
        if mu0 is not None and mu0 > 0:
            flux_direct = flux_toa * mu0 * np.exp(-tau_cumsum / mu0)
        else:
            flux_direct = np.zeros(n_levels)

        # Thermal emission source
        if thermal_emission is None:
            thermal_emission = np.zeros(n_layers)

        # Phase function expansion (Henyey-Greenstein approximation)
        def hg_phase(mu, mu_prime, g_val):
            """Henyey-Greenstein phase function."""
            cos_theta = mu * mu_prime + np.sqrt(1 - mu**2) * np.sqrt(1 - mu_prime**2)
            return (1 - g_val**2) / (1 + g_val**2 - 2 * g_val * cos_theta) ** 1.5 / (4 * np.pi)

        # Simple iterative solution (Gauss-Seidel)
        for iteration in range(20):
            I_up_old = I_up.copy()
            I_down_old = I_down.copy()

            # Downward pass (from TOA to surface)
            I_down[0, :] = 0  # No incoming diffuse at TOA

            for i in range(n_layers):
                dtau = tau[i]
                w = omega[i]
                g_val = g[i]

                for j, mu_j in enumerate(self.mu_down):
                    # Transmission factor
                    trans = np.exp(-dtau / mu_j)

                    # Scattering source (simplified isotropic + forward peak)
                    J_scat = 0.0
                    if w > 0:
                        # Contribution from all directions
                        for k, mu_k in enumerate(self.mu_down):
                            J_scat += self.w_down[k] * I_down[i, k] * (1 + g_val * mu_j * mu_k)
                        for k, mu_k in enumerate(self.mu_up):
                            J_scat += self.w_up[k] * I_up[i + 1, k] * (1 + g_val * mu_j * (-mu_k))

                        J_scat *= w / 2

                        # Solar source
                        if mu0 is not None and mu0 > 0:
                            J_scat += (
                                w * flux_direct[i] / (4 * np.pi) * (1 + g_val * mu_j * (-mu0))
                            )

                    # Thermal source
                    J_thermal = (1 - w) * thermal_emission[i] / (4 * np.pi)

                    # Total source
                    J_total = J_scat + J_thermal

                    # Update intensity
                    if dtau / mu_j > 1e-4:
                        I_down[i + 1, j] = (
                            I_down[i, j] * trans
                            + J_total * (1 - trans)
                        )
                    else:
                        I_down[i + 1, j] = I_down[i, j] + J_total * dtau / mu_j

            # Surface boundary condition
            for j in range(n_mu):
                # Reflected diffuse
                reflected = 0.0
                for k in range(n_mu):
                    reflected += (
                        surface_albedo / np.pi * self.w_down[k] * self.mu_down[k] * I_down[-1, k]
                    )
                # Reflected direct
                if mu0 is not None:
                    reflected += surface_albedo * flux_direct[-1] * mu0 / np.pi

                # Surface emission
                I_up[-1, j] = reflected + surface_emission / np.pi

            # Upward pass (from surface to TOA)
            for i in range(n_layers - 1, -1, -1):
                dtau = tau[i]
                w = omega[i]
                g_val = g[i]

                for j, mu_j in enumerate(self.mu_up):
                    trans = np.exp(-dtau / mu_j)

                    J_scat = 0.0
                    if w > 0:
                        for k, mu_k in enumerate(self.mu_up):
                            J_scat += self.w_up[k] * I_up[i + 1, k] * (1 + g_val * mu_j * mu_k)
                        for k, mu_k in enumerate(self.mu_down):
                            J_scat += self.w_down[k] * I_down[i, k] * (1 + g_val * mu_j * (-mu_k))

                        J_scat *= w / 2

                        if mu0 is not None and mu0 > 0:
                            J_scat += (
                                w * flux_direct[i + 1] / (4 * np.pi) * (1 + g_val * mu_j * mu0)
                            )

                    J_thermal = (1 - w) * thermal_emission[i] / (4 * np.pi)
                    J_total = J_scat + J_thermal

                    if dtau / mu_j > 1e-4:
                        I_up[i, j] = I_up[i + 1, j] * trans + J_total * (1 - trans)
                    else:
                        I_up[i, j] = I_up[i + 1, j] + J_total * dtau / mu_j

            # Check convergence
            max_diff = max(
                np.max(np.abs(I_up - I_up_old)),
                np.max(np.abs(I_down - I_down_old)),
            )
            if max_diff < 1e-6:
                break

        # Compute fluxes from intensities
        flux_up = np.zeros(n_levels)
        flux_down = np.zeros(n_levels)

        for i in range(n_levels):
            flux_up[i] = 2 * np.pi * np.sum(self.w_up * self.mu_up * I_up[i, :])
            flux_down[i] = 2 * np.pi * np.sum(self.w_down * self.mu_down * I_down[i, :])

        return DiscreteOrdinatesResult(
            flux_up=flux_up,
            flux_down=flux_down,
            flux_direct=flux_direct,
            intensity=np.stack([I_up, I_down], axis=-1),
        )
