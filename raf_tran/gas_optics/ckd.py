"""
Correlated-k distribution method for gas absorption.

This module implements the correlated-k distribution method for efficient
spectral integration of gas absorption in atmospheric radiative transfer.

The correlated-k method reduces the spectral integration from thousands of
monochromatic calculations to a small number (typically 8-16) of pseudo-
monochromatic calculations per spectral band.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap


@dataclass
class CKDTable:
    """
    Correlated-k distribution lookup table.

    Stores k-distribution data for a single gas species as a function of
    pressure, temperature, and g-point.

    Attributes
    ----------
    gas_name : str
        Name of the absorbing gas (e.g., 'H2O', 'CO2', 'O3')
    wavenumber_bounds : tuple
        (lower, upper) wavenumber bounds in cm⁻¹
    pressures : ndarray
        Reference pressure levels in Pa, shape (n_press,)
    temperatures : ndarray
        Reference temperature levels in K, shape (n_temp,)
    g_points : ndarray
        Gauss-Legendre quadrature points in [0, 1], shape (n_g,)
    g_weights : ndarray
        Gauss-Legendre quadrature weights, shape (n_g,)
    k_coefficients : ndarray
        Absorption coefficients in m^2/mol, shape (n_press, n_temp, n_g)
    """

    gas_name: str
    wavenumber_bounds: Tuple[float, float]
    pressures: np.ndarray
    temperatures: np.ndarray
    g_points: np.ndarray
    g_weights: np.ndarray
    k_coefficients: np.ndarray

    @property
    def n_g_points(self) -> int:
        """Number of g-points (quadrature points)."""
        return len(self.g_points)

    def interpolate_k(
        self, pressure: np.ndarray, temperature: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate k-coefficients to given pressure and temperature.

        Uses bilinear interpolation in log(p)-T space.

        Parameters
        ----------
        pressure : array_like
            Pressure in Pa, shape (n_layers,)
        temperature : array_like
            Temperature in K, shape (n_layers,)

        Returns
        -------
        k : ndarray
            Interpolated absorption coefficients, shape (n_layers, n_g)
        """
        pressure = np.asarray(pressure)
        temperature = np.asarray(temperature)

        # Use log pressure for interpolation
        log_p = np.log(pressure)
        log_p_ref = np.log(self.pressures)

        n_layers = len(pressure)
        k_interp = np.zeros((n_layers, self.n_g_points))

        for i in range(n_layers):
            # Find pressure indices
            p_idx = np.searchsorted(log_p_ref, log_p[i]) - 1
            p_idx = np.clip(p_idx, 0, len(self.pressures) - 2)

            # Find temperature indices
            t_idx = np.searchsorted(self.temperatures, temperature[i]) - 1
            t_idx = np.clip(t_idx, 0, len(self.temperatures) - 2)

            # Interpolation weights
            dp = (log_p[i] - log_p_ref[p_idx]) / (
                log_p_ref[p_idx + 1] - log_p_ref[p_idx]
            )
            dt = (temperature[i] - self.temperatures[t_idx]) / (
                self.temperatures[t_idx + 1] - self.temperatures[t_idx]
            )

            dp = np.clip(dp, 0, 1)
            dt = np.clip(dt, 0, 1)

            # Bilinear interpolation
            k00 = self.k_coefficients[p_idx, t_idx, :]
            k01 = self.k_coefficients[p_idx, t_idx + 1, :]
            k10 = self.k_coefficients[p_idx + 1, t_idx, :]
            k11 = self.k_coefficients[p_idx + 1, t_idx + 1, :]

            k_interp[i, :] = (
                (1 - dp) * (1 - dt) * k00
                + (1 - dp) * dt * k01
                + dp * (1 - dt) * k10
                + dp * dt * k11
            )

        return k_interp


def generate_gauss_legendre(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Gauss-Legendre quadrature points and weights for [0, 1].

    Parameters
    ----------
    n_points : int
        Number of quadrature points

    Returns
    -------
    points : ndarray
        Quadrature points in [0, 1]
    weights : ndarray
        Quadrature weights (sum to 1)
    """
    # Get Gauss-Legendre points and weights for [-1, 1]
    points_11, weights_11 = np.polynomial.legendre.leggauss(n_points)

    # Transform to [0, 1]
    points = 0.5 * (points_11 + 1)
    weights = 0.5 * weights_11

    return points, weights


@dataclass
class GasOptics:
    """
    Gas optics calculator using correlated-k method.

    Combines absorption from multiple gas species.

    Attributes
    ----------
    ckd_tables : dict
        Dictionary of CKDTable objects keyed by gas name
    """

    ckd_tables: Dict[str, CKDTable] = field(default_factory=dict)

    def add_gas(self, ckd_table: CKDTable) -> None:
        """Add a gas species to the calculator."""
        self.ckd_tables[ckd_table.gas_name] = ckd_table

    def compute_optical_depth(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: Dict[str, np.ndarray],
        dz: np.ndarray,
        number_density: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical depth for all layers and g-points.

        Parameters
        ----------
        pressure : array_like
            Layer pressure in Pa, shape (n_layers,)
        temperature : array_like
            Layer temperature in K, shape (n_layers,)
        vmr : dict
            Volume mixing ratios by gas name, each shape (n_layers,)
        dz : array_like
            Layer thickness in m, shape (n_layers,)
        number_density : array_like
            Air number density in molecules/m³, shape (n_layers,)

        Returns
        -------
        tau : ndarray
            Optical depth, shape (n_layers, n_g)
        g_weights : ndarray
            Quadrature weights, shape (n_g,)
        """
        if not self.ckd_tables:
            raise ValueError("No gas species added to GasOptics")

        # Get first table to determine n_g_points
        first_table = next(iter(self.ckd_tables.values()))
        n_layers = len(pressure)
        n_g = first_table.n_g_points

        tau_total = np.zeros((n_layers, n_g))

        for gas_name, ckd_table in self.ckd_tables.items():
            if gas_name not in vmr:
                continue

            # Interpolate k-coefficients
            k = ckd_table.interpolate_k(pressure, temperature)  # (n_layers, n_g)

            # Gas column amount: N_gas = vmr * n_air * dz [molecules/m^2]
            gas_vmr = np.asarray(vmr[gas_name])
            column_amount = gas_vmr * number_density * dz  # molecules/m^2

            # Convert to mol/m^2 for k in m^2/mol
            from raf_tran.utils.constants import AVOGADRO

            column_amount_mol = column_amount / AVOGADRO  # mol/m^2

            # Optical depth: tau = k * column_amount
            tau_gas = k * column_amount_mol[:, np.newaxis]
            tau_total += tau_gas

        return tau_total, first_table.g_weights


def compute_optical_depth(
    k_coefficients: np.ndarray,
    column_amount: np.ndarray,
) -> np.ndarray:
    """
    Compute optical depth from k-coefficients and column amount.

    tau = k x column_amount

    Parameters
    ----------
    k_coefficients : array_like
        Absorption coefficient in m^2/mol, shape (n_layers, n_g)
    column_amount : array_like
        Column amount in mol/m^2, shape (n_layers,)

    Returns
    -------
    tau : ndarray
        Optical depth, shape (n_layers, n_g)
    """
    k_coefficients = np.asarray(k_coefficients)
    column_amount = np.asarray(column_amount)

    return k_coefficients * column_amount[:, np.newaxis]


def create_simple_ckd_table(
    gas_name: str,
    wavenumber_bounds: Tuple[float, float],
    n_g_points: int = 8,
    reference_k: float = 1e-24,
) -> CKDTable:
    """
    Create a simple CKD table with uniform absorption.

    This is primarily for testing; real applications should use
    pre-computed tables from spectroscopic databases.

    Parameters
    ----------
    gas_name : str
        Name of the gas
    wavenumber_bounds : tuple
        (lower, upper) wavenumber bounds in cm⁻¹
    n_g_points : int
        Number of g-points (default: 8)
    reference_k : float
        Reference absorption coefficient in m^2/mol

    Returns
    -------
    ckd_table : CKDTable
        Simple CKD table
    """
    # Reference pressure and temperature grid
    pressures = np.array([1e5, 5e4, 2e4, 1e4, 5e3, 2e3, 1e3, 500, 200, 100])
    temperatures = np.array([200, 220, 240, 260, 280, 300, 320])

    g_points, g_weights = generate_gauss_legendre(n_g_points)

    # Create k-coefficient array
    # In reality, this would come from line-by-line calculations
    # Here we use a simple pressure-dependent model
    n_p, n_t, n_g = len(pressures), len(temperatures), n_g_points
    k_coefficients = np.zeros((n_p, n_t, n_g))

    for i, p in enumerate(pressures):
        for j, t in enumerate(temperatures):
            # Pressure and temperature dependence
            p_factor = (p / 1e5) ** 0.5
            t_factor = (300 / t) ** 1.5

            # g-point dependence (k increases with g)
            for k in range(n_g):
                k_coefficients[i, j, k] = reference_k * p_factor * t_factor * (1 + k)

    return CKDTable(
        gas_name=gas_name,
        wavenumber_bounds=wavenumber_bounds,
        pressures=pressures,
        temperatures=temperatures,
        g_points=g_points,
        g_weights=g_weights,
        k_coefficients=k_coefficients,
    )


@jit
def compute_optical_depth_jax(
    k_coefficients: jnp.ndarray,
    column_amount: jnp.ndarray,
) -> jnp.ndarray:
    """
    JAX-accelerated optical depth calculation.

    Parameters
    ----------
    k_coefficients : jnp.ndarray
        Absorption coefficient, shape (n_layers, n_g)
    column_amount : jnp.ndarray
        Column amount, shape (n_layers,)

    Returns
    -------
    tau : jnp.ndarray
        Optical depth, shape (n_layers, n_g)
    """
    return k_coefficients * column_amount[:, jnp.newaxis]
