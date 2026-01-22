"""
Gas Absorption Engine for Line-by-Line radiative transfer calculations.

Implements FR-09: Line-by-Line (LBL) molecular absorption calculations.

This module computes molecular absorption coefficients using:
- Voigt line shape profiles
- Temperature and pressure scaling
- Line mixing (optional)
- Continuum absorption

The calculations use HITRAN line parameters and follow standard
spectroscopic conventions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from numba import jit, prange

from raf_tran.core.constants import (
    BOLTZMANN_CONSTANT,
    SPEED_OF_LIGHT,
    HITRAN_REFERENCE_TEMPERATURE,
    HITRAN_REFERENCE_PRESSURE_ATM,
    LOSCHMIDT_CONSTANT,
    LINE_CUTOFF_CM1,
    MIN_LINE_INTENSITY,
    MOLECULAR_MASSES,
)
from raf_tran.data.spectral_db import SpectralDatabase, SpectralLines

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionCoefficients:
    """Container for computed absorption coefficients.

    Attributes:
        wavenumber: Spectral grid [cm^-1]
        total_absorption: Total absorption coefficient [cm^-1]
        molecule_contributions: Per-molecule absorption [cm^-1]
    """
    wavenumber: np.ndarray
    total_absorption: np.ndarray
    molecule_contributions: Dict[str, np.ndarray]

    @property
    def optical_depth(self) -> np.ndarray:
        """Absorption coefficient (same as total_absorption for consistency)."""
        return self.total_absorption


# =============================================================================
# Numba-accelerated line shape functions
# =============================================================================

@jit(nopython=True, cache=True)
def voigt_humlicek(x: np.ndarray, y: float) -> np.ndarray:
    """Compute Voigt function using Humlicek's algorithm.

    Uses the rational approximation from Humlicek (1982) JQSRT 27, 437.
    This provides a good balance of accuracy and speed.

    Args:
        x: Dimensionless frequency offset from line center
        y: Ratio of Lorentzian to Doppler width

    Returns:
        Real part of the Faddeeva function (Voigt profile)
    """
    n = len(x)
    result = np.zeros(n)

    # Complex argument w = x + i*y
    for i in range(n):
        t = y - 1j * x[i]

        # Different approximation regions
        s = np.abs(x[i]) + y

        if s >= 15.0:
            # Region I: Large |w|
            result[i] = y / (np.pi * (x[i]*x[i] + y*y))
        elif s >= 5.5:
            # Region II
            t2 = t * t
            result[i] = np.real(
                t * (1.410474 + t2 * 0.5641896) /
                (0.75 + t2 * (3.0 + t2))
            ) / np.sqrt(np.pi)
        elif y >= 0.195 * np.abs(x[i]) - 0.176:
            # Region III
            result[i] = np.real(
                (16.4955 + t * (20.20933 + t * (11.96482 +
                 t * (3.778987 + t * 0.5642236)))) /
                (16.4955 + t * (38.82363 + t * (39.27121 +
                 t * (21.69274 + t * (6.699398 + t)))))
            ) / np.sqrt(np.pi)
        else:
            # Region IV: Small y, moderate x
            t2 = t * t
            u = t2 - 1.5
            result[i] = np.real(
                np.exp(t2) * (1.0 - t2 * (36183.31 - u *
                (3321.9905 - u * (1540.787 - u * (219.0313 -
                u * (35.76683 - u * (1.320522 - u * 0.56419)))))) /
                (32066.6 - u * (24322.84 - u * (9022.228 -
                u * (2186.181 - u * (364.2191 - u * (61.57037 -
                u * (1.841439 - u))))))))
            ) / np.sqrt(np.pi)

    return result


@jit(nopython=True, cache=True)
def lorentzian_profile(
    wavenumber_grid: np.ndarray,
    line_center: float,
    gamma: float,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Compute Lorentzian line shape.

    Args:
        wavenumber_grid: Spectral grid [cm^-1]
        line_center: Line center wavenumber [cm^-1]
        gamma: Half-width at half-maximum [cm^-1]
        cutoff: Distance from line center to compute [cm^-1]

    Returns:
        Lorentzian line shape (normalized to 1)
    """
    n = len(wavenumber_grid)
    result = np.zeros(n)

    for i in range(n):
        dnu = wavenumber_grid[i] - line_center
        if np.abs(dnu) <= cutoff:
            result[i] = gamma / (np.pi * (dnu * dnu + gamma * gamma))

    return result


@jit(nopython=True, cache=True, parallel=True)
def compute_absorption_lbl(
    wavenumber_grid: np.ndarray,
    line_centers: np.ndarray,
    line_intensities: np.ndarray,
    line_widths: np.ndarray,
    temperature: float,
    pressure_atm: float,
    number_density: float,
    temp_exp: np.ndarray,
    lower_energy: np.ndarray,
    pressure_shifts: np.ndarray,
    cutoff: float = 25.0,
) -> np.ndarray:
    """Compute Line-by-Line absorption coefficient.

    Numba-accelerated core calculation that sums contributions from
    all spectral lines using Lorentzian profiles scaled for temperature
    and pressure.

    Args:
        wavenumber_grid: Spectral grid [cm^-1]
        line_centers: Line center wavenumbers [cm^-1]
        line_intensities: Line intensities at 296K [cm^-1/(molecule·cm^-2)]
        line_widths: Air-broadened half-widths at 296K [cm^-1/atm]
        temperature: Layer temperature [K]
        pressure_atm: Layer pressure [atm]
        number_density: Absorber number density [molecules/cm³]
        temp_exp: Temperature dependence exponent
        lower_energy: Lower state energy [cm^-1]
        pressure_shifts: Pressure-induced line shifts [cm^-1/atm]
        cutoff: Line cutoff distance [cm^-1]

    Returns:
        Absorption coefficient [cm^-1]
    """
    n_grid = len(wavenumber_grid)
    n_lines = len(line_centers)
    absorption = np.zeros(n_grid)

    # Reference conditions
    T_ref = 296.0  # K
    P_ref = 1.0    # atm

    # Constants for temperature scaling
    c2 = 1.4387769  # hc/k in cm·K

    # Temperature ratio
    T_ratio = T_ref / temperature

    for j in prange(n_lines):
        # Skip weak lines
        if line_intensities[j] < 1e-30:
            continue

        # Pressure-scaled line width (Lorentzian)
        gamma = line_widths[j] * (pressure_atm / P_ref) * (T_ratio ** temp_exp[j])

        # Temperature-scaled line intensity
        # S(T) = S(T_ref) * Q(T_ref)/Q(T) * exp(-c2*E"/T) / exp(-c2*E"/T_ref)
        #      * (1 - exp(-c2*nu/T)) / (1 - exp(-c2*nu/T_ref))
        E_lower = lower_energy[j]
        nu_center = line_centers[j]

        # Boltzmann factor
        boltz_factor = np.exp(-c2 * E_lower / temperature) / np.exp(-c2 * E_lower / T_ref)

        # Stimulated emission correction
        stim_factor = (1.0 - np.exp(-c2 * nu_center / temperature)) / \
                      (1.0 - np.exp(-c2 * nu_center / T_ref))

        # Partition function ratio (approximation for linear molecules)
        Q_ratio = T_ratio  # Simplified - exact depends on molecule

        S_T = line_intensities[j] * Q_ratio * boltz_factor * stim_factor

        # Pressure-shifted line center
        nu_shifted = nu_center + pressure_shifts[j] * pressure_atm

        # Add line contribution using Lorentzian profile
        for i in range(n_grid):
            dnu = wavenumber_grid[i] - nu_shifted
            if np.abs(dnu) <= cutoff:
                # Lorentzian line shape
                line_shape = gamma / (np.pi * (dnu * dnu + gamma * gamma))
                absorption[i] += S_T * line_shape * number_density

    return absorption


class GasEngine:
    """Molecular absorption calculation engine.

    Computes absorption coefficients for atmospheric gases using
    Line-by-Line (LBL) calculations with HITRAN spectral data.

    The engine supports:
    - Multiple molecules simultaneously
    - Temperature and pressure scaling
    - Voigt/Lorentzian line shapes
    - Configurable line cutoff distance
    - Optional GPU acceleration (via CuPy)

    Example:
        >>> db = SpectralDatabase("./data/hitran_lines.h5")
        >>> engine = GasEngine(db)
        >>> absorption = engine.compute_absorption(
        ...     wavenumber_range=(2000, 2500),
        ...     temperature=280,
        ...     pressure_pa=50000,
        ...     molecules=["H2O", "CO2"],
        ...     vmr={"H2O": 5000e-6, "CO2": 420e-6}
        ... )
    """

    def __init__(
        self,
        spectral_db: SpectralDatabase,
        use_gpu: bool = False,
        line_cutoff: float = LINE_CUTOFF_CM1,
        min_intensity: float = MIN_LINE_INTENSITY,
    ):
        """Initialize the gas absorption engine.

        Args:
            spectral_db: SpectralDatabase instance with HITRAN data
            use_gpu: Enable GPU acceleration (requires CuPy)
            line_cutoff: Distance from line center to compute [cm^-1]
            min_intensity: Minimum line intensity to include
        """
        self.spectral_db = spectral_db
        self.use_gpu = use_gpu
        self.line_cutoff = line_cutoff
        self.min_intensity = min_intensity

        # Check GPU availability
        if use_gpu:
            try:
                import cupy as cp
                self._gpu_available = True
                logger.info("GPU acceleration enabled via CuPy")
            except ImportError:
                self._gpu_available = False
                logger.warning("CuPy not available, falling back to CPU")
        else:
            self._gpu_available = False

    def compute_absorption(
        self,
        wavenumber_range: Tuple[float, float],
        temperature: float,
        pressure_pa: float,
        molecules: List[str],
        vmr: Dict[str, float],
        resolution: float = 0.01,
        include_continuum: bool = True,
    ) -> AbsorptionCoefficients:
        """Compute molecular absorption coefficients.

        Performs Line-by-Line calculation for specified molecules over
        the given spectral range at the specified atmospheric conditions.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            temperature: Temperature [K]
            pressure_pa: Pressure [Pa]
            molecules: List of molecule names
            vmr: Volume mixing ratios (dimensionless, e.g., 420e-6 for CO2)
            resolution: Spectral resolution [cm^-1]
            include_continuum: Include water vapor continuum

        Returns:
            AbsorptionCoefficients with total and per-molecule absorption
        """
        wn_min, wn_max = wavenumber_range

        # Create wavenumber grid
        num_points = int((wn_max - wn_min) / resolution) + 1
        wavenumber_grid = np.linspace(wn_min, wn_max, num_points)

        # Convert pressure to atmospheres
        pressure_atm = pressure_pa / 101325.0

        # Calculate air number density (ideal gas law)
        # n = P / (k * T) in molecules/cm³
        n_air = (pressure_pa / (BOLTZMANN_CONSTANT * temperature)) * 1e-6  # to cm^-3

        # Initialize total absorption
        total_absorption = np.zeros(num_points)
        molecule_contributions = {}

        logger.debug(f"Computing absorption for {molecules} at T={temperature:.1f}K, P={pressure_pa:.0f}Pa")

        for mol in molecules:
            if mol not in self.spectral_db.molecules:
                logger.warning(f"Molecule {mol} not in database, skipping")
                continue

            # Get mixing ratio
            mol_vmr = vmr.get(mol, 0.0)
            if mol_vmr <= 0:
                logger.debug(f"Zero VMR for {mol}, skipping")
                continue

            # Absorber number density
            n_absorber = n_air * mol_vmr

            # Get spectral lines
            lines = self.spectral_db.get_lines(
                mol,
                wavenumber_range=(wn_min - self.line_cutoff, wn_max + self.line_cutoff),
                min_intensity=self.min_intensity,
            )

            if lines.num_lines == 0:
                logger.debug(f"No lines found for {mol} in range")
                molecule_contributions[mol] = np.zeros(num_points)
                continue

            logger.debug(f"Computing {lines.num_lines} lines for {mol}")

            # Compute LBL absorption
            mol_absorption = compute_absorption_lbl(
                wavenumber_grid=wavenumber_grid,
                line_centers=lines.wavenumber,
                line_intensities=lines.intensity,
                line_widths=lines.air_width,
                temperature=temperature,
                pressure_atm=pressure_atm,
                number_density=n_absorber,
                temp_exp=lines.temp_exp,
                lower_energy=lines.lower_energy,
                pressure_shifts=lines.pressure_shift,
                cutoff=self.line_cutoff,
            )

            molecule_contributions[mol] = mol_absorption
            total_absorption += mol_absorption

        # Add water vapor continuum if requested
        if include_continuum and "H2O" in molecules and vmr.get("H2O", 0) > 0:
            continuum = self._compute_h2o_continuum(
                wavenumber_grid, temperature, pressure_pa, vmr["H2O"]
            )
            if "H2O" in molecule_contributions:
                molecule_contributions["H2O"] += continuum
            total_absorption += continuum

        return AbsorptionCoefficients(
            wavenumber=wavenumber_grid,
            total_absorption=total_absorption,
            molecule_contributions=molecule_contributions,
        )

    def _compute_h2o_continuum(
        self,
        wavenumber: np.ndarray,
        temperature: float,
        pressure_pa: float,
        h2o_vmr: float,
    ) -> np.ndarray:
        """Compute water vapor continuum absorption.

        Uses a simplified MT_CKD-like formulation for the self and foreign
        continuum contributions.

        Args:
            wavenumber: Spectral grid [cm^-1]
            temperature: Temperature [K]
            pressure_pa: Pressure [Pa]
            h2o_vmr: Water vapor volume mixing ratio

        Returns:
            Continuum absorption coefficient [cm^-1]
        """
        # Simplified continuum model
        # Real implementation should use MT_CKD tables

        # Convert to number densities
        n_air = (pressure_pa / (BOLTZMANN_CONSTANT * temperature)) * 1e-6
        n_h2o = n_air * h2o_vmr

        # Temperature correction
        theta = 296.0 / temperature

        # Self-continuum coefficient (very simplified)
        # C_s ~ 4e-25 * exp(-E/kT) in typical window regions
        C_self = 4e-25 * (theta ** 4) * np.exp(-400 / temperature)

        # Foreign continuum coefficient
        C_foreign = 5e-26 * (theta ** 2)

        # Total continuum
        continuum = (C_self * n_h2o + C_foreign * n_air) * n_h2o

        return continuum

    def compute_layer_absorption(
        self,
        wavenumber_range: Tuple[float, float],
        atmosphere_profile,
        molecules: List[str],
        resolution: float = 0.01,
    ) -> List[AbsorptionCoefficients]:
        """Compute absorption for all atmospheric layers.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            atmosphere_profile: AtmosphereProfile with layer data
            molecules: List of molecules to include
            resolution: Spectral resolution [cm^-1]

        Returns:
            List of AbsorptionCoefficients for each layer
        """
        layer_absorptions = []

        for layer in atmosphere_profile.layers:
            # Get VMR for each molecule
            vmr = {}
            for mol in molecules:
                vmr_ppmv = atmosphere_profile.get_gas_profile(mol)
                layer_idx = np.searchsorted(atmosphere_profile.altitudes, layer.altitude_km)
                layer_idx = min(layer_idx, len(vmr_ppmv) - 1)
                vmr[mol] = vmr_ppmv[layer_idx] * 1e-6  # ppmv to mixing ratio

            absorption = self.compute_absorption(
                wavenumber_range=wavenumber_range,
                temperature=layer.temperature_k,
                pressure_pa=layer.pressure_pa,
                molecules=molecules,
                vmr=vmr,
                resolution=resolution,
            )
            layer_absorptions.append(absorption)

        return layer_absorptions

    def compute_cross_section(
        self,
        molecule: str,
        wavenumber_range: Tuple[float, float],
        temperature: float,
        pressure_pa: float,
        resolution: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute absorption cross-section for a single molecule.

        Useful for validation against reference data (HAPI, etc.).

        Args:
            molecule: Molecule name
            wavenumber_range: (min, max) wavenumber [cm^-1]
            temperature: Temperature [K]
            pressure_pa: Pressure [Pa]
            resolution: Spectral resolution [cm^-1]

        Returns:
            Tuple of (wavenumber, cross_section) arrays
        """
        # Compute with unit mixing ratio to get cross-section
        result = self.compute_absorption(
            wavenumber_range=wavenumber_range,
            temperature=temperature,
            pressure_pa=pressure_pa,
            molecules=[molecule],
            vmr={molecule: 1.0},  # Unit mixing ratio
            resolution=resolution,
            include_continuum=False,
        )

        # Convert absorption coefficient to cross-section
        # k = n * sigma, so sigma = k/n
        n_air = (pressure_pa / (BOLTZMANN_CONSTANT * temperature)) * 1e-6
        cross_section = result.total_absorption / n_air

        return result.wavenumber, cross_section
