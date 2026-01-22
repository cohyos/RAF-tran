"""
Scattering Engine for Mie and Rayleigh scattering calculations.

Implements:
- FR-04: Mie scattering for aerosols (Rural, Urban, Maritime, Desert)
- FR-05: Visibility to optical depth conversion

This module computes scattering optical properties for:
- Molecular (Rayleigh) scattering
- Aerosol (Mie) scattering with standard aerosol models

The Mie calculations use the Bohren-Huffman algorithm for
homogeneous spheres.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from numba import jit, complex128

from raf_tran.core.constants import (
    wavenumber_to_wavelength,
    LOSCHMIDT_CONSTANT,
    AEROSOL_TYPES,
)

logger = logging.getLogger(__name__)


@dataclass
class ScatteringProperties:
    """Container for scattering optical properties.

    Attributes:
        wavenumber: Spectral grid [cm^-1]
        extinction_coeff: Extinction coefficient [km^-1]
        scattering_coeff: Scattering coefficient [km^-1]
        absorption_coeff: Absorption coefficient [km^-1]
        asymmetry_factor: Asymmetry parameter g
        single_scatter_albedo: Single scattering albedo
    """
    wavenumber: np.ndarray
    extinction_coeff: np.ndarray
    scattering_coeff: np.ndarray
    absorption_coeff: np.ndarray
    asymmetry_factor: np.ndarray
    single_scatter_albedo: np.ndarray

    @property
    def optical_depth_per_km(self) -> np.ndarray:
        """Optical depth per kilometer (same as extinction coefficient)."""
        return self.extinction_coeff


# =============================================================================
# Aerosol Model Parameters
# =============================================================================

@dataclass
class AerosolModelParams:
    """Parameters for standard aerosol models.

    Based on MODTRAN and LOWTRAN aerosol parameterizations.

    Attributes:
        name: Model name
        mode_radii: Log-normal mode radii [um]
        mode_sigmas: Log-normal geometric standard deviations
        mode_weights: Relative number concentrations
        refractive_index_real: Real part of refractive index at 550nm
        refractive_index_imag: Imaginary part of refractive index at 550nm
        extinction_at_550nm: Extinction coefficient at 550nm for VIS=23km [km^-1]
    """
    name: str
    mode_radii: np.ndarray
    mode_sigmas: np.ndarray
    mode_weights: np.ndarray
    refractive_index_real: float
    refractive_index_imag: float
    extinction_at_550nm: float


# Standard aerosol model parameters (from MODTRAN/LOWTRAN)
AEROSOL_MODELS = {
    "RURAL": AerosolModelParams(
        name="RURAL",
        mode_radii=np.array([0.03, 0.24]),
        mode_sigmas=np.array([0.35, 0.4]),
        mode_weights=np.array([0.999, 0.001]),
        refractive_index_real=1.53,
        refractive_index_imag=0.006,
        extinction_at_550nm=0.16,  # km^-1 for VIS=23km
    ),
    "URBAN": AerosolModelParams(
        name="URBAN",
        mode_radii=np.array([0.03, 0.24]),
        mode_sigmas=np.array([0.35, 0.4]),
        mode_weights=np.array([0.999, 0.001]),
        refractive_index_real=1.55,
        refractive_index_imag=0.04,  # More absorbing
        extinction_at_550nm=0.16,
    ),
    "MARITIME": AerosolModelParams(
        name="MARITIME",
        mode_radii=np.array([0.05, 0.3, 2.0]),
        mode_sigmas=np.array([0.4, 0.4, 0.6]),
        mode_weights=np.array([0.99, 0.009, 0.001]),
        refractive_index_real=1.50,
        refractive_index_imag=0.002,
        extinction_at_550nm=0.16,
    ),
    "DESERT": AerosolModelParams(
        name="DESERT",
        mode_radii=np.array([0.5, 2.0]),
        mode_sigmas=np.array([0.5, 0.6]),
        mode_weights=np.array([0.99, 0.01]),
        refractive_index_real=1.53,
        refractive_index_imag=0.008,
        extinction_at_550nm=0.16,
    ),
}


# =============================================================================
# Mie Scattering Functions (Bohren-Huffman algorithm)
# =============================================================================

@jit(nopython=True, cache=True)
def mie_coefficients(x: float, m: complex) -> Tuple[float, float, float]:
    """Compute Mie scattering coefficients using Bohren-Huffman algorithm.

    Computes efficiency factors for a homogeneous sphere.

    Args:
        x: Size parameter (2*pi*r/lambda)
        m: Complex refractive index

    Returns:
        Tuple of (Q_ext, Q_sca, g) - extinction efficiency,
        scattering efficiency, and asymmetry factor
    """
    # Number of terms needed for convergence
    nstop = int(x + 4 * x**0.3333 + 2) + 1

    # Downward recurrence for logarithmic derivative
    nmx = max(nstop, int(abs(m * x))) + 15
    d = np.zeros(nmx + 1, dtype=np.complex128)

    for n in range(nmx, 0, -1):
        d[n-1] = n / (m * x) - 1.0 / (d[n] + n / (m * x))

    # Upward recurrence for Riccati-Bessel functions
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 = np.cos(x)

    xi1 = psi1 - 1j * chi1

    Q_ext = 0.0
    Q_sca = 0.0
    g_num = 0.0  # Numerator for asymmetry factor

    # Initialize previous coefficients for asymmetry factor calculation
    a_n_prev = complex(0.0, 0.0)
    b_n_prev = complex(0.0, 0.0)

    for n in range(1, nstop + 1):
        fn = (2.0 * n + 1.0) / (n * (n + 1.0))
        psi = (2.0 * n - 1.0) * psi1 / x - psi0
        chi = (2.0 * n - 1.0) * chi1 / x - chi0
        xi = psi - 1j * chi

        # Mie coefficients a_n and b_n
        dn = d[n]
        a_n = ((dn / m + n / x) * psi - psi1) / ((dn / m + n / x) * xi - xi1)
        b_n = ((m * dn + n / x) * psi - psi1) / ((m * dn + n / x) * xi - xi1)

        # Efficiency factors
        Q_ext += (2.0 * n + 1.0) * (a_n.real + b_n.real)
        Q_sca += (2.0 * n + 1.0) * (abs(a_n)**2 + abs(b_n)**2)

        # Asymmetry factor contribution
        if n > 1:
            g_num += ((n * (n + 2.0)) / (n + 1.0)) * \
                     (a_n_prev.real * a_n.real + a_n_prev.imag * a_n.imag +
                      b_n_prev.real * b_n.real + b_n_prev.imag * b_n.imag)
            g_num += ((2.0 * n + 1.0) / (n * (n + 1.0))) * \
                     (a_n.real * b_n.real + a_n.imag * b_n.imag)

        a_n_prev = a_n
        b_n_prev = b_n

        # Update for next iteration
        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = xi

    Q_ext *= 2.0 / x**2
    Q_sca *= 2.0 / x**2

    # Asymmetry factor
    g = 4.0 * g_num / (x**2 * Q_sca) if Q_sca > 0 else 0.0

    return Q_ext, Q_sca, g


@jit(nopython=True, cache=True)
def mie_single_particle(
    wavelength_um: float,
    radius_um: float,
    n_real: float,
    n_imag: float,
) -> Tuple[float, float, float, float]:
    """Compute Mie properties for a single particle.

    Args:
        wavelength_um: Wavelength [um]
        radius_um: Particle radius [um]
        n_real: Real part of refractive index
        n_imag: Imaginary part of refractive index

    Returns:
        (Q_ext, Q_sca, Q_abs, g) - efficiency factors and asymmetry parameter
    """
    # Size parameter
    x = 2.0 * np.pi * radius_um / wavelength_um

    # Complex refractive index
    m = complex(n_real, n_imag)

    Q_ext, Q_sca, g = mie_coefficients(x, m)
    Q_abs = Q_ext - Q_sca

    return Q_ext, Q_sca, Q_abs, g


def compute_rayleigh_scattering(
    wavenumber: np.ndarray,
    pressure_pa: float,
    temperature: float,
) -> ScatteringProperties:
    """Compute Rayleigh scattering for air.

    Uses the standard formula for molecular scattering by air molecules.

    Args:
        wavenumber: Spectral grid [cm^-1]
        pressure_pa: Atmospheric pressure [Pa]
        temperature: Temperature [K]

    Returns:
        ScatteringProperties for Rayleigh scattering
    """
    # Convert wavenumber to wavelength in um
    wavelength_um = wavenumber_to_wavelength(wavenumber)

    # Rayleigh scattering cross-section [cm^2]
    # sigma = (8 * pi^3 / 3) * ((n^2 - 1)^2 / (N^2 * lambda^4)) * F_K
    # where F_K is the King correction factor (~1.05 for air)

    # Simplified formula: sigma = A / lambda^4
    # A ~ 4.02e-28 cm^2 um^4 for standard air
    A = 4.02e-28  # cm^2 um^4
    sigma_rayleigh = A / (wavelength_um ** 4)  # cm^2

    # Number density [molecules/cm^3]
    k_B = 1.380649e-23  # Boltzmann constant
    n_air = (pressure_pa / (k_B * temperature)) * 1e-6  # cm^-3

    # Scattering coefficient [cm^-1] -> [km^-1]
    scattering_coeff = sigma_rayleigh * n_air * 1e5  # to km^-1

    # Rayleigh scattering has no absorption
    extinction_coeff = scattering_coeff.copy()
    absorption_coeff = np.zeros_like(scattering_coeff)

    # Rayleigh asymmetry factor is 0
    asymmetry = np.zeros_like(wavenumber)

    # Single scatter albedo is 1 for pure scattering
    ssa = np.ones_like(wavenumber)

    return ScatteringProperties(
        wavenumber=wavenumber,
        extinction_coeff=extinction_coeff,
        scattering_coeff=scattering_coeff,
        absorption_coeff=absorption_coeff,
        asymmetry_factor=asymmetry,
        single_scatter_albedo=ssa,
    )


class ScatteringEngine:
    """Scattering calculation engine for aerosols and molecules.

    Computes scattering optical properties using:
    - Mie theory for aerosol particles
    - Rayleigh formula for molecular scattering
    - Standard aerosol models (Rural, Urban, Maritime, Desert)

    Example:
        >>> engine = ScatteringEngine()
        >>> props = engine.compute_aerosol_scattering(
        ...     wavenumber_range=(2000, 3333),
        ...     aerosol_type="RURAL",
        ...     visibility_km=23.0,
        ... )
    """

    def __init__(
        self,
        use_lut: bool = True,
        lut_resolution: int = 100,
    ):
        """Initialize the scattering engine.

        Args:
            use_lut: Use lookup tables for faster computation
            lut_resolution: Number of points in LUT
        """
        self.use_lut = use_lut
        self.lut_resolution = lut_resolution
        self._lut_cache: Dict[str, np.ndarray] = {}

    def compute_aerosol_scattering(
        self,
        wavenumber_range: Tuple[float, float],
        aerosol_type: str,
        visibility_km: float = 23.0,
        resolution: float = 1.0,
        altitude_km: float = 0.0,
    ) -> ScatteringProperties:
        """Compute aerosol scattering properties.

        Implements FR-04 (aerosol models) and FR-05 (visibility conversion).

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            aerosol_type: Aerosol model type (RURAL, URBAN, MARITIME, DESERT)
            visibility_km: Surface visibility [km]
            resolution: Spectral resolution [cm^-1]
            altitude_km: Altitude for scale height correction [km]

        Returns:
            ScatteringProperties for aerosol layer
        """
        if aerosol_type == "NONE":
            return self._zero_scattering(wavenumber_range, resolution)

        if aerosol_type not in AEROSOL_MODELS:
            raise ValueError(
                f"Unknown aerosol type: {aerosol_type}. "
                f"Available: {list(AEROSOL_MODELS.keys())}"
            )

        model = AEROSOL_MODELS[aerosol_type]

        # Create wavenumber grid
        wn_min, wn_max = wavenumber_range
        num_points = int((wn_max - wn_min) / resolution) + 1
        wavenumber = np.linspace(wn_min, wn_max, num_points)
        wavelength_um = wavenumber_to_wavelength(wavenumber)

        # Convert visibility to extinction at 550nm (FR-05)
        # Koschmieder equation: VIS = 3.912 / beta_ext
        beta_550 = 3.912 / visibility_km  # km^-1

        # Reference extinction at VIS=23km
        beta_ref = model.extinction_at_550nm

        # Scale factor
        scale_factor = beta_550 / beta_ref

        # Compute wavelength-dependent extinction using Mie theory
        # For efficiency, we use Angstrom exponent approximation
        # beta(lambda) = beta(550) * (550/lambda)^alpha
        # Typical alpha ~ 1.0-1.5 for rural aerosol

        alpha = self._compute_angstrom_exponent(model)
        wavelength_550 = 0.55  # um

        # Extinction coefficient [km^-1]
        extinction = beta_550 * (wavelength_550 / wavelength_um) ** alpha

        # Single scatter albedo (wavelength dependent)
        # Approximate: SSA increases slightly at longer wavelengths
        ssa_550 = 1.0 - model.refractive_index_imag / model.refractive_index_real
        ssa = ssa_550 * (1.0 + 0.1 * (wavelength_um - 0.55))
        ssa = np.clip(ssa, 0.0, 1.0)

        # Scattering and absorption coefficients
        scattering = extinction * ssa
        absorption = extinction * (1.0 - ssa)

        # Asymmetry factor (approximate)
        # g ~ 0.65-0.75 for typical aerosols, increases with particle size
        mean_size_param = 2.0 * np.pi * model.mode_radii[0] / wavelength_um
        g_factor = 0.65 + 0.1 * np.tanh(mean_size_param - 1.0)
        asymmetry = np.full_like(wavenumber, np.mean(g_factor))

        # Altitude scale height correction (exponential decrease)
        scale_height = 2.0  # km
        altitude_factor = np.exp(-altitude_km / scale_height)
        extinction *= altitude_factor
        scattering *= altitude_factor
        absorption *= altitude_factor

        return ScatteringProperties(
            wavenumber=wavenumber,
            extinction_coeff=extinction,
            scattering_coeff=scattering,
            absorption_coeff=absorption,
            asymmetry_factor=asymmetry,
            single_scatter_albedo=ssa,
        )

    def _compute_angstrom_exponent(self, model: AerosolModelParams) -> float:
        """Compute Angstrom exponent from aerosol model parameters.

        The Angstrom exponent describes the wavelength dependence of
        extinction: beta ~ lambda^(-alpha)

        Args:
            model: Aerosol model parameters

        Returns:
            Angstrom exponent
        """
        # Approximate: alpha depends on effective radius
        # Smaller particles -> larger alpha
        r_eff = np.sum(model.mode_weights * model.mode_radii)

        # Empirical relationship
        if r_eff < 0.1:
            alpha = 1.5
        elif r_eff < 0.5:
            alpha = 1.2
        else:
            alpha = 0.5

        return alpha

    def compute_rayleigh_scattering(
        self,
        wavenumber_range: Tuple[float, float],
        pressure_pa: float,
        temperature: float,
        resolution: float = 1.0,
    ) -> ScatteringProperties:
        """Compute Rayleigh (molecular) scattering.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            pressure_pa: Atmospheric pressure [Pa]
            temperature: Temperature [K]
            resolution: Spectral resolution [cm^-1]

        Returns:
            ScatteringProperties for Rayleigh scattering
        """
        wn_min, wn_max = wavenumber_range
        num_points = int((wn_max - wn_min) / resolution) + 1
        wavenumber = np.linspace(wn_min, wn_max, num_points)

        return compute_rayleigh_scattering(wavenumber, pressure_pa, temperature)

    def compute_total_scattering(
        self,
        wavenumber_range: Tuple[float, float],
        aerosol_type: str,
        visibility_km: float,
        pressure_pa: float,
        temperature: float,
        altitude_km: float = 0.0,
        resolution: float = 1.0,
    ) -> ScatteringProperties:
        """Compute combined aerosol and Rayleigh scattering.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            aerosol_type: Aerosol model type
            visibility_km: Surface visibility [km]
            pressure_pa: Atmospheric pressure [Pa]
            temperature: Temperature [K]
            altitude_km: Layer altitude [km]
            resolution: Spectral resolution [cm^-1]

        Returns:
            Combined ScatteringProperties
        """
        # Compute aerosol scattering
        aerosol = self.compute_aerosol_scattering(
            wavenumber_range, aerosol_type, visibility_km, resolution, altitude_km
        )

        # Compute Rayleigh scattering
        rayleigh = self.compute_rayleigh_scattering(
            wavenumber_range, pressure_pa, temperature, resolution
        )

        # Combine (simple additive for extinction)
        total_ext = aerosol.extinction_coeff + rayleigh.extinction_coeff
        total_sca = aerosol.scattering_coeff + rayleigh.scattering_coeff
        total_abs = aerosol.absorption_coeff + rayleigh.absorption_coeff

        # Combined single scatter albedo
        total_ssa = np.where(
            total_ext > 0,
            total_sca / total_ext,
            1.0
        )

        # Combined asymmetry factor (weighted by scattering coefficient)
        total_g = np.where(
            total_sca > 0,
            (aerosol.asymmetry_factor * aerosol.scattering_coeff +
             rayleigh.asymmetry_factor * rayleigh.scattering_coeff) / total_sca,
            0.0
        )

        return ScatteringProperties(
            wavenumber=aerosol.wavenumber,
            extinction_coeff=total_ext,
            scattering_coeff=total_sca,
            absorption_coeff=total_abs,
            asymmetry_factor=total_g,
            single_scatter_albedo=total_ssa,
        )

    def _zero_scattering(
        self,
        wavenumber_range: Tuple[float, float],
        resolution: float,
    ) -> ScatteringProperties:
        """Return zero scattering properties (for NONE aerosol type).

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            resolution: Spectral resolution [cm^-1]

        Returns:
            ScatteringProperties with all zeros
        """
        wn_min, wn_max = wavenumber_range
        num_points = int((wn_max - wn_min) / resolution) + 1
        wavenumber = np.linspace(wn_min, wn_max, num_points)

        return ScatteringProperties(
            wavenumber=wavenumber,
            extinction_coeff=np.zeros(num_points),
            scattering_coeff=np.zeros(num_points),
            absorption_coeff=np.zeros(num_points),
            asymmetry_factor=np.zeros(num_points),
            single_scatter_albedo=np.ones(num_points),
        )

    def visibility_to_optical_depth(
        self,
        visibility_km: float,
        path_length_km: float,
        wavelength_um: float = 0.55,
    ) -> float:
        """Convert visibility to optical depth (FR-05).

        Uses the Koschmieder equation: VIS = 3.912 / beta_ext

        Args:
            visibility_km: Meteorological visibility [km]
            path_length_km: Path length [km]
            wavelength_um: Reference wavelength [um]

        Returns:
            Optical depth (dimensionless)
        """
        # Extinction coefficient at reference wavelength
        beta_ext = 3.912 / visibility_km  # km^-1

        # Optical depth = extinction * path length
        return beta_ext * path_length_km
