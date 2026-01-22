"""
Gas absorption module using correlated-k distribution method.

Implements simplified correlated-k absorption coefficients for
major atmospheric absorbers: H2O, CO2, O3, N2O, CH4, CO.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from enum import Enum

from ..utils.constants import (
    BOLTZMANN_CONSTANT,
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
)


class AbsorberSpecies(Enum):
    """Supported absorbing species."""
    H2O = "h2o"
    CO2 = "co2"
    O3 = "o3"
    N2O = "n2o"
    CH4 = "ch4"
    CO = "co"


@dataclass
class CKDCoefficients:
    """
    Correlated-k distribution coefficients for a spectral band.

    Attributes
    ----------
    wavenumber_min : float
        Lower wavenumber bound (cm^-1)
    wavenumber_max : float
        Upper wavenumber bound (cm^-1)
    g_points : np.ndarray
        Gauss-Legendre quadrature points (cumulative probability)
    weights : np.ndarray
        Gauss-Legendre weights
    k_coeffs : np.ndarray
        Absorption coefficients at each g-point (cm^2/molecule)
    reference_temperature : float
        Reference temperature (K)
    reference_pressure : float
        Reference pressure (Pa)
    """
    wavenumber_min: float
    wavenumber_max: float
    g_points: np.ndarray
    weights: np.ndarray
    k_coeffs: np.ndarray
    reference_temperature: float = 296.0
    reference_pressure: float = 101325.0


class GasAbsorption:
    """
    Gas absorption calculator using correlated-k distribution method.

    This implements a simplified CKD approach suitable for broadband
    radiative transfer calculations.
    """

    # Number of g-points for quadrature (8-point Gauss-Legendre)
    N_GPOINTS = 8

    # Absorption band definitions for major species
    # Each band: (wavenumber_min, wavenumber_max, peak_cross_section_cm2)
    ABSORPTION_BANDS = {
        AbsorberSpecies.H2O: [
            (0, 350, 1e-22),       # Rotational band
            (1200, 2000, 5e-21),   # 6.3 μm band
            (3000, 4200, 2e-20),   # 2.7 μm band
            (5000, 7500, 1e-20),   # 1.38 μm band
        ],
        AbsorberSpecies.CO2: [
            (580, 760, 5e-19),     # 15 μm band (strongest)
            (900, 1100, 1e-22),    # 10 μm band
            (1900, 2200, 5e-20),   # 4.3 μm band
            (3500, 3800, 1e-21),   # 2.7 μm band
        ],
        AbsorberSpecies.O3: [
            (980, 1080, 1e-17),    # 9.6 μm band
            (14000, 40000, 1e-17), # Hartley-Huggins UV bands
        ],
        AbsorberSpecies.N2O: [
            (1200, 1350, 1e-18),   # 7.8 μm band
            (2150, 2280, 5e-19),   # 4.5 μm band
        ],
        AbsorberSpecies.CH4: [
            (1200, 1400, 1e-19),   # 7.7 μm band
            (2800, 3200, 5e-19),   # 3.3 μm band
        ],
    }

    def __init__(self, species: Optional[List[AbsorberSpecies]] = None):
        """
        Initialize gas absorption calculator.

        Parameters
        ----------
        species : list of AbsorberSpecies, optional
            Species to include. Default includes all major absorbers.
        """
        if species is None:
            self.species = list(AbsorberSpecies)
        else:
            self.species = species

        # Initialize CKD coefficients for each species
        self.ckd_coefficients = self._initialize_ckd_coefficients()

        # Gauss-Legendre quadrature
        self.g_points, self.weights = np.polynomial.legendre.leggauss(self.N_GPOINTS)
        # Transform from [-1, 1] to [0, 1]
        self.g_points = 0.5 * (self.g_points + 1)
        self.weights = 0.5 * self.weights

    def _initialize_ckd_coefficients(self) -> Dict[AbsorberSpecies, List[CKDCoefficients]]:
        """
        Initialize CKD coefficients for all species and bands.

        Uses simplified parameterization based on band-averaged properties.
        """
        g_points, weights = np.polynomial.legendre.leggauss(self.N_GPOINTS)
        g_points = 0.5 * (g_points + 1)
        weights = 0.5 * weights

        coefficients = {}

        for species in self.species:
            if species not in self.ABSORPTION_BANDS:
                continue

            species_coeffs = []
            for band in self.ABSORPTION_BANDS[species]:
                wn_min, wn_max, peak_k = band

                # Generate k-distribution (log-normal distribution of k values)
                # k values span several orders of magnitude
                k_min = peak_k * 1e-4
                k_max = peak_k

                # Log-spaced k values at g-points
                log_k = np.linspace(np.log(k_min), np.log(k_max), self.N_GPOINTS)
                k_coeffs = np.exp(log_k)

                ckd = CKDCoefficients(
                    wavenumber_min=wn_min,
                    wavenumber_max=wn_max,
                    g_points=g_points,
                    weights=weights,
                    k_coeffs=k_coeffs,
                )
                species_coeffs.append(ckd)

            coefficients[species] = species_coeffs

        return coefficients

    def _temperature_correction(self, k_ref: np.ndarray,
                                  temperature: float,
                                  reference_temp: float = 296.0) -> np.ndarray:
        """
        Apply temperature correction to absorption coefficients.

        Uses simplified temperature dependence for line strengths.

        Parameters
        ----------
        k_ref : np.ndarray
            Reference absorption coefficients
        temperature : float
            Actual temperature (K)
        reference_temp : float
            Reference temperature (K)

        Returns
        -------
        np.ndarray
            Temperature-corrected coefficients
        """
        # Simplified temperature dependence
        # S(T) = S(T_ref) * (T_ref/T)^n * exp(-E/k * (1/T - 1/T_ref))
        # Using n=1.5 and effective E/k = 500 K as typical values
        n_exponent = 1.5
        effective_energy = 500  # K

        temp_ratio = reference_temp / temperature
        exp_factor = np.exp(-effective_energy * (1/temperature - 1/reference_temp))

        return k_ref * (temp_ratio ** n_exponent) * exp_factor

    def _pressure_broadening(self, k_ref: np.ndarray,
                              pressure: float,
                              temperature: float,
                              reference_pressure: float = 101325.0,
                              reference_temp: float = 296.0) -> np.ndarray:
        """
        Apply pressure broadening correction.

        Parameters
        ----------
        k_ref : np.ndarray
            Reference absorption coefficients
        pressure : float
            Actual pressure (Pa)
        temperature : float
            Actual temperature (K)
        reference_pressure : float
            Reference pressure (Pa)
        reference_temp : float
            Reference temperature (K)

        Returns
        -------
        np.ndarray
            Pressure-corrected coefficients
        """
        # Lorentz half-width pressure dependence
        # γ(p,T) = γ_ref * (p/p_ref) * (T_ref/T)^n
        n_exponent = 0.75  # Typical temperature exponent

        pressure_ratio = pressure / reference_pressure
        temp_ratio = (reference_temp / temperature) ** n_exponent

        # For CKD, pressure mainly affects the k-distribution shape
        # Higher pressure -> more absorption in line wings
        return k_ref * pressure_ratio * temp_ratio

    def compute_optical_depth(self,
                               species: AbsorberSpecies,
                               wavenumber: float,
                               column_density: float,
                               temperature: float,
                               pressure: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical depth for a species at given conditions.

        Parameters
        ----------
        species : AbsorberSpecies
            Absorbing species
        wavenumber : float
            Wavenumber (cm^-1)
        column_density : float
            Column density (molecules/m^2)
        temperature : float
            Temperature (K)
        pressure : float
            Pressure (Pa)

        Returns
        -------
        optical_depths : np.ndarray
            Optical depth at each g-point
        weights : np.ndarray
            Quadrature weights
        """
        if species not in self.ckd_coefficients:
            return np.zeros(self.N_GPOINTS), self.weights

        # Find relevant absorption band
        for ckd in self.ckd_coefficients[species]:
            if ckd.wavenumber_min <= wavenumber <= ckd.wavenumber_max:
                # Apply temperature and pressure corrections
                k_corrected = self._temperature_correction(
                    ckd.k_coeffs, temperature, ckd.reference_temperature
                )
                k_corrected = self._pressure_broadening(
                    k_corrected, pressure, temperature,
                    ckd.reference_pressure, ckd.reference_temperature
                )

                # Convert column density from molecules/m^2 to molecules/cm^2
                column_density_cm2 = column_density * 1e-4

                # Optical depth = k * N
                optical_depth = k_corrected * column_density_cm2

                return optical_depth, ckd.weights

        # Wavenumber not in any band - no absorption
        return np.zeros(self.N_GPOINTS), self.weights

    def compute_total_optical_depth(self,
                                     wavenumber: float,
                                     layer_properties: Dict,
                                     vmr_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total optical depth from all species for a layer.

        Parameters
        ----------
        wavenumber : float
            Wavenumber (cm^-1)
        layer_properties : dict
            Layer properties (temperature, pressure, path_length)
        vmr_dict : dict
            Volume mixing ratios for each species

        Returns
        -------
        total_optical_depth : np.ndarray
            Combined optical depth at each g-point
        weights : np.ndarray
            Quadrature weights
        """
        total_tau = np.zeros(self.N_GPOINTS)

        temperature = layer_properties["temperature"]
        pressure = layer_properties["pressure"]
        path_length = layer_properties["path_length"]

        # Number density (molecules/m^3)
        n_air = pressure / (BOLTZMANN_CONSTANT * temperature)

        for species in self.species:
            species_name = species.value.lower()
            vmr_key = f"{species_name}_vmr"

            if vmr_key in vmr_dict and vmr_dict[vmr_key] is not None:
                vmr = vmr_dict[vmr_key]
                # Column density for the layer
                column_density = n_air * vmr * path_length

                tau, weights = self.compute_optical_depth(
                    species, wavenumber, column_density, temperature, pressure
                )
                total_tau += tau

        return total_tau, self.weights

    def transmittance(self, optical_depth: np.ndarray,
                       weights: np.ndarray) -> float:
        """
        Compute band-averaged transmittance from g-point optical depths.

        Parameters
        ----------
        optical_depth : np.ndarray
            Optical depths at each g-point
        weights : np.ndarray
            Quadrature weights

        Returns
        -------
        float
            Band-averaged transmittance
        """
        return np.sum(weights * np.exp(-optical_depth))

    def absorption_coefficient_spectrum(self,
                                         species: AbsorberSpecies,
                                         wavenumber_grid: np.ndarray,
                                         temperature: float = 296.0,
                                         pressure: float = 101325.0) -> np.ndarray:
        """
        Get absorption coefficient spectrum (band-averaged k values).

        Parameters
        ----------
        species : AbsorberSpecies
            Species to compute
        wavenumber_grid : np.ndarray
            Wavenumber grid (cm^-1)
        temperature : float
            Temperature (K)
        pressure : float
            Pressure (Pa)

        Returns
        -------
        np.ndarray
            Effective absorption coefficient at each wavenumber (cm^2/molecule)
        """
        k_spectrum = np.zeros_like(wavenumber_grid)

        if species not in self.ckd_coefficients:
            return k_spectrum

        for i, wn in enumerate(wavenumber_grid):
            for ckd in self.ckd_coefficients[species]:
                if ckd.wavenumber_min <= wn <= ckd.wavenumber_max:
                    k_corrected = self._temperature_correction(
                        ckd.k_coeffs, temperature, ckd.reference_temperature
                    )
                    k_corrected = self._pressure_broadening(
                        k_corrected, pressure, temperature,
                        ckd.reference_pressure, ckd.reference_temperature
                    )
                    # Effective k is weighted average
                    k_spectrum[i] = np.sum(ckd.weights * k_corrected)
                    break

        return k_spectrum
