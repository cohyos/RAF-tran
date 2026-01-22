"""
Atmospheric profile and layer definitions for radiative transfer.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum

from ..utils.constants import (
    STANDARD_PRESSURE,
    STANDARD_TEMPERATURE,
    GRAVITY,
    DRY_AIR_MOLAR_MASS,
    BOLTZMANN_CONSTANT,
    AVOGADRO_NUMBER,
)


class AtmosphereModel(Enum):
    """Standard atmosphere models."""
    US_STANDARD_1976 = "us_standard_1976"
    TROPICAL = "tropical"
    MIDLATITUDE_SUMMER = "midlatitude_summer"
    MIDLATITUDE_WINTER = "midlatitude_winter"
    SUBARCTIC_SUMMER = "subarctic_summer"
    SUBARCTIC_WINTER = "subarctic_winter"
    CUSTOM = "custom"


@dataclass
class AtmosphericProfile:
    """
    Represents a vertical atmospheric profile.

    Attributes
    ----------
    altitude : np.ndarray
        Altitude levels in meters (from surface to TOA)
    pressure : np.ndarray
        Pressure at each level in Pa
    temperature : np.ndarray
        Temperature at each level in K
    h2o_vmr : np.ndarray
        Water vapor volume mixing ratio
    co2_vmr : np.ndarray
        CO2 volume mixing ratio
    o3_vmr : np.ndarray
        Ozone volume mixing ratio
    n2o_vmr : np.ndarray, optional
        N2O volume mixing ratio
    ch4_vmr : np.ndarray, optional
        Methane volume mixing ratio
    co_vmr : np.ndarray, optional
        Carbon monoxide volume mixing ratio
    """
    altitude: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    h2o_vmr: np.ndarray
    co2_vmr: np.ndarray
    o3_vmr: np.ndarray
    n2o_vmr: Optional[np.ndarray] = None
    ch4_vmr: Optional[np.ndarray] = None
    co_vmr: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate profile dimensions."""
        n_levels = len(self.altitude)
        for attr_name in ['pressure', 'temperature', 'h2o_vmr', 'co2_vmr', 'o3_vmr']:
            attr = getattr(self, attr_name)
            if len(attr) != n_levels:
                raise ValueError(f"{attr_name} must have {n_levels} elements")

    @property
    def n_levels(self) -> int:
        """Number of altitude levels."""
        return len(self.altitude)

    @property
    def n_layers(self) -> int:
        """Number of atmospheric layers."""
        return self.n_levels - 1

    def number_density(self) -> np.ndarray:
        """
        Calculate total air number density at each level.

        Returns
        -------
        np.ndarray
            Number density in molecules/m^3
        """
        return self.pressure / (BOLTZMANN_CONSTANT * self.temperature)

    def layer_path_length(self) -> np.ndarray:
        """
        Calculate geometric path length through each layer (vertical).

        Returns
        -------
        np.ndarray
            Path lengths in meters
        """
        return np.diff(self.altitude)

    def column_density(self, species: str) -> np.ndarray:
        """
        Calculate column density for a species through each layer.

        Parameters
        ----------
        species : str
            Species name (h2o, co2, o3, etc.)

        Returns
        -------
        np.ndarray
            Column density in molecules/m^2 for each layer
        """
        vmr_attr = f"{species.lower()}_vmr"
        if not hasattr(self, vmr_attr):
            raise ValueError(f"Unknown species: {species}")

        vmr = getattr(self, vmr_attr)
        if vmr is None:
            return np.zeros(self.n_layers)

        # Layer-averaged values
        n_air = self.number_density()
        n_species = n_air * vmr

        # Average over layer
        n_avg = 0.5 * (n_species[:-1] + n_species[1:])
        path_length = self.layer_path_length()

        return n_avg * path_length


class Atmosphere:
    """
    Main atmosphere class for radiative transfer calculations.

    Provides methods to create standard atmosphere profiles and
    compute atmospheric optical properties.
    """

    def __init__(self, profile: Optional[AtmosphericProfile] = None,
                 model: AtmosphereModel = AtmosphereModel.US_STANDARD_1976):
        """
        Initialize atmosphere.

        Parameters
        ----------
        profile : AtmosphericProfile, optional
            Custom atmospheric profile
        model : AtmosphereModel
            Standard atmosphere model to use if no profile provided
        """
        if profile is not None:
            self.profile = profile
            self.model = AtmosphereModel.CUSTOM
        else:
            self.model = model
            self.profile = self._create_standard_profile(model)

    def _create_standard_profile(self, model: AtmosphereModel) -> AtmosphericProfile:
        """Create a standard atmosphere profile."""
        if model == AtmosphereModel.US_STANDARD_1976:
            return self._us_standard_1976()
        elif model == AtmosphereModel.TROPICAL:
            return self._tropical_atmosphere()
        elif model == AtmosphereModel.MIDLATITUDE_SUMMER:
            return self._midlatitude_summer()
        elif model == AtmosphereModel.MIDLATITUDE_WINTER:
            return self._midlatitude_winter()
        else:
            return self._us_standard_1976()

    def _us_standard_1976(self) -> AtmosphericProfile:
        """
        US Standard Atmosphere 1976.

        Simplified version with key atmospheric layers.
        """
        # Altitude levels (m) - from surface to 100 km
        altitude = np.array([
            0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
            11000, 12000, 15000, 20000, 25000, 30000, 35000, 40000, 45000,
            50000, 60000, 70000, 80000, 90000, 100000
        ], dtype=float)

        # Temperature (K) - US Standard 1976 approximation
        temperature = np.array([
            288.15, 281.65, 275.15, 268.65, 262.15, 255.65, 249.15, 242.65,
            236.15, 229.65, 223.15, 216.65, 216.65, 216.65, 216.65, 221.65,
            226.65, 237.05, 251.05, 264.15, 270.65, 245.45, 219.65, 198.65,
            186.95, 195.08
        ])

        # Pressure (Pa) - hydrostatic approximation
        pressure = STANDARD_PRESSURE * np.exp(-altitude / 8500)

        # Water vapor VMR (decreasing with altitude)
        h2o_vmr = 0.01 * np.exp(-altitude / 2000)
        h2o_vmr[altitude > 15000] = 3e-6  # Stratospheric value

        # CO2 VMR (well-mixed)
        co2_vmr = np.full_like(altitude, 420e-6)

        # Ozone VMR (stratospheric peak)
        o3_vmr = 1e-7 * np.exp(-((altitude - 25000) / 8000) ** 2) * 10
        o3_vmr += 3e-8  # Background

        # N2O (decreasing above troposphere)
        n2o_vmr = 335e-9 * np.exp(-altitude / 50000)

        # CH4 (decreasing above troposphere)
        ch4_vmr = 1.9e-6 * np.exp(-altitude / 60000)

        return AtmosphericProfile(
            altitude=altitude,
            pressure=pressure,
            temperature=temperature,
            h2o_vmr=h2o_vmr,
            co2_vmr=co2_vmr,
            o3_vmr=o3_vmr,
            n2o_vmr=n2o_vmr,
            ch4_vmr=ch4_vmr,
        )

    def _tropical_atmosphere(self) -> AtmosphericProfile:
        """Tropical atmosphere profile."""
        base = self._us_standard_1976()
        # Tropical modifications
        temperature = base.temperature + 10 * np.exp(-base.altitude / 5000)
        h2o_vmr = base.h2o_vmr * 2  # Higher humidity in tropics

        return AtmosphericProfile(
            altitude=base.altitude,
            pressure=base.pressure,
            temperature=temperature,
            h2o_vmr=h2o_vmr,
            co2_vmr=base.co2_vmr,
            o3_vmr=base.o3_vmr,
            n2o_vmr=base.n2o_vmr,
            ch4_vmr=base.ch4_vmr,
        )

    def _midlatitude_summer(self) -> AtmosphericProfile:
        """Midlatitude summer atmosphere profile."""
        base = self._us_standard_1976()
        temperature = base.temperature + 5 * np.exp(-base.altitude / 8000)
        h2o_vmr = base.h2o_vmr * 1.5

        return AtmosphericProfile(
            altitude=base.altitude,
            pressure=base.pressure,
            temperature=temperature,
            h2o_vmr=h2o_vmr,
            co2_vmr=base.co2_vmr,
            o3_vmr=base.o3_vmr,
            n2o_vmr=base.n2o_vmr,
            ch4_vmr=base.ch4_vmr,
        )

    def _midlatitude_winter(self) -> AtmosphericProfile:
        """Midlatitude winter atmosphere profile."""
        base = self._us_standard_1976()
        temperature = base.temperature - 10 * np.exp(-base.altitude / 5000)
        h2o_vmr = base.h2o_vmr * 0.5  # Drier in winter

        return AtmosphericProfile(
            altitude=base.altitude,
            pressure=base.pressure,
            temperature=temperature,
            h2o_vmr=h2o_vmr,
            co2_vmr=base.co2_vmr,
            o3_vmr=base.o3_vmr,
            n2o_vmr=base.n2o_vmr,
            ch4_vmr=base.ch4_vmr,
        )

    @property
    def n_levels(self) -> int:
        """Number of altitude levels."""
        return self.profile.n_levels

    @property
    def n_layers(self) -> int:
        """Number of atmospheric layers."""
        return self.profile.n_layers

    def get_layer_properties(self, layer_idx: int) -> Dict:
        """
        Get averaged properties for a specific layer.

        Parameters
        ----------
        layer_idx : int
            Layer index (0 = surface layer)

        Returns
        -------
        dict
            Layer properties including temperature, pressure, path length
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(f"Layer index must be 0-{self.n_layers - 1}")

        p = self.profile

        return {
            "altitude_bottom": p.altitude[layer_idx],
            "altitude_top": p.altitude[layer_idx + 1],
            "pressure": 0.5 * (p.pressure[layer_idx] + p.pressure[layer_idx + 1]),
            "temperature": 0.5 * (p.temperature[layer_idx] + p.temperature[layer_idx + 1]),
            "path_length": p.altitude[layer_idx + 1] - p.altitude[layer_idx],
            "h2o_vmr": 0.5 * (p.h2o_vmr[layer_idx] + p.h2o_vmr[layer_idx + 1]),
            "co2_vmr": 0.5 * (p.co2_vmr[layer_idx] + p.co2_vmr[layer_idx + 1]),
            "o3_vmr": 0.5 * (p.o3_vmr[layer_idx] + p.o3_vmr[layer_idx + 1]),
        }

    def surface_temperature(self) -> float:
        """Return surface temperature in K."""
        return self.profile.temperature[0]

    def surface_pressure(self) -> float:
        """Return surface pressure in Pa."""
        return self.profile.pressure[0]
