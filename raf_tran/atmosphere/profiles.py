"""
Atmospheric profile models.

Implements standard atmospheric profiles including US Standard Atmosphere 1976
and MODTRAN model atmospheres (tropical, midlatitude, subarctic).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from raf_tran.utils.constants import (
    EARTH_SURFACE_GRAVITY,
    GAS_CONSTANT,
    DRY_AIR_MOLAR_MASS,
    STANDARD_PRESSURE,
    STANDARD_TEMPERATURE,
)


@dataclass
class AtmosphericLayer:
    """
    Represents a single atmospheric layer.

    Attributes
    ----------
    z_bottom : float
        Bottom altitude in meters
    z_top : float
        Top altitude in meters
    temperature : float
        Mean layer temperature in Kelvin
    pressure : float
        Mean layer pressure in Pa
    density : float
        Mean layer density in kg/m³
    h2o_vmr : float
        Water vapor volume mixing ratio
    co2_vmr : float
        CO2 volume mixing ratio
    o3_vmr : float
        Ozone volume mixing ratio
    """

    z_bottom: float
    z_top: float
    temperature: float
    pressure: float
    density: float
    h2o_vmr: float = 0.0
    co2_vmr: float = 420e-6  # ~420 ppm default
    o3_vmr: float = 0.0

    @property
    def thickness(self) -> float:
        """Layer geometric thickness in meters."""
        return self.z_top - self.z_bottom

    @property
    def z_mid(self) -> float:
        """Layer midpoint altitude in meters."""
        return (self.z_bottom + self.z_top) / 2


class AtmosphereProfile(ABC):
    """
    Abstract base class for atmospheric profiles.

    Subclasses must implement methods to return temperature, pressure,
    and gas concentrations as functions of altitude.
    """

    @abstractmethod
    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get temperature at given altitudes.

        Parameters
        ----------
        altitude : array_like
            Altitude in meters

        Returns
        -------
        temperature : ndarray
            Temperature in Kelvin
        """
        pass

    @abstractmethod
    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get pressure at given altitudes.

        Parameters
        ----------
        altitude : array_like
            Altitude in meters

        Returns
        -------
        pressure : ndarray
            Pressure in Pa
        """
        pass

    def density(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get air density at given altitudes using ideal gas law.

        Parameters
        ----------
        altitude : array_like
            Altitude in meters

        Returns
        -------
        density : ndarray
            Air density in kg/m³
        """
        T = self.temperature(altitude)
        P = self.pressure(altitude)
        # ρ = PM / RT
        return P * DRY_AIR_MOLAR_MASS / (GAS_CONSTANT * T)

    def number_density(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get total number density at given altitudes.

        Parameters
        ----------
        altitude : array_like
            Altitude in meters

        Returns
        -------
        n : ndarray
            Number density in molecules/m³
        """
        from raf_tran.utils.constants import AVOGADRO

        T = self.temperature(altitude)
        P = self.pressure(altitude)
        # n = P / kT = P * N_A / RT
        return P * AVOGADRO / (GAS_CONSTANT * T)

    @abstractmethod
    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """Get water vapor volume mixing ratio."""
        pass

    @abstractmethod
    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """Get ozone volume mixing ratio."""
        pass

    def co2_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get CO2 volume mixing ratio.

        Default assumes well-mixed CO2 at ~420 ppm.
        """
        return np.full_like(np.asarray(altitude, dtype=float), 420e-6)

    def create_layers(
        self, z_levels: np.ndarray
    ) -> list[AtmosphericLayer]:
        """
        Create atmospheric layers from altitude levels.

        Parameters
        ----------
        z_levels : array_like
            Altitude levels in meters (must be monotonically increasing)

        Returns
        -------
        layers : list of AtmosphericLayer
            List of atmospheric layers from bottom to top
        """
        z_levels = np.asarray(z_levels)
        layers = []

        for i in range(len(z_levels) - 1):
            z_bot = z_levels[i]
            z_top = z_levels[i + 1]
            z_mid = (z_bot + z_top) / 2

            layer = AtmosphericLayer(
                z_bottom=z_bot,
                z_top=z_top,
                temperature=float(self.temperature(np.array([z_mid]))[0]),
                pressure=float(self.pressure(np.array([z_mid]))[0]),
                density=float(self.density(np.array([z_mid]))[0]),
                h2o_vmr=float(self.h2o_vmr(np.array([z_mid]))[0]),
                co2_vmr=float(self.co2_vmr(np.array([z_mid]))[0]),
                o3_vmr=float(self.o3_vmr(np.array([z_mid]))[0]),
            )
            layers.append(layer)

        return layers


class StandardAtmosphere(AtmosphereProfile):
    """
    US Standard Atmosphere 1976.

    A piecewise linear temperature profile with hydrostatic pressure.
    Valid from 0 to 86 km altitude.

    References
    ----------
    NOAA/NASA/USAF, U.S. Standard Atmosphere, 1976
    """

    # Geopotential altitude levels (m) and temperature gradients (K/m)
    # (base altitude, base temperature, lapse rate)
    _LAYERS = [
        (0, 288.15, -0.0065),  # Troposphere
        (11000, 216.65, 0.0),  # Tropopause
        (20000, 216.65, 0.001),  # Stratosphere 1
        (32000, 228.65, 0.0028),  # Stratosphere 2
        (47000, 270.65, 0.0),  # Stratopause
        (51000, 270.65, -0.0028),  # Mesosphere 1
        (71000, 214.65, -0.002),  # Mesosphere 2
        (86000, 186.95, 0.0),  # Upper limit
    ]

    def __init__(self):
        """Initialize US Standard Atmosphere 1976."""
        # Precompute base pressures for each layer
        self._base_pressures = [STANDARD_PRESSURE]

        for i in range(len(self._LAYERS) - 1):
            z_b, T_b, L = self._LAYERS[i]
            z_next = self._LAYERS[i + 1][0]
            P_b = self._base_pressures[-1]

            if L == 0:
                # Isothermal layer
                P_next = P_b * np.exp(
                    -EARTH_SURFACE_GRAVITY
                    * DRY_AIR_MOLAR_MASS
                    * (z_next - z_b)
                    / (GAS_CONSTANT * T_b)
                )
            else:
                # Linear temperature gradient
                T_next = T_b + L * (z_next - z_b)
                P_next = P_b * (T_next / T_b) ** (
                    -EARTH_SURFACE_GRAVITY * DRY_AIR_MOLAR_MASS / (GAS_CONSTANT * L)
                )

            self._base_pressures.append(P_next)

    def _get_layer_index(self, altitude: float) -> int:
        """Get the index of the atmospheric layer containing the altitude."""
        for i in range(len(self._LAYERS) - 1):
            if altitude < self._LAYERS[i + 1][0]:
                return i
        return len(self._LAYERS) - 2

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        """Get temperature at given altitudes."""
        altitude = np.asarray(altitude)
        result = np.zeros_like(altitude, dtype=float)

        for i, z in enumerate(altitude.flat):
            layer_idx = self._get_layer_index(z)
            z_b, T_b, L = self._LAYERS[layer_idx]
            result.flat[i] = T_b + L * (z - z_b)

        return result

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        """Get pressure at given altitudes."""
        altitude = np.asarray(altitude)
        result = np.zeros_like(altitude, dtype=float)

        for i, z in enumerate(altitude.flat):
            layer_idx = self._get_layer_index(z)
            z_b, T_b, L = self._LAYERS[layer_idx]
            P_b = self._base_pressures[layer_idx]

            if L == 0:
                result.flat[i] = P_b * np.exp(
                    -EARTH_SURFACE_GRAVITY
                    * DRY_AIR_MOLAR_MASS
                    * (z - z_b)
                    / (GAS_CONSTANT * T_b)
                )
            else:
                T = T_b + L * (z - z_b)
                result.flat[i] = P_b * (T / T_b) ** (
                    -EARTH_SURFACE_GRAVITY * DRY_AIR_MOLAR_MASS / (GAS_CONSTANT * L)
                )

        return result

    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get water vapor volume mixing ratio.

        Simple exponential decrease with altitude scale height of ~2 km.
        Surface value ~1% (humid conditions).
        """
        altitude = np.asarray(altitude)
        # Exponential decay with 2 km scale height
        surface_vmr = 0.01  # 1% at surface
        scale_height = 2000.0  # meters
        return surface_vmr * np.exp(-altitude / scale_height)

    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """
        Get ozone volume mixing ratio.

        Chapman layer profile with peak around 22-25 km.
        Calibrated to give ~300 DU total column.
        """
        altitude = np.asarray(altitude)
        # Simplified ozone profile (Chapman layer approximation)
        # Parameters tuned to give ~300 DU total column ozone
        peak_altitude = 23000.0  # m
        peak_vmr = 8e-6  # ~8 ppmv at peak
        width = 3200.0  # m (tuned to give ~300 DU column)

        return peak_vmr * np.exp(-((altitude - peak_altitude) ** 2) / (2 * width**2))


class MidlatitudeSummer(AtmosphereProfile):
    """
    Midlatitude summer atmospheric profile.

    Based on MODTRAN/LOWTRAN model atmosphere.
    """

    # Altitude levels (km), Temperature (K), Pressure (mbar), H2O (ppmv), O3 (ppmv)
    _DATA = np.array(
        [
            [0, 294.0, 1013.0, 14000, 0.030],
            [1, 290.0, 902.0, 9300, 0.035],
            [2, 285.0, 802.0, 5900, 0.040],
            [3, 279.0, 710.0, 3300, 0.045],
            [4, 273.0, 628.0, 1900, 0.050],
            [5, 267.0, 554.0, 1000, 0.055],
            [6, 261.0, 487.0, 610, 0.070],
            [7, 255.0, 426.0, 370, 0.100],
            [8, 248.0, 372.0, 210, 0.160],
            [9, 242.0, 324.0, 120, 0.280],
            [10, 235.0, 281.0, 64, 0.500],
            [11, 229.0, 243.0, 22, 0.950],
            [12, 222.0, 209.0, 6.0, 1.80],
            [13, 216.0, 179.0, 1.8, 3.30],
            [14, 216.0, 153.0, 1.0, 5.00],
            [15, 216.0, 130.0, 0.76, 6.40],
            [16, 216.0, 111.0, 0.64, 7.30],
            [17, 216.0, 95.0, 0.56, 7.70],
            [18, 216.0, 81.2, 0.50, 7.80],
            [19, 217.0, 69.5, 0.49, 7.50],
            [20, 218.0, 59.5, 0.45, 6.90],
            [25, 224.0, 25.5, 0.26, 4.50],
            [30, 234.0, 11.1, 0.14, 1.60],
            [35, 245.0, 5.0, 0.10, 0.60],
            [40, 258.0, 2.3, 0.075, 0.30],
            [45, 270.0, 1.1, 0.060, 0.17],
            [50, 276.0, 0.52, 0.050, 0.10],
        ]
    )

    # Scale factor to achieve realistic ~320 DU total column ozone
    _O3_SCALE = 0.337

    def __init__(self):
        """Initialize midlatitude summer atmosphere."""
        self._z = self._DATA[:, 0] * 1000  # km to m
        self._T = self._DATA[:, 1]
        self._P = self._DATA[:, 2] * 100  # mbar to Pa
        self._h2o = self._DATA[:, 3] * 1e-6  # ppmv to vmr
        self._o3 = self._DATA[:, 4] * 1e-6 * self._O3_SCALE  # ppmv to vmr, scaled

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        """Get temperature at given altitudes."""
        return np.interp(np.asarray(altitude), self._z, self._T)

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        """Get pressure at given altitudes."""
        # Use log interpolation for pressure
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._P)))

    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """Get water vapor volume mixing ratio."""
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._h2o)))

    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        """Get ozone volume mixing ratio."""
        return np.interp(np.asarray(altitude), self._z, self._o3)


class MidlatitudeWinter(AtmosphereProfile):
    """
    Midlatitude winter atmospheric profile.

    Based on MODTRAN/LOWTRAN model atmosphere.
    """

    _DATA = np.array(
        [
            [0, 272.2, 1018.0, 3500, 0.028],
            [1, 268.7, 898.0, 2500, 0.030],
            [2, 265.2, 791.0, 1800, 0.032],
            [3, 261.7, 693.0, 1200, 0.034],
            [4, 255.7, 608.0, 660, 0.038],
            [5, 249.7, 531.0, 380, 0.045],
            [6, 243.7, 462.0, 220, 0.060],
            [7, 237.7, 400.0, 120, 0.090],
            [8, 231.7, 345.0, 68, 0.160],
            [9, 225.7, 297.0, 36, 0.300],
            [10, 219.7, 255.0, 19, 0.550],
            [11, 219.2, 218.0, 10, 1.10],
            [12, 218.7, 186.0, 5.4, 2.00],
            [13, 218.2, 159.0, 2.9, 3.20],
            [14, 217.7, 136.0, 1.8, 4.50],
            [15, 217.2, 116.0, 1.1, 5.50],
            [16, 216.7, 98.1, 0.72, 6.20],
            [17, 216.2, 83.5, 0.52, 6.50],
            [18, 215.7, 71.1, 0.40, 6.40],
            [19, 215.2, 60.6, 0.34, 6.00],
            [20, 215.2, 51.7, 0.28, 5.40],
            [25, 217.4, 24.3, 0.14, 2.80],
            [30, 227.8, 11.0, 0.085, 1.10],
            [35, 243.2, 4.9, 0.060, 0.40],
            [40, 258.5, 2.2, 0.050, 0.20],
            [45, 265.7, 1.1, 0.040, 0.12],
            [50, 270.6, 0.52, 0.035, 0.075],
        ]
    )

    # Scale factor to achieve realistic ~380 DU total column ozone
    _O3_SCALE = 0.517

    def __init__(self):
        """Initialize midlatitude winter atmosphere."""
        self._z = self._DATA[:, 0] * 1000
        self._T = self._DATA[:, 1]
        self._P = self._DATA[:, 2] * 100
        self._h2o = self._DATA[:, 3] * 1e-6
        self._o3 = self._DATA[:, 4] * 1e-6 * self._O3_SCALE

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._T)

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._P)))

    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._h2o)))

    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._o3)


class TropicalAtmosphere(AtmosphereProfile):
    """
    Tropical atmospheric profile.

    Based on MODTRAN/LOWTRAN model atmosphere.
    """

    _DATA = np.array(
        [
            [0, 300.0, 1013.0, 19000, 0.028],
            [1, 294.0, 904.0, 13000, 0.030],
            [2, 288.0, 805.0, 9300, 0.032],
            [3, 284.0, 715.0, 4700, 0.034],
            [4, 277.0, 633.0, 2700, 0.038],
            [5, 270.0, 559.0, 1500, 0.045],
            [6, 264.0, 492.0, 850, 0.060],
            [7, 257.0, 432.0, 470, 0.100],
            [8, 250.0, 378.0, 250, 0.175],
            [9, 244.0, 329.0, 120, 0.320],
            [10, 237.0, 286.0, 50, 0.570],
            [11, 230.0, 247.0, 17, 1.10],
            [12, 224.0, 213.0, 6.0, 2.00],
            [13, 217.0, 182.0, 1.8, 3.20],
            [14, 210.0, 156.0, 1.0, 4.50],
            [15, 204.0, 132.0, 0.76, 5.80],
            [16, 197.0, 111.0, 0.64, 7.00],
            [17, 195.0, 93.7, 0.56, 7.80],
            [18, 199.0, 78.9, 0.50, 8.10],
            [19, 203.0, 66.6, 0.49, 8.00],
            [20, 207.0, 56.5, 0.45, 7.50],
            [25, 224.0, 25.0, 0.26, 4.50],
            [30, 234.0, 11.1, 0.14, 1.60],
            [35, 245.0, 5.0, 0.10, 0.60],
            [40, 258.0, 2.3, 0.075, 0.30],
            [45, 270.0, 1.1, 0.060, 0.17],
            [50, 276.0, 0.52, 0.050, 0.10],
        ]
    )

    # Scale factor to achieve realistic ~270 DU total column ozone
    _O3_SCALE = 0.273

    def __init__(self):
        """Initialize tropical atmosphere."""
        self._z = self._DATA[:, 0] * 1000
        self._T = self._DATA[:, 1]
        self._P = self._DATA[:, 2] * 100
        self._h2o = self._DATA[:, 3] * 1e-6
        self._o3 = self._DATA[:, 4] * 1e-6 * self._O3_SCALE

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._T)

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._P)))

    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._h2o)))

    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._o3)


class SubarcticSummer(AtmosphereProfile):
    """Subarctic summer atmospheric profile."""

    _DATA = np.array(
        [
            [0, 287.0, 1010.0, 8500, 0.040],
            [1, 282.0, 896.0, 6000, 0.042],
            [2, 276.0, 792.0, 4200, 0.045],
            [3, 271.0, 696.0, 2700, 0.048],
            [4, 266.0, 608.0, 1700, 0.052],
            [5, 260.0, 531.0, 1000, 0.058],
            [6, 253.0, 462.0, 540, 0.075],
            [7, 246.0, 400.0, 290, 0.110],
            [8, 239.0, 345.0, 130, 0.190],
            [9, 232.0, 297.0, 47, 0.340],
            [10, 225.0, 255.0, 22, 0.600],
            [11, 225.0, 218.0, 8.0, 1.20],
            [12, 225.0, 186.0, 2.4, 2.20],
            [13, 225.0, 159.0, 0.80, 3.30],
            [14, 225.0, 136.0, 0.34, 4.40],
            [15, 225.0, 116.0, 0.18, 5.30],
            [16, 225.0, 98.1, 0.12, 5.90],
            [17, 225.0, 83.5, 0.085, 6.20],
            [18, 225.0, 71.1, 0.068, 6.20],
            [19, 225.0, 60.6, 0.057, 5.90],
            [20, 225.0, 51.7, 0.050, 5.30],
            [25, 228.0, 24.3, 0.028, 2.80],
            [30, 235.0, 11.0, 0.018, 1.10],
            [35, 247.0, 4.9, 0.014, 0.40],
            [40, 262.0, 2.2, 0.012, 0.20],
            [45, 274.0, 1.1, 0.010, 0.12],
            [50, 277.0, 0.52, 0.0090, 0.075],
        ]
    )

    # Scale factor to achieve realistic ~350 DU total column ozone
    _O3_SCALE = 0.490

    def __init__(self):
        self._z = self._DATA[:, 0] * 1000
        self._T = self._DATA[:, 1]
        self._P = self._DATA[:, 2] * 100
        self._h2o = self._DATA[:, 3] * 1e-6
        self._o3 = self._DATA[:, 4] * 1e-6 * self._O3_SCALE

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._T)

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._P)))

    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._h2o)))

    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._o3)


class SubarcticWinter(AtmosphereProfile):
    """Subarctic winter atmospheric profile."""

    _DATA = np.array(
        [
            [0, 257.1, 1013.0, 1200, 0.040],
            [1, 259.1, 887.8, 1000, 0.042],
            [2, 255.9, 777.5, 800, 0.045],
            [3, 252.7, 679.8, 590, 0.048],
            [4, 247.7, 593.2, 350, 0.052],
            [5, 240.9, 515.8, 160, 0.058],
            [6, 234.1, 446.7, 68, 0.075],
            [7, 227.3, 385.3, 36, 0.110],
            [8, 220.6, 330.8, 19, 0.190],
            [9, 217.2, 282.9, 10, 0.340],
            [10, 217.2, 241.8, 5.4, 0.600],
            [11, 217.2, 206.7, 2.9, 1.20],
            [12, 217.2, 176.6, 1.5, 2.20],
            [13, 217.2, 151.0, 0.79, 3.30],
            [14, 217.2, 129.1, 0.42, 4.40],
            [15, 217.2, 110.3, 0.22, 5.30],
            [16, 216.6, 94.31, 0.12, 5.90],
            [17, 216.0, 80.58, 0.076, 6.20],
            [18, 215.4, 68.82, 0.054, 6.20],
            [19, 214.8, 58.75, 0.040, 5.90],
            [20, 214.1, 50.14, 0.032, 5.30],
            [25, 211.0, 22.56, 0.015, 2.80],
            [30, 216.5, 10.2, 0.0094, 1.10],
            [35, 227.5, 4.7, 0.0069, 0.40],
            [40, 243.2, 2.2, 0.0058, 0.20],
            [45, 258.5, 1.1, 0.0050, 0.12],
            [50, 265.7, 0.52, 0.0044, 0.075],
        ]
    )

    # Scale factor to achieve realistic ~400 DU total column ozone
    _O3_SCALE = 0.560

    def __init__(self):
        self._z = self._DATA[:, 0] * 1000
        self._T = self._DATA[:, 1]
        self._P = self._DATA[:, 2] * 100
        self._h2o = self._DATA[:, 3] * 1e-6
        self._o3 = self._DATA[:, 4] * 1e-6 * self._O3_SCALE

    def temperature(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._T)

    def pressure(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._P)))

    def h2o_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(np.asarray(altitude), self._z, np.log(self._h2o)))

    def o3_vmr(self, altitude: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(altitude), self._z, self._o3)
