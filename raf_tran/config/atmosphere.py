"""
Standard atmosphere profiles and custom profile loading.

Implements FR-01 (standard models) and FR-02 (custom radiosonde profiles).

Standard models included:
- US Standard 1976
- Tropical
- Mid-Latitude Summer/Winter
- Sub-Arctic Summer/Winter

Data based on AFGL atmospheric models (Anderson et al., 1986).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import csv


@dataclass
class AtmosphereLayer:
    """Single atmospheric layer properties.

    Attributes:
        altitude_km: Layer center altitude [km]
        pressure_pa: Pressure [Pa]
        temperature_k: Temperature [K]
        density_kg_m3: Air density [kg/m³]
        h2o_vmr: Water vapor volume mixing ratio [ppmv]
        co2_vmr: CO2 volume mixing ratio [ppmv]
        o3_vmr: Ozone volume mixing ratio [ppmv]
        n2o_vmr: N2O volume mixing ratio [ppmv]
        co_vmr: CO volume mixing ratio [ppmv]
        ch4_vmr: Methane volume mixing ratio [ppmv]
    """
    altitude_km: float
    pressure_pa: float
    temperature_k: float
    density_kg_m3: float = 0.0
    h2o_vmr: float = 0.0
    co2_vmr: float = 420.0  # 2024 global average
    o3_vmr: float = 0.0
    n2o_vmr: float = 0.332
    co_vmr: float = 0.1
    ch4_vmr: float = 1.9


@dataclass
class AtmosphereProfile:
    """Complete atmospheric profile.

    Attributes:
        name: Profile identifier
        layers: List of atmospheric layers from ground to TOA
        gas_concentrations: Override gas concentrations (ppmv)
    """
    name: str
    layers: List[AtmosphereLayer]
    gas_concentrations: Dict[str, float] = field(default_factory=dict)

    @property
    def altitudes(self) -> np.ndarray:
        """Get altitude array [km]."""
        return np.array([layer.altitude_km for layer in self.layers])

    @property
    def pressures(self) -> np.ndarray:
        """Get pressure array [Pa]."""
        return np.array([layer.pressure_pa for layer in self.layers])

    @property
    def temperatures(self) -> np.ndarray:
        """Get temperature array [K]."""
        return np.array([layer.temperature_k for layer in self.layers])

    @property
    def num_layers(self) -> int:
        """Number of layers."""
        return len(self.layers)

    def get_gas_profile(self, gas: str) -> np.ndarray:
        """Get mixing ratio profile for a specific gas.

        Args:
            gas: Gas name (e.g., 'H2O', 'CO2')

        Returns:
            Volume mixing ratio profile [ppmv]
        """
        attr_map = {
            "H2O": "h2o_vmr",
            "CO2": "co2_vmr",
            "O3": "o3_vmr",
            "N2O": "n2o_vmr",
            "CO": "co_vmr",
            "CH4": "ch4_vmr",
        }

        if gas in self.gas_concentrations:
            # Return uniform override concentration
            return np.full(self.num_layers, self.gas_concentrations[gas])

        if gas in attr_map:
            return np.array([getattr(layer, attr_map[gas]) for layer in self.layers])

        raise ValueError(f"Unknown gas: {gas}")

    def interpolate_to_altitudes(self, altitudes_km: np.ndarray) -> "AtmosphereProfile":
        """Interpolate profile to new altitude grid.

        Args:
            altitudes_km: New altitude grid [km]

        Returns:
            Interpolated atmosphere profile
        """
        orig_alt = self.altitudes
        orig_press = self.pressures
        orig_temp = self.temperatures

        # Log-interpolate pressure
        log_press = np.interp(altitudes_km, orig_alt, np.log(orig_press))
        new_press = np.exp(log_press)

        # Linear interpolate temperature
        new_temp = np.interp(altitudes_km, orig_alt, orig_temp)

        # Create new layers
        new_layers = []
        for i, alt in enumerate(altitudes_km):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=new_press[i],
                temperature_k=new_temp[i],
            )

            # Interpolate gas concentrations
            for gas in ["H2O", "CO2", "O3", "N2O", "CO", "CH4"]:
                orig_gas = self.get_gas_profile(gas)
                new_val = np.interp(alt, orig_alt, orig_gas)
                attr_name = f"{gas.lower()}_vmr"
                if hasattr(layer, attr_name):
                    setattr(layer, attr_name, new_val)

            # Calculate density from ideal gas law
            R = 287.05  # J/(kg·K) for dry air
            layer.density_kg_m3 = new_press[i] / (R * new_temp[i])

            new_layers.append(layer)

        return AtmosphereProfile(
            name=f"{self.name}_interpolated",
            layers=new_layers,
            gas_concentrations=self.gas_concentrations.copy(),
        )

    def apply_gas_overrides(self, overrides: Dict[str, float]) -> None:
        """Apply custom gas concentration overrides (FR-03).

        Args:
            overrides: Dictionary of gas names to concentrations [ppmv]
        """
        self.gas_concentrations.update(overrides)


class StandardAtmospheres:
    """Factory for standard atmosphere profiles (FR-01).

    Standard atmosphere data based on AFGL (1986) profiles.
    Altitudes from 0 to 100 km in standard layers.
    """

    # Standard altitude grid [km]
    STANDARD_ALTITUDES = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45,
        50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100
    ])

    @classmethod
    def us_standard_1976(cls) -> AtmosphereProfile:
        """US Standard Atmosphere 1976.

        Reference: U.S. Standard Atmosphere, 1976, NOAA-S/T76-1562.
        """
        # Temperature profile [K]
        temperatures = np.array([
            288.15, 281.65, 275.15, 268.65, 262.15, 255.65, 249.15, 242.65,
            236.15, 229.65, 223.15, 216.65, 216.65, 216.65, 216.65, 216.65,
            216.65, 216.65, 216.65, 216.65, 216.65, 217.65, 218.65, 219.65,
            220.65, 221.65, 226.65, 237.05, 251.05, 265.05, 270.65, 260.65,
            247.02, 233.29, 219.59, 208.40, 198.64, 188.89, 186.87, 188.42,
            195.08
        ])

        # Pressure profile [Pa]
        pressures = np.array([
            101325, 89876, 79501, 70121, 61660, 54048, 47217, 41105,
            35651, 30800, 26499, 22699, 19399, 16579, 14170, 12111,
            10352, 8849.5, 7565.0, 6467.0, 5529.0, 4729.0, 4047.0, 3467.0,
            2972.0, 2549.0, 1197.0, 574.6, 287.1, 149.1, 79.78, 42.53,
            21.96, 10.93, 5.221, 2.388, 1.052, 0.4457, 0.1836, 0.0760,
            0.0320
        ])

        # Water vapor profile [ppmv] - decreases with altitude
        h2o = np.array([
            7750, 6070, 4630, 3330, 2220, 1520, 1030, 669,
            434, 271, 186, 118, 66, 37.5, 21.6, 12.4,
            7.25, 4.33, 2.62, 1.60, 1.00, 0.75, 0.56, 0.42,
            0.32, 0.24, 0.048, 0.0096, 0.0048, 0.0048, 0.0048, 0.0048,
            0.0096, 0.024, 0.048, 0.096, 0.19, 0.38, 0.48, 0.48, 0.48
        ])

        # Ozone profile [ppmv] - peak around 25-30 km
        o3 = np.array([
            0.027, 0.029, 0.032, 0.036, 0.043, 0.054, 0.067, 0.084,
            0.106, 0.133, 0.167, 0.219, 0.304, 0.420, 0.545, 0.730,
            1.01, 1.38, 1.84, 2.43, 3.14, 3.88, 4.62, 5.30,
            5.86, 6.22, 7.76, 8.80, 8.50, 6.00, 4.00, 2.50,
            1.50, 0.92, 0.50, 0.27, 0.14, 0.074, 0.038, 0.020, 0.010
        ])

        layers = []
        for i, alt in enumerate(cls.STANDARD_ALTITUDES):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=pressures[i],
                temperature_k=temperatures[i],
                h2o_vmr=h2o[i],
                o3_vmr=o3[i],
            )
            # Calculate density
            R = 287.05
            layer.density_kg_m3 = pressures[i] / (R * temperatures[i])
            layers.append(layer)

        return AtmosphereProfile(name="US_STANDARD_1976", layers=layers)

    @classmethod
    def tropical(cls) -> AtmosphereProfile:
        """Tropical atmosphere profile.

        Reference: AFGL-TR-86-0110 (Anderson et al., 1986)
        """
        temperatures = np.array([
            300.0, 294.0, 288.0, 284.0, 277.0, 270.0, 264.0, 257.0,
            250.0, 244.0, 237.0, 230.0, 224.0, 217.0, 210.0, 204.0,
            197.0, 195.0, 199.0, 203.0, 207.0, 211.0, 215.0, 217.0,
            219.0, 221.0, 232.0, 243.0, 254.0, 265.0, 270.0, 264.0,
            253.0, 236.0, 219.0, 210.0, 199.0, 190.0, 188.0, 187.0,
            187.0
        ])

        pressures = np.array([
            101300, 90400, 80500, 71500, 63300, 55900, 49200, 43200,
            37800, 32900, 28600, 24700, 21300, 18200, 15600, 13200,
            11100, 9370, 7890, 6660, 5650, 4800, 4090, 3500,
            3000, 2570, 1220, 600, 305, 159, 85.2, 45.6,
            23.7, 11.9, 5.75, 2.69, 1.22, 0.542, 0.238, 0.105, 0.047
        ])

        h2o = np.array([
            19000, 13000, 9300, 4700, 2200, 1500, 850, 540,
            380, 210, 120, 46, 18, 8.2, 3.7, 1.8,
            0.85, 0.40, 0.19, 0.095, 0.045, 0.030, 0.020, 0.013,
            0.0087, 0.0058, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029,
            0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029, 0.0029
        ])

        o3 = np.array([
            0.028, 0.030, 0.034, 0.040, 0.044, 0.049, 0.057, 0.069,
            0.090, 0.110, 0.130, 0.180, 0.250, 0.330, 0.410, 0.560,
            0.810, 1.20, 1.80, 2.60, 3.60, 4.80, 6.20, 7.40,
            8.30, 8.80, 9.40, 9.00, 7.60, 5.30, 3.30, 2.00,
            1.20, 0.70, 0.40, 0.22, 0.12, 0.065, 0.035, 0.020, 0.010
        ])

        layers = []
        for i, alt in enumerate(cls.STANDARD_ALTITUDES):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=pressures[i],
                temperature_k=temperatures[i],
                h2o_vmr=h2o[i],
                o3_vmr=o3[i],
            )
            R = 287.05
            layer.density_kg_m3 = pressures[i] / (R * temperatures[i])
            layers.append(layer)

        return AtmosphereProfile(name="TROPICAL", layers=layers)

    @classmethod
    def mid_latitude_summer(cls) -> AtmosphereProfile:
        """Mid-latitude summer atmosphere profile.

        Reference: AFGL-TR-86-0110 (Anderson et al., 1986)
        """
        temperatures = np.array([
            294.0, 290.0, 285.0, 279.0, 273.0, 267.0, 261.0, 255.0,
            248.0, 242.0, 235.0, 229.0, 222.0, 216.0, 216.0, 216.0,
            216.0, 216.0, 216.0, 217.0, 218.0, 219.0, 220.0, 222.0,
            223.0, 224.0, 234.0, 245.0, 258.0, 270.0, 276.0, 268.0,
            254.0, 240.0, 226.0, 212.0, 200.0, 189.0, 186.0, 186.0, 187.0
        ])

        pressures = np.array([
            101300, 90200, 80200, 71000, 62800, 55400, 48700, 42700,
            37300, 32400, 28100, 24300, 20800, 17800, 15300, 13000,
            11100, 9500, 8120, 6950, 5950, 5100, 4370, 3760,
            3220, 2770, 1320, 651, 332, 176, 95.1, 51.4,
            27.1, 13.9, 6.91, 3.32, 1.55, 0.714, 0.328, 0.152, 0.070
        ])

        h2o = np.array([
            14000, 9300, 5900, 3300, 1900, 1000, 610, 370,
            210, 120, 64, 22, 6.0, 1.8, 1.0, 0.74,
            0.64, 0.56, 0.50, 0.45, 0.41, 0.37, 0.34, 0.31,
            0.29, 0.26, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,
            0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20
        ])

        o3 = np.array([
            0.060, 0.058, 0.056, 0.054, 0.052, 0.050, 0.050, 0.052,
            0.058, 0.070, 0.090, 0.130, 0.200, 0.300, 0.450, 0.670,
            0.990, 1.50, 2.20, 3.00, 4.00, 5.00, 6.00, 6.80,
            7.40, 7.80, 8.60, 8.40, 7.20, 5.20, 3.40, 2.10,
            1.30, 0.80, 0.47, 0.26, 0.14, 0.075, 0.040, 0.021, 0.011
        ])

        layers = []
        for i, alt in enumerate(cls.STANDARD_ALTITUDES):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=pressures[i],
                temperature_k=temperatures[i],
                h2o_vmr=h2o[i],
                o3_vmr=o3[i],
            )
            R = 287.05
            layer.density_kg_m3 = pressures[i] / (R * temperatures[i])
            layers.append(layer)

        return AtmosphereProfile(name="MID_LATITUDE_SUMMER", layers=layers)

    @classmethod
    def mid_latitude_winter(cls) -> AtmosphereProfile:
        """Mid-latitude winter atmosphere profile.

        Reference: AFGL-TR-86-0110 (Anderson et al., 1986)
        """
        temperatures = np.array([
            272.2, 268.7, 265.2, 261.7, 255.7, 249.7, 243.7, 237.7,
            231.7, 225.7, 219.7, 219.2, 218.7, 218.2, 217.7, 217.2,
            216.7, 216.2, 215.7, 215.2, 215.2, 215.2, 215.2, 215.2,
            215.2, 215.2, 217.4, 227.8, 243.2, 258.5, 265.7, 260.6,
            250.8, 240.5, 229.7, 219.0, 208.3, 198.6, 188.9, 186.9, 188.4
        ])

        pressures = np.array([
            101800, 89700, 78900, 69300, 60800, 53200, 46400, 40300,
            34900, 30100, 25900, 22300, 19200, 16500, 14200, 12200,
            10400, 8960, 7690, 6600, 5660, 4860, 4170, 3580,
            3070, 2640, 1290, 648, 333, 177, 96.5, 52.7,
            28.0, 14.5, 7.26, 3.52, 1.65, 0.757, 0.344, 0.159, 0.074
        ])

        h2o = np.array([
            3500, 2500, 1800, 1200, 660, 380, 220, 150,
            94, 46, 18, 8.2, 3.7, 1.8, 0.85, 0.40,
            0.19, 0.095, 0.045, 0.030, 0.020, 0.013, 0.0087, 0.0058,
            0.0038, 0.0026, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010,
            0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010
        ])

        o3 = np.array([
            0.028, 0.029, 0.030, 0.032, 0.034, 0.037, 0.042, 0.050,
            0.063, 0.082, 0.110, 0.150, 0.200, 0.280, 0.400, 0.580,
            0.850, 1.30, 1.90, 2.80, 3.90, 5.00, 6.10, 7.00,
            7.60, 7.90, 8.30, 7.80, 6.40, 4.50, 2.80, 1.70,
            1.00, 0.60, 0.35, 0.20, 0.11, 0.060, 0.032, 0.017, 0.009
        ])

        layers = []
        for i, alt in enumerate(cls.STANDARD_ALTITUDES):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=pressures[i],
                temperature_k=temperatures[i],
                h2o_vmr=h2o[i],
                o3_vmr=o3[i],
            )
            R = 287.05
            layer.density_kg_m3 = pressures[i] / (R * temperatures[i])
            layers.append(layer)

        return AtmosphereProfile(name="MID_LATITUDE_WINTER", layers=layers)

    @classmethod
    def sub_arctic_summer(cls) -> AtmosphereProfile:
        """Sub-arctic summer atmosphere profile.

        Reference: AFGL-TR-86-0110 (Anderson et al., 1986)
        """
        temperatures = np.array([
            287.0, 282.0, 276.0, 271.0, 266.0, 260.0, 253.0, 246.0,
            239.0, 232.0, 225.0, 225.0, 225.0, 225.0, 225.0, 225.0,
            225.0, 225.0, 225.0, 225.0, 225.0, 225.0, 225.0, 226.0,
            228.0, 235.0, 247.0, 262.0, 274.0, 277.0, 276.0, 264.0,
            252.0, 240.0, 228.0, 216.0, 204.0, 193.0, 186.0, 186.0, 187.0
        ])

        pressures = np.array([
            101000, 89600, 79200, 69800, 61400, 53800, 46900, 40700,
            35200, 30300, 25900, 22100, 18900, 16200, 13800, 11800,
            10100, 8610, 7350, 6280, 5370, 4580, 3910, 3340,
            2860, 2430, 1180, 594, 311, 167, 91.2, 49.8,
            26.5, 13.7, 6.88, 3.35, 1.58, 0.731, 0.336, 0.157, 0.074
        ])

        h2o = np.array([
            9100, 6000, 4200, 2700, 1700, 1000, 540, 290,
            130, 54, 21, 11, 6.4, 3.4, 1.8, 0.98,
            0.52, 0.28, 0.15, 0.080, 0.043, 0.029, 0.019, 0.013,
            0.0084, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056,
            0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056
        ])

        o3 = np.array([
            0.035, 0.038, 0.042, 0.048, 0.055, 0.063, 0.074, 0.089,
            0.110, 0.130, 0.170, 0.220, 0.300, 0.430, 0.600, 0.830,
            1.20, 1.70, 2.40, 3.30, 4.30, 5.30, 6.30, 7.20,
            7.80, 8.10, 8.30, 7.50, 6.00, 4.20, 2.70, 1.70,
            1.00, 0.60, 0.35, 0.20, 0.11, 0.060, 0.032, 0.017, 0.009
        ])

        layers = []
        for i, alt in enumerate(cls.STANDARD_ALTITUDES):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=pressures[i],
                temperature_k=temperatures[i],
                h2o_vmr=h2o[i],
                o3_vmr=o3[i],
            )
            R = 287.05
            layer.density_kg_m3 = pressures[i] / (R * temperatures[i])
            layers.append(layer)

        return AtmosphereProfile(name="SUB_ARCTIC_SUMMER", layers=layers)

    @classmethod
    def sub_arctic_winter(cls) -> AtmosphereProfile:
        """Sub-arctic winter atmosphere profile.

        Reference: AFGL-TR-86-0110 (Anderson et al., 1986)
        """
        temperatures = np.array([
            257.1, 259.1, 256.0, 253.0, 247.0, 240.0, 234.0, 227.0,
            221.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0, 217.0,
            216.0, 216.0, 216.0, 217.0, 218.0, 219.0, 220.0, 222.0,
            224.0, 226.0, 236.0, 248.0, 260.0, 269.0, 270.0, 263.0,
            250.0, 238.0, 225.0, 213.0, 201.0, 191.0, 185.0, 185.0, 186.0
        ])

        pressures = np.array([
            101300, 88800, 77700, 68000, 59300, 51500, 44600, 38500,
            33100, 28600, 24600, 21200, 18200, 15700, 13500, 11600,
            9950, 8540, 7320, 6280, 5390, 4620, 3970, 3400,
            2920, 2510, 1240, 630, 328, 177, 97.5, 53.5,
            28.6, 14.9, 7.52, 3.67, 1.74, 0.808, 0.372, 0.174, 0.082
        ])

        h2o = np.array([
            1200, 1200, 940, 680, 410, 200, 98, 54,
            29, 13, 4.2, 1.5, 0.60, 0.24, 0.097, 0.039,
            0.016, 0.0064, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026,
            0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026,
            0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026, 0.0026
        ])

        o3 = np.array([
            0.040, 0.044, 0.050, 0.057, 0.066, 0.077, 0.091, 0.110,
            0.140, 0.180, 0.250, 0.350, 0.500, 0.700, 1.00, 1.50,
            2.20, 3.10, 4.20, 5.40, 6.60, 7.60, 8.30, 8.80,
            9.00, 9.00, 8.50, 7.50, 5.90, 4.10, 2.60, 1.60,
            0.96, 0.57, 0.33, 0.19, 0.11, 0.060, 0.032, 0.017, 0.009
        ])

        layers = []
        for i, alt in enumerate(cls.STANDARD_ALTITUDES):
            layer = AtmosphereLayer(
                altitude_km=alt,
                pressure_pa=pressures[i],
                temperature_k=temperatures[i],
                h2o_vmr=h2o[i],
                o3_vmr=o3[i],
            )
            R = 287.05
            layer.density_kg_m3 = pressures[i] / (R * temperatures[i])
            layers.append(layer)

        return AtmosphereProfile(name="SUB_ARCTIC_WINTER", layers=layers)

    @classmethod
    def get_profile(cls, model_name: str) -> AtmosphereProfile:
        """Get atmosphere profile by name.

        Args:
            model_name: Model identifier (case-insensitive)

        Returns:
            AtmosphereProfile for the specified model

        Raises:
            ValueError: If model name is not recognized
        """
        model_map = {
            "US_STANDARD_1976": cls.us_standard_1976,
            "TROPICAL": cls.tropical,
            "MID_LATITUDE_SUMMER": cls.mid_latitude_summer,
            "MID_LATITUDE_WINTER": cls.mid_latitude_winter,
            "SUB_ARCTIC_SUMMER": cls.sub_arctic_summer,
            "SUB_ARCTIC_WINTER": cls.sub_arctic_winter,
        }

        model_upper = model_name.upper().replace(" ", "_").replace("-", "_")
        if model_upper not in model_map:
            raise ValueError(
                f"Unknown atmosphere model: {model_name}. "
                f"Available models: {list(model_map.keys())}"
            )

        return model_map[model_upper]()

    @classmethod
    def load_radiosonde(cls, csv_path: str) -> AtmosphereProfile:
        """Load custom radiosonde profile from CSV (FR-02).

        Expected CSV format:
            altitude_km,pressure_pa,temperature_k,h2o_ppmv
            0.0,101325,288.15,7750
            1.0,89876,281.65,6070
            ...

        Args:
            csv_path: Path to CSV file

        Returns:
            AtmosphereProfile from radiosonde data
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Radiosonde file not found: {csv_path}")

        layers = []
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer = AtmosphereLayer(
                    altitude_km=float(row.get("altitude_km", row.get("alt", 0))),
                    pressure_pa=float(row.get("pressure_pa", row.get("pressure", 101325))),
                    temperature_k=float(row.get("temperature_k", row.get("temp", 288.15))),
                    h2o_vmr=float(row.get("h2o_ppmv", row.get("h2o", 0))),
                )
                # Calculate density
                R = 287.05
                layer.density_kg_m3 = layer.pressure_pa / (R * layer.temperature_k)
                layers.append(layer)

        # Sort by altitude
        layers.sort(key=lambda x: x.altitude_km)

        return AtmosphereProfile(
            name=f"RADIOSONDE_{path.stem}",
            layers=layers,
        )
