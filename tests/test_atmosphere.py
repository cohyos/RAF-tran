"""Tests for atmospheric profile models."""

import numpy as np
import pytest

from raf_tran.atmosphere import (
    StandardAtmosphere,
    MidlatitudeSummer,
    MidlatitudeWinter,
    TropicalAtmosphere,
    SubarcticSummer,
    SubarcticWinter,
)


class TestStandardAtmosphere:
    """Tests for US Standard Atmosphere 1976."""

    @pytest.fixture
    def atmosphere(self):
        return StandardAtmosphere()

    def test_surface_temperature(self, atmosphere):
        """Test temperature at sea level."""
        T = atmosphere.temperature(np.array([0.0]))
        assert np.isclose(T[0], 288.15, rtol=1e-3)

    def test_surface_pressure(self, atmosphere):
        """Test pressure at sea level."""
        P = atmosphere.pressure(np.array([0.0]))
        assert np.isclose(P[0], 101325, rtol=1e-3)

    def test_tropopause_temperature(self, atmosphere):
        """Test temperature at tropopause (~11 km)."""
        T = atmosphere.temperature(np.array([11000.0]))
        assert np.isclose(T[0], 216.65, rtol=1e-3)

    def test_temperature_decreases_in_troposphere(self, atmosphere):
        """Test that temperature decreases in troposphere."""
        altitudes = np.array([0, 5000, 10000])
        T = atmosphere.temperature(altitudes)
        assert T[0] > T[1] > T[2]

    def test_pressure_decreases_with_altitude(self, atmosphere):
        """Test that pressure decreases with altitude."""
        altitudes = np.array([0, 10000, 20000, 30000])
        P = atmosphere.pressure(altitudes)
        assert all(P[i] > P[i + 1] for i in range(len(P) - 1))

    def test_density_positive(self, atmosphere):
        """Test that density is always positive."""
        altitudes = np.linspace(0, 50000, 100)
        rho = atmosphere.density(altitudes)
        assert all(rho > 0)

    def test_h2o_decreases_with_altitude(self, atmosphere):
        """Test that water vapor decreases with altitude."""
        altitudes = np.array([0, 5000, 10000])
        h2o = atmosphere.h2o_vmr(altitudes)
        assert h2o[0] > h2o[1] > h2o[2]

    def test_o3_profile_has_peak(self, atmosphere):
        """Test that ozone has a peak in the stratosphere."""
        altitudes = np.linspace(0, 50000, 100)
        o3 = atmosphere.o3_vmr(altitudes)
        peak_idx = np.argmax(o3)
        peak_altitude = altitudes[peak_idx]
        # Ozone peak should be around 20-25 km
        assert 15000 < peak_altitude < 30000

    def test_create_layers(self, atmosphere):
        """Test layer creation."""
        z_levels = np.array([0, 1000, 2000, 5000, 10000])
        layers = atmosphere.create_layers(z_levels)

        assert len(layers) == 4
        assert layers[0].z_bottom == 0
        assert layers[0].z_top == 1000
        assert layers[-1].z_top == 10000


class TestModelAtmospheres:
    """Tests for MODTRAN model atmospheres."""

    @pytest.mark.parametrize(
        "AtmosphereClass",
        [
            MidlatitudeSummer,
            MidlatitudeWinter,
            TropicalAtmosphere,
            SubarcticSummer,
            SubarcticWinter,
        ],
    )
    def test_surface_values_reasonable(self, AtmosphereClass):
        """Test that surface values are physically reasonable."""
        atm = AtmosphereClass()
        z0 = np.array([0.0])

        T = atm.temperature(z0)[0]
        P = atm.pressure(z0)[0]

        # Temperature should be between 250-310 K at surface
        assert 250 < T < 310

        # Pressure should be around 101325 Pa at surface
        assert 90000 < P < 105000

    @pytest.mark.parametrize(
        "AtmosphereClass",
        [
            MidlatitudeSummer,
            MidlatitudeWinter,
            TropicalAtmosphere,
            SubarcticSummer,
            SubarcticWinter,
        ],
    )
    def test_pressure_monotonic(self, AtmosphereClass):
        """Test that pressure decreases monotonically with altitude."""
        atm = AtmosphereClass()
        altitudes = np.linspace(0, 50000, 50)
        P = atm.pressure(altitudes)

        assert all(P[i] >= P[i + 1] for i in range(len(P) - 1))

    def test_tropical_warmer_than_subarctic(self):
        """Test that tropical atmosphere is warmer than subarctic."""
        tropical = TropicalAtmosphere()
        subarctic = SubarcticWinter()

        z = np.array([0.0, 5000.0, 10000.0])

        T_tropical = tropical.temperature(z)
        T_subarctic = subarctic.temperature(z)

        assert all(T_tropical > T_subarctic)

    def test_summer_warmer_than_winter(self):
        """Test that summer atmosphere is warmer than winter."""
        summer = MidlatitudeSummer()
        winter = MidlatitudeWinter()

        z = np.array([0.0, 5000.0])

        T_summer = summer.temperature(z)
        T_winter = winter.temperature(z)

        assert all(T_summer > T_winter)


class TestUSStandardAtmosphere1976Benchmark:
    """
    Rigorous benchmark tests for US Standard Atmosphere 1976.

    Reference values from:
    US Standard Atmosphere, 1976. NOAA-S/T 76-1562.
    National Oceanic and Atmospheric Administration.

    These tests validate the implementation against published values
    at specific altitudes from the official tables.
    """

    @pytest.fixture
    def atmosphere(self):
        return StandardAtmosphere()

    # Reference table values from US Standard Atmosphere 1976
    # Format: (altitude_m, temperature_K, pressure_Pa, density_kg_m3)
    REFERENCE_VALUES = [
        (0, 288.150, 101325.0, 1.2250),
        (1000, 281.651, 89874.6, 1.1117),
        (2000, 275.154, 79495.2, 1.0066),
        (3000, 268.659, 70108.5, 0.9093),
        (4000, 262.166, 61640.2, 0.8194),
        (5000, 255.676, 54019.9, 0.7364),
        (6000, 249.187, 47181.0, 0.6601),
        (7000, 242.700, 41060.7, 0.5900),
        (8000, 236.215, 35599.8, 0.5258),
        (9000, 229.733, 30742.5, 0.4671),
        (10000, 223.252, 26436.3, 0.4135),
        (11000, 216.774, 22632.1, 0.3648),
        # Tropopause starts - isothermal layer
        (12000, 216.650, 19330.4, 0.3119),
        (15000, 216.650, 12044.6, 0.1948),
        (20000, 216.650, 5474.89, 0.08891),
        # Stratosphere warming begins
        (25000, 221.552, 2511.02, 0.03996),
        (30000, 226.509, 1171.87, 0.01841),
        (35000, 236.513, 558.924, 0.008463),
        (40000, 250.350, 277.522, 0.003996),
        (45000, 264.164, 143.135, 0.001966),
        (50000, 270.650, 75.9448, 0.001027),
    ]

    @pytest.mark.parametrize("altitude,T_ref,P_ref,rho_ref", REFERENCE_VALUES)
    def test_temperature_benchmark(self, atmosphere, altitude, T_ref, P_ref, rho_ref):
        """Test temperature against US Standard Atmosphere 1976 tables."""
        T = atmosphere.temperature(np.array([float(altitude)]))[0]
        # Allow 0.5% tolerance for implementation differences
        assert np.isclose(T, T_ref, rtol=0.005), \
            f"Temperature at {altitude}m: got {T:.3f}K, expected {T_ref:.3f}K"

    @pytest.mark.parametrize("altitude,T_ref,P_ref,rho_ref", REFERENCE_VALUES)
    def test_pressure_benchmark(self, atmosphere, altitude, T_ref, P_ref, rho_ref):
        """Test pressure against US Standard Atmosphere 1976 tables."""
        P = atmosphere.pressure(np.array([float(altitude)]))[0]
        # Allow 1% tolerance for numerical integration differences
        assert np.isclose(P, P_ref, rtol=0.01), \
            f"Pressure at {altitude}m: got {P:.3f}Pa, expected {P_ref:.3f}Pa"

    @pytest.mark.parametrize("altitude,T_ref,P_ref,rho_ref", REFERENCE_VALUES)
    def test_density_benchmark(self, atmosphere, altitude, T_ref, P_ref, rho_ref):
        """Test density against US Standard Atmosphere 1976 tables."""
        rho = atmosphere.density(np.array([float(altitude)]))[0]
        # Allow 5% tolerance (error accumulates from T and P, larger at high altitude)
        assert np.isclose(rho, rho_ref, rtol=0.05), \
            f"Density at {altitude}m: got {rho:.6f}kg/m3, expected {rho_ref:.6f}kg/m3"

    def test_ideal_gas_law_consistency(self, atmosphere):
        """Test that rho = P/(R*T) is satisfied (ideal gas law)."""
        R_air = 287.05  # J/(kg*K) - specific gas constant for dry air
        altitudes = np.linspace(0, 50000, 51)

        T = atmosphere.temperature(altitudes)
        P = atmosphere.pressure(altitudes)
        rho = atmosphere.density(altitudes)

        # Compute density from ideal gas law
        rho_computed = P / (R_air * T)

        # Should match within 0.1%
        assert np.allclose(rho, rho_computed, rtol=0.001), \
            "Density does not satisfy ideal gas law P = rho*R*T"

    def test_lapse_rate_troposphere(self, atmosphere):
        """Test tropospheric lapse rate is approximately -6.5 K/km."""
        z = np.array([0, 1000, 5000, 10000], dtype=float)
        T = atmosphere.temperature(z)

        # Calculate average lapse rate in troposphere
        dT_dz = (T[-1] - T[0]) / (z[-1] - z[0])  # K/m
        lapse_rate_km = dT_dz * 1000  # K/km

        # Standard lapse rate is -6.5 K/km
        assert np.isclose(lapse_rate_km, -6.5, atol=0.2), \
            f"Tropospheric lapse rate: got {lapse_rate_km:.2f} K/km, expected -6.5 K/km"

    def test_tropopause_isothermal(self, atmosphere):
        """Test that tropopause region is isothermal (11-20 km)."""
        z = np.linspace(11000, 20000, 20)
        T = atmosphere.temperature(z)

        # Temperature should be nearly constant (~216.65 K)
        T_mean = T.mean()
        T_std = T.std()

        assert T_std < 0.5, \
            f"Tropopause not isothermal: std={T_std:.3f}K (expected <0.5K)"
        assert np.isclose(T_mean, 216.65, atol=1.0), \
            f"Tropopause temperature: got {T_mean:.2f}K, expected ~216.65K"

    def test_pressure_scale_height(self, atmosphere):
        """Test that pressure decreases exponentially with reasonable scale height."""
        # In a non-isothermal atmosphere, scale height varies with altitude
        # At sea level, H ≈ RT/Mg ≈ 8.4 km for T=288K
        # But due to lapse rate, effective scale height is different

        z = np.array([0, 5000], dtype=float)
        P = atmosphere.pressure(z)

        # Calculate effective scale height from pressure ratio
        # P(z) = P(0) * exp(-z/H) => H = -z / ln(P(z)/P(0))
        H_eff = -5000 / np.log(P[1] / P[0])

        # Effective scale height should be between 7 and 9 km
        assert 7000 < H_eff < 9000, \
            f"Effective scale height: {H_eff/1000:.2f} km (expected 7-9 km)"

    def test_number_density_at_sea_level(self, atmosphere):
        """Test number density at sea level (Loschmidt constant)."""
        n = atmosphere.number_density(np.array([0.0]))[0]

        # Loschmidt constant: 2.687e25 molecules/m^3 at STP (273.15 K, 101325 Pa)
        # At 288.15 K, n should be slightly lower: n = P/(k*T)
        k_B = 1.380649e-23  # Boltzmann constant, J/K
        n_expected = 101325 / (k_B * 288.15)  # ~2.547e25 m^-3

        assert np.isclose(n, n_expected, rtol=0.01), \
            f"Number density at SL: got {n:.3e}m^-3, expected {n_expected:.3e}m^-3"

    def test_column_density(self, atmosphere):
        """Test total column density (molecules/m^2)."""
        # Integrate from 0 to 100 km
        z = np.linspace(0, 100000, 1001)
        n = atmosphere.number_density(z)
        dz = z[1] - z[0]

        # Simple trapezoidal integration (use trapezoid for NumPy 2.0+)
        try:
            column = np.trapezoid(n, z)
        except AttributeError:
            column = np.trapz(n, z)

        # Expected column: ~2.1e29 molecules/m^2
        # This is related to the surface pressure: N = P/(m_air * g)
        m_air = 4.81e-26  # kg, average molecular mass of air
        g = 9.80665  # m/s^2
        column_expected = 101325 / (m_air * g)  # ~2.15e29

        assert np.isclose(column, column_expected, rtol=0.05), \
            f"Column density: got {column:.3e}m^-2, expected ~{column_expected:.3e}m^-2"
