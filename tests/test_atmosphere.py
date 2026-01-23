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
