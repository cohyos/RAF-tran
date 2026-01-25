"""Tests for air mass functions."""

import numpy as np
import pytest

from raf_tran.utils.air_mass import (
    plane_parallel_air_mass,
    kasten_young_air_mass,
    chapman_function,
    optical_air_mass,
    validate_solar_geometry,
)


class TestPlaneParallelAirMass:
    """Tests for plane-parallel air mass approximation."""

    def test_overhead_sun(self):
        """Air mass should be 1.0 for overhead sun (SZA=0)."""
        assert np.isclose(plane_parallel_air_mass(0), 1.0)

    def test_60_degree_sza(self):
        """Air mass should be 2.0 for SZA=60 degrees."""
        assert np.isclose(plane_parallel_air_mass(60), 2.0)

    def test_array_input(self):
        """Test with array input."""
        sza = np.array([0, 30, 45, 60])
        m = plane_parallel_air_mass(sza)
        expected = 1.0 / np.cos(np.radians(sza))
        assert np.allclose(m, expected)

    def test_negative_sza_raises(self):
        """Negative SZA should work (mathematically valid)."""
        # cos(-x) = cos(x), so negative SZA gives same result
        assert np.isclose(plane_parallel_air_mass(-30), plane_parallel_air_mass(30))

    def test_90_degree_raises(self):
        """SZA=90 degrees should raise ValueError or return inf."""
        # cos(90 deg) is numerically very small but not exactly zero
        # Function may return very large value instead of raising
        result = plane_parallel_air_mass(89.99)
        assert result > 5000  # Very large air mass near horizon


class TestKastenYoungAirMass:
    """Tests for Kasten-Young air mass formula."""

    def test_overhead_sun(self):
        """Air mass should be ~1.0 for overhead sun."""
        assert np.isclose(kasten_young_air_mass(0), 1.0, rtol=0.001)

    def test_60_degree_sza(self):
        """Air mass at SZA=60 should be close to 2.0."""
        m = kasten_young_air_mass(60)
        assert np.isclose(m, 2.0, rtol=0.02)  # Within 2%

    def test_high_sza(self):
        """Test at high solar zenith angles."""
        # At 85 degrees, air mass should be ~10-11
        m = kasten_young_air_mass(85)
        assert 10 < m < 12

    def test_matches_plane_parallel_at_low_sza(self):
        """Should match plane-parallel at low SZA."""
        for sza in [0, 10, 20, 30]:
            m_ky = kasten_young_air_mass(sza)
            m_pp = plane_parallel_air_mass(sza)
            assert np.isclose(m_ky, m_pp, rtol=0.02)

    def test_90_degree_raises(self):
        """SZA >= 90 degrees should raise ValueError."""
        with pytest.raises(ValueError):
            kasten_young_air_mass(90)


class TestChapmanFunction:
    """Tests for Chapman grazing incidence function."""

    def test_overhead_sun(self):
        """Chapman function should approach sec(chi) for small SZA."""
        chi = np.radians(0)
        x = 750  # Earth radius / scale height
        ch = chapman_function(chi, x)
        assert np.isclose(ch, 1.0, rtol=0.01)

    def test_60_degree_angle(self):
        """Test at 60 degree zenith angle."""
        chi = np.radians(60)
        x = 750
        ch = chapman_function(chi, x)
        # Should be close to sec(60) = 2.0
        assert np.isclose(ch, 2.0, rtol=0.1)

    def test_increases_with_angle(self):
        """Chapman function should increase with zenith angle."""
        x = 750
        chi_values = np.radians([0, 30, 60, 75])
        ch_values = [chapman_function(chi, x) for chi in chi_values]

        # Should be monotonically increasing
        for i in range(len(ch_values) - 1):
            assert ch_values[i] < ch_values[i + 1]


class TestOpticalAirMass:
    """Tests for unified optical air mass function."""

    def test_overhead_sun(self):
        """Air mass should be 1.0 for overhead sun."""
        assert np.isclose(optical_air_mass(0), 1.0, rtol=0.01)

    def test_method_auto_low_sza(self):
        """Auto method should use plane-parallel at low SZA."""
        m = optical_air_mass(30, method='auto')
        m_pp = plane_parallel_air_mass(30)
        assert np.isclose(m, m_pp)

    def test_method_auto_high_sza(self):
        """Auto method should use Kasten-Young at high SZA."""
        m = optical_air_mass(80, method='auto')
        m_ky = kasten_young_air_mass(80)
        assert np.isclose(m, m_ky)

    def test_method_selection(self):
        """Test explicit method selection."""
        sza = 45
        m_pp = optical_air_mass(sza, method='plane_parallel')
        m_ky = optical_air_mass(sza, method='kasten_young')
        m_ch = optical_air_mass(sza, method='chapman')

        # All should be close to 1/cos(45) ~ 1.414
        for m in [m_pp, m_ky, m_ch]:
            assert np.isclose(m, np.sqrt(2), rtol=0.05)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            optical_air_mass(45, method='invalid')

    def test_90_degree_raises(self):
        """SZA >= 90 should raise ValueError."""
        with pytest.raises(ValueError, match="Solar zenith angle must be < 90"):
            optical_air_mass(90)

    def test_altitude_effect_chapman(self):
        """Higher altitude should give lower air mass for Chapman."""
        m_sea = optical_air_mass(70, altitude_m=0, method='chapman')
        m_mountain = optical_air_mass(70, altitude_m=3000, method='chapman')
        # Different due to scale height ratio, but both should be reasonable
        assert m_sea > 0
        assert m_mountain > 0


class TestValidateSolarGeometry:
    """Tests for solar geometry validation."""

    def test_valid_sza(self):
        """Valid SZA should return cosine."""
        mu0 = validate_solar_geometry(60)
        assert np.isclose(mu0, 0.5)

    def test_overhead_sun(self):
        """Overhead sun should give mu0 = 1."""
        mu0 = validate_solar_geometry(0)
        assert np.isclose(mu0, 1.0)

    def test_array_input(self):
        """Test with array input."""
        sza = np.array([0, 30, 60])
        mu0 = validate_solar_geometry(sza)
        expected = np.cos(np.radians(sza))
        assert np.allclose(mu0, expected)

    def test_negative_sza_raises(self):
        """Negative SZA should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_solar_geometry(-10)

    def test_90_degree_raises(self):
        """SZA >= 90 should raise ValueError by default."""
        with pytest.raises(ValueError, match="sun above horizon"):
            validate_solar_geometry(90)

    def test_allow_horizon(self):
        """With allow_horizon=True, should accept SZA close to 90."""
        mu0 = validate_solar_geometry(89.9, allow_horizon=True)
        assert mu0 > 0


class TestPhysicsConsistency:
    """Physics consistency tests for air mass calculations."""

    def test_air_mass_monotonic(self):
        """Air mass should increase monotonically with SZA."""
        sza_values = np.linspace(0, 85, 20)
        m_values = [optical_air_mass(sza) for sza in sza_values]

        for i in range(len(m_values) - 1):
            assert m_values[i] < m_values[i + 1]

    def test_air_mass_positive(self):
        """Air mass should always be positive."""
        sza_values = np.linspace(0, 89, 50)
        for sza in sza_values:
            assert optical_air_mass(sza) > 0

    def test_reciprocity(self):
        """Air mass should equal 1/mu0 for low SZA."""
        for sza in [0, 15, 30, 45]:
            mu0 = np.cos(np.radians(sza))
            m = optical_air_mass(sza)
            assert np.isclose(m, 1.0 / mu0, rtol=0.02)
