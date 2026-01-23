"""Tests for utility functions."""

import numpy as np
import pytest

from raf_tran.utils import (
    SPEED_OF_LIGHT,
    PLANCK_CONSTANT,
    BOLTZMANN_CONSTANT,
    STEFAN_BOLTZMANN,
    planck_function,
    wavenumber_to_wavelength,
    wavelength_to_wavenumber,
)
from raf_tran.utils.spectral import planck_function_wavenumber, stefan_boltzmann_flux


class TestConstants:
    """Tests for physical constants."""

    def test_speed_of_light(self):
        """Test speed of light value."""
        assert np.isclose(SPEED_OF_LIGHT, 299792458.0, rtol=1e-9)

    def test_planck_constant(self):
        """Test Planck constant value."""
        assert np.isclose(PLANCK_CONSTANT, 6.62607015e-34, rtol=1e-9)

    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        assert np.isclose(BOLTZMANN_CONSTANT, 1.380649e-23, rtol=1e-9)

    def test_stefan_boltzmann(self):
        """Test Stefan-Boltzmann constant value."""
        assert np.isclose(STEFAN_BOLTZMANN, 5.670374419e-8, rtol=1e-6)

    def test_stefan_boltzmann_derived(self):
        """Test Stefan-Boltzmann constant from other constants."""
        # σ = 2π⁵k⁴/(15h³c²)
        sigma_derived = (
            2 * np.pi**5 * BOLTZMANN_CONSTANT**4
            / (15 * PLANCK_CONSTANT**3 * SPEED_OF_LIGHT**2)
        )
        assert np.isclose(STEFAN_BOLTZMANN, sigma_derived, rtol=1e-5)


class TestWavelengthConversions:
    """Tests for wavelength/wavenumber conversions."""

    def test_wavelength_to_wavenumber(self):
        """Test wavelength to wavenumber conversion."""
        # 10 μm = 1000 cm⁻¹
        wl = np.array([10.0])
        wn = wavelength_to_wavenumber(wl)
        assert np.isclose(wn[0], 1000.0)

    def test_wavenumber_to_wavelength(self):
        """Test wavenumber to wavelength conversion."""
        # 1000 cm⁻¹ = 10 μm
        wn = np.array([1000.0])
        wl = wavenumber_to_wavelength(wn)
        assert np.isclose(wl[0], 10.0)

    def test_round_trip_conversion(self):
        """Test round-trip wavelength conversion."""
        wl_original = np.array([0.5, 1.0, 5.0, 10.0, 15.0])
        wl_roundtrip = wavenumber_to_wavelength(wavelength_to_wavenumber(wl_original))
        assert np.allclose(wl_original, wl_roundtrip)


class TestPlanckFunction:
    """Tests for Planck blackbody function."""

    def test_planck_at_peak(self):
        """Test Planck function at Wien's displacement peak."""
        T = 5778  # Solar temperature
        # Wien's displacement law: λ_max * T = 2.898e-3 m·K
        lambda_max = 2.898e-3 / T  # ~500 nm

        wavelengths = np.linspace(lambda_max * 0.5, lambda_max * 2, 100)
        B = planck_function(wavelengths, T)

        # Maximum should be near lambda_max
        max_idx = np.argmax(B)
        assert np.isclose(wavelengths[max_idx], lambda_max, rtol=0.1)

    def test_planck_stefan_boltzmann(self):
        """Test Planck function integrates to Stefan-Boltzmann law."""
        T = 300  # K
        # Integrate B over wavelength: ∫B dλ = σT⁴/π (per steradian)

        # Use wide wavelength range (0.1 μm to 1000 μm)
        wavelengths = np.logspace(-7, -3, 10000)  # meters
        B = planck_function(wavelengths, T)

        # Integrate using trapezoidal rule
        trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        integral = trapz_func(B, wavelengths)

        # Expected: σT⁴/π
        expected = STEFAN_BOLTZMANN * T**4 / np.pi

        assert np.isclose(integral, expected, rtol=0.01)

    def test_planck_zero_temperature(self):
        """Test Planck function returns zero for T=0."""
        wavelengths = np.array([1e-6, 5e-6, 10e-6])
        B = planck_function(wavelengths, 0.0)
        assert np.allclose(B, 0)

    def test_planck_negative_temperature(self):
        """Test Planck function handles negative temperature."""
        wavelengths = np.array([1e-6])
        B = planck_function(wavelengths, -100.0)
        assert np.allclose(B, 0)

    def test_planck_wavenumber(self):
        """Test Planck function in wavenumber space."""
        T = 300
        wavenumber = np.array([500, 1000, 1500])  # cm⁻¹

        B = planck_function_wavenumber(wavenumber, T)

        assert all(B > 0)
        # At 300 K, peak is around 600 cm⁻¹
        # So B(1000) should be significant


class TestStefanBoltzmannFlux:
    """Tests for Stefan-Boltzmann flux calculation."""

    def test_stefan_boltzmann_flux(self):
        """Test Stefan-Boltzmann flux calculation."""
        T = 300  # K
        flux = stefan_boltzmann_flux(T)

        expected = STEFAN_BOLTZMANN * T**4
        assert np.isclose(flux, expected)

    def test_flux_temperature_dependence(self):
        """Test T⁴ dependence of flux."""
        T1 = 200
        T2 = 400

        F1 = stefan_boltzmann_flux(T1)
        F2 = stefan_boltzmann_flux(T2)

        # F2/F1 should be (T2/T1)⁴ = 16
        assert np.isclose(F2 / F1, 16.0)
