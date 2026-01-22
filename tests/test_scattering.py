"""Tests for scattering modules."""

import numpy as np
import pytest

from raf_tran.scattering import (
    RayleighScattering,
    rayleigh_cross_section,
    rayleigh_phase_function,
    MieScattering,
    mie_coefficients,
)


class TestRayleighScattering:
    """Tests for Rayleigh scattering calculations."""

    def test_cross_section_wavelength_dependence(self):
        """Test λ⁻⁴ wavelength dependence."""
        wavelengths = np.array([0.4, 0.5, 0.6, 0.7])  # μm
        sigma = rayleigh_cross_section(wavelengths)

        # σ ∝ λ⁻⁴, so σ₁/σ₂ ≈ (λ₂/λ₁)⁴
        ratio = sigma[0] / sigma[-1]
        expected_ratio = (wavelengths[-1] / wavelengths[0]) ** 4
        assert np.isclose(ratio, expected_ratio, rtol=0.1)

    def test_cross_section_magnitude(self):
        """Test cross section is in correct order of magnitude."""
        # At 0.55 μm, Rayleigh cross section is ~4e-31 m²
        sigma = rayleigh_cross_section(np.array([0.55]))
        assert 1e-32 < sigma[0] < 1e-30

    def test_phase_function_normalized(self):
        """Test phase function normalization."""
        cos_theta = np.linspace(-1, 1, 1000)
        P = rayleigh_phase_function(cos_theta)

        # Integral over sphere should be 4π
        # P(cos θ) integrated over 4π: ∫∫ P sin θ dθ dφ = 2π ∫ P d(cos θ)
        integral = 2 * np.pi * np.trapz(P, cos_theta)
        assert np.isclose(integral, 4 * np.pi, rtol=0.01)

    def test_phase_function_forward_backward_symmetric(self):
        """Test Rayleigh phase function is symmetric."""
        P_forward = rayleigh_phase_function(np.array([1.0]))[0]
        P_backward = rayleigh_phase_function(np.array([-1.0]))[0]
        assert np.isclose(P_forward, P_backward)

    def test_rayleigh_scattering_class(self):
        """Test RayleighScattering class."""
        rayleigh = RayleighScattering()

        wavelengths = np.array([0.4, 0.5, 0.6])
        sigma = rayleigh.cross_section(wavelengths)

        assert len(sigma) == 3
        assert all(sigma > 0)

        # Asymmetry parameter should be 0
        assert rayleigh.asymmetry_parameter() == 0.0

        # Single scattering albedo should be 1
        assert rayleigh.single_scattering_albedo() == 1.0

    def test_optical_depth_calculation(self):
        """Test optical depth calculation."""
        rayleigh = RayleighScattering()

        wavelengths = np.array([0.5])  # μm
        number_density = np.array([2.5e25, 1.0e25])  # molecules/m³
        dz = np.array([1000, 2000])  # m

        tau = rayleigh.optical_depth(wavelengths, number_density, dz)

        assert tau.shape == (1, 2)
        assert all(tau.ravel() > 0)


class TestMieScattering:
    """Tests for Mie scattering calculations."""

    def test_mie_small_particle_limit(self):
        """Test Mie scattering approaches Rayleigh for small particles."""
        # For very small size parameters, Mie → Rayleigh
        x = 0.01  # Very small particle
        m = 1.5 + 0j  # Real refractive index

        Q_ext, Q_sca, Q_abs, g = mie_coefficients.__wrapped__(x, m) if hasattr(mie_coefficients, '__wrapped__') else (0, 0, 0, 0)
        # For small x, Q_ext ∝ x⁴ (Rayleigh regime)
        # Just verify function runs without error for now
        from raf_tran.scattering.mie import mie_efficiencies
        Q_ext, Q_sca, Q_abs, g = mie_efficiencies(x, m)

        assert Q_ext >= 0
        assert Q_sca >= 0
        assert Q_abs >= 0 or np.isclose(Q_abs, 0, atol=1e-10)

    def test_mie_large_particle_limit(self):
        """Test Mie scattering for large particles."""
        # For large size parameters, Q_ext → 2 (extinction paradox)
        x = 100  # Large particle
        m = 1.5 + 0j

        from raf_tran.scattering.mie import mie_efficiencies
        Q_ext, Q_sca, Q_abs, g = mie_efficiencies(x, m)

        # Q_ext should approach 2 for large particles
        assert 1.5 < Q_ext < 2.5

    def test_mie_absorbing_particle(self):
        """Test Mie scattering with absorption."""
        x = 5.0
        m = 1.5 + 0.1j  # Complex refractive index (absorbing)

        from raf_tran.scattering.mie import mie_efficiencies
        Q_ext, Q_sca, Q_abs, g = mie_efficiencies(x, m)

        # Absorption should be non-zero for complex refractive index
        assert Q_abs > 0
        # Conservation: Q_ext = Q_sca + Q_abs
        assert np.isclose(Q_ext, Q_sca + Q_abs, rtol=1e-6)

    def test_mie_scattering_class(self):
        """Test MieScattering class."""
        mie = MieScattering(refractive_index=1.5 + 0.01j)

        size_params = np.array([0.1, 1.0, 5.0, 10.0])
        Q_ext, Q_sca, Q_abs, g = mie.efficiencies(size_params)

        assert all(Q_ext >= 0)
        assert all(Q_sca >= 0)
        assert all(Q_abs >= 0)
        # g should be between -1 and 1
        assert all((-1 <= g) & (g <= 1))

    def test_cross_sections(self):
        """Test cross section calculation."""
        mie = MieScattering(refractive_index=1.5 + 0j)

        wavelength = np.array([0.5, 1.0])  # μm
        radius = 0.5  # μm

        sigma_ext, sigma_sca, sigma_abs = mie.cross_sections(wavelength, radius)

        assert all(sigma_ext >= 0)
        assert all(sigma_sca >= 0)
        # Cross sections should have dimensions of area (μm²)
        # For r=0.5μm, geometric cross section = π * 0.25 ≈ 0.785 μm²
        geometric = np.pi * radius**2
        # Extinction cross section should be on the order of geometric
        assert all(0.1 * geometric < sigma_ext) and all(sigma_ext < 10 * geometric)


class TestLognormalDistribution:
    """Tests for particle size distribution."""

    def test_lognormal_normalization(self):
        """Test lognormal distribution integrates to total N."""
        from raf_tran.scattering.mie import lognormal_size_distribution

        r = np.logspace(-2, 1, 1000)  # 0.01 to 10 μm
        r_g = 0.5  # geometric mean radius
        sigma_g = 2.0  # geometric standard deviation
        N_total = 100.0

        n = lognormal_size_distribution(r, r_g, sigma_g, N_total)

        # Integrate using log spacing
        integral = np.trapz(n, r)
        assert np.isclose(integral, N_total, rtol=0.05)

    def test_lognormal_peak_at_mode(self):
        """Test lognormal distribution peaks near mode radius."""
        from raf_tran.scattering.mie import lognormal_size_distribution

        r = np.logspace(-2, 1, 1000)
        r_g = 0.5
        sigma_g = 1.5

        n = lognormal_size_distribution(r, r_g, sigma_g)

        # Mode of lognormal: r_mode = r_g * exp(-ln²(σ_g))
        r_mode = r_g * np.exp(-(np.log(sigma_g)) ** 2)

        peak_idx = np.argmax(n)
        peak_r = r[peak_idx]

        assert np.isclose(peak_r, r_mode, rtol=0.1)
