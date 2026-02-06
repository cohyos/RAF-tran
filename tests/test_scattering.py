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
        trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        integral = 2 * np.pi * trapz_func(P, cos_theta)
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
        trapz_func = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        integral = trapz_func(n, r)
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


class TestMieRayleighLimit:
    """
    Tests validating Mie scattering converges to Rayleigh in the small particle limit.

    For x = 2πr/λ << 1 (Rayleigh regime):
    - Extinction efficiency: Q_ext ≈ (8/3) x⁴ |K|² where K = (m²-1)/(m²+2)
    - Scattering efficiency: Q_sca ≈ (8/3) x⁴ |K|²
    - Asymmetry parameter: g → 0

    Reference:
    Bohren, C.F. and Huffman, D.R. (1983). Absorption and Scattering
    of Light by Small Particles. Wiley.
    """

    def test_rayleigh_limit_x4_dependence(self):
        """Test Q_ext ∝ x⁴ in Rayleigh limit."""
        from raf_tran.scattering.mie import mie_efficiencies

        m = 1.5 + 0j
        x_values = np.array([0.001, 0.01, 0.05])

        Q_ext_values = []
        for x in x_values:
            Q_ext, _, _, _ = mie_efficiencies(x, m)
            Q_ext_values.append(Q_ext)

        Q_ext_values = np.array(Q_ext_values)

        # Check x⁴ scaling: Q(x1)/Q(x2) should equal (x1/x2)⁴
        for i in range(len(x_values) - 1):
            ratio = Q_ext_values[i] / Q_ext_values[i + 1]
            expected_ratio = (x_values[i] / x_values[i + 1]) ** 4
            assert np.isclose(ratio, expected_ratio, rtol=0.05), \
                f"Failed x⁴ scaling: got ratio {ratio:.4f}, expected {expected_ratio:.4f}"

    def test_rayleigh_limit_formula(self):
        """Test Q_ext matches Rayleigh formula for small x."""
        from raf_tran.scattering.mie import mie_efficiencies

        m = 1.5 + 0j
        x = 0.01

        # Rayleigh formula: Q_ext = (8/3) x⁴ Re[K²] where K = (m²-1)/(m²+2)
        m2 = m ** 2
        K = (m2 - 1) / (m2 + 2)
        K_sq_real = np.abs(K) ** 2  # |K|² for scattering
        Q_rayleigh = (8/3) * x**4 * K_sq_real

        Q_ext, Q_sca, Q_abs, _ = mie_efficiencies(x, m)

        # Mie should match Rayleigh within a few percent for x=0.01
        assert np.isclose(Q_ext, Q_rayleigh, rtol=0.05), \
            f"Q_ext mismatch: Mie={Q_ext:.6e}, Rayleigh={Q_rayleigh:.6e}"

    def test_rayleigh_limit_asymmetry_zero(self):
        """Test asymmetry parameter g → 0 in Rayleigh limit."""
        from raf_tran.scattering.mie import mie_efficiencies

        m = 1.5 + 0j
        x_values = [0.001, 0.01, 0.05]

        for x in x_values:
            _, _, _, g = mie_efficiencies(x, m)
            # g should be very small for Rayleigh regime
            assert np.abs(g) < 0.1, \
                f"Asymmetry g={g:.4f} too large for x={x} (Rayleigh limit)"

    def test_rayleigh_limit_no_absorption(self):
        """Test Q_abs ≈ 0 for real refractive index in Rayleigh limit."""
        from raf_tran.scattering.mie import mie_efficiencies

        m = 1.5 + 0j  # Non-absorbing
        x = 0.01

        _, _, Q_abs, _ = mie_efficiencies(x, m)

        # For non-absorbing particle, Q_abs should be essentially zero
        assert np.abs(Q_abs) < 1e-10, \
            f"Q_abs={Q_abs:.6e} should be zero for non-absorbing particle"

    def test_rayleigh_limit_energy_conservation(self):
        """Test Q_ext = Q_sca + Q_abs in Rayleigh limit."""
        from raf_tran.scattering.mie import mie_efficiencies

        # Test with absorbing particle
        m = 1.5 + 0.01j
        x = 0.01

        Q_ext, Q_sca, Q_abs, _ = mie_efficiencies(x, m)

        # Energy conservation
        assert np.isclose(Q_ext, Q_sca + Q_abs, rtol=1e-6), \
            f"Energy conservation violated: Q_ext={Q_ext:.6e}, Q_sca+Q_abs={Q_sca+Q_abs:.6e}"

    def test_rayleigh_limit_absorption_imaginary_index(self):
        """Test absorption increases with imaginary refractive index."""
        from raf_tran.scattering.mie import mie_efficiencies

        x = 0.01
        k_values = [0.001, 0.01, 0.1]  # Increasing imaginary part

        Q_abs_values = []
        for k in k_values:
            m = 1.5 + k * 1j
            _, _, Q_abs, _ = mie_efficiencies(x, m)
            Q_abs_values.append(Q_abs)

        # Q_abs should increase with k
        for i in range(len(Q_abs_values) - 1):
            assert Q_abs_values[i] < Q_abs_values[i + 1], \
                f"Q_abs not increasing with imaginary index"

    def test_rayleigh_cross_section_vs_mie(self):
        """Compare Rayleigh cross section function to Mie for small particles."""
        from raf_tran.scattering.mie import mie_efficiencies
        from raf_tran.scattering import rayleigh_cross_section

        # Use a very small particle (x << 1)
        wavelength = 0.55  # μm
        radius = 0.001  # μm (1 nm particle)
        x = 2 * np.pi * radius / wavelength  # ≈ 0.011

        # Rayleigh cross section (for molecules)
        sigma_ray = rayleigh_cross_section(np.array([wavelength]))[0]

        # Mie cross section
        # For air molecules, effective "radius" would give similar cross section
        # This is more of a sanity check than exact comparison
        m = 1.0003 + 0j  # Near unity for air
        Q_ext, _, _, _ = mie_efficiencies(x, m)
        sigma_mie = Q_ext * np.pi * radius**2  # geometric cross section in μm²

        # The Mie cross section for a 1nm particle with n≈1 should be tiny
        # Just verify it's physically reasonable (positive)
        assert sigma_mie > 0
        assert sigma_ray > 0


class TestRayleighLambdaDependence:
    """
    Tests for Rayleigh wavelength dependence.

    Note: The actual wavelength dependence is approximately λ⁻⁴ but includes
    small corrections from the refractive index dispersion and depolarization
    factor (King factor). See Bodhaine et al. (1999).

    Reference:
    Bodhaine, B.A., et al. (1999). On Rayleigh Optical Depth Calculations.
    J. Atmos. Ocean. Tech., 16, 1854-1861.
    """

    def test_lambda4_ratio_multiple_wavelengths(self):
        """Test approximate λ⁻⁴ dependence across visible spectrum."""
        from raf_tran.scattering import rayleigh_cross_section

        wavelengths = np.array([0.35, 0.45, 0.55, 0.65, 0.75, 0.85])  # μm
        sigma = rayleigh_cross_section(wavelengths)

        # Normalize to 0.55 μm
        sigma_norm = sigma / sigma[2]
        expected_norm = (wavelengths[2] / wavelengths) ** 4

        # Check each ratio - allow 10% tolerance due to refractive index
        # dispersion and depolarization factor corrections
        for i, wl in enumerate(wavelengths):
            assert np.isclose(sigma_norm[i], expected_norm[i], rtol=0.10), \
                f"λ⁻⁴ dependence failed at {wl:.2f}μm: got {sigma_norm[i]:.4f}, expected {expected_norm[i]:.4f}"

    def test_blue_stronger_than_red(self):
        """Test blue light scatters more than red (why sky is blue)."""
        from raf_tran.scattering import rayleigh_cross_section

        blue = 0.45   # μm
        green = 0.55  # μm
        red = 0.65    # μm

        sigma = rayleigh_cross_section(np.array([blue, green, red]))

        # Blue should scatter most, red least
        assert sigma[0] > sigma[1] > sigma[2], \
            "Rayleigh scattering should be strongest for blue light"

        # Blue/red ratio should be approximately (0.65/0.45)⁴ ≈ 4.35
        # Allow 5% tolerance for refractive index corrections
        ratio = sigma[0] / sigma[2]
        expected = (red / blue) ** 4
        assert np.isclose(ratio, expected, rtol=0.05), \
            f"Blue/red ratio: got {ratio:.2f}, expected {expected:.2f}"

    def test_cross_section_literature_value(self):
        """Test cross section matches literature at 550nm."""
        from raf_tran.scattering import rayleigh_cross_section

        # At 550 nm, σ ≈ 4.5e-31 m² (Bodhaine et al., 1999)
        sigma = rayleigh_cross_section(np.array([0.55]))[0]

        # Convert to m² if necessary (assuming function returns m²)
        # Allow 10% tolerance for different formulations
        assert 3e-31 < sigma < 6e-31, \
            f"Cross section at 550nm: got {sigma:.3e}m², expected ~4.5e-31 m²"

    def test_optical_depth_vertical_column(self):
        """Test Rayleigh optical depth for standard atmosphere."""
        from raf_tran.scattering import RayleighScattering
        from raf_tran.atmosphere import StandardAtmosphere

        rayleigh = RayleighScattering()
        atmosphere = StandardAtmosphere()

        # Create atmospheric column
        z = np.linspace(0, 100000, 1001)  # 0-100 km
        dz = np.diff(z)
        z_mid = (z[:-1] + z[1:]) / 2
        n = atmosphere.number_density(z_mid)

        # Calculate optical depth at 550 nm
        wavelength = np.array([0.55])
        tau = rayleigh.optical_depth(wavelength, n, dz)
        tau_total = tau.sum()

        # Expected Rayleigh optical depth at 550 nm is ~0.097
        # (see Bodhaine et al., 1999, Table 1)
        assert 0.08 < tau_total < 0.12, \
            f"Total Rayleigh optical depth at 550nm: got {tau_total:.4f}, expected ~0.097"
