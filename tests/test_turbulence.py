"""Tests for atmospheric turbulence module."""

import numpy as np
import pytest

from raf_tran.turbulence import (
    # Cn2 profiles
    hufnagel_valley_cn2,
    slc_day_cn2,
    slc_night_cn2,
    cn2_from_weather,
    # Propagation parameters
    fried_parameter,
    scintillation_index,
    rytov_variance,
    beam_wander_variance,
    coherence_time,
    strehl_ratio,
    # Spectra
    kolmogorov_spectrum,
    von_karman_spectrum,
    structure_function,
    # Phase-aware functions
    phase_variance,
    tilt_removed_phase_variance,
    angle_of_arrival_variance,
    zernike_variance,
    phase_power_spectrum,
    residual_phase_variance_ao,
    long_exposure_strehl,
)


class TestHufnagelValleyCn2:
    """Tests for Hufnagel-Valley Cn2 profile."""

    def test_ground_level(self):
        """Test Cn2 at ground level."""
        cn2_ground = 1.7e-14  # Default
        cn2 = hufnagel_valley_cn2(0, cn2_ground=cn2_ground)
        # Should be close to ground value
        assert np.isclose(cn2, cn2_ground, rtol=0.01)

    def test_decreases_with_altitude(self):
        """Cn2 should generally decrease with altitude."""
        altitudes = [0, 100, 1000, 5000]
        cn2_values = [hufnagel_valley_cn2(h) for h in altitudes]

        # Overall trend should be decreasing
        assert cn2_values[0] > cn2_values[-1]

    def test_typical_range(self):
        """Cn2 values should be in reasonable range."""
        for h in [0, 1000, 5000, 10000]:
            cn2 = hufnagel_valley_cn2(h)
            # Typical range: 10^-18 to 10^-12
            assert 1e-18 < cn2 < 1e-12
        # At very high altitude (20km), Cn2 can be extremely small
        cn2_high = hufnagel_valley_cn2(20000)
        assert cn2_high > 0

    def test_array_input(self):
        """Test with array input."""
        altitudes = np.array([0, 1000, 5000, 10000])
        cn2 = hufnagel_valley_cn2(altitudes)
        assert cn2.shape == altitudes.shape
        assert all(cn2 > 0)

    def test_wind_speed_effect(self):
        """Higher wind speed should increase tropopause turbulence."""
        h = 10000  # 10 km - tropopause region
        cn2_low_wind = hufnagel_valley_cn2(h, wind_speed_rms=10)
        cn2_high_wind = hufnagel_valley_cn2(h, wind_speed_rms=30)
        assert cn2_high_wind > cn2_low_wind


class TestSLCCn2:
    """Tests for SLC day/night Cn2 models."""

    def test_slc_day_ground(self):
        """Test SLC daytime at ground level."""
        cn2 = slc_day_cn2(0)
        # Should return value at h=1m (avoiding singularity)
        assert cn2 > 0
        assert cn2 < 1e-10  # Reasonable upper bound

    def test_slc_night_ground(self):
        """Test SLC nighttime at ground level."""
        cn2 = slc_night_cn2(0)
        assert cn2 > 0

    def test_day_stronger_than_night_near_surface(self):
        """Daytime turbulence should be stronger near surface."""
        for h in [10, 50, 100]:
            cn2_day = slc_day_cn2(h)
            cn2_night = slc_night_cn2(h)
            assert cn2_day > cn2_night

    def test_slc_typical_range(self):
        """SLC values should be in reasonable range."""
        for h in [0, 100, 500, 1000, 5000]:
            cn2_day = slc_day_cn2(h)
            cn2_night = slc_night_cn2(h)
            assert 1e-20 < cn2_day < 1e-10
            assert 1e-20 < cn2_night < 1e-10

    def test_array_input(self):
        """Test SLC with array input."""
        altitudes = np.array([0, 100, 1000, 5000])
        cn2_day = slc_day_cn2(altitudes)
        cn2_night = slc_night_cn2(altitudes)
        assert cn2_day.shape == altitudes.shape
        assert cn2_night.shape == altitudes.shape


class TestCn2FromWeather:
    """Tests for weather-based Cn2 estimation."""

    def test_basic_output(self):
        """Test basic function output."""
        # Use higher altitude where values are within bounds
        cn2 = cn2_from_weather(1000, 280, 5)
        assert cn2 > 0
        assert 1e-18 <= cn2 <= 1e-12

    def test_temperature_effect(self):
        """Lower temperature should increase Cn2 (density effect)."""
        # Use conditions where output is not clipped
        cn2_warm = cn2_from_weather(2000, 270, 2, solar_elevation_deg=20)
        cn2_cold = cn2_from_weather(2000, 240, 2, solar_elevation_deg=20)
        # Formula has T^2 in denominator squared, so cold -> higher Cn2
        # But output is clipped, so test relative order when possible
        assert cn2_cold >= cn2_warm

    def test_solar_effect(self):
        """Higher sun should increase Cn2 (convection)."""
        # Use altitude where we can see the effect
        cn2_low_sun = cn2_from_weather(3000, 270, 2, solar_elevation_deg=10)
        cn2_high_sun = cn2_from_weather(3000, 270, 2, solar_elevation_deg=60)
        assert cn2_high_sun >= cn2_low_sun

    def test_altitude_effect(self):
        """Higher altitude should have lower Cn2."""
        cn2_low = cn2_from_weather(100, 288, 5)
        cn2_high = cn2_from_weather(5000, 260, 5)
        assert cn2_high < cn2_low


class TestFriedParameter:
    """Tests for Fried parameter calculation."""

    def test_basic_output(self):
        """Test basic Fried parameter calculation."""
        wavelength = 0.5e-6  # 500 nm
        cn2_int = 1e-13  # m^(1/3)
        r0 = fried_parameter(wavelength, cn2_int)
        assert r0 > 0
        # Typical r0 at 500 nm: 5-30 cm
        assert 0.01 < r0 < 1.0

    def test_wavelength_dependence(self):
        """r0 should increase with wavelength (r0 ~ lambda^1.2)."""
        cn2_int = 1e-13
        r0_visible = fried_parameter(0.5e-6, cn2_int)
        r0_infrared = fried_parameter(2.0e-6, cn2_int)
        assert r0_infrared > r0_visible

        # Check ~lambda^1.2 scaling
        ratio = r0_infrared / r0_visible
        expected_ratio = (2.0 / 0.5)**(6/5)
        assert np.isclose(ratio, expected_ratio, rtol=0.05)

    def test_cn2_dependence(self):
        """r0 should decrease with stronger turbulence."""
        wavelength = 0.5e-6
        r0_weak = fried_parameter(wavelength, 1e-14)
        r0_strong = fried_parameter(wavelength, 1e-12)
        assert r0_weak > r0_strong


class TestRytovVariance:
    """Tests for Rytov variance calculation."""

    def test_basic_output(self):
        """Test basic Rytov variance calculation."""
        sigma_r2 = rytov_variance(1.55e-6, 1e-15, 10000)
        assert sigma_r2 > 0

    def test_weak_fluctuation_regime(self):
        """Test weak fluctuation conditions."""
        # Short path, low Cn2 -> weak fluctuations (sigma_R^2 < 0.3)
        sigma_r2 = rytov_variance(1.55e-6, 1e-17, 1000)
        assert sigma_r2 < 0.3

    def test_strong_fluctuation_regime(self):
        """Test strong fluctuation conditions."""
        # Long path, high Cn2 -> strong fluctuations
        sigma_r2 = rytov_variance(0.5e-6, 1e-14, 50000)
        assert sigma_r2 > 5

    def test_path_length_dependence(self):
        """Rytov variance should increase with path length."""
        wavelength = 1.55e-6
        cn2 = 1e-15
        sigma_short = rytov_variance(wavelength, cn2, 1000)
        sigma_long = rytov_variance(wavelength, cn2, 10000)
        assert sigma_long > sigma_short


class TestScintillationIndex:
    """Tests for scintillation index calculation."""

    def test_basic_output(self):
        """Test basic scintillation index calculation."""
        si = scintillation_index(1.55e-6, 1e-15, 10000)
        assert si > 0
        assert si <= 1.0  # Saturates at 1

    def test_weak_regime(self):
        """In weak regime, scintillation should be small."""
        si = scintillation_index(1.55e-6, 1e-17, 1000)
        assert si < 0.1

    def test_saturation(self):
        """Scintillation should be bounded even in strong turbulence."""
        # In very strong turbulence, scintillation saturates
        # This is correct physics - Andrews-Phillips model
        si = scintillation_index(0.5e-6, 1e-13, 50000)
        # Scintillation should stay bounded (saturation at 1.0)
        assert 0 < si <= 1.0

        # Test that increasing turbulence initially increases scintillation
        si_weak = scintillation_index(1.55e-6, 1e-17, 2000)
        si_moderate = scintillation_index(1.55e-6, 1e-16, 2000)
        assert si_moderate > si_weak

    def test_aperture_averaging(self):
        """Larger aperture should reduce scintillation."""
        si_point = scintillation_index(1.55e-6, 1e-15, 10000, aperture_diameter_m=None)
        si_large = scintillation_index(1.55e-6, 1e-15, 10000, aperture_diameter_m=0.5)
        assert si_large <= si_point


class TestBeamWander:
    """Tests for beam wander variance calculation."""

    def test_basic_output(self):
        """Test basic beam wander calculation."""
        sigma_bw2 = beam_wander_variance(1.55e-6, 1e-15, 10000, 0.1)
        assert sigma_bw2 >= 0

    def test_increases_with_turbulence(self):
        """Beam wander should increase with turbulence."""
        sigma_weak = beam_wander_variance(1.55e-6, 1e-16, 10000, 0.1)
        sigma_strong = beam_wander_variance(1.55e-6, 1e-14, 10000, 0.1)
        assert sigma_strong > sigma_weak

    def test_increases_with_path(self):
        """Beam wander should increase with path length."""
        sigma_short = beam_wander_variance(1.55e-6, 1e-15, 1000, 0.1)
        sigma_long = beam_wander_variance(1.55e-6, 1e-15, 10000, 0.1)
        assert sigma_long > sigma_short


class TestCoherenceTime:
    """Tests for atmospheric coherence time."""

    def test_basic_output(self):
        """Test basic coherence time calculation."""
        tau0 = coherence_time(0.5e-6, 1e-13, 10)
        assert tau0 > 0
        # Typical: 1-100 ms
        assert 1e-4 < tau0 < 1

    def test_wavelength_dependence(self):
        """Longer wavelength -> longer coherence time."""
        tau_visible = coherence_time(0.5e-6, 1e-13, 10)
        tau_ir = coherence_time(2.0e-6, 1e-13, 10)
        assert tau_ir > tau_visible

    def test_wind_dependence(self):
        """Higher wind -> shorter coherence time."""
        tau_calm = coherence_time(0.5e-6, 1e-13, 5)
        tau_windy = coherence_time(0.5e-6, 1e-13, 20)
        assert tau_calm > tau_windy


class TestStrehlRatio:
    """Tests for Strehl ratio calculation."""

    def test_from_wavefront_error(self):
        """Test Strehl from RMS wavefront error."""
        # Small error -> high Strehl
        S = strehl_ratio(0.5e-6, rms_wavefront_error_m=50e-9)
        assert 0 < S < 1
        assert S > 0.5  # Small error

    def test_from_turbulence(self):
        """Test Strehl from r0 and aperture."""
        S = strehl_ratio(0.5e-6, r0=0.1, aperture_diameter_m=0.5)
        assert 0 < S < 1

    def test_diffraction_limited(self):
        """Zero wavefront error -> Strehl = 1."""
        S = strehl_ratio(0.5e-6, rms_wavefront_error_m=0)
        assert np.isclose(S, 1.0)

    def test_strehl_bounds(self):
        """Strehl should always be between 0 and 1."""
        S1 = strehl_ratio(0.5e-6, rms_wavefront_error_m=100e-9)
        S2 = strehl_ratio(0.5e-6, r0=0.05, aperture_diameter_m=1.0)
        assert 0 <= S1 <= 1
        assert 0 <= S2 <= 1

    def test_missing_params_raises(self):
        """Should raise if neither wavefront error nor r0/D specified."""
        with pytest.raises(ValueError):
            strehl_ratio(0.5e-6)


class TestKolmogorovSpectrum:
    """Tests for Kolmogorov turbulence spectrum."""

    def test_basic_output(self):
        """Test basic spectrum output."""
        kappa = np.logspace(-2, 4, 100)
        cn2 = 1e-15
        phi = kolmogorov_spectrum(kappa, cn2)
        assert all(phi >= 0)

    def test_power_law(self):
        """Spectrum should follow kappa^(-11/3) power law."""
        kappa = np.array([10, 100, 1000])
        cn2 = 1e-15
        phi = kolmogorov_spectrum(kappa, cn2)

        # Check -11/3 slope
        log_kappa = np.log10(kappa)
        log_phi = np.log10(phi)
        slope = (log_phi[-1] - log_phi[0]) / (log_kappa[-1] - log_kappa[0])
        assert np.isclose(slope, -11/3, rtol=0.01)

    def test_cn2_scaling(self):
        """Spectrum should scale linearly with Cn2."""
        kappa = 100
        phi1 = kolmogorov_spectrum(kappa, 1e-15)
        phi2 = kolmogorov_spectrum(kappa, 2e-15)
        assert np.isclose(phi2 / phi1, 2.0)


class TestVonKarmanSpectrum:
    """Tests for von Karman turbulence spectrum."""

    def test_basic_output(self):
        """Test basic spectrum output."""
        kappa = np.logspace(-2, 4, 100)
        cn2 = 1e-15
        phi = von_karman_spectrum(kappa, cn2)
        assert all(phi >= 0)

    def test_matches_kolmogorov_in_inertial(self):
        """Should match Kolmogorov in inertial range."""
        kappa = np.array([1, 10, 100])  # Inertial range
        cn2 = 1e-15
        phi_vk = von_karman_spectrum(kappa, cn2, L0=100, l0=0.001)
        phi_k = kolmogorov_spectrum(kappa, cn2)

        # Should be close in inertial range
        for i in range(len(kappa)):
            assert np.isclose(phi_vk[i], phi_k[i], rtol=0.3)

    def test_outer_scale_effect(self):
        """Large outer scale should increase low-frequency power."""
        kappa = 0.01  # Low frequency
        cn2 = 1e-15
        phi_small_L0 = von_karman_spectrum(kappa, cn2, L0=10)
        phi_large_L0 = von_karman_spectrum(kappa, cn2, L0=1000)
        # Larger L0 -> more low-frequency power
        assert phi_large_L0 > phi_small_L0


class TestStructureFunction:
    """Tests for turbulence structure function."""

    def test_basic_output(self):
        """Test basic structure function output."""
        r = np.array([0.01, 0.1, 1.0])
        cn2 = 1e-15
        D = structure_function(r, cn2)
        assert all(D >= 0)

    def test_r_2_3_scaling(self):
        """Structure function should scale as r^(2/3)."""
        r = np.array([0.1, 1.0])
        cn2 = 1e-15
        D = structure_function(r, cn2)

        # D(r2)/D(r1) = (r2/r1)^(2/3)
        ratio = D[1] / D[0]
        expected = (r[1] / r[0])**(2/3)
        assert np.isclose(ratio, expected, rtol=0.01)

    def test_cn2_definition(self):
        """Structure function at r=1m should equal Cn2."""
        cn2 = 1e-15
        D = structure_function(1.0, cn2)
        assert np.isclose(D, cn2, rtol=0.01)


class TestPhysicsConsistency:
    """Cross-module physics consistency tests."""

    def test_fried_scintillation_relationship(self):
        """Smaller r0 should correlate with higher scintillation."""
        wavelength = 0.5e-6
        path = 10000

        cn2_weak = 1e-16
        cn2_strong = 1e-14

        r0_weak = fried_parameter(wavelength, cn2_weak)
        r0_strong = fried_parameter(wavelength, cn2_strong)
        si_weak = scintillation_index(wavelength, cn2_weak, path)
        si_strong = scintillation_index(wavelength, cn2_strong, path)

        assert r0_weak > r0_strong
        assert si_weak < si_strong

    def test_hv_integrated_typical_values(self):
        """HV integrated Cn2 should give typical r0 values."""
        # Integrate HV profile
        altitudes = np.linspace(0, 20000, 100)
        cn2_profile = hufnagel_valley_cn2(altitudes)
        dh = np.diff(altitudes)
        cn2_mid = 0.5 * (cn2_profile[:-1] + cn2_profile[1:])
        cn2_integrated = np.sum(cn2_mid * dh)

        # Calculate r0 at 500 nm
        r0 = fried_parameter(0.5e-6, cn2_integrated)

        # HV 5/7 should give r0 ~ 5 cm at 500 nm
        assert 0.03 < r0 < 0.15  # 3-15 cm range

    def test_day_night_turbulence_comparison(self):
        """Daytime should have stronger turbulence overall."""
        altitudes = np.array([10, 100, 500, 1000])

        cn2_day = slc_day_cn2(altitudes)
        cn2_night = slc_night_cn2(altitudes)

        # Daytime should be stronger on average in boundary layer
        assert np.mean(cn2_day) > np.mean(cn2_night)


# =============================================================================
# Phase-Aware Functions Tests
# =============================================================================

class TestPhaseVariance:
    """Tests for phase variance calculation."""

    def test_basic_output(self):
        """Test basic phase variance calculation."""
        sigma_phi2 = phase_variance(0.5e-6, 1e-13, 0.5)
        assert sigma_phi2 > 0

    def test_aperture_dependence(self):
        """Larger aperture should have more phase variance."""
        sigma_small = phase_variance(0.5e-6, 1e-13, 0.1)
        sigma_large = phase_variance(0.5e-6, 1e-13, 1.0)
        assert sigma_large > sigma_small

    def test_wavelength_dependence(self):
        """Shorter wavelength should have more phase variance."""
        sigma_visible = phase_variance(0.5e-6, 1e-13, 0.5)
        sigma_ir = phase_variance(2.0e-6, 1e-13, 0.5)
        assert sigma_visible > sigma_ir


class TestTiltRemovedPhaseVariance:
    """Tests for tilt-removed phase variance."""

    def test_basic_output(self):
        """Test basic tilt-removed variance calculation."""
        sigma_tr = tilt_removed_phase_variance(0.5e-6, 1e-13, 0.5)
        assert sigma_tr > 0

    def test_less_than_total(self):
        """Tilt-removed should be less than total variance."""
        sigma_total = phase_variance(0.5e-6, 1e-13, 0.5)
        sigma_tr = tilt_removed_phase_variance(0.5e-6, 1e-13, 0.5)
        assert sigma_tr < sigma_total

    def test_noll_ratio(self):
        """Tilt-removed should be ~13% of total (Noll 1976)."""
        sigma_total = phase_variance(0.5e-6, 1e-13, 0.5)
        sigma_tr = tilt_removed_phase_variance(0.5e-6, 1e-13, 0.5)
        ratio = sigma_tr / sigma_total
        # 0.134 / 1.03 ~ 0.13
        assert np.isclose(ratio, 0.134 / 1.03, rtol=0.1)


class TestAngleOfArrivalVariance:
    """Tests for angle-of-arrival variance."""

    def test_basic_output(self):
        """Test basic angle-of-arrival calculation."""
        sigma_alpha2 = angle_of_arrival_variance(0.5e-6, 1e-13, 0.5)
        assert sigma_alpha2 > 0

    def test_aperture_effect(self):
        """Larger aperture should have smaller angle variance."""
        sigma_small = angle_of_arrival_variance(0.5e-6, 1e-13, 0.1)
        sigma_large = angle_of_arrival_variance(0.5e-6, 1e-13, 1.0)
        # Image motion is averaged over larger aperture
        assert sigma_large < sigma_small


class TestZernikeVariance:
    """Tests for Zernike coefficient variance."""

    def test_basic_output(self):
        """Test basic Zernike variance calculation."""
        for j in [1, 2, 3, 4, 5]:
            sigma_j2 = zernike_variance(j, 0.5e-6, 1e-13, 0.5)
            assert sigma_j2 > 0

    def test_piston_largest(self):
        """Piston (j=1) should have largest variance."""
        sigma_1 = zernike_variance(1, 0.5e-6, 1e-13, 0.5)
        sigma_2 = zernike_variance(2, 0.5e-6, 1e-13, 0.5)
        sigma_4 = zernike_variance(4, 0.5e-6, 1e-13, 0.5)
        assert sigma_1 > sigma_2
        assert sigma_2 > sigma_4

    def test_tip_tilt_equal(self):
        """Tip and tilt should have equal variance."""
        sigma_tip = zernike_variance(2, 0.5e-6, 1e-13, 0.5)
        sigma_tilt = zernike_variance(3, 0.5e-6, 1e-13, 0.5)
        assert np.isclose(sigma_tip, sigma_tilt)

    def test_invalid_index_raises(self):
        """Invalid Zernike index should raise error."""
        with pytest.raises(ValueError):
            zernike_variance(0, 0.5e-6, 1e-13, 0.5)


class TestPhasePowerSpectrum:
    """Tests for phase power spectrum."""

    def test_basic_output(self):
        """Test basic phase spectrum output."""
        f = np.logspace(-2, 3, 50)
        W_phi = phase_power_spectrum(f, 0.5e-6, 1e-13)
        assert all(W_phi >= 0)

    def test_power_law(self):
        """Spectrum should follow f^(-11/3) power law."""
        f = np.array([0.1, 1.0, 10.0])
        W_phi = phase_power_spectrum(f, 0.5e-6, 1e-13)

        # Check slope
        log_f = np.log10(f)
        log_W = np.log10(W_phi)
        slope = (log_W[-1] - log_W[0]) / (log_f[-1] - log_f[0])
        assert np.isclose(slope, -11/3, rtol=0.05)


class TestResidualPhaseVarianceAO:
    """Tests for AO residual phase variance."""

    def test_basic_output(self):
        """Test basic residual variance calculation."""
        for n in [1, 3, 5, 10]:
            sigma_res = residual_phase_variance_ao(n, 0.5e-6, 1e-13, 0.5)
            assert sigma_res > 0

    def test_decreases_with_modes(self):
        """More corrected modes -> less residual."""
        sigma_1 = residual_phase_variance_ao(1, 0.5e-6, 1e-13, 0.5)
        sigma_3 = residual_phase_variance_ao(3, 0.5e-6, 1e-13, 0.5)
        sigma_10 = residual_phase_variance_ao(10, 0.5e-6, 1e-13, 0.5)
        assert sigma_1 > sigma_3
        assert sigma_3 > sigma_10


class TestLongExposureStrehl:
    """Tests for long-exposure Strehl ratio."""

    def test_basic_output(self):
        """Test basic Strehl calculation."""
        S = long_exposure_strehl(0.5e-6, 1e-13, 0.5)
        assert 0 <= S <= 1

    def test_large_r0_high_strehl(self):
        """Good seeing (large r0) should give high Strehl."""
        # Weak turbulence, small aperture
        S = long_exposure_strehl(2.0e-6, 1e-15, 0.1)
        assert S > 0.5

    def test_small_r0_low_strehl(self):
        """Poor seeing (small r0) should give low Strehl."""
        # Strong turbulence, large aperture
        S = long_exposure_strehl(0.5e-6, 1e-12, 2.0)
        assert S < 0.1

    def test_bounds(self):
        """Strehl should always be bounded [0, 1]."""
        for cn2 in [1e-16, 1e-14, 1e-12]:
            for D in [0.1, 0.5, 2.0]:
                S = long_exposure_strehl(0.5e-6, cn2, D)
                assert 0 <= S <= 1
