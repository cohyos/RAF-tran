"""
Physics Validity Checks for RAF-Tran

These tests verify that the simulation produces physically plausible results
by checking against known atmospheric radiative transfer principles.

Reference: MODTRAN, libRadtran, and standard atmospheric physics.
"""

import pytest
import numpy as np
from raf_tran.core.simulation import Simulation
from raf_tran.config.atmosphere import StandardAtmospheres
from raf_tran.core.scattering_engine import compute_rayleigh_scattering


class TestTransmittancePhysics:
    """Verify transmittance follows physical laws."""

    def test_transmittance_bounds(self):
        """Transmittance must be between 0 and 1."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 5.0},
            "spectral": {"min_wavenumber": 2000, "max_wavenumber": 3000, "resolution": 2.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert np.all(result.transmittance >= 0), "Transmittance cannot be negative"
        assert np.all(result.transmittance <= 1), "Transmittance cannot exceed 1"

    def test_longer_path_lower_transmittance(self):
        """Longer optical paths should have lower transmittance (Beer-Lambert law)."""
        base_config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 2.0},
        }

        # Short path
        config_short = base_config.copy()
        config_short["geometry"] = {**base_config["geometry"], "path_length_km": 1.0}
        sim_short = Simulation(config_short)
        result_short = sim_short.run()

        # Long path
        config_long = base_config.copy()
        config_long["geometry"] = {**base_config["geometry"], "path_length_km": 10.0}
        sim_long = Simulation(config_long)
        result_long = sim_long.run()

        # Mean transmittance should decrease with path length
        assert np.mean(result_long.transmittance) < np.mean(result_short.transmittance), \
            "Longer path should have lower mean transmittance"

    def test_optical_depth_path_scaling(self):
        """Optical depth should scale linearly with path length."""
        base_config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 2.0},
        }

        # Path 1 km
        config_1km = base_config.copy()
        config_1km["geometry"] = {**base_config["geometry"], "path_length_km": 1.0}
        sim_1km = Simulation(config_1km)
        result_1km = sim_1km.run()

        # Path 2 km
        config_2km = base_config.copy()
        config_2km["geometry"] = {**base_config["geometry"], "path_length_km": 2.0}
        sim_2km = Simulation(config_2km)
        result_2km = sim_2km.run()

        # Optical depth should roughly double (within numerical tolerance)
        ratio = np.mean(result_2km.optical_depth) / np.mean(result_1km.optical_depth)
        assert 1.5 < ratio < 2.5, f"Optical depth ratio should be ~2, got {ratio:.2f}"

    def test_transmittance_optical_depth_relation(self):
        """Verify T = exp(-tau) relationship."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 2.0},
        }
        sim = Simulation(config)
        result = sim.run()

        # For moderate optical depths, check Beer-Lambert
        moderate_tau = (result.optical_depth > 0.01) & (result.optical_depth < 10)
        if np.any(moderate_tau):
            expected_trans = np.exp(-result.optical_depth[moderate_tau])
            actual_trans = result.transmittance[moderate_tau]
            np.testing.assert_allclose(
                actual_trans, expected_trans, rtol=0.1,
                err_msg="Transmittance should follow Beer-Lambert law"
            )


class TestAbsorptionPhysics:
    """Verify absorption calculations are physically correct."""

    def test_h2o_absorption_band(self):
        """H2O should show absorption over longer atmospheric path."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 10.0},
            "spectral": {"min_wavenumber": 2000, "max_wavenumber": 3000, "resolution": 2.0},
        }
        sim = Simulation(config)
        result = sim.run()

        # With synthetic database, absorption may be weaker than real HITRAN
        # But should see some absorption with longer path (transmittance < 1.0)
        min_trans = np.min(result.transmittance)
        assert min_trans < 0.999, \
            f"Should see some H2O absorption, got min transmittance={min_trans:.6f}"

    def test_co2_absorption_band(self):
        """CO2 should show absorption around 4.3 μm (~2350 cm⁻¹)."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2300, "max_wavenumber": 2400, "resolution": 2.0},
        }
        sim = Simulation(config)
        result = sim.run()

        # CO2 band should show absorption
        mean_trans = np.mean(result.transmittance)
        assert mean_trans < 0.95, f"CO2 band should show absorption, got mean T={mean_trans:.3f}"

    def test_window_region_high_transmittance(self):
        """Atmospheric windows should have high transmittance."""
        # 3-5 μm window region (~2500-3000 cm⁻¹, avoiding strong bands)
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2700, "resolution": 2.0},
        }
        sim = Simulation(config)
        result = sim.run()

        # Window should have some high transmittance regions
        max_trans = np.max(result.transmittance)
        assert max_trans > 0.7, f"Window region should have high transmittance, got max T={max_trans:.3f}"


class TestScatteringPhysics:
    """Verify scattering calculations are physically correct."""

    def test_rayleigh_wavelength_dependence(self):
        """Rayleigh scattering should scale as λ⁻⁴."""
        # Test at two wavelengths
        wn1 = np.array([2000.0])  # 5 μm
        wn2 = np.array([4000.0])  # 2.5 μm

        props1 = compute_rayleigh_scattering(wn1, 101325.0, 288.0)
        props2 = compute_rayleigh_scattering(wn2, 101325.0, 288.0)

        # Ratio should be (λ1/λ2)^4 = (5/2.5)^4 = 16
        # Or in terms of wavenumber: (wn2/wn1)^4 = 16
        ratio = props2.extinction_coeff[0] / props1.extinction_coeff[0]
        expected_ratio = (4000 / 2000) ** 4

        np.testing.assert_allclose(
            ratio, expected_ratio, rtol=0.1,
            err_msg=f"Rayleigh should scale as λ⁻⁴, got ratio {ratio:.1f}, expected {expected_ratio:.1f}"
        )

    def test_rayleigh_pressure_scaling(self):
        """Rayleigh scattering should scale linearly with pressure."""
        wn = np.array([3000.0])

        props_1atm = compute_rayleigh_scattering(wn, 101325.0, 288.0)
        props_half = compute_rayleigh_scattering(wn, 50662.5, 288.0)

        ratio = props_1atm.extinction_coeff[0] / props_half.extinction_coeff[0]
        np.testing.assert_allclose(
            ratio, 2.0, rtol=0.05,
            err_msg="Rayleigh scattering should scale linearly with pressure"
        )

    def test_aerosol_reduces_transmittance(self):
        """Aerosols should reduce transmittance."""
        base_config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 5.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 2.0},
        }

        # Without aerosols
        config_clear = base_config.copy()
        config_clear["atmosphere"] = {
            "model": "US_STANDARD_1976",
            "aerosols": {"type": "NONE"}
        }
        sim_clear = Simulation(config_clear)
        result_clear = sim_clear.run()

        # With aerosols (low visibility)
        config_hazy = base_config.copy()
        config_hazy["atmosphere"] = {
            "model": "US_STANDARD_1976",
            "aerosols": {"type": "RURAL", "visibility_km": 5.0}
        }
        sim_hazy = Simulation(config_hazy)
        result_hazy = sim_hazy.run()

        assert np.mean(result_hazy.transmittance) < np.mean(result_clear.transmittance), \
            "Aerosols should reduce mean transmittance"


class TestAtmospherePhysics:
    """Verify atmospheric profile physics."""

    def test_pressure_decreases_with_altitude(self):
        """Pressure should decrease with altitude."""
        atm = StandardAtmospheres.get_profile("US_STANDARD_1976")

        pressures = [layer.pressure_pa for layer in atm.layers]
        altitudes = [layer.altitude_km for layer in atm.layers]

        # Pressure should monotonically decrease with altitude
        for i in range(1, len(pressures)):
            if altitudes[i] > altitudes[i-1]:
                assert pressures[i] <= pressures[i-1], \
                    f"Pressure should decrease with altitude: P({altitudes[i]} km) > P({altitudes[i-1]} km)"

    def test_tropical_warmer_than_standard(self):
        """Tropical atmosphere should be warmer than US Standard at surface."""
        std = StandardAtmospheres.us_standard_1976()
        trop = StandardAtmospheres.tropical()

        std_surface_temp = std.layers[0].temperature_k
        trop_surface_temp = trop.layers[0].temperature_k

        assert trop_surface_temp > std_surface_temp, \
            f"Tropical ({trop_surface_temp} K) should be warmer than US Standard ({std_surface_temp} K)"

    def test_winter_colder_than_summer(self):
        """Winter atmosphere should be colder than summer."""
        summer = StandardAtmospheres.mid_latitude_summer()
        winter = StandardAtmospheres.mid_latitude_winter()

        summer_temp = summer.layers[0].temperature_k
        winter_temp = winter.layers[0].temperature_k

        assert winter_temp < summer_temp, \
            f"Winter ({winter_temp} K) should be colder than summer ({summer_temp} K)"


class TestEnergyConservation:
    """Verify energy conservation principles."""

    def test_radiance_positive(self):
        """Radiance must be non-negative."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 3000, "resolution": 2.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert np.all(result.radiance >= 0), "Radiance cannot be negative"

    def test_thermal_emission_increases_with_temperature(self):
        """Warmer atmospheres should emit more thermal radiation."""
        base_config = {
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 3000, "resolution": 2.0},
        }

        # Tropical (warmer)
        config_warm = base_config.copy()
        config_warm["atmosphere"] = {"model": "TROPICAL"}
        sim_warm = Simulation(config_warm)
        result_warm = sim_warm.run()

        # Subarctic winter (colder)
        config_cold = base_config.copy()
        config_cold["atmosphere"] = {"model": "SUB_ARCTIC_WINTER"}
        sim_cold = Simulation(config_cold)
        result_cold = sim_cold.run()

        # Thermal emission should be higher for warmer atmosphere
        warm_emission = np.mean(result_warm.thermal_emission)
        cold_emission = np.mean(result_cold.thermal_emission)

        assert warm_emission >= cold_emission, \
            f"Warmer atmosphere should emit more: tropical={warm_emission:.2e}, subarctic={cold_emission:.2e}"
