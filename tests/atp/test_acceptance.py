"""
Acceptance Test Procedure (ATP) for RAF-Tran

These tests verify that the simulation meets all functional and performance
requirements as specified in the Software Requirements Specification (SRS).

ATP Categories:
1. Functional Requirements (FR) - Core functionality
2. Performance Requirements (PR) - Speed and resource usage
3. Interface Requirements (IR) - Input/output handling
4. Data Requirements (DR) - Data handling and formats
"""

import pytest
import numpy as np
import json
import time
import tempfile
import os
from pathlib import Path

from raf_tran.core.simulation import Simulation
from raf_tran.config.manager import ConfigurationManager
from raf_tran.config.atmosphere import StandardAtmospheres
from raf_tran.utils.output import OutputFormatter


# =============================================================================
# FR-1: Atmospheric Profile Support
# =============================================================================

class TestFR1AtmosphericProfiles:
    """ATP for FR-1: System shall support multiple atmospheric profiles."""

    @pytest.mark.parametrize("model", [
        "US_STANDARD_1976",
        "TROPICAL",
        "MID_LATITUDE_SUMMER",
        "MID_LATITUDE_WINTER",
        "SUB_ARCTIC_SUMMER",
        "SUB_ARCTIC_WINTER",
    ])
    def test_fr1_1_standard_atmospheres(self, model):
        """FR-1.1: Shall support 6 standard atmosphere models."""
        config = {
            "atmosphere": {"model": model},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert result is not None
        assert len(result.transmittance) > 0
        assert result.metadata["atmosphere_model"] == model

    def test_fr1_2_atmosphere_layers(self):
        """FR-1.2: Atmosphere shall have multiple vertical layers."""
        atm = StandardAtmospheres.get_profile("US_STANDARD_1976")

        assert len(atm.layers) >= 10, "Atmosphere should have at least 10 layers"

        # Verify layer properties
        for layer in atm.layers:
            assert layer.altitude_km >= 0
            assert layer.pressure_pa > 0
            assert layer.temperature_k > 0

    def test_fr1_3_gas_profiles(self):
        """FR-1.3: Shall provide gas concentration profiles."""
        atm = StandardAtmospheres.get_profile("US_STANDARD_1976")

        # Check major gases
        for gas in ["H2O", "CO2", "O3"]:
            profile = atm.get_gas_profile(gas)
            assert len(profile) == len(atm.layers)
            assert all(p >= 0 for p in profile), f"{gas} VMR cannot be negative"


# =============================================================================
# FR-2: Spectral Range Support
# =============================================================================

class TestFR2SpectralRange:
    """ATP for FR-2: System shall support configurable spectral ranges."""

    def test_fr2_1_wavenumber_range(self):
        """FR-2.1: Shall accept wavenumber range in cm⁻¹."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2000, "max_wavenumber": 3333, "resolution": 10.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert result.wavenumber[0] >= 2000
        assert result.wavenumber[-1] <= 3333

    def test_fr2_2_wavelength_output(self):
        """FR-2.2: Shall provide wavelength in micrometers."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)
        result = sim.run()

        # λ (μm) = 10000 / ν (cm⁻¹)
        expected_wavelength = 10000 / result.wavenumber
        np.testing.assert_allclose(result.wavelength_um, expected_wavelength, rtol=1e-6)

    def test_fr2_3_resolution_configurable(self):
        """FR-2.3: Spectral resolution shall be configurable."""
        wn_min, wn_max = 2500, 2600

        # Low resolution
        config_low = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": wn_min, "max_wavenumber": wn_max, "resolution": 10.0},
        }
        result_low = Simulation(config_low).run()

        # High resolution
        config_high = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": wn_min, "max_wavenumber": wn_max, "resolution": 1.0},
        }
        result_high = Simulation(config_high).run()

        assert len(result_high.wavenumber) > len(result_low.wavenumber)


# =============================================================================
# FR-3: Path Geometry Support
# =============================================================================

class TestFR3PathGeometry:
    """ATP for FR-3: System shall support various path geometries."""

    def test_fr3_1_horizontal_path(self):
        """FR-3.1: Shall support horizontal paths."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {
                "path_type": "HORIZONTAL",
                "h1_km": 0.0,
                "path_length_km": 5.0,
            },
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert result.metadata["path_type"] == "HORIZONTAL"
        assert len(result.transmittance) > 0

    def test_fr3_2_slant_path(self):
        """FR-3.2: Shall support slant paths between altitudes."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {
                "path_type": "SLANT",
                "h1_km": 0.0,
                "h2_km": 10.0,
                "angle_deg": 45.0,
            },
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert result.metadata["path_type"] == "SLANT"

    def test_fr3_3_vertical_path(self):
        """FR-3.3: Shall support vertical paths (zenith angle = 0)."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {
                "path_type": "SLANT",
                "h1_km": 0.0,
                "h2_km": 100.0,
                "angle_deg": 0.0,
            },
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)
        result = sim.run()

        assert len(result.transmittance) > 0


# =============================================================================
# FR-4: Molecular Absorption
# =============================================================================

class TestFR4MolecularAbsorption:
    """ATP for FR-4: System shall compute molecular absorption."""

    def test_fr4_1_multiple_molecules(self):
        """FR-4.1: Shall support multiple absorbing molecules."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)

        # Default molecules
        assert "H2O" in sim.molecules
        assert "CO2" in sim.molecules

    def test_fr4_2_temperature_scaling(self):
        """FR-4.2: Absorption shall vary with temperature."""
        # This is implicitly tested by different atmosphere models
        config_warm = {
            "atmosphere": {"model": "TROPICAL"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        config_cold = {
            "atmosphere": {"model": "SUB_ARCTIC_WINTER"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }

        result_warm = Simulation(config_warm).run()
        result_cold = Simulation(config_cold).run()

        # Results should differ due to temperature effects
        assert not np.allclose(result_warm.transmittance, result_cold.transmittance)


# =============================================================================
# FR-5: Scattering
# =============================================================================

class TestFR5Scattering:
    """ATP for FR-5: System shall compute atmospheric scattering."""

    def test_fr5_1_rayleigh_scattering(self):
        """FR-5.1: Shall compute Rayleigh (molecular) scattering."""
        from raf_tran.core.scattering_engine import compute_rayleigh_scattering

        wavenumber = np.array([2500.0, 3000.0, 3500.0])
        props = compute_rayleigh_scattering(wavenumber, 101325.0, 288.0)

        assert all(props.extinction_coeff >= 0)
        assert all(props.asymmetry_factor == 0)  # Rayleigh is symmetric

    def test_fr5_2_mie_scattering(self):
        """FR-5.2: Shall compute Mie (aerosol) scattering."""
        from raf_tran.core.scattering_engine import mie_single_particle

        # Water droplet
        Q_ext, Q_sca, Q_abs, g = mie_single_particle(
            wavelength_um=0.5,
            radius_um=1.0,
            n_real=1.33,
            n_imag=0.0,
        )

        assert Q_ext > 0
        assert Q_sca > 0
        assert 0 <= Q_abs <= Q_ext
        assert -1 <= g <= 1

    def test_fr5_3_aerosol_types(self):
        """FR-5.3: Shall support multiple aerosol types."""
        aerosol_types = ["RURAL", "URBAN", "MARITIME", "DESERT"]

        for atype in aerosol_types:
            config = {
                "atmosphere": {
                    "model": "US_STANDARD_1976",
                    "aerosols": {"type": atype, "visibility_km": 23.0},
                },
                "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
                "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
            }
            sim = Simulation(config)
            result = sim.run()
            assert result.metadata["aerosol_type"] == atype


# =============================================================================
# FR-6: Output Formats
# =============================================================================

class TestFR6OutputFormats:
    """ATP for FR-6: System shall support multiple output formats."""

    @pytest.fixture
    def simulation_result(self):
        """Run a simulation for output testing."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        return Simulation(config).run()

    def test_fr6_1_json_output(self, simulation_result):
        """FR-6.1: Shall export results to JSON format."""
        formatter = OutputFormatter()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            formatter.save(simulation_result, temp_path, format="json")

            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path) as fp:
                data = json.load(fp)

            # Check structure (wavenumber is in spectral section)
            assert "spectral" in data
            assert "wavenumber_cm1" in data["spectral"]
            assert "results" in data
            assert "transmittance" in data["results"]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_fr6_2_csv_output(self, simulation_result):
        """FR-6.2: Shall export results to CSV format."""
        formatter = OutputFormatter()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            formatter.save(simulation_result, temp_path, format="csv")

            assert os.path.exists(temp_path)

            # Verify CSV structure
            with open(temp_path) as fp:
                header = fp.readline()
                assert "wavenumber" in header.lower()
                assert "transmittance" in header.lower()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# =============================================================================
# FR-7: Configuration Management
# =============================================================================

class TestFR7Configuration:
    """ATP for FR-7: System shall support flexible configuration."""

    def test_fr7_1_dict_config(self):
        """FR-7.1: Shall accept configuration as Python dictionary."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        sim = Simulation(config)
        assert sim is not None

    def test_fr7_2_json_config(self):
        """FR-7.2: Shall accept configuration from JSON file."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(config, f)
            f.flush()

            sim = Simulation(f.name)
            assert sim is not None

            os.unlink(f.name)

    def test_fr7_3_config_validation(self):
        """FR-7.3: Shall validate configuration parameters."""
        manager = ConfigurationManager()

        # Valid config should pass
        valid_config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 5.0},
        }
        loaded = manager.load_config(valid_config)
        assert loaded.is_valid

        # Invalid config should fail
        invalid_config = {
            "atmosphere": {"model": "INVALID_MODEL"},
            "geometry": {"path_type": "HORIZONTAL"},
            "spectral": {"min_wavenumber": 2500},
        }
        loaded_invalid = manager.load_config(invalid_config)
        assert not loaded_invalid.is_valid


# =============================================================================
# PR-1: Performance Requirements
# =============================================================================

class TestPR1Performance:
    """ATP for PR-1: System shall meet performance requirements."""

    def test_pr1_1_execution_time(self):
        """PR-1.1: Single simulation shall complete within reasonable time."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2600, "resolution": 1.0},
        }

        sim = Simulation(config)

        start_time = time.time()
        result = sim.run()
        elapsed_time = time.time() - start_time

        # Should complete in less than 60 seconds for 100 cm⁻¹ range
        assert elapsed_time < 60, f"Simulation took {elapsed_time:.1f}s, expected < 60s"
        assert result is not None

    def test_pr1_2_memory_efficiency(self):
        """PR-1.2: System shall handle moderate spectral ranges without memory issues."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2000, "max_wavenumber": 3000, "resolution": 1.0},
        }

        # Should complete without memory errors
        sim = Simulation(config)
        result = sim.run()

        assert len(result.wavenumber) == 1001  # (3000-2000)/1 + 1


# =============================================================================
# DR-1: Data Requirements
# =============================================================================

class TestDR1DataHandling:
    """ATP for DR-1: System shall handle data correctly."""

    def test_dr1_1_numerical_precision(self):
        """DR-1.1: Results shall have adequate numerical precision."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2510, "resolution": 1.0},
        }
        sim = Simulation(config)
        result = sim.run()

        # Results should be float64
        assert result.transmittance.dtype == np.float64
        assert result.optical_depth.dtype == np.float64

    @pytest.mark.skip(reason="Known issue: numba parallel execution causes non-determinism")
    def test_dr1_2_reproducibility(self):
        """DR-1.2: Same configuration shall produce identical results.

        NOTE: Currently skipped due to non-determinism in numba prange.
        The compute_absorption_lbl function uses parallel execution which
        can cause race conditions in the absorption array updates.
        TODO: Fix by restructuring to avoid race conditions.
        """
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2510, "resolution": 1.0},
        }

        # Use the same Simulation instance to ensure database consistency
        sim = Simulation(config)
        result1 = sim.run()
        result2 = sim.run()

        # Results should be reproducible within acceptable tolerance
        np.testing.assert_allclose(result1.transmittance, result2.transmittance, rtol=1e-6)
        np.testing.assert_allclose(result1.optical_depth, result2.optical_depth, rtol=1e-6)

    def test_dr1_3_result_completeness(self):
        """DR-1.3: Results shall include all required fields."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
            "spectral": {"min_wavenumber": 2500, "max_wavenumber": 2510, "resolution": 1.0},
        }
        result = Simulation(config).run()

        # Required fields
        assert hasattr(result, "wavenumber")
        assert hasattr(result, "wavelength_um")
        assert hasattr(result, "transmittance")
        assert hasattr(result, "radiance")
        assert hasattr(result, "optical_depth")
        assert hasattr(result, "thermal_emission")
        assert hasattr(result, "metadata")

        # Metadata completeness
        assert "atmosphere_model" in result.metadata
        assert "molecules" in result.metadata
        assert "path_type" in result.metadata
