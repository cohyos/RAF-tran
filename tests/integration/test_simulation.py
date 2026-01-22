"""
Integration tests for full simulation pipeline.

Tests the complete workflow from configuration to output,
matching the benchmark scenarios from SRS Section 6.2.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from raf_tran import Simulation, SimulationConfig
from raf_tran.config.manager import ConfigurationManager


class TestSimulationBasic:
    """Basic simulation functionality tests."""

    def test_simulation_creation(self):
        """Test simulation can be created with default config."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "spectral": {
                "min_wavenumber": 2000,
                "max_wavenumber": 2100,
                "resolution": 1.0,
            },
        }
        sim = Simulation(config)
        assert sim is not None
        assert sim.config.atmosphere.model == "US_STANDARD_1976"

    def test_simulation_from_config_object(self):
        """Test simulation with SimulationConfig object."""
        config = SimulationConfig()
        config.atmosphere.model = "TROPICAL"
        config.spectral.min_wavenumber = 2500
        config.spectral.max_wavenumber = 2600

        sim = Simulation(config)
        assert sim.config.atmosphere.model == "TROPICAL"

    def test_quick_transmittance(self):
        """Test quick transmittance calculation."""
        wavenumber, transmittance = Simulation.quick_transmittance(
            wavenumber_range=(2000, 2100),
            atmosphere_model="US_STANDARD_1976",
            path_length_km=1.0,
            resolution=1.0,
        )

        assert len(wavenumber) > 0
        assert len(transmittance) == len(wavenumber)
        assert all(0 <= t <= 1 for t in transmittance)


class TestBenchmarkScenario1:
    """
    Benchmark Scenario 1: Horizontal path at sea level (Baseline test).

    From SRS Section 6.2:
    - Geometry: 0 km altitude, 1 km horizontal path
    - Atmosphere: US Standard 1976
    - Aerosols: None (gas absorption only)
    - Spectral: MWIR band
    - Criterion: RMSE < 0.01 in transmittance
    """

    @pytest.fixture
    def baseline_config(self):
        """Configuration for baseline test."""
        return {
            "simulation_config": {
                "offline_mode": True,
            },
            "atmosphere": {
                "model": "US_STANDARD_1976",
                "aerosols": {"type": "NONE"},
            },
            "geometry": {
                "path_type": "HORIZONTAL",
                "h1_km": 0.0,
                "path_length_km": 1.0,
            },
            "spectral": {
                "min_wavenumber": 2000,
                "max_wavenumber": 3333,
                "resolution": 1.0,  # Lower resolution for faster testing
            },
        }

    def test_baseline_runs(self, baseline_config):
        """Test that baseline scenario executes successfully."""
        sim = Simulation(baseline_config)
        result = sim.run()

        assert result is not None
        assert len(result.transmittance) > 0
        assert result.metadata["atmosphere_model"] == "US_STANDARD_1976"
        assert result.metadata["aerosol_type"] == "NONE"

    def test_baseline_transmittance_range(self, baseline_config):
        """Test that transmittance values are physical."""
        sim = Simulation(baseline_config)
        result = sim.run()

        # Transmittance must be between 0 and 1
        assert all(result.transmittance >= 0)
        assert all(result.transmittance <= 1)

    def test_baseline_optical_depth_positive(self, baseline_config):
        """Test that optical depth is non-negative."""
        sim = Simulation(baseline_config)
        result = sim.run()

        assert all(result.optical_depth >= 0)

    def test_baseline_spectral_features(self, baseline_config):
        """Test that spectral features are present (absorption bands)."""
        sim = Simulation(baseline_config)
        result = sim.run()

        # There should be variation in transmittance (absorption features)
        trans_std = np.std(result.transmittance)
        assert trans_std > 0.01, "Should see absorption features"


class TestBenchmarkScenario2:
    """
    Benchmark Scenario 2: Aerosol loading test.

    From SRS Section 6.2:
    - Geometry: 0 km altitude, horizontal path
    - Atmosphere: US Standard 1976
    - Aerosols: Rural, visibility 5 km
    - Tests scattering engine
    """

    @pytest.fixture
    def aerosol_config(self):
        """Configuration for aerosol test."""
        return {
            "atmosphere": {
                "model": "US_STANDARD_1976",
                "aerosols": {
                    "type": "RURAL",
                    "visibility_km": 5.0,
                },
            },
            "geometry": {
                "path_type": "HORIZONTAL",
                "h1_km": 0.0,
                "path_length_km": 1.0,
            },
            "spectral": {
                "min_wavenumber": 2000,
                "max_wavenumber": 2500,
                "resolution": 2.0,
            },
        }

    def test_aerosol_effect(self, aerosol_config):
        """Test that aerosols reduce transmittance."""
        # With aerosols
        sim_with_aerosol = Simulation(aerosol_config)
        result_with = sim_with_aerosol.run()

        # Without aerosols
        aerosol_config["atmosphere"]["aerosols"]["type"] = "NONE"
        sim_no_aerosol = Simulation(aerosol_config)
        result_without = sim_no_aerosol.run()

        # Aerosols should reduce mean transmittance
        mean_with = np.mean(result_with.transmittance)
        mean_without = np.mean(result_without.transmittance)
        assert mean_with < mean_without


class TestBenchmarkScenario3:
    """
    Benchmark Scenario 3: Full scene (slant path, humid atmosphere).

    From SRS Section 6.2:
    - Geometry: Ground to 10 km, slant path
    - Atmosphere: Mid-Latitude Summer
    - Aerosols: Rural, visibility 23 km
    - Full integration test
    """

    @pytest.fixture
    def full_scene_config(self):
        """Configuration for full scene test."""
        return {
            "atmosphere": {
                "model": "MID_LATITUDE_SUMMER",
                "aerosols": {
                    "type": "RURAL",
                    "visibility_km": 23.0,
                },
            },
            "geometry": {
                "path_type": "SLANT",
                "h1_km": 0.0,
                "h2_km": 10.0,
                "angle_deg": 45.0,
            },
            "spectral": {
                "min_wavenumber": 2000,
                "max_wavenumber": 2200,
                "resolution": 2.0,
            },
        }

    def test_full_scene_runs(self, full_scene_config):
        """Test that full scene scenario executes."""
        sim = Simulation(full_scene_config)
        result = sim.run()

        assert result is not None
        assert result.metadata["atmosphere_model"] == "MID_LATITUDE_SUMMER"
        assert result.metadata["path_type"] == "SLANT"

    def test_slant_path_longer_optical_depth(self, full_scene_config):
        """Test that slant path has larger optical depth than vertical."""
        # Slant path
        sim_slant = Simulation(full_scene_config)
        result_slant = sim_slant.run()

        # Vertical path (same altitude range)
        full_scene_config["geometry"]["path_type"] = "VERTICAL"
        full_scene_config["geometry"]["angle_deg"] = 0.0
        sim_vertical = Simulation(full_scene_config)
        result_vertical = sim_vertical.run()

        # Slant path should have larger optical depth
        mean_tau_slant = np.mean(result_slant.optical_depth)
        mean_tau_vertical = np.mean(result_vertical.optical_depth)
        assert mean_tau_slant > mean_tau_vertical


class TestOutputFormats:
    """Test output formatting and file generation."""

    @pytest.fixture
    def simple_result(self):
        """Run a simple simulation for output testing."""
        config = {
            "atmosphere": {"model": "US_STANDARD_1976"},
            "spectral": {
                "min_wavenumber": 2000,
                "max_wavenumber": 2050,
                "resolution": 5.0,
            },
        }
        sim = Simulation(config)
        return sim.run()

    def test_json_output(self, simple_result):
        """Test JSON output format."""
        from raf_tran.utils.output import OutputFormatter
        formatter = OutputFormatter()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            result_path = formatter.save(simple_result, output_path, format="json")
            assert Path(result_path).exists()

            import json
            with open(result_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "results" in data
            assert "transmittance" in data["results"]
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_csv_output(self, simple_result):
        """Test CSV output format."""
        from raf_tran.utils.output import OutputFormatter
        formatter = OutputFormatter()

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_path = f.name

        try:
            result_path = formatter.save(simple_result, output_path, format="csv")
            assert Path(result_path).exists()

            # Check CSV has header and data
            with open(result_path) as f:
                lines = f.readlines()
            assert len(lines) > 1  # Header + data
            assert "transmittance" in lines[0]
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestConfigurationManager:
    """Test configuration management."""

    def test_example_config_creation(self):
        """Test example configuration generation."""
        manager = ConfigurationManager()
        example = manager.create_example_config()

        assert "simulation_config" in example
        assert "atmosphere" in example
        assert "geometry" in example
        assert "spectral" in example

    def test_config_validation(self):
        """Test configuration validation."""
        config = SimulationConfig()
        errors = config.validate()

        # Default config should be valid
        assert len(errors) == 0

    def test_invalid_config_detected(self):
        """Test that invalid configurations are detected."""
        config = SimulationConfig()
        config.spectral.min_wavenumber = 3000
        config.spectral.max_wavenumber = 2000  # Invalid: min > max

        errors = config.validate()
        assert len(errors) > 0
        assert any("min_wavenumber" in e for e in errors)
