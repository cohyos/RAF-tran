"""
Unit tests for atmosphere module.

Tests FR-01 (standard models) and FR-02 (custom profiles).
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import csv

from raf_tran.config.atmosphere import (
    AtmosphereProfile,
    AtmosphereLayer,
    StandardAtmospheres,
)


class TestAtmosphereLayer:
    """Tests for AtmosphereLayer dataclass."""

    def test_layer_creation(self):
        """Test basic layer creation."""
        layer = AtmosphereLayer(
            altitude_km=5.0,
            pressure_pa=50000.0,
            temperature_k=260.0,
            h2o_vmr=1000.0,
        )
        assert layer.altitude_km == 5.0
        assert layer.pressure_pa == 50000.0
        assert layer.temperature_k == 260.0
        assert layer.h2o_vmr == 1000.0

    def test_layer_defaults(self):
        """Test default values."""
        layer = AtmosphereLayer(
            altitude_km=0.0,
            pressure_pa=101325.0,
            temperature_k=288.0,
        )
        assert layer.co2_vmr == 420.0  # Current CO2 level
        assert layer.density_kg_m3 == 0.0  # Default


class TestAtmosphereProfile:
    """Tests for AtmosphereProfile."""

    @pytest.fixture
    def simple_profile(self):
        """Create a simple test profile."""
        layers = [
            AtmosphereLayer(0.0, 101325, 288.0, h2o_vmr=7750),
            AtmosphereLayer(5.0, 54000, 256.0, h2o_vmr=1500),
            AtmosphereLayer(10.0, 26500, 223.0, h2o_vmr=186),
        ]
        return AtmosphereProfile(name="TEST", layers=layers)

    def test_profile_properties(self, simple_profile):
        """Test profile property accessors."""
        assert simple_profile.num_layers == 3
        assert len(simple_profile.altitudes) == 3
        assert len(simple_profile.pressures) == 3
        assert len(simple_profile.temperatures) == 3

    def test_get_gas_profile(self, simple_profile):
        """Test gas profile retrieval."""
        h2o = simple_profile.get_gas_profile("H2O")
        assert len(h2o) == 3
        assert h2o[0] == 7750.0
        assert h2o[2] == 186.0

    def test_gas_override(self, simple_profile):
        """Test gas concentration override (FR-03)."""
        simple_profile.apply_gas_overrides({"CO2": 500.0})
        co2 = simple_profile.get_gas_profile("CO2")
        assert all(co2 == 500.0)

    def test_interpolation(self, simple_profile):
        """Test altitude interpolation."""
        new_alts = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        interp_profile = simple_profile.interpolate_to_altitudes(new_alts)

        assert interp_profile.num_layers == 5
        assert interp_profile.altitudes[2] == 5.0

    def test_unknown_gas_raises(self, simple_profile):
        """Test that unknown gas raises ValueError."""
        with pytest.raises(ValueError, match="Unknown gas"):
            simple_profile.get_gas_profile("UNKNOWN_GAS")


class TestStandardAtmospheres:
    """Tests for standard atmosphere models (FR-01)."""

    @pytest.mark.parametrize("model_name", [
        "US_STANDARD_1976",
        "TROPICAL",
        "MID_LATITUDE_SUMMER",
        "MID_LATITUDE_WINTER",
        "SUB_ARCTIC_SUMMER",
        "SUB_ARCTIC_WINTER",
    ])
    def test_all_models_load(self, model_name):
        """Test that all standard models can be loaded."""
        profile = StandardAtmospheres.get_profile(model_name)
        assert profile is not None
        assert profile.name == model_name
        assert profile.num_layers > 0

    def test_us_standard_sea_level(self):
        """Test US Standard 1976 sea level values."""
        profile = StandardAtmospheres.us_standard_1976()

        # Sea level values
        assert profile.layers[0].altitude_km == 0.0
        assert abs(profile.layers[0].pressure_pa - 101325) < 100
        assert abs(profile.layers[0].temperature_k - 288.15) < 0.5

    def test_tropical_higher_temperature(self):
        """Test that tropical is warmer at surface."""
        tropical = StandardAtmospheres.tropical()
        us_std = StandardAtmospheres.us_standard_1976()

        assert tropical.layers[0].temperature_k > us_std.layers[0].temperature_k

    def test_invalid_model_raises(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError, match="Unknown atmosphere model"):
            StandardAtmospheres.get_profile("INVALID_MODEL")

    def test_radiosonde_loading(self):
        """Test custom radiosonde CSV loading (FR-02)."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['altitude_km', 'pressure_pa', 'temperature_k', 'h2o_ppmv'])
            writer.writerow([0.0, 101325, 288.0, 7500])
            writer.writerow([1.0, 89876, 281.0, 6000])
            writer.writerow([2.0, 79500, 275.0, 4500])
            temp_path = f.name

        try:
            profile = StandardAtmospheres.load_radiosonde(temp_path)
            assert profile.num_layers == 3
            assert profile.name.startswith("RADIOSONDE_")
        finally:
            Path(temp_path).unlink()

    def test_radiosonde_not_found_raises(self):
        """Test that missing radiosonde file raises error."""
        with pytest.raises(FileNotFoundError):
            StandardAtmospheres.load_radiosonde("/nonexistent/file.csv")

    def test_altitude_ordering(self):
        """Test that all profiles have ascending altitudes."""
        for model_name in ["US_STANDARD_1976", "TROPICAL", "MID_LATITUDE_SUMMER"]:
            profile = StandardAtmospheres.get_profile(model_name)
            altitudes = profile.altitudes
            assert np.all(np.diff(altitudes) >= 0), f"{model_name} altitudes not ascending"
