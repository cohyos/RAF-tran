"""
Configuration Manager for RAF-Tran simulations.

Handles loading, validation, and application of simulation configurations.
Supports both standard atmosphere models and custom radiosonde profiles.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from raf_tran.config.settings import SimulationConfig
from raf_tran.config.atmosphere import AtmosphereProfile, StandardAtmospheres
from raf_tran.core.constants import ATMOSPHERE_MODELS, AEROSOL_TYPES

logger = logging.getLogger(__name__)


class OfflineModeError(Exception):
    """Raised when network access is attempted in offline mode (FR-13)."""
    pass


@dataclass
class LoadedConfiguration:
    """Container for fully loaded and validated configuration.

    Attributes:
        config: The simulation configuration settings
        atmosphere: Loaded atmosphere profile
        is_valid: Whether the configuration passed validation
        validation_errors: List of validation error messages
    """
    config: SimulationConfig
    atmosphere: AtmosphereProfile
    is_valid: bool
    validation_errors: list


class ConfigurationManager:
    """Manages simulation configurations (FR-01 through FR-06).

    This class handles:
    - Loading configurations from JSON/YAML/dict
    - Loading and validating atmosphere profiles
    - Applying gas concentration overrides
    - Enforcing offline mode restrictions (FR-13)

    Example:
        >>> manager = ConfigurationManager()
        >>> loaded = manager.load_config({
        ...     "atmosphere": {"model": "US_STANDARD_1976"},
        ...     "spectral": {"min_wavenumber": 2000, "max_wavenumber": 2500}
        ... })
        >>> if loaded.is_valid:
        ...     print(f"Loaded {loaded.atmosphere.num_layers} atmosphere layers")
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            base_path: Base path for relative file references.
                      Defaults to current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self._offline_mode = False

    @property
    def offline_mode(self) -> bool:
        """Whether offline mode is enabled."""
        return self._offline_mode

    @offline_mode.setter
    def offline_mode(self, value: bool):
        """Set offline mode and configure environment."""
        self._offline_mode = value
        if value:
            # Set environment variables to prevent network access (FR-13)
            os.environ["RAF_TRAN_OFFLINE_MODE"] = "True"
            os.environ["RADIS_OFFLINE_MODE"] = "True"
            logger.info("Offline mode enabled - network access blocked")

    def _check_network_access(self) -> None:
        """Check if network access is allowed.

        Raises:
            OfflineModeError: If offline mode is enabled (FR-13)
        """
        if self._offline_mode:
            raise OfflineModeError(
                "Network access is blocked in offline mode. "
                "Please use local database files only."
            )

    def load_config(
        self,
        config_source: Dict[str, Any] | str,
    ) -> LoadedConfiguration:
        """Load and validate a complete configuration.

        Args:
            config_source: Configuration dictionary, JSON path, or YAML path

        Returns:
            LoadedConfiguration with parsed config and atmosphere profile
        """
        # Parse configuration
        if isinstance(config_source, dict):
            config = SimulationConfig.from_dict(config_source)
        elif isinstance(config_source, str):
            path = Path(config_source)
            if path.suffix.lower() == '.json':
                config = SimulationConfig.from_json(config_source)
            elif path.suffix.lower() in ('.yaml', '.yml'):
                config = SimulationConfig.from_yaml(config_source)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        else:
            raise TypeError(f"Invalid config source type: {type(config_source)}")

        # Set offline mode from config
        self.offline_mode = config.system.offline_mode

        # Validate configuration
        validation_errors = config.validate()

        # Load atmosphere profile
        try:
            atmosphere = self._load_atmosphere(config)
        except Exception as e:
            validation_errors.append(f"Atmosphere loading failed: {e}")
            atmosphere = StandardAtmospheres.us_standard_1976()  # Fallback

        # Apply gas concentration overrides (FR-03)
        if config.atmosphere.custom_concentrations:
            atmosphere.apply_gas_overrides(config.atmosphere.custom_concentrations)
            logger.info(
                f"Applied gas overrides: {config.atmosphere.custom_concentrations}"
            )

        is_valid = len(validation_errors) == 0

        if not is_valid:
            for error in validation_errors:
                logger.warning(f"Configuration validation error: {error}")

        return LoadedConfiguration(
            config=config,
            atmosphere=atmosphere,
            is_valid=is_valid,
            validation_errors=validation_errors,
        )

    def _load_atmosphere(self, config: SimulationConfig) -> AtmosphereProfile:
        """Load atmosphere profile based on configuration.

        Args:
            config: Simulation configuration

        Returns:
            Loaded AtmosphereProfile
        """
        # Check for custom radiosonde profile (FR-02)
        if config.atmosphere.custom_profile_path:
            profile_path = self.base_path / config.atmosphere.custom_profile_path
            logger.info(f"Loading custom radiosonde profile from {profile_path}")
            return StandardAtmospheres.load_radiosonde(str(profile_path))

        # Load standard atmosphere model (FR-01)
        logger.info(f"Loading standard atmosphere: {config.atmosphere.model}")
        return StandardAtmospheres.get_profile(config.atmosphere.model)

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_path.

        Implements NFR-07 for portable relative paths.

        Args:
            path: Relative or absolute path string

        Returns:
            Resolved absolute Path
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return (self.base_path / p).resolve()

    def validate_database_path(self, config: SimulationConfig) -> bool:
        """Validate that the spectral database exists.

        Args:
            config: Simulation configuration

        Returns:
            True if database file exists
        """
        db_path = self.resolve_path(config.system.database_path)
        return db_path.exists()

    @staticmethod
    def create_example_config() -> Dict[str, Any]:
        """Create an example configuration dictionary.

        Useful for documentation and testing.

        Returns:
            Example configuration matching the API specification
        """
        return {
            "simulation_config": {
                "offline_mode": True,
                "database_path": "./data/spectral_db/hitran_lines.h5",
                "use_gpu": False,
            },
            "atmosphere": {
                "model": "MID_LATITUDE_SUMMER",
                "aerosols": {
                    "type": "RURAL",
                    "visibility_km": 23.0,
                },
                "custom_concentrations": {
                    "CO2": 420.0,
                },
            },
            "geometry": {
                "path_type": "SLANT",
                "h1_km": 0,
                "h2_km": 10,
                "angle_deg": 45,
            },
            "spectral": {
                "min_wavenumber": 2000,
                "max_wavenumber": 2500,
                "resolution": 0.01,
            },
            "output": {
                "format": "json",
                "output_path": "./output",
                "quantities": ["transmittance", "radiance", "optical_depth"],
            },
        }

    def save_example_config(self, output_path: str) -> None:
        """Save an example configuration file.

        Args:
            output_path: Path to save the example JSON
        """
        example = self.create_example_config()
        with open(output_path, 'w') as f:
            json.dump(example, f, indent=2)
        logger.info(f"Saved example configuration to {output_path}")
