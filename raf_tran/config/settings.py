"""
Simulation configuration data structures.

Defines the complete configuration schema for RAF-Tran simulations,
matching the API specification in the SRS document.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from pathlib import Path
import json
import yaml


@dataclass
class SystemConfig:
    """System-level configuration settings.

    Attributes:
        offline_mode: If True, block all network access (FR-13)
        database_path: Path to local HDF5 spectral database (FR-12)
        use_gpu: Enable GPU acceleration for LBL calculations (NFR-02)
        num_threads: Number of CPU threads for parallel computation
        cache_size_mb: Maximum memory for caching spectral data
    """
    offline_mode: bool = True
    database_path: str = "./data/spectral_db/hitran_lines.h5"
    use_gpu: bool = False
    num_threads: int = 4
    cache_size_mb: int = 2048


@dataclass
class AerosolConfig:
    """Aerosol and visibility configuration.

    Attributes:
        type: Aerosol model type (NONE, RURAL, URBAN, MARITIME, DESERT) (FR-04)
        visibility_km: Visibility at ground level in km (FR-05)
        custom_properties: Optional custom aerosol optical properties
    """
    type: str = "NONE"
    visibility_km: float = 23.0
    custom_properties: Optional[Dict[str, Any]] = None


@dataclass
class CloudConfig:
    """Cloud layer configuration.

    Attributes:
        enabled: Whether clouds are included in simulation
        base_altitude_km: Cloud base altitude (FR-06)
        top_altitude_km: Cloud top altitude (FR-06)
        liquid_water_content: LWC in g/m³
        rain_rate_mm_hr: Precipitation rate (FR-06)
    """
    enabled: bool = False
    base_altitude_km: float = 2.0
    top_altitude_km: float = 4.0
    liquid_water_content: float = 0.2
    rain_rate_mm_hr: float = 0.0


@dataclass
class AtmosphereConfig:
    """Atmosphere model configuration.

    Attributes:
        model: Standard atmosphere model name (FR-01)
        custom_profile_path: Path to custom radiosonde CSV file (FR-02)
        custom_concentrations: Override gas mixing ratios in ppm (FR-03)
        aerosols: Aerosol configuration
        clouds: Cloud configuration
    """
    model: str = "US_STANDARD_1976"
    custom_profile_path: Optional[str] = None
    custom_concentrations: Dict[str, float] = field(default_factory=dict)
    aerosols: AerosolConfig = field(default_factory=AerosolConfig)
    clouds: CloudConfig = field(default_factory=CloudConfig)


@dataclass
class GeometryConfig:
    """Path geometry configuration.

    Attributes:
        path_type: Type of viewing path (HORIZONTAL, SLANT, VERTICAL) (FR-07)
        h1_km: Observer/start altitude in km
        h2_km: Target/end altitude in km
        angle_deg: Zenith angle in degrees (0=vertical, 90=horizontal)
        path_length_km: Horizontal path length (for HORIZONTAL type)
        earth_curvature: Include Earth curvature correction (FR-08)
    """
    path_type: str = "HORIZONTAL"
    h1_km: float = 0.0
    h2_km: float = 0.0
    angle_deg: float = 0.0
    path_length_km: float = 1.0
    earth_curvature: bool = True


@dataclass
class SpectralConfig:
    """Spectral calculation parameters.

    Attributes:
        min_wavenumber: Start wavenumber in cm^-1 (FR-11)
        max_wavenumber: End wavenumber in cm^-1 (FR-11)
        resolution: Spectral resolution in cm^-1 (FR-11)
        line_cutoff: Distance from line center for calculation in cm^-1
    """
    min_wavenumber: float = 2000.0
    max_wavenumber: float = 3333.0
    resolution: float = 0.01
    line_cutoff: float = 25.0


@dataclass
class OutputConfig:
    """Output format configuration.

    Attributes:
        format: Output file format (csv, json, netcdf)
        output_path: Directory for output files
        include_intermediate: Save intermediate results (layer-by-layer)
        quantities: List of quantities to output
    """
    format: str = "json"
    output_path: str = "./output"
    include_intermediate: bool = False
    quantities: list = field(default_factory=lambda: [
        "transmittance",
        "radiance",
        "optical_depth"
    ])


@dataclass
class SimulationConfig:
    """Complete simulation configuration.

    This is the top-level configuration object that matches the JSON API
    specification from the SRS document.

    Example JSON input:
        {
            "simulation_config": {"offline_mode": true, "database_path": "./data/hitran_lines.h5"},
            "atmosphere": {"model": "MID_LATITUDE_SUMMER", "aerosols": {"type": "RURAL"}},
            "geometry": {"path_type": "SLANT", "h1_km": 0, "h2_km": 10, "angle_deg": 45},
            "spectral": {"min_wavenumber": 2000, "max_wavenumber": 2500, "resolution": 0.01}
        }
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """Create SimulationConfig from a dictionary.

        Supports the JSON API format specified in the SRS document.

        Args:
            config_dict: Configuration dictionary

        Returns:
            SimulationConfig instance
        """
        # Handle legacy 'simulation_config' key for system settings
        system_dict = config_dict.get("simulation_config", config_dict.get("system", {}))

        # Parse system config
        system = SystemConfig(
            offline_mode=system_dict.get("offline_mode", True),
            database_path=system_dict.get("database_path", "./data/spectral_db/hitran_lines.h5"),
            use_gpu=system_dict.get("use_gpu", False),
            num_threads=system_dict.get("num_threads", 4),
            cache_size_mb=system_dict.get("cache_size_mb", 2048),
        )

        # Parse atmosphere config
        atmo_dict = config_dict.get("atmosphere", {})
        aerosol_dict = atmo_dict.get("aerosols", {})
        cloud_dict = atmo_dict.get("clouds", {})

        aerosols = AerosolConfig(
            type=aerosol_dict.get("type", "NONE"),
            visibility_km=aerosol_dict.get("visibility_km", 23.0),
            custom_properties=aerosol_dict.get("custom_properties"),
        )

        clouds = CloudConfig(
            enabled=cloud_dict.get("enabled", False),
            base_altitude_km=cloud_dict.get("base_altitude_km", 2.0),
            top_altitude_km=cloud_dict.get("top_altitude_km", 4.0),
            liquid_water_content=cloud_dict.get("liquid_water_content", 0.2),
            rain_rate_mm_hr=cloud_dict.get("rain_rate_mm_hr", 0.0),
        )

        atmosphere = AtmosphereConfig(
            model=atmo_dict.get("model", "US_STANDARD_1976"),
            custom_profile_path=atmo_dict.get("custom_profile_path"),
            custom_concentrations=atmo_dict.get("custom_concentrations", {}),
            aerosols=aerosols,
            clouds=clouds,
        )

        # Parse geometry config
        geom_dict = config_dict.get("geometry", {})
        geometry = GeometryConfig(
            path_type=geom_dict.get("path_type", "HORIZONTAL"),
            h1_km=geom_dict.get("h1_km", 0.0),
            h2_km=geom_dict.get("h2_km", 0.0),
            angle_deg=geom_dict.get("angle_deg", 0.0),
            path_length_km=geom_dict.get("path_length_km", 1.0),
            earth_curvature=geom_dict.get("earth_curvature", True),
        )

        # Parse spectral config
        spec_dict = config_dict.get("spectral", {})
        spectral = SpectralConfig(
            min_wavenumber=spec_dict.get("min_wavenumber", 2000.0),
            max_wavenumber=spec_dict.get("max_wavenumber", 3333.0),
            resolution=spec_dict.get("resolution", 0.01),
            line_cutoff=spec_dict.get("line_cutoff", 25.0),
        )

        # Parse output config
        out_dict = config_dict.get("output", {})
        output = OutputConfig(
            format=out_dict.get("format", "json"),
            output_path=out_dict.get("output_path", "./output"),
            include_intermediate=out_dict.get("include_intermediate", False),
            quantities=out_dict.get("quantities", ["transmittance", "radiance", "optical_depth"]),
        )

        return cls(
            system=system,
            atmosphere=atmosphere,
            geometry=geometry,
            spectral=spectral,
            output=output,
        )

    @classmethod
    def from_json(cls, json_path: str) -> "SimulationConfig":
        """Load configuration from a JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            SimulationConfig instance
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SimulationConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            SimulationConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary
        """
        return {
            "simulation_config": {
                "offline_mode": self.system.offline_mode,
                "database_path": self.system.database_path,
                "use_gpu": self.system.use_gpu,
                "num_threads": self.system.num_threads,
                "cache_size_mb": self.system.cache_size_mb,
            },
            "atmosphere": {
                "model": self.atmosphere.model,
                "custom_profile_path": self.atmosphere.custom_profile_path,
                "custom_concentrations": self.atmosphere.custom_concentrations,
                "aerosols": {
                    "type": self.atmosphere.aerosols.type,
                    "visibility_km": self.atmosphere.aerosols.visibility_km,
                },
                "clouds": {
                    "enabled": self.atmosphere.clouds.enabled,
                    "base_altitude_km": self.atmosphere.clouds.base_altitude_km,
                    "top_altitude_km": self.atmosphere.clouds.top_altitude_km,
                    "liquid_water_content": self.atmosphere.clouds.liquid_water_content,
                    "rain_rate_mm_hr": self.atmosphere.clouds.rain_rate_mm_hr,
                },
            },
            "geometry": {
                "path_type": self.geometry.path_type,
                "h1_km": self.geometry.h1_km,
                "h2_km": self.geometry.h2_km,
                "angle_deg": self.geometry.angle_deg,
                "path_length_km": self.geometry.path_length_km,
                "earth_curvature": self.geometry.earth_curvature,
            },
            "spectral": {
                "min_wavenumber": self.spectral.min_wavenumber,
                "max_wavenumber": self.spectral.max_wavenumber,
                "resolution": self.spectral.resolution,
                "line_cutoff": self.spectral.line_cutoff,
            },
            "output": {
                "format": self.output.format,
                "output_path": self.output.output_path,
                "include_intermediate": self.output.include_intermediate,
                "quantities": self.output.quantities,
            },
        }

    def to_json(self, json_path: str, indent: int = 2) -> None:
        """Save configuration to JSON file.

        Args:
            json_path: Output file path
            indent: JSON indentation level
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    def validate(self) -> list:
        """Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate spectral range
        if self.spectral.min_wavenumber >= self.spectral.max_wavenumber:
            errors.append("min_wavenumber must be less than max_wavenumber")

        if self.spectral.resolution <= 0:
            errors.append("spectral resolution must be positive")

        # Validate geometry
        if self.geometry.h1_km < 0 or self.geometry.h2_km < 0:
            errors.append("altitude values must be non-negative")

        if self.geometry.path_type == "SLANT":
            if not (0 <= self.geometry.angle_deg <= 180):
                errors.append("zenith angle must be between 0 and 180 degrees")

        # Validate atmosphere model
        valid_models = [
            "US_STANDARD_1976", "TROPICAL",
            "MID_LATITUDE_SUMMER", "MID_LATITUDE_WINTER",
            "SUB_ARCTIC_SUMMER", "SUB_ARCTIC_WINTER",
        ]
        if self.atmosphere.model not in valid_models and not self.atmosphere.custom_profile_path:
            errors.append(f"Invalid atmosphere model: {self.atmosphere.model}")

        # Validate aerosol type
        valid_aerosols = ["NONE", "RURAL", "URBAN", "MARITIME", "DESERT"]
        if self.atmosphere.aerosols.type not in valid_aerosols:
            errors.append(f"Invalid aerosol type: {self.atmosphere.aerosols.type}")

        # Validate visibility
        if self.atmosphere.aerosols.visibility_km <= 0:
            errors.append("visibility must be positive")

        # Validate database path in offline mode
        if self.system.offline_mode:
            db_path = Path(self.system.database_path)
            # Don't error if path doesn't exist yet - it may be generated

        return errors
