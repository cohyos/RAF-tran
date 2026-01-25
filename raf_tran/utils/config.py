"""
Configuration file support for RAF-tran.

Provides YAML and JSON configuration file loading and validation
for atmospheric simulations.

Usage
-----
>>> from raf_tran.utils.config import load_config, SimulationConfig
>>> config = load_config("simulation.yaml")
>>> print(config.atmosphere.surface_pressure)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Try to import YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class AtmosphereConfig:
    """Atmospheric profile configuration."""

    model: str = "us_standard_1976"  # us_standard_1976, tropical, midlatitude_summer, etc.
    surface_pressure: float = 1013.25  # hPa
    surface_temperature: float = 288.15  # K
    scale_height: float = 8.5  # km
    n_layers: int = 50
    top_altitude: float = 50.0  # km

    # Gas mixing ratios (ppmv unless noted)
    co2_ppmv: float = 420.0
    ch4_ppbv: float = 1900.0
    n2o_ppbv: float = 335.0
    o3_column_du: float = 300.0  # Dobson units


@dataclass
class AerosolConfig:
    """Aerosol configuration."""

    enabled: bool = True
    model: str = "rural"  # rural, urban, maritime, desert, volcanic
    aod_550: float = 0.1  # AOD at 550 nm
    angstrom_exponent: float = 1.3
    single_scatter_albedo: float = 0.95
    asymmetry_parameter: float = 0.7
    scale_height: float = 2.0  # km


@dataclass
class CloudConfig:
    """Cloud configuration."""

    enabled: bool = False
    cloud_type: str = "stratus"  # stratus, cumulus, cirrus
    cloud_base: float = 2.0  # km
    cloud_top: float = 3.0  # km
    optical_depth: float = 10.0
    effective_radius: float = 10.0  # micrometers
    liquid_water_path: float = 100.0  # g/m^2


@dataclass
class SurfaceConfig:
    """Surface configuration."""

    albedo: float = 0.2
    spectral_albedo: bool = False
    surface_type: str = "grass"  # grass, forest, ocean, snow, desert, urban
    emissivity: float = 0.98
    temperature: Optional[float] = None  # If None, use atmosphere surface temp


@dataclass
class SolarConfig:
    """Solar/illumination configuration."""

    solar_zenith_angle: float = 30.0  # degrees
    solar_azimuth_angle: float = 180.0  # degrees
    day_of_year: int = 172  # Summer solstice
    solar_constant: float = 1361.0  # W/m^2
    earth_sun_distance: float = 1.0  # AU


@dataclass
class SpectralConfig:
    """Spectral configuration."""

    wavelength_min: float = 0.25  # micrometers
    wavelength_max: float = 4.0  # micrometers
    n_wavelengths: int = 100
    wavelengths: Optional[List[float]] = None  # Explicit wavelength list
    spectral_resolution: Optional[float] = None  # nm


@dataclass
class SolverConfig:
    """RTE solver configuration."""

    method: str = "delta_eddington"  # eddington, quadrature, delta_eddington
    n_streams: int = 2  # For future multi-stream support
    convergence_threshold: float = 1e-6
    max_iterations: int = 100


@dataclass
class OutputConfig:
    """Output configuration."""

    compute_fluxes: bool = True
    compute_radiances: bool = False
    compute_heating_rates: bool = True
    output_levels: str = "all"  # all, surface, toa, custom
    custom_levels: Optional[List[float]] = None  # km


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    name: str = "unnamed_simulation"
    description: str = ""

    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    aerosol: AerosolConfig = field(default_factory=AerosolConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")

        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)


def _dict_to_config(data: Dict[str, Any]) -> SimulationConfig:
    """Convert dictionary to SimulationConfig."""
    config = SimulationConfig(
        name=data.get('name', 'unnamed_simulation'),
        description=data.get('description', ''),
    )

    if 'atmosphere' in data:
        config.atmosphere = AtmosphereConfig(**data['atmosphere'])
    if 'aerosol' in data:
        config.aerosol = AerosolConfig(**data['aerosol'])
    if 'cloud' in data:
        config.cloud = CloudConfig(**data['cloud'])
    if 'surface' in data:
        config.surface = SurfaceConfig(**data['surface'])
    if 'solar' in data:
        config.solar = SolarConfig(**data['solar'])
    if 'spectral' in data:
        config.spectral = SpectralConfig(**data['spectral'])
    if 'solver' in data:
        config.solver = SolverConfig(**data['solver'])
    if 'output' in data:
        config.output = OutputConfig(**data['output'])

    return config


def load_config(path: Union[str, Path]) -> SimulationConfig:
    """
    Load simulation configuration from YAML or JSON file.

    Parameters
    ----------
    path : str or Path
        Path to configuration file (.yaml, .yml, or .json)

    Returns
    -------
    config : SimulationConfig
        Loaded configuration

    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist
    ValueError
        If file format is not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in ('.yaml', '.yml'):
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML support. Install with: pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f)
    elif suffix == '.json':
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .yaml, .yml, or .json")

    return _dict_to_config(data)


def create_default_config(path: Union[str, Path] = "simulation_config.yaml") -> SimulationConfig:
    """
    Create and save a default configuration file.

    Parameters
    ----------
    path : str or Path
        Output path for configuration file

    Returns
    -------
    config : SimulationConfig
        Default configuration
    """
    config = SimulationConfig(
        name="default_simulation",
        description="Default RAF-tran simulation configuration",
    )

    path = Path(path)
    if path.suffix.lower() in ('.yaml', '.yml'):
        config.to_yaml(path)
    else:
        config.to_json(path)

    return config


def validate_config(config: SimulationConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Parameters
    ----------
    config : SimulationConfig
        Configuration to validate

    Returns
    -------
    issues : list of str
        List of validation issues (empty if valid)
    """
    issues = []

    # Atmosphere validation
    if config.atmosphere.surface_pressure <= 0:
        issues.append("Surface pressure must be positive")
    if config.atmosphere.surface_temperature <= 0:
        issues.append("Surface temperature must be positive")
    if config.atmosphere.n_layers < 1:
        issues.append("Number of layers must be at least 1")
    if config.atmosphere.top_altitude <= 0:
        issues.append("Top altitude must be positive")

    # Solar validation
    if not 0 <= config.solar.solar_zenith_angle <= 90:
        issues.append("Solar zenith angle must be between 0 and 90 degrees")
    if not 1 <= config.solar.day_of_year <= 366:
        issues.append("Day of year must be between 1 and 366")

    # Spectral validation
    if config.spectral.wavelength_min >= config.spectral.wavelength_max:
        issues.append("Wavelength min must be less than wavelength max")
    if config.spectral.n_wavelengths < 1:
        issues.append("Number of wavelengths must be at least 1")

    # Surface validation
    if not 0 <= config.surface.albedo <= 1:
        issues.append("Surface albedo must be between 0 and 1")
    if not 0 <= config.surface.emissivity <= 1:
        issues.append("Surface emissivity must be between 0 and 1")

    # Aerosol validation
    if config.aerosol.enabled:
        if config.aerosol.aod_550 < 0:
            issues.append("AOD must be non-negative")
        if not 0 <= config.aerosol.single_scatter_albedo <= 1:
            issues.append("Single scatter albedo must be between 0 and 1")

    # Cloud validation
    if config.cloud.enabled:
        if config.cloud.cloud_base >= config.cloud.cloud_top:
            issues.append("Cloud base must be below cloud top")
        if config.cloud.optical_depth < 0:
            issues.append("Cloud optical depth must be non-negative")

    return issues
