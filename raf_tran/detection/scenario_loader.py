"""
Detection Scenario Configuration Loader
========================================

This module provides YAML-based configuration loading for detection
scenarios, enabling reproducible studies with specified detectors,
targets, and environmental conditions.

Example YAML structure:
    name: "Fighter Detection Study"
    scenario:
      sensor_altitude_km: 0.0
      target_altitude_km: 10.0
      visibility_km: 23.0
    detectors:
      - name: "InSb MWIR"
        type: "insb"
    targets:
      - name: "Fighter"
        type: "fighter"
        aspect: "rear"
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
import yaml

from raf_tran.detectors import InSbDetector, MCTDetector, FPADetector
from raf_tran.targets import generic_fighter, generic_transport, generic_uav, AircraftSignature


@dataclass
class DetectorConfig:
    """Configuration for an FPA detector."""
    name: str
    type: str  # 'insb', 'mct', 'mct_mwir', 'mct_lwir'
    pixel_pitch: float = 15.0  # um
    f_number: float = 2.0
    integration_time: float = 10.0  # ms
    spectral_band: Optional[tuple] = None  # Override default band
    focal_length_mm: Optional[float] = None
    aperture_mm: Optional[float] = None


@dataclass
class TargetConfig:
    """Configuration for an aircraft target."""
    name: str
    type: str  # 'fighter', 'transport', 'uav'
    aspect: str = 'rear'
    afterburner: bool = False
    mach: float = 0.9
    characteristic_dimension_m: Optional[float] = None  # Override default


@dataclass
class ScenarioParams:
    """Environmental and geometry parameters."""
    sensor_altitude_km: float = 0.0
    target_altitude_km: float = 10.0
    visibility_km: float = 23.0
    humidity_percent: float = 50.0
    snr_threshold: float = 5.0
    max_range_km: float = 100.0


@dataclass
class DetectionScenario:
    """
    Complete detection scenario configuration.

    Attributes
    ----------
    name : str
        Scenario identifier
    description : str
        Detailed description
    scenario : ScenarioParams
        Environmental parameters
    detectors : List[DetectorConfig]
        List of detector configurations
    targets : List[TargetConfig]
        List of target configurations
    analysis_types : List[str]
        Types of analysis to perform
    johnson_probability : float
        Target probability for Johnson criteria
    """
    name: str
    description: str = ""
    scenario: ScenarioParams = field(default_factory=ScenarioParams)
    detectors: List[DetectorConfig] = field(default_factory=list)
    targets: List[TargetConfig] = field(default_factory=list)
    analysis_types: List[str] = field(default_factory=lambda: ['detection_range'])
    johnson_probability: float = 0.5


def create_detector(config: DetectorConfig) -> FPADetector:
    """
    Create an FPA detector from configuration.

    Parameters
    ----------
    config : DetectorConfig
        Detector configuration

    Returns
    -------
    detector : FPADetector
        Configured detector instance
    """
    detector_type = config.type.lower()

    if detector_type == 'insb':
        detector = InSbDetector(
            name=config.name,
            pixel_pitch=config.pixel_pitch,
            f_number=config.f_number,
            integration_time=config.integration_time,
        )
        if config.spectral_band:
            detector.spectral_band = config.spectral_band
    elif detector_type in ('mct', 'mct_lwir', 'hgcdte'):
        # Default MCT is LWIR
        detector = MCTDetector(
            name=config.name,
            spectral_band=config.spectral_band or (8.0, 12.0),
            pixel_pitch=config.pixel_pitch,
            f_number=config.f_number,
            integration_time=config.integration_time,
        )
    elif detector_type == 'mct_mwir':
        detector = MCTDetector(
            name=config.name,
            spectral_band=config.spectral_band or (3.0, 5.0),
            pixel_pitch=config.pixel_pitch,
            f_number=config.f_number,
            integration_time=config.integration_time,
        )
    else:
        raise ValueError(f"Unknown detector type: {config.type}")

    return detector


def create_target(config: TargetConfig) -> AircraftSignature:
    """
    Create an aircraft signature from configuration.

    Parameters
    ----------
    config : TargetConfig
        Target configuration

    Returns
    -------
    target : AircraftSignature
        Configured target instance
    """
    target_type = config.type.lower()

    if target_type == 'fighter':
        target = generic_fighter(
            aspect=config.aspect,
            afterburner=config.afterburner,
            mach=config.mach,
        )
    elif target_type == 'transport':
        target = generic_transport(
            aspect=config.aspect,
            mach=config.mach,
        )
    elif target_type == 'uav':
        target = generic_uav(
            aspect=config.aspect,
            mach=config.mach,
        )
    else:
        raise ValueError(f"Unknown target type: {config.type}")

    # Override name if specified
    if config.name:
        target.name = config.name

    # Override characteristic dimension if specified
    if config.characteristic_dimension_m is not None:
        target.characteristic_dimension_m = config.characteristic_dimension_m

    return target


def load_scenario(filepath: str) -> DetectionScenario:
    """
    Load detection scenario from YAML file.

    Parameters
    ----------
    filepath : str
        Path to YAML configuration file

    Returns
    -------
    scenario : DetectionScenario
        Loaded scenario configuration
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # Parse scenario parameters
    scenario_data = data.get('scenario', {})
    scenario_params = ScenarioParams(
        sensor_altitude_km=scenario_data.get('sensor_altitude_km', 0.0),
        target_altitude_km=scenario_data.get('target_altitude_km', 10.0),
        visibility_km=scenario_data.get('visibility_km', 23.0),
        humidity_percent=scenario_data.get('humidity_percent', 50.0),
        snr_threshold=scenario_data.get('snr_threshold', 5.0),
        max_range_km=scenario_data.get('max_range_km', 100.0),
    )

    # Parse detectors
    detectors = []
    for det_data in data.get('detectors', []):
        spectral_band = det_data.get('spectral_band')
        if spectral_band and isinstance(spectral_band, list):
            spectral_band = tuple(spectral_band)
        detectors.append(DetectorConfig(
            name=det_data.get('name', 'Detector'),
            type=det_data.get('type', 'insb'),
            pixel_pitch=det_data.get('pixel_pitch', 15.0),
            f_number=det_data.get('f_number', 2.0),
            integration_time=det_data.get('integration_time', 10.0),
            spectral_band=spectral_band,
            focal_length_mm=det_data.get('focal_length_mm'),
            aperture_mm=det_data.get('aperture_mm'),
        ))

    # Parse targets
    targets = []
    for tgt_data in data.get('targets', []):
        targets.append(TargetConfig(
            name=tgt_data.get('name', 'Target'),
            type=tgt_data.get('type', 'fighter'),
            aspect=tgt_data.get('aspect', 'rear'),
            afterburner=tgt_data.get('afterburner', False),
            mach=tgt_data.get('mach', 0.9),
            characteristic_dimension_m=tgt_data.get('characteristic_dimension_m'),
        ))

    return DetectionScenario(
        name=data.get('name', 'Unnamed Scenario'),
        description=data.get('description', ''),
        scenario=scenario_params,
        detectors=detectors,
        targets=targets,
        analysis_types=data.get('analysis_types', ['detection_range']),
        johnson_probability=data.get('johnson_probability', 0.5),
    )


def save_scenario(scenario: DetectionScenario, filepath: str):
    """
    Save detection scenario to YAML file.

    Parameters
    ----------
    scenario : DetectionScenario
        Scenario configuration to save
    filepath : str
        Output file path
    """
    data = {
        'name': scenario.name,
        'description': scenario.description,
        'scenario': asdict(scenario.scenario),
        'detectors': [asdict(d) for d in scenario.detectors],
        'targets': [asdict(t) for t in scenario.targets],
        'analysis_types': scenario.analysis_types,
        'johnson_probability': scenario.johnson_probability,
    }

    # Clean up None values
    for det in data['detectors']:
        det = {k: v for k, v in det.items() if v is not None}
    for tgt in data['targets']:
        tgt = {k: v for k, v in tgt.items() if v is not None}

    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def create_default_scenario() -> DetectionScenario:
    """
    Create a default detection scenario for testing.

    Returns
    -------
    scenario : DetectionScenario
        Default scenario with InSb/MCT detectors and fighter target
    """
    return DetectionScenario(
        name="Default Detection Scenario",
        description="MWIR vs LWIR comparison for fighter aircraft",
        scenario=ScenarioParams(),
        detectors=[
            DetectorConfig(name="InSb MWIR", type="insb"),
            DetectorConfig(name="MCT LWIR", type="mct"),
        ],
        targets=[
            TargetConfig(name="Fighter (Rear)", type="fighter", aspect="rear"),
        ],
        analysis_types=['detection_range', 'recognition_range'],
    )
