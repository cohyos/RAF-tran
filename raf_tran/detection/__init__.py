"""Detection range calculations for IR systems."""

from raf_tran.detection.range_equation import (
    DetectionResult,
    calculate_detection_range,
    calculate_snr_vs_range,
    atmospheric_transmission_ir,
    # Geometry functions
    slant_range_from_altitudes,
    elevation_angle,
    mean_path_altitude,
    # Slant path functions
    atmospheric_transmission_slant,
    calculate_detection_range_slant,
    scan_altitude_performance,
)

from raf_tran.detection.johnson_criteria import (
    RecognitionTask,
    JohnsonResult,
    JOHNSON_CYCLES,
    TARGET_DIMENSIONS,
    detection_probability,
    cycles_on_target,
    range_for_cycles,
    range_for_probability,
    calculate_recognition_ranges,
    calculate_probability_vs_range,
    johnson_analysis,
)

from raf_tran.detection.scenario_loader import (
    DetectorConfig,
    TargetConfig,
    ScenarioParams,
    MonteCarloScenarioConfig,
    DetectionScenario,
    create_detector,
    create_target,
    load_scenario,
    save_scenario,
    create_default_scenario,
)

from raf_tran.detection.monte_carlo import (
    DistributionType,
    ParameterDistribution,
    MonteCarloConfig,
    MonteCarloResult,
    sample_parameter,
    monte_carlo_detection_range,
    monte_carlo_multi_detector,
    default_monte_carlo_config,
)

__all__ = [
    # Range equation
    'DetectionResult',
    'calculate_detection_range',
    'calculate_snr_vs_range',
    'atmospheric_transmission_ir',
    # Geometry
    'slant_range_from_altitudes',
    'elevation_angle',
    'mean_path_altitude',
    # Slant path
    'atmospheric_transmission_slant',
    'calculate_detection_range_slant',
    'scan_altitude_performance',
    # Johnson criteria
    'RecognitionTask',
    'JohnsonResult',
    'JOHNSON_CYCLES',
    'TARGET_DIMENSIONS',
    'detection_probability',
    'cycles_on_target',
    'range_for_cycles',
    'range_for_probability',
    'calculate_recognition_ranges',
    'calculate_probability_vs_range',
    'johnson_analysis',
    # Scenario configuration
    'DetectorConfig',
    'TargetConfig',
    'ScenarioParams',
    'MonteCarloScenarioConfig',
    'DetectionScenario',
    'create_detector',
    'create_target',
    'load_scenario',
    'save_scenario',
    'create_default_scenario',
    # Monte Carlo simulation
    'DistributionType',
    'ParameterDistribution',
    'MonteCarloConfig',
    'MonteCarloResult',
    'sample_parameter',
    'monte_carlo_detection_range',
    'monte_carlo_multi_detector',
    'default_monte_carlo_config',
]
