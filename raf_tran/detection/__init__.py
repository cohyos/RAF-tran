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

__all__ = [
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
]
