"""Detection range calculations for IR systems."""

from raf_tran.detection.range_equation import (
    DetectionResult,
    calculate_detection_range,
    calculate_snr_vs_range,
    atmospheric_transmission_ir,
)

__all__ = [
    'DetectionResult',
    'calculate_detection_range',
    'calculate_snr_vs_range',
    'atmospheric_transmission_ir',
]
