"""Detector models for infrared sensing applications."""

from raf_tran.detectors.fpa import (
    FPADetector,
    InSbDetector,
    MCTDetector,
    detector_from_type,
)

__all__ = [
    'FPADetector',
    'InSbDetector',
    'MCTDetector',
    'detector_from_type',
]
