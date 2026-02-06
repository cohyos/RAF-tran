"""
Johnson Criteria for Detection Probability
==========================================

This module implements the Johnson criteria for calculating detection,
recognition, and identification probabilities for IR imaging systems.

The Johnson criteria relate the number of resolution cycles on a target
to the probability of completing a recognition task:
- Detection: Is something there? (~1 cycle)
- Orientation: Which way is it facing? (~1.4 cycles)
- Recognition: What type is it? (~4 cycles)
- Identification: What specific model? (~6.4 cycles)

The probability function uses the Target Transfer Probability (TTP)
function from Night Vision and Electronic Sensors Directorate (NVESD).

References:
- Johnson, J. (1958). Analysis of image forming systems
- NVESD (1975). NVTherm model documentation
- Holst, G.C. (2008). Electro-Optical Imaging System Performance
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
import numpy as np


class RecognitionTask(Enum):
    """Recognition tasks per Johnson criteria."""
    DETECTION = "detection"           # Is something there?
    ORIENTATION = "orientation"       # Which way is it facing?
    RECOGNITION = "recognition"       # What type/class is it?
    IDENTIFICATION = "identification" # What specific model/individual?


@dataclass
class JohnsonResult:
    """
    Result of Johnson criteria calculation.

    Attributes
    ----------
    range_km : float
        Range at which the task can be performed (km)
    cycles_on_target : float
        Number of resolution cycles across target
    probability : float
        Probability of completing the task (0-1)
    task : RecognitionTask
        The recognition task being evaluated
    """
    range_km: float
    cycles_on_target: float
    probability: float
    task: RecognitionTask


# Johnson criteria cycle requirements for 50% probability
# These are the N50 values from standard Johnson criteria
JOHNSON_CYCLES = {
    RecognitionTask.DETECTION: 1.0,       # 0.75-1.5 cycles
    RecognitionTask.ORIENTATION: 1.4,     # 1.0-3.0 cycles
    RecognitionTask.RECOGNITION: 4.0,     # 3.0-6.0 cycles
    RecognitionTask.IDENTIFICATION: 6.4,  # 6.0-8.0 cycles
}


# Characteristic dimensions for common targets (meters)
# These represent the critical dimension for recognition
TARGET_DIMENSIONS = {
    'fighter': 15.0,      # Fighter aircraft length
    'transport': 60.0,    # Large transport/commercial aircraft
    'uav': 3.0,           # Medium UAV wingspan
    'drone': 0.5,         # Small drone
    'helicopter': 12.0,   # Helicopter length
    'missile': 4.0,       # Cruise missile
    'vehicle': 4.5,       # Ground vehicle
    'person': 1.8,        # Standing person
}


def detection_probability(
    cycles_on_target: float,
    task: RecognitionTask = RecognitionTask.DETECTION,
) -> float:
    """
    Calculate probability of completing a recognition task.

    Uses the Target Transfer Probability (TTP) function:
    P = (N/N50)^E / (1 + (N/N50)^E)

    where:
    - N = cycles on target
    - N50 = cycles required for 50% probability
    - E = 2.7 (empirically determined exponent)

    Parameters
    ----------
    cycles_on_target : float
        Number of resolution cycles across the target
    task : RecognitionTask
        The recognition task (detection, orientation, recognition, identification)

    Returns
    -------
    probability : float
        Probability of completing the task (0-1)

    Examples
    --------
    >>> detection_probability(1.0, RecognitionTask.DETECTION)
    0.5  # 50% at 1 cycle for detection

    >>> detection_probability(4.0, RecognitionTask.RECOGNITION)
    0.5  # 50% at 4 cycles for recognition
    """
    if cycles_on_target <= 0:
        return 0.0

    n50 = JOHNSON_CYCLES[task]
    exponent = 2.7  # TTP function exponent

    ratio = cycles_on_target / n50
    probability = (ratio ** exponent) / (1 + ratio ** exponent)

    return np.clip(probability, 0.0, 1.0)


def cycles_on_target(
    target_dimension_m: float,
    range_m: float,
    detector_ifov_mrad: float,
) -> float:
    """
    Calculate number of resolution cycles across target.

    A cycle is defined as one line pair (bright-dark pair).
    Cycles = target_angular_size / (2 * IFOV)

    Parameters
    ----------
    target_dimension_m : float
        Target characteristic dimension (m)
    range_m : float
        Distance to target (m)
    detector_ifov_mrad : float
        Detector instantaneous field of view (mrad)

    Returns
    -------
    cycles : float
        Number of resolution cycles across target

    Notes
    -----
    The factor of 2 accounts for the Nyquist criterion:
    one cycle requires two pixels (samples).
    """
    if range_m <= 0 or detector_ifov_mrad <= 0:
        return 0.0

    # Target angular size in mrad
    target_angular_size_mrad = (target_dimension_m / range_m) * 1000.0

    # Cycles across target
    cycles = target_angular_size_mrad / (2.0 * detector_ifov_mrad)

    return max(cycles, 0.0)


def range_for_cycles(
    target_dimension_m: float,
    detector_ifov_mrad: float,
    cycles_required: float,
) -> float:
    """
    Calculate range at which a given number of cycles is achieved.

    Parameters
    ----------
    target_dimension_m : float
        Target characteristic dimension (m)
    detector_ifov_mrad : float
        Detector instantaneous field of view (mrad)
    cycles_required : float
        Required number of cycles on target

    Returns
    -------
    range_m : float
        Maximum range for achieving required cycles (m)
    """
    if cycles_required <= 0 or detector_ifov_mrad <= 0:
        return float('inf')

    # target_angular_size_mrad = 2 * cycles * ifov_mrad
    # target_dimension_m / range_m * 1000 = 2 * cycles * ifov_mrad
    # range_m = target_dimension_m * 1000 / (2 * cycles * ifov_mrad)

    range_m = (target_dimension_m * 1000.0) / (2.0 * cycles_required * detector_ifov_mrad)
    return range_m


def range_for_probability(
    target_dimension_m: float,
    detector_ifov_mrad: float,
    task: RecognitionTask,
    probability: float = 0.5,
) -> float:
    """
    Calculate range at which target can be recognized with given probability.

    Inverts the TTP function to find cycles required for target probability,
    then converts to range.

    Parameters
    ----------
    target_dimension_m : float
        Target characteristic dimension (m)
    detector_ifov_mrad : float
        Detector instantaneous field of view (mrad)
    task : RecognitionTask
        The recognition task
    probability : float
        Target probability (0-1), default 0.5

    Returns
    -------
    range_m : float
        Maximum range for achieving target probability (m)

    Raises
    ------
    ValueError
        If probability is not in (0, 1)
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be in (0, 1)")

    n50 = JOHNSON_CYCLES[task]
    exponent = 2.7

    # Invert TTP function: P = r^E / (1 + r^E) where r = N/N50
    # P * (1 + r^E) = r^E
    # P = r^E - P * r^E = r^E * (1 - P)
    # r^E = P / (1 - P)
    # r = (P / (1 - P))^(1/E)

    p_ratio = probability / (1 - probability)
    ratio = p_ratio ** (1.0 / exponent)
    cycles_required = n50 * ratio

    return range_for_cycles(target_dimension_m, detector_ifov_mrad, cycles_required)


def calculate_recognition_ranges(
    target_dimension_m: float,
    detector_ifov_mrad: float,
    probability: float = 0.5,
) -> Dict[RecognitionTask, float]:
    """
    Calculate recognition ranges for all tasks.

    Parameters
    ----------
    target_dimension_m : float
        Target characteristic dimension (m)
    detector_ifov_mrad : float
        Detector instantaneous field of view (mrad)
    probability : float
        Target probability (0-1)

    Returns
    -------
    ranges : Dict[RecognitionTask, float]
        Range in meters for each recognition task
    """
    ranges = {}
    for task in RecognitionTask:
        ranges[task] = range_for_probability(
            target_dimension_m, detector_ifov_mrad, task, probability
        )
    return ranges


def calculate_probability_vs_range(
    target_dimension_m: float,
    detector_ifov_mrad: float,
    ranges_m: np.ndarray,
    task: RecognitionTask = RecognitionTask.DETECTION,
) -> np.ndarray:
    """
    Calculate detection probability as a function of range.

    Parameters
    ----------
    target_dimension_m : float
        Target characteristic dimension (m)
    detector_ifov_mrad : float
        Detector instantaneous field of view (mrad)
    ranges_m : np.ndarray
        Array of ranges to evaluate (m)
    task : RecognitionTask
        The recognition task

    Returns
    -------
    probabilities : np.ndarray
        Probability at each range
    """
    probabilities = np.zeros_like(ranges_m)

    for i, range_m in enumerate(ranges_m):
        if range_m > 0:
            cycles = cycles_on_target(target_dimension_m, range_m, detector_ifov_mrad)
            probabilities[i] = detection_probability(cycles, task)
        else:
            probabilities[i] = 1.0

    return probabilities


def johnson_analysis(
    target_dimension_m: float,
    detector_ifov_mrad: float,
    snr_detection_range_m: float,
    probability: float = 0.5,
) -> Dict[str, JohnsonResult]:
    """
    Perform complete Johnson criteria analysis.

    Compares SNR-based detection range with recognition ranges
    to determine which task limits performance.

    Parameters
    ----------
    target_dimension_m : float
        Target characteristic dimension (m)
    detector_ifov_mrad : float
        Detector instantaneous field of view (mrad)
    snr_detection_range_m : float
        SNR-limited detection range (m)
    probability : float
        Target probability for recognition tasks

    Returns
    -------
    results : Dict[str, JohnsonResult]
        Results for each task including SNR-limited detection
    """
    results = {}

    # SNR-limited detection range
    cycles_at_snr_range = cycles_on_target(
        target_dimension_m, snr_detection_range_m, detector_ifov_mrad
    )
    det_prob_at_snr_range = detection_probability(
        cycles_at_snr_range, RecognitionTask.DETECTION
    )

    results['snr_detection'] = JohnsonResult(
        range_km=snr_detection_range_m / 1000.0,
        cycles_on_target=cycles_at_snr_range,
        probability=det_prob_at_snr_range,
        task=RecognitionTask.DETECTION,
    )

    # Recognition ranges for each task
    for task in RecognitionTask:
        task_range_m = range_for_probability(
            target_dimension_m, detector_ifov_mrad, task, probability
        )
        # Effective range is minimum of SNR and resolution limits
        effective_range_m = min(task_range_m, snr_detection_range_m)

        cycles = cycles_on_target(
            target_dimension_m, effective_range_m, detector_ifov_mrad
        )
        prob = detection_probability(cycles, task)

        results[task.value] = JohnsonResult(
            range_km=effective_range_m / 1000.0,
            cycles_on_target=cycles,
            probability=prob,
            task=task,
        )

    return results
