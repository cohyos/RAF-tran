"""
FPA Analysis and Comparison
============================

Functions for comparing FPAs, computing performance metrics, and generating
trade-study data for system design.
"""

from typing import List, Dict, Optional, Tuple
import math

from raf_tran.fpa_library.models import (
    FPASpec, SpectralBand, DetectorType, CoolingType, Vendor,
    ApplicationDomain,
)


# =============================================================================
# Performance Metrics
# =============================================================================

def compute_ifov_urad(fpa: FPASpec, focal_length_mm: float) -> float:
    """
    Compute Instantaneous Field of View (IFOV) in microradians.

    Parameters
    ----------
    fpa : FPASpec
    focal_length_mm : float

    Returns
    -------
    ifov_urad : float
    """
    return fpa.pixel_pitch_um / focal_length_mm * 1000.0


def compute_fov_deg(fpa: FPASpec, focal_length_mm: float) -> Optional[Tuple[float, float]]:
    """Compute full Field of View in degrees (horizontal, vertical)."""
    return fpa.fov_at_focal_length(focal_length_mm)


def compute_dri_ranges(fpa: FPASpec, focal_length_mm: float,
                       target_size_m: float = 2.3) -> Dict[str, float]:
    """
    Compute Detection, Recognition, and Identification ranges using
    Johnson criteria.

    Parameters
    ----------
    fpa : FPASpec
    focal_length_mm : float
    target_size_m : float
        Critical target dimension (default 2.3m for NATO standard vehicle)

    Returns
    -------
    ranges : dict
        Keys: 'detection_m', 'recognition_m', 'identification_m'
    """
    return {
        'detection_m': fpa.johnson_criteria_range(target_size_m, focal_length_mm, cycles=1.0),
        'recognition_m': fpa.johnson_criteria_range(target_size_m, focal_length_mm, cycles=3.0),
        'identification_m': fpa.johnson_criteria_range(target_size_m, focal_length_mm, cycles=6.0),
    }


def compute_swap_score(fpa: FPASpec) -> Optional[float]:
    """
    Compute a SWaP-C figure of merit (higher = better performance per unit SWaP).

    Score = (total_pixels / 1e6) / (weight_kg * power_W)

    Returns None if weight or power data is unavailable.
    """
    if fpa.total_pixels is None:
        return None
    weight_g = fpa.weight_g
    power_w = fpa.power_steady_w or fpa.power_w
    if weight_g is None or power_w is None or power_w == 0:
        return None
    weight_kg = weight_g / 1000.0
    return (fpa.total_pixels / 1e6) / (weight_kg * power_w)


def compute_sensitivity_score(fpa: FPASpec) -> Optional[float]:
    """
    Compute sensitivity figure of merit (higher = better).

    Score = total_pixels / (NETD_mK * pitch_um^2)

    Returns None if NETD is unavailable.
    """
    if fpa.netd_mk is None or fpa.total_pixels is None:
        return None
    return fpa.total_pixels / (fpa.netd_mk * fpa.pixel_pitch_um**2)


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_fpas(fpas: List[FPASpec],
                 focal_length_mm: float = 100.0,
                 target_size_m: float = 2.3) -> List[Dict]:
    """
    Generate a detailed comparison table for a list of FPAs.

    Parameters
    ----------
    fpas : list of FPASpec
    focal_length_mm : float
        Reference focal length for range calculations
    target_size_m : float
        Reference target size for DRI calculations

    Returns
    -------
    comparison : list of dict
        Each dict contains all comparison metrics for one FPA
    """
    results = []
    for fpa in fpas:
        row = {
            'name': fpa.name,
            'vendor': fpa.vendor.value,
            'band': fpa.spectral_band.value,
            'detector': fpa.detector_type.value,
            'resolution': fpa.resolution_str,
            'megapixels': fpa.megapixels,
            'pitch_um': fpa.pixel_pitch_um,
            'netd_mk': fpa.netd_mk,
            'cooling': fpa.cooling.value,
            'weight_g': fpa.weight_g,
            'power_w': fpa.power_steady_w or fpa.power_w,
            'ifov_urad': compute_ifov_urad(fpa, focal_length_mm),
        }

        # DRI ranges
        dri = compute_dri_ranges(fpa, focal_length_mm, target_size_m)
        row['detection_km'] = dri['detection_m'] / 1000
        row['recognition_km'] = dri['recognition_m'] / 1000
        row['identification_km'] = dri['identification_m'] / 1000

        # Scores
        row['swap_score'] = compute_swap_score(fpa)
        row['sensitivity_score'] = compute_sensitivity_score(fpa)

        results.append(row)

    return results


def rank_fpas(fpas: List[FPASpec],
              metric: str = 'sensitivity_score',
              ascending: bool = False) -> List[Tuple[FPASpec, Optional[float]]]:
    """
    Rank FPAs by a given metric.

    Parameters
    ----------
    fpas : list of FPASpec
    metric : str
        One of: 'sensitivity_score', 'swap_score', 'netd_mk', 'megapixels',
        'pixel_pitch_um', 'identification_range'
    ascending : bool
        If True, rank lowest first (useful for NETD, pitch)

    Returns
    -------
    ranked : list of (FPASpec, score) tuples, sorted
    """
    scored = []
    for fpa in fpas:
        if metric == 'sensitivity_score':
            score = compute_sensitivity_score(fpa)
        elif metric == 'swap_score':
            score = compute_swap_score(fpa)
        elif metric == 'netd_mk':
            score = fpa.netd_mk
        elif metric == 'megapixels':
            score = fpa.megapixels
        elif metric == 'pixel_pitch_um':
            score = fpa.pixel_pitch_um
        elif metric == 'identification_range':
            score = fpa.johnson_criteria_range(2.3, 100.0, 6.0)
        else:
            score = None
        scored.append((fpa, score))

    # Filter None scores, sort
    with_scores = [(f, s) for f, s in scored if s is not None]
    without_scores = [(f, None) for f, s in scored if s is None]

    with_scores.sort(key=lambda x: x[1], reverse=not ascending)
    return with_scores + without_scores


# =============================================================================
# Trade Study Helpers
# =============================================================================

def pitch_miniaturization_factor(pitch_um: float, reference_pitch_um: float = 15.0) -> float:
    """
    Compute the optical miniaturization factor relative to a reference pitch.

    A smaller pitch allows proportionally smaller optics for the same IFOV.

    Parameters
    ----------
    pitch_um : float
        Target pixel pitch
    reference_pitch_um : float
        Reference pitch (default 15 um)

    Returns
    -------
    factor : float
        Miniaturization factor (e.g., 0.33 for 5um vs 15um)
    """
    return pitch_um / reference_pitch_um


def hot_reliability_gain(operating_temp_k: float, reference_temp_k: float = 77.0) -> Dict[str, float]:
    """
    Estimate reliability and readiness gains from HOT operation.

    Parameters
    ----------
    operating_temp_k : float
        HOT operating temperature
    reference_temp_k : float
        Reference cryogenic temperature (default 77K for InSb)

    Returns
    -------
    gains : dict
        'mttf_multiplier': Estimated cooler life extension factor
        'cooldown_reduction': Estimated cooldown time reduction factor
        'power_reduction': Estimated power reduction factor
    """
    temp_ratio = operating_temp_k / reference_temp_k
    return {
        'mttf_multiplier': temp_ratio ** 1.5,
        'cooldown_reduction': 1.0 / temp_ratio,
        'power_reduction': 1.0 / temp_ratio ** 0.8,
    }


def spectral_band_comparison() -> Dict[str, Dict[str, str]]:
    """
    Return a comparison of IR spectral bands and their characteristics.

    Returns
    -------
    comparison : dict of band name -> properties dict
    """
    return {
        'SWIR (0.9-1.7 um)': {
            'signal_level': 'Low (reflected sunlight)',
            'atmospheric_transmission': 'Good',
            'day_night': 'Day only (no thermal)',
            'typical_detector': 'InGaAs',
            'cooling': 'TEC or uncooled',
            'applications': 'Laser designation, imaging through glass',
        },
        'MWIR (3.0-5.0 um)': {
            'signal_level': 'Medium (thermal + reflected)',
            'atmospheric_transmission': 'Good (3-5 um window)',
            'day_night': 'Day and night',
            'typical_detector': 'InSb, XBn, MCT, T2SL',
            'cooling': 'Cooled (77-150K)',
            'applications': 'Long-range surveillance, missile seekers',
        },
        'LWIR (8.0-14.0 um)': {
            'signal_level': 'High (peak thermal at ~300K)',
            'atmospheric_transmission': 'Good (8-14 um window)',
            'day_night': 'Day and night',
            'typical_detector': 'MCT, T2SL, VOx (uncooled)',
            'cooling': 'Cooled or uncooled',
            'applications': 'Fire control, DVE, missile warning',
        },
        'Dual MW/LW': {
            'signal_level': 'Both bands simultaneously',
            'atmospheric_transmission': 'Both windows',
            'day_night': 'Day and night, obscurant penetration',
            'typical_detector': 'Dual-band MCT/T2SL',
            'cooling': 'Cooled',
            'applications': '3rd Gen FLIR, all-weather targeting',
        },
    }
