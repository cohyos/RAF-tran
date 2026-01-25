"""
Infrared Detection Range Equations
==================================

This module provides functions for calculating detection range
of IR targets based on the IR range equation.

The fundamental equation relates:
- Target signature (radiant intensity)
- Atmospheric transmission
- Detector sensitivity (NEI, D*)
- Required SNR for detection

References:
- Holst, G.C. (2008). Electro-Optical Imaging System Performance
- Lloyd, J.M. (1975). Thermal Imaging Systems
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.optimize import brentq

from raf_tran.detectors.fpa import FPADetector
from raf_tran.targets.aircraft import AircraftSignature


# =============================================================================
# Geometry Functions
# =============================================================================

def slant_range_from_altitudes(
    sensor_altitude_km: float,
    target_altitude_km: float,
    horizontal_range_km: float,
) -> float:
    """
    Calculate slant range from sensor and target altitudes.

    Parameters
    ----------
    sensor_altitude_km : float
        Sensor altitude (km)
    target_altitude_km : float
        Target altitude (km)
    horizontal_range_km : float
        Horizontal (ground) range (km)

    Returns
    -------
    slant_range_km : float
        Slant range (km)
    """
    delta_h = abs(target_altitude_km - sensor_altitude_km)
    slant_range = np.sqrt(horizontal_range_km**2 + delta_h**2)
    return slant_range


def elevation_angle(
    sensor_altitude_km: float,
    target_altitude_km: float,
    horizontal_range_km: float,
) -> float:
    """
    Calculate elevation angle from sensor to target.

    Parameters
    ----------
    sensor_altitude_km : float
        Sensor altitude (km)
    target_altitude_km : float
        Target altitude (km)
    horizontal_range_km : float
        Horizontal range (km)

    Returns
    -------
    elevation_deg : float
        Elevation angle (degrees), positive if looking up
    """
    delta_h = target_altitude_km - sensor_altitude_km
    if horizontal_range_km == 0:
        return 90.0 if delta_h > 0 else -90.0
    elevation_rad = np.arctan2(delta_h, horizontal_range_km)
    return np.degrees(elevation_rad)


def mean_path_altitude(
    sensor_altitude_km: float,
    target_altitude_km: float,
) -> float:
    """
    Calculate mean altitude along the path.

    Parameters
    ----------
    sensor_altitude_km : float
        Sensor altitude (km)
    target_altitude_km : float
        Target altitude (km)

    Returns
    -------
    mean_alt_km : float
        Mean path altitude (km)
    """
    return (sensor_altitude_km + target_altitude_km) / 2.0


# =============================================================================
# Atmospheric Transmission with Slant Path
# =============================================================================

def atmospheric_transmission_slant(
    sensor_altitude_km: float,
    target_altitude_km: float,
    slant_range_km: float,
    wavelength_min: float,
    wavelength_max: float,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    n_segments: int = 10,
) -> float:
    """
    Calculate atmospheric transmission along a slant path.

    Integrates extinction along the path, accounting for
    altitude-dependent atmospheric properties.

    Parameters
    ----------
    sensor_altitude_km : float
        Sensor altitude (km)
    target_altitude_km : float
        Target altitude (km)
    slant_range_km : float
        Slant range (km)
    wavelength_min : float
        Minimum wavelength (um)
    wavelength_max : float
        Maximum wavelength (um)
    visibility_km : float
        Surface meteorological visibility (km)
    humidity_percent : float
        Surface relative humidity (%)
    n_segments : int
        Number of path segments for integration

    Returns
    -------
    transmission : float
        Atmospheric transmission (0-1)
    """
    if slant_range_km <= 0:
        return 1.0

    center_wl = (wavelength_min + wavelength_max) / 2

    # Divide path into segments
    segment_length = slant_range_km / n_segments

    # Altitude at each segment midpoint
    alt_min = min(sensor_altitude_km, target_altitude_km)
    alt_max = max(sensor_altitude_km, target_altitude_km)

    total_optical_depth = 0.0

    for i in range(n_segments):
        # Fraction along path (0 to 1)
        frac = (i + 0.5) / n_segments

        # Altitude at this segment
        if sensor_altitude_km <= target_altitude_km:
            seg_alt = sensor_altitude_km + frac * (target_altitude_km - sensor_altitude_km)
        else:
            seg_alt = sensor_altitude_km - frac * (sensor_altitude_km - target_altitude_km)

        # Altitude-dependent extinction
        h_scale_aerosol = 1.2  # km scale height for aerosols
        h_scale_h2o = 2.0      # km scale height for water vapor

        # Aerosol extinction (decreases with altitude)
        beta_aerosol_surface = 3.912 / visibility_km
        beta_aerosol_surface *= (0.55 / center_wl) ** 0.5
        beta_aerosol = beta_aerosol_surface * np.exp(-seg_alt / h_scale_aerosol)

        # Water vapor (decreases with altitude)
        humidity_factor = humidity_percent / 50.0 * np.exp(-seg_alt / h_scale_h2o)

        if 3.0 <= center_wl <= 5.0:
            # MWIR 3-5um: Good atmospheric window, H2O absorption mainly at edges
            # The 3.4-4.8um core is relatively clean
            beta_h2o = 0.06 * humidity_factor
            beta_co2 = 0.015 if center_wl > 4.2 else 0.005
        elif 8.0 <= center_wl <= 12.0:
            # LWIR 8-12um: Very clean window, some ozone at 9.6um
            beta_h2o = 0.05 * humidity_factor
            beta_co2 = 0.01
        else:
            beta_h2o = 0.2 * humidity_factor
            beta_co2 = 0.01

        # CO2 also decreases with altitude (but slower)
        beta_co2 *= np.exp(-seg_alt / 8.0)

        # Total extinction for this segment
        beta_total = beta_aerosol + beta_h2o + beta_co2
        total_optical_depth += beta_total * segment_length

    transmission = np.exp(-total_optical_depth)
    return np.clip(transmission, 0.0, 1.0)


@dataclass
class DetectionResult:
    """
    Result of detection range calculation.

    Attributes
    ----------
    detection_range_m : float
        Maximum detection range in meters
    detection_range_km : float
        Maximum detection range in kilometers
    snr_at_range : float
        SNR at detection range (should equal threshold)
    target_irradiance : float
        Target irradiance at detection range (W/cm^2)
    atmospheric_transmission : float
        Atmospheric transmission at detection range
    """
    detection_range_m: float
    snr_at_range: float
    target_irradiance: float
    atmospheric_transmission: float

    @property
    def detection_range_km(self) -> float:
        """Detection range in kilometers."""
        return self.detection_range_m / 1000.0


def atmospheric_transmission_ir(
    range_km: float,
    wavelength_min: float,
    wavelength_max: float,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    altitude_km: float = 5.0,
) -> float:
    """
    Calculate atmospheric transmission in IR bands.

    Uses simplified models for:
    - Molecular absorption (H2O, CO2)
    - Aerosol scattering

    Parameters
    ----------
    range_km : float
        Slant range in km
    wavelength_min : float
        Minimum wavelength (um)
    wavelength_max : float
        Maximum wavelength (um)
    visibility_km : float
        Meteorological visibility (km)
    humidity_percent : float
        Relative humidity (%)
    altitude_km : float
        Mean path altitude (km)

    Returns
    -------
    transmission : float
        Atmospheric transmission (0-1)
    """
    center_wl = (wavelength_min + wavelength_max) / 2

    # Altitude scaling for water vapor (exponential decrease)
    h_scale = 2.0  # km scale height for water vapor
    humidity_factor = humidity_percent / 50.0 * np.exp(-altitude_km / h_scale)

    # Aerosol extinction coefficient (km^-1)
    # Based on Koschmieder: visibility = 3.912 / extinction
    beta_aerosol = 3.912 / visibility_km

    # Wavelength scaling for aerosol scattering
    # Approximate lambda^-1 dependence in IR
    beta_aerosol *= (0.55 / center_wl) ** 0.5  # Reduced dependence in IR

    # Molecular absorption coefficients (simplified, km^-1)
    if 3.0 <= center_wl <= 5.0:
        # MWIR 3-5um: Good atmospheric window, H2O absorption mainly at edges
        # The 3.4-4.8um core is relatively clean
        beta_h2o = 0.06 * humidity_factor
        beta_co2 = 0.015 if center_wl > 4.2 else 0.005
    elif 8.0 <= center_wl <= 12.0:
        # LWIR 8-12um: Very clean window, some ozone at 9.6um
        beta_h2o = 0.05 * humidity_factor
        beta_co2 = 0.01
    else:
        # Outside main windows
        beta_h2o = 0.2 * humidity_factor
        beta_co2 = 0.01

    # Total extinction
    beta_total = beta_aerosol + beta_h2o + beta_co2

    # Beer-Lambert transmission
    transmission = np.exp(-beta_total * range_km)

    return np.clip(transmission, 0.0, 1.0)


def calculate_snr_vs_range(
    detector: FPADetector,
    target: AircraftSignature,
    ranges_km: np.ndarray,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    altitude_km: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate SNR as a function of range.

    Parameters
    ----------
    detector : FPADetector
        Detector specification
    target : AircraftSignature
        Target signature
    ranges_km : np.ndarray
        Array of ranges to calculate (km)
    visibility_km : float
        Meteorological visibility (km)
    humidity_percent : float
        Relative humidity (%)
    altitude_km : float
        Mean path altitude (km)

    Returns
    -------
    snr : np.ndarray
        Signal-to-noise ratio at each range
    irradiance : np.ndarray
        Target irradiance at each range (W/cm^2)
    transmission : np.ndarray
        Atmospheric transmission at each range
    """
    wl_min, wl_max = detector.spectral_band

    snr = np.zeros_like(ranges_km)
    irradiance = np.zeros_like(ranges_km)
    transmission = np.zeros_like(ranges_km)

    nei = detector.noise_equivalent_irradiance()

    for i, range_km in enumerate(ranges_km):
        range_m = range_km * 1000.0

        # Atmospheric transmission
        trans = atmospheric_transmission_ir(
            range_km, wl_min, wl_max,
            visibility_km, humidity_percent, altitude_km
        )
        transmission[i] = trans

        # Target irradiance at detector (W/m^2 -> W/cm^2)
        irrad = target.irradiance_at_range(
            range_m, wl_min, wl_max, trans
        ) * 1e-4  # W/m^2 to W/cm^2

        irradiance[i] = irrad

        # SNR
        snr[i] = irrad / nei

    return snr, irradiance, transmission


def calculate_detection_range(
    detector: FPADetector,
    target: AircraftSignature,
    snr_threshold: float = 5.0,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    altitude_km: float = 5.0,
    max_range_km: float = 100.0,
) -> DetectionResult:
    """
    Calculate maximum detection range for given SNR threshold.

    Uses root-finding to solve: SNR(R) = threshold

    Parameters
    ----------
    detector : FPADetector
        Detector specification
    target : AircraftSignature
        Target signature
    snr_threshold : float
        Required SNR for detection (typically 3-10)
    visibility_km : float
        Meteorological visibility (km)
    humidity_percent : float
        Relative humidity (%)
    altitude_km : float
        Mean path altitude (km)
    max_range_km : float
        Maximum range to search (km)

    Returns
    -------
    result : DetectionResult
        Detection range and associated parameters
    """
    wl_min, wl_max = detector.spectral_band
    nei = detector.noise_equivalent_irradiance()

    def snr_minus_threshold(range_km: float) -> float:
        """SNR - threshold (zero at detection range)."""
        if range_km <= 0:
            return 1e10  # Very high SNR at zero range

        range_m = range_km * 1000.0

        trans = atmospheric_transmission_ir(
            range_km, wl_min, wl_max,
            visibility_km, humidity_percent, altitude_km
        )

        irrad = target.irradiance_at_range(
            range_m, wl_min, wl_max, trans
        ) * 1e-4

        snr = irrad / nei
        return snr - snr_threshold

    # Check if detection is possible at max range
    snr_at_max = snr_minus_threshold(max_range_km) + snr_threshold
    if snr_at_max >= snr_threshold:
        # Can detect at max range, return max
        range_m = max_range_km * 1000.0
        trans = atmospheric_transmission_ir(
            max_range_km, wl_min, wl_max,
            visibility_km, humidity_percent, altitude_km
        )
        irrad = target.irradiance_at_range(
            range_m, wl_min, wl_max, trans
        ) * 1e-4

        return DetectionResult(
            detection_range_m=range_m,
            snr_at_range=snr_at_max,
            target_irradiance=irrad,
            atmospheric_transmission=trans,
        )

    # Check if detection is possible at minimum range
    min_range_km = 0.1
    snr_at_min = snr_minus_threshold(min_range_km) + snr_threshold
    if snr_at_min < snr_threshold:
        # Cannot detect even at close range
        return DetectionResult(
            detection_range_m=0.0,
            snr_at_range=snr_at_min,
            target_irradiance=0.0,
            atmospheric_transmission=1.0,
        )

    # Find detection range using root finding
    try:
        detection_range_km = brentq(
            snr_minus_threshold,
            min_range_km,
            max_range_km,
            xtol=0.01  # 10m precision
        )
    except ValueError:
        # Root finding failed, use interpolation
        detection_range_km = max_range_km / 2

    range_m = detection_range_km * 1000.0
    trans = atmospheric_transmission_ir(
        detection_range_km, wl_min, wl_max,
        visibility_km, humidity_percent, altitude_km
    )
    irrad = target.irradiance_at_range(
        range_m, wl_min, wl_max, trans
    ) * 1e-4

    return DetectionResult(
        detection_range_m=range_m,
        snr_at_range=snr_threshold,
        target_irradiance=irrad,
        atmospheric_transmission=trans,
    )


def compare_detectors(
    detectors: List[FPADetector],
    target: AircraftSignature,
    snr_threshold: float = 5.0,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    altitude_km: float = 5.0,
) -> List[DetectionResult]:
    """
    Compare detection range of multiple detectors.

    Parameters
    ----------
    detectors : List[FPADetector]
        List of detector specifications
    target : AircraftSignature
        Target signature
    snr_threshold : float
        Required SNR for detection
    visibility_km : float
        Meteorological visibility (km)
    humidity_percent : float
        Relative humidity (%)
    altitude_km : float
        Mean path altitude (km)

    Returns
    -------
    results : List[DetectionResult]
        Detection result for each detector
    """
    results = []
    for detector in detectors:
        result = calculate_detection_range(
            detector, target, snr_threshold,
            visibility_km, humidity_percent, altitude_km
        )
        results.append(result)
    return results


# =============================================================================
# Altitude-Aware Detection Functions
# =============================================================================

def calculate_detection_range_slant(
    detector: FPADetector,
    target: AircraftSignature,
    sensor_altitude_km: float,
    target_altitude_km: float,
    snr_threshold: float = 5.0,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    max_range_km: float = 100.0,
) -> DetectionResult:
    """
    Calculate maximum detection range with slant path geometry.

    Parameters
    ----------
    detector : FPADetector
        Detector specification
    target : AircraftSignature
        Target signature
    sensor_altitude_km : float
        Sensor platform altitude (km)
    target_altitude_km : float
        Target altitude (km)
    snr_threshold : float
        Required SNR for detection
    visibility_km : float
        Surface meteorological visibility (km)
    humidity_percent : float
        Surface relative humidity (%)
    max_range_km : float
        Maximum slant range to search (km)

    Returns
    -------
    result : DetectionResult
        Detection range and associated parameters
    """
    wl_min, wl_max = detector.spectral_band
    nei = detector.noise_equivalent_irradiance()

    def snr_minus_threshold(slant_range_km: float) -> float:
        """SNR - threshold (zero at detection range)."""
        if slant_range_km <= 0:
            return 1e10

        range_m = slant_range_km * 1000.0

        trans = atmospheric_transmission_slant(
            sensor_altitude_km, target_altitude_km, slant_range_km,
            wl_min, wl_max, visibility_km, humidity_percent
        )

        irrad = target.irradiance_at_range(
            range_m, wl_min, wl_max, trans
        ) * 1e-4

        snr = irrad / nei
        return snr - snr_threshold

    # Check if detection is possible at max range
    snr_at_max = snr_minus_threshold(max_range_km) + snr_threshold
    if snr_at_max >= snr_threshold:
        range_m = max_range_km * 1000.0
        trans = atmospheric_transmission_slant(
            sensor_altitude_km, target_altitude_km, max_range_km,
            wl_min, wl_max, visibility_km, humidity_percent
        )
        irrad = target.irradiance_at_range(
            range_m, wl_min, wl_max, trans
        ) * 1e-4

        return DetectionResult(
            detection_range_m=range_m,
            snr_at_range=snr_at_max,
            target_irradiance=irrad,
            atmospheric_transmission=trans,
        )

    # Check if detection is possible at minimum range
    min_range_km = 0.1
    snr_at_min = snr_minus_threshold(min_range_km) + snr_threshold
    if snr_at_min < snr_threshold:
        return DetectionResult(
            detection_range_m=0.0,
            snr_at_range=snr_at_min,
            target_irradiance=0.0,
            atmospheric_transmission=1.0,
        )

    # Find detection range using root finding
    try:
        detection_range_km = brentq(
            snr_minus_threshold,
            min_range_km,
            max_range_km,
            xtol=0.01
        )
    except ValueError:
        detection_range_km = max_range_km / 2

    range_m = detection_range_km * 1000.0
    trans = atmospheric_transmission_slant(
        sensor_altitude_km, target_altitude_km, detection_range_km,
        wl_min, wl_max, visibility_km, humidity_percent
    )
    irrad = target.irradiance_at_range(
        range_m, wl_min, wl_max, trans
    ) * 1e-4

    return DetectionResult(
        detection_range_m=range_m,
        snr_at_range=snr_threshold,
        target_irradiance=irrad,
        atmospheric_transmission=trans,
    )


def scan_altitude_performance(
    detector: FPADetector,
    target: AircraftSignature,
    sensor_altitudes_km: np.ndarray,
    target_altitudes_km: np.ndarray,
    snr_threshold: float = 5.0,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    max_range_km: float = 100.0,
) -> np.ndarray:
    """
    Scan detection range across sensor and target altitude combinations.

    Parameters
    ----------
    detector : FPADetector
        Detector specification
    target : AircraftSignature
        Target signature
    sensor_altitudes_km : np.ndarray
        Array of sensor altitudes to scan (km)
    target_altitudes_km : np.ndarray
        Array of target altitudes to scan (km)
    snr_threshold : float
        Required SNR for detection
    visibility_km : float
        Surface visibility (km)
    humidity_percent : float
        Surface humidity (%)
    max_range_km : float
        Maximum range to search (km)

    Returns
    -------
    detection_ranges : np.ndarray
        2D array of detection ranges (km), shape (len(sensor_altitudes), len(target_altitudes))
    """
    n_sensor = len(sensor_altitudes_km)
    n_target = len(target_altitudes_km)
    detection_ranges = np.zeros((n_sensor, n_target))

    for i, sensor_alt in enumerate(sensor_altitudes_km):
        for j, target_alt in enumerate(target_altitudes_km):
            result = calculate_detection_range_slant(
                detector, target,
                sensor_alt, target_alt,
                snr_threshold, visibility_km, humidity_percent, max_range_km
            )
            detection_ranges[i, j] = result.detection_range_km

    return detection_ranges
