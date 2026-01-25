"""
Monte Carlo Simulation for Detection Range Uncertainty
======================================================

This module provides Monte Carlo simulation capabilities for estimating
detection range uncertainty by sampling target, sensor, and atmospheric
parameters.

Based on EO-simplified-simulation Monte Carlo implementation.

Features:
- Multiple distribution types (uniform, normal, triangular, lognormal)
- Sampling of target, sensor, and atmospheric parameters
- Confidence interval calculation
- Sensitivity analysis support

References:
- EO-simplified-simulation Monte Carlo module
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Tuple, Callable
import numpy as np
from copy import deepcopy

from raf_tran.detectors.fpa import FPADetector
from raf_tran.targets.aircraft import AircraftSignature
from raf_tran.detection.range_equation import (
    calculate_detection_range,
    calculate_detection_range_slant,
)


class DistributionType(Enum):
    """Supported probability distributions for parameter sampling."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"


@dataclass
class ParameterDistribution:
    """
    Configuration for a sampled parameter.

    Attributes
    ----------
    name : str
        Parameter name
    distribution : DistributionType
        Type of probability distribution
    mean : float
        Mean or central value
    std : Optional[float]
        Standard deviation (for normal/lognormal)
    min_val : Optional[float]
        Minimum value (for uniform/triangular)
    max_val : Optional[float]
        Maximum value (for uniform/triangular)
    """
    name: str
    distribution: DistributionType
    mean: float
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation.

    Attributes
    ----------
    n_samples : int
        Number of Monte Carlo samples
    seed : Optional[int]
        Random seed for reproducibility
    target_params : List[ParameterDistribution]
        Target parameter variations
    sensor_params : List[ParameterDistribution]
        Sensor parameter variations
    atmosphere_params : List[ParameterDistribution]
        Atmospheric parameter variations
    """
    n_samples: int = 1000
    seed: Optional[int] = None

    # Target parameter variations
    target_params: List[ParameterDistribution] = field(default_factory=lambda: [
        ParameterDistribution(
            "delta_temp", DistributionType.NORMAL,
            mean=0.0, std=50.0
        ),  # K variation in exhaust temp
        ParameterDistribution(
            "area_factor", DistributionType.LOGNORMAL,
            mean=1.0, std=0.15
        ),  # Multiplicative factor on areas
        ParameterDistribution(
            "emissivity_factor", DistributionType.TRIANGULAR,
            mean=1.0, min_val=0.9, max_val=1.1
        ),  # Emissivity variation
    ])

    # Sensor parameter variations
    sensor_params: List[ParameterDistribution] = field(default_factory=lambda: [
        ParameterDistribution(
            "d_star_factor", DistributionType.LOGNORMAL,
            mean=1.0, std=0.1
        ),  # D* variation
        ParameterDistribution(
            "read_noise_factor", DistributionType.LOGNORMAL,
            mean=1.0, std=0.1
        ),  # Read noise variation
    ])

    # Atmospheric parameter variations
    atmosphere_params: List[ParameterDistribution] = field(default_factory=lambda: [
        ParameterDistribution(
            "visibility_km", DistributionType.NORMAL,
            mean=23.0, std=5.0
        ),
        ParameterDistribution(
            "humidity_percent", DistributionType.NORMAL,
            mean=50.0, std=10.0
        ),
    ])


@dataclass
class MonteCarloResult:
    """
    Results from Monte Carlo simulation.

    Attributes
    ----------
    detection_ranges_km : np.ndarray
        Array of detection ranges from each sample
    mean_range_km : float
        Mean detection range
    std_range_km : float
        Standard deviation of detection range
    median_range_km : float
        Median detection range
    p5_range_km : float
        5th percentile (lower bound of 90% CI)
    p25_range_km : float
        25th percentile
    p75_range_km : float
        75th percentile
    p95_range_km : float
        95th percentile (upper bound of 90% CI)
    sampled_params : Dict[str, np.ndarray]
        Sampled parameter values for sensitivity analysis
    """
    detection_ranges_km: np.ndarray
    mean_range_km: float
    std_range_km: float
    median_range_km: float
    p5_range_km: float
    p25_range_km: float
    p75_range_km: float
    p95_range_km: float
    sampled_params: Dict[str, np.ndarray]

    @property
    def confidence_interval_90(self) -> Tuple[float, float]:
        """90% confidence interval [p5, p95]."""
        return (self.p5_range_km, self.p95_range_km)

    @property
    def confidence_interval_50(self) -> Tuple[float, float]:
        """50% confidence interval [p25, p75]."""
        return (self.p25_range_km, self.p75_range_km)

    def format_result(self, confidence: float = 0.90) -> str:
        """Format result as 'mean [low-high]' string."""
        if confidence == 0.90:
            low, high = self.confidence_interval_90
        else:
            low, high = self.confidence_interval_50
        return f"{self.mean_range_km:.1f} [{low:.1f}-{high:.1f}]"


def sample_parameter(
    param: ParameterDistribution,
    rng: np.random.Generator,
) -> float:
    """
    Sample a single value from the parameter distribution.

    Parameters
    ----------
    param : ParameterDistribution
        Parameter distribution configuration
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    value : float
        Sampled value
    """
    if param.distribution == DistributionType.UNIFORM:
        return rng.uniform(param.min_val, param.max_val)
    elif param.distribution == DistributionType.NORMAL:
        value = rng.normal(param.mean, param.std)
        # Clip to reasonable bounds if specified
        if param.min_val is not None:
            value = max(value, param.min_val)
        if param.max_val is not None:
            value = min(value, param.max_val)
        return value
    elif param.distribution == DistributionType.TRIANGULAR:
        return rng.triangular(param.min_val, param.mean, param.max_val)
    elif param.distribution == DistributionType.LOGNORMAL:
        # For lognormal, std is the sigma of the underlying normal
        return param.mean * np.exp(rng.normal(0, param.std))
    else:
        raise ValueError(f"Unknown distribution type: {param.distribution}")


def apply_target_perturbations(
    target: AircraftSignature,
    samples: Dict[str, float],
) -> AircraftSignature:
    """
    Apply sampled perturbations to target signature.

    Parameters
    ----------
    target : AircraftSignature
        Base target signature
    samples : Dict[str, float]
        Sampled parameter values

    Returns
    -------
    perturbed_target : AircraftSignature
        Target with perturbations applied
    """
    perturbed = deepcopy(target)

    if 'delta_temp' in samples:
        perturbed.exhaust_temp += samples['delta_temp']
        perturbed.nozzle_temp += samples['delta_temp'] * 0.7  # Nozzle tracks exhaust

    if 'area_factor' in samples:
        perturbed.exhaust_area *= samples['area_factor']
        perturbed.nozzle_area *= samples['area_factor']
        perturbed.skin_area *= samples['area_factor']

    if 'emissivity_factor' in samples:
        perturbed.exhaust_emissivity *= samples['emissivity_factor']
        perturbed.exhaust_emissivity = min(perturbed.exhaust_emissivity, 1.0)
        perturbed.skin_emissivity *= samples['emissivity_factor']
        perturbed.skin_emissivity = min(perturbed.skin_emissivity, 1.0)

    return perturbed


def apply_detector_perturbations(
    detector: FPADetector,
    samples: Dict[str, float],
) -> FPADetector:
    """
    Apply sampled perturbations to detector.

    Parameters
    ----------
    detector : FPADetector
        Base detector
    samples : Dict[str, float]
        Sampled parameter values

    Returns
    -------
    perturbed_detector : FPADetector
        Detector with perturbations applied
    """
    perturbed = deepcopy(detector)

    if 'd_star_factor' in samples:
        perturbed.d_star *= samples['d_star_factor']

    if 'read_noise_factor' in samples:
        perturbed.read_noise *= samples['read_noise_factor']

    if 'dark_current_factor' in samples:
        perturbed.dark_current *= samples['dark_current_factor']

    return perturbed


def monte_carlo_detection_range(
    detector: FPADetector,
    target: AircraftSignature,
    config: MonteCarloConfig,
    snr_threshold: float = 5.0,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    sensor_altitude_km: float = 0.0,
    target_altitude_km: float = 10.0,
    use_slant_path: bool = True,
) -> MonteCarloResult:
    """
    Run Monte Carlo ensemble for detection range.

    Samples target, sensor, and atmospheric parameters to produce
    distributions of detection ranges.

    Parameters
    ----------
    detector : FPADetector
        Base detector configuration
    target : AircraftSignature
        Base target signature
    config : MonteCarloConfig
        Monte Carlo configuration
    snr_threshold : float
        SNR threshold for detection
    visibility_km : float
        Base visibility (km)
    humidity_percent : float
        Base humidity (%)
    sensor_altitude_km : float
        Sensor altitude (km)
    target_altitude_km : float
        Target altitude (km)
    use_slant_path : bool
        Use slant path atmospheric model

    Returns
    -------
    result : MonteCarloResult
        Monte Carlo simulation results
    """
    rng = np.random.default_rng(config.seed)

    detection_ranges = []
    sampled_params: Dict[str, List[float]] = {}

    # Initialize sampled_params dict
    for param_list in [config.target_params, config.sensor_params, config.atmosphere_params]:
        for param in param_list:
            sampled_params[param.name] = []

    for i in range(config.n_samples):
        # Sample all parameters
        target_samples = {
            p.name: sample_parameter(p, rng)
            for p in config.target_params
        }
        sensor_samples = {
            p.name: sample_parameter(p, rng)
            for p in config.sensor_params
        }
        atm_samples = {
            p.name: sample_parameter(p, rng)
            for p in config.atmosphere_params
        }

        # Store samples
        for name, val in {**target_samples, **sensor_samples, **atm_samples}.items():
            sampled_params[name].append(val)

        # Apply perturbations
        perturbed_target = apply_target_perturbations(target, target_samples)
        perturbed_detector = apply_detector_perturbations(detector, sensor_samples)

        # Get atmospheric parameters
        vis = atm_samples.get('visibility_km', visibility_km)
        hum = atm_samples.get('humidity_percent', humidity_percent)

        # Clip to reasonable values
        vis = max(vis, 1.0)  # Min 1 km visibility
        hum = np.clip(hum, 10.0, 100.0)  # 10-100% humidity

        # Calculate detection range
        if use_slant_path:
            result = calculate_detection_range_slant(
                perturbed_detector, perturbed_target,
                sensor_altitude_km, target_altitude_km,
                snr_threshold, vis, hum,
            )
        else:
            mean_alt = (sensor_altitude_km + target_altitude_km) / 2
            result = calculate_detection_range(
                perturbed_detector, perturbed_target,
                snr_threshold, vis, hum, mean_alt,
            )

        detection_ranges.append(result.detection_range_km)

    ranges = np.array(detection_ranges)

    return MonteCarloResult(
        detection_ranges_km=ranges,
        mean_range_km=float(np.mean(ranges)),
        std_range_km=float(np.std(ranges)),
        median_range_km=float(np.median(ranges)),
        p5_range_km=float(np.percentile(ranges, 5)),
        p25_range_km=float(np.percentile(ranges, 25)),
        p75_range_km=float(np.percentile(ranges, 75)),
        p95_range_km=float(np.percentile(ranges, 95)),
        sampled_params={k: np.array(v) for k, v in sampled_params.items()},
    )


def monte_carlo_multi_detector(
    detectors: List[FPADetector],
    target: AircraftSignature,
    config: MonteCarloConfig,
    snr_threshold: float = 5.0,
    visibility_km: float = 23.0,
    humidity_percent: float = 50.0,
    sensor_altitude_km: float = 0.0,
    target_altitude_km: float = 10.0,
) -> Dict[str, MonteCarloResult]:
    """
    Run Monte Carlo for multiple detectors with correlated samples.

    Uses the same atmospheric and target perturbations across all
    detectors for fair comparison.

    Parameters
    ----------
    detectors : List[FPADetector]
        List of detectors to compare
    target : AircraftSignature
        Target signature
    config : MonteCarloConfig
        Monte Carlo configuration
    snr_threshold : float
        SNR threshold
    visibility_km : float
        Base visibility
    humidity_percent : float
        Base humidity
    sensor_altitude_km : float
        Sensor altitude
    target_altitude_km : float
        Target altitude

    Returns
    -------
    results : Dict[str, MonteCarloResult]
        Results keyed by detector name
    """
    rng = np.random.default_rng(config.seed)

    # Storage for each detector
    results_data: Dict[str, List[float]] = {d.name: [] for d in detectors}
    sampled_params: Dict[str, List[float]] = {}

    # Initialize sampled_params
    for param_list in [config.target_params, config.atmosphere_params]:
        for param in param_list:
            sampled_params[param.name] = []

    for i in range(config.n_samples):
        # Sample shared parameters (target and atmosphere)
        target_samples = {
            p.name: sample_parameter(p, rng)
            for p in config.target_params
        }
        atm_samples = {
            p.name: sample_parameter(p, rng)
            for p in config.atmosphere_params
        }

        # Store shared samples
        for name, val in {**target_samples, **atm_samples}.items():
            sampled_params[name].append(val)

        # Apply target perturbations (shared)
        perturbed_target = apply_target_perturbations(target, target_samples)

        # Get atmospheric parameters
        vis = atm_samples.get('visibility_km', visibility_km)
        hum = atm_samples.get('humidity_percent', humidity_percent)
        vis = max(vis, 1.0)
        hum = np.clip(hum, 10.0, 100.0)

        # Calculate for each detector with independent sensor noise
        for detector in detectors:
            sensor_samples = {
                p.name: sample_parameter(p, rng)
                for p in config.sensor_params
            }
            perturbed_detector = apply_detector_perturbations(detector, sensor_samples)

            result = calculate_detection_range_slant(
                perturbed_detector, perturbed_target,
                sensor_altitude_km, target_altitude_km,
                snr_threshold, vis, hum,
            )
            results_data[detector.name].append(result.detection_range_km)

    # Build results
    results = {}
    for detector in detectors:
        ranges = np.array(results_data[detector.name])
        results[detector.name] = MonteCarloResult(
            detection_ranges_km=ranges,
            mean_range_km=float(np.mean(ranges)),
            std_range_km=float(np.std(ranges)),
            median_range_km=float(np.median(ranges)),
            p5_range_km=float(np.percentile(ranges, 5)),
            p25_range_km=float(np.percentile(ranges, 25)),
            p75_range_km=float(np.percentile(ranges, 75)),
            p95_range_km=float(np.percentile(ranges, 95)),
            sampled_params={k: np.array(v) for k, v in sampled_params.items()},
        )

    return results


def default_monte_carlo_config(n_samples: int = 1000, seed: Optional[int] = None) -> MonteCarloConfig:
    """
    Create a default Monte Carlo configuration.

    Parameters
    ----------
    n_samples : int
        Number of samples
    seed : Optional[int]
        Random seed

    Returns
    -------
    config : MonteCarloConfig
        Default configuration
    """
    return MonteCarloConfig(n_samples=n_samples, seed=seed)
