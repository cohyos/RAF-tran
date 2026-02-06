"""
Sensor Optimizer Module
=======================

Parameter sweep and performance optimization for sensor assemblies.
Enables searching across FPA choices and operating parameter ranges to
find optimal sensor configurations for given mission requirements.

Optimization Modes
------------------
- Single-FPA parameter sweep: vary optics/operating params for one FPA
- Multi-FPA comparison: find best FPA for given constraints
- Pareto-optimal frontier: identify non-dominated configurations
- Constraint-based search: find all valid configs meeting requirements

All functions return results sorted by a user-specified objective metric.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
import math
import itertools

from raf_tran.fpa_library.models import FPASpec, SpectralBand
from raf_tran.fpa_library.sensor import (
    SensorAssembly, OpticsConfig, OperatingParams, ValidationResult,
)
from raf_tran.fpa_library.database import get_fpa_database, search_fpas


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ParameterRange:
    """
    A range of values to sweep for a single parameter.

    Parameters
    ----------
    name : str
        Parameter name (e.g., 'focal_length_mm', 'f_number')
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    steps : int
        Number of steps in the sweep
    log_scale : bool
        If True, use logarithmic spacing
    """
    name: str
    min_val: float
    max_val: float
    steps: int = 10
    log_scale: bool = False

    @property
    def values(self) -> List[float]:
        """Generate the sweep values."""
        if self.steps <= 1:
            return [self.min_val]
        if self.log_scale:
            log_min = math.log10(self.min_val)
            log_max = math.log10(self.max_val)
            return [10 ** (log_min + i * (log_max - log_min) / (self.steps - 1))
                    for i in range(self.steps)]
        else:
            return [self.min_val + i * (self.max_val - self.min_val) / (self.steps - 1)
                    for i in range(self.steps)]


@dataclass
class OptimizationConstraint:
    """
    A constraint that must be satisfied for a valid configuration.

    Parameters
    ----------
    metric : str
        Key from performance_summary() dict
    min_val : float, optional
        Minimum acceptable value (inclusive)
    max_val : float, optional
        Maximum acceptable value (inclusive)
    """
    metric: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def is_satisfied(self, value: Any) -> bool:
        if value is None:
            return False
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False
        return True


@dataclass
class OptimizationResult:
    """
    Result of a sensor optimization run.

    Attributes
    ----------
    sensor : SensorAssembly
        The optimized sensor configuration
    score : float
        Objective metric value
    summary : dict
        Full performance summary
    rank : int
        Position in the sorted results (1 = best)
    """
    sensor: SensorAssembly
    score: float
    summary: Dict[str, Any]
    rank: int = 0

    def __repr__(self) -> str:
        return (f"OptimizationResult(rank={self.rank}, score={self.score:.4f}, "
                f"fpa={self.sensor.fpa.name}, "
                f"f={self.sensor.optics.focal_length_mm}mm, "
                f"F/{self.sensor.optics.f_number})")


# =============================================================================
# Sweep Functions
# =============================================================================

def sweep_focal_length(
    fpa: FPASpec,
    f_number: float,
    focal_lengths_mm: List[float],
    operating: Optional[OperatingParams] = None,
    objective: str = 'identification_range_km',
    constraints: Optional[List[OptimizationConstraint]] = None,
    only_valid: bool = True,
) -> List[OptimizationResult]:
    """
    Sweep focal length for a single FPA and rank by objective.

    Parameters
    ----------
    fpa : FPASpec
    f_number : float
    focal_lengths_mm : list of float
    operating : OperatingParams, optional
    objective : str
        Key from performance_summary() to optimize (higher = better)
    constraints : list of OptimizationConstraint, optional
    only_valid : bool
        If True, exclude configurations with validation errors

    Returns
    -------
    results : list of OptimizationResult, sorted best-first
    """
    if operating is None:
        operating = OperatingParams()

    results = []
    for fl in focal_lengths_mm:
        optics = OpticsConfig(focal_length_mm=fl, f_number=f_number)
        sensor = SensorAssembly(fpa, optics, operating, validate_on_init=True)

        if only_valid and not sensor.is_valid:
            continue

        summary = sensor.performance_summary()
        score = summary.get(objective)
        if score is None:
            continue

        if constraints and not _check_constraints(summary, constraints):
            continue

        results.append(OptimizationResult(sensor=sensor, score=score, summary=summary))

    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


def sweep_parameters(
    fpa: FPASpec,
    param_ranges: List[ParameterRange],
    base_optics: Optional[OpticsConfig] = None,
    base_operating: Optional[OperatingParams] = None,
    objective: str = 'identification_range_km',
    constraints: Optional[List[OptimizationConstraint]] = None,
    only_valid: bool = True,
    max_results: int = 100,
) -> List[OptimizationResult]:
    """
    Multi-parameter sweep over optics and operating parameters.

    Supported parameter names:
    - Optics: 'focal_length_mm', 'f_number', 'transmission'
    - Operating: 'integration_time_ms', 'frame_rate_hz', 'tdi_stages', 'num_frames_avg'

    Parameters
    ----------
    fpa : FPASpec
    param_ranges : list of ParameterRange
    base_optics : OpticsConfig, optional
        Starting optics (swept params override)
    base_operating : OperatingParams, optional
        Starting operating params (swept params override)
    objective : str
    constraints : list of OptimizationConstraint, optional
    only_valid : bool
    max_results : int

    Returns
    -------
    results : list of OptimizationResult, sorted best-first
    """
    if base_optics is None:
        base_optics = OpticsConfig(focal_length_mm=100.0, f_number=2.0)
    if base_operating is None:
        base_operating = OperatingParams()

    optics_params = {'focal_length_mm', 'f_number', 'transmission'}
    oper_params = {'integration_time_ms', 'frame_rate_hz', 'tdi_stages', 'num_frames_avg'}

    # Generate all combinations
    param_values = [pr.values for pr in param_ranges]
    param_names = [pr.name for pr in param_ranges]

    results = []
    for combo in itertools.product(*param_values):
        overrides = dict(zip(param_names, combo))

        # Build optics
        opt_kwargs = {
            'focal_length_mm': base_optics.focal_length_mm,
            'f_number': base_optics.f_number,
            'transmission': base_optics.transmission,
            'spectral_filter': base_optics.spectral_filter,
            'num_elements': base_optics.num_elements,
            'obscuration_ratio': base_optics.obscuration_ratio,
        }
        for k in optics_params & overrides.keys():
            opt_kwargs[k] = overrides[k]
        optics = OpticsConfig(**opt_kwargs)

        # Build operating
        op_kwargs = {
            'integration_time_ms': base_operating.integration_time_ms,
            'frame_rate_hz': base_operating.frame_rate_hz,
            'gain': base_operating.gain,
            'window': base_operating.window,
            'tdi_stages': int(base_operating.tdi_stages),
            'num_frames_avg': int(base_operating.num_frames_avg),
            'nuc_enabled': base_operating.nuc_enabled,
        }
        for k in oper_params & overrides.keys():
            val = overrides[k]
            if k in ('tdi_stages', 'num_frames_avg'):
                val = max(1, int(round(val)))
            op_kwargs[k] = val
        operating = OperatingParams(**op_kwargs)

        sensor = SensorAssembly(fpa, optics, operating, validate_on_init=True)

        if only_valid and not sensor.is_valid:
            continue

        summary = sensor.performance_summary()
        score = summary.get(objective)
        if score is None:
            continue

        if constraints and not _check_constraints(summary, constraints):
            continue

        results.append(OptimizationResult(sensor=sensor, score=score, summary=summary))

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:max_results]
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


# =============================================================================
# Multi-FPA Optimization
# =============================================================================

def find_best_fpa(
    optics: OpticsConfig,
    operating: Optional[OperatingParams] = None,
    objective: str = 'identification_range_km',
    constraints: Optional[List[OptimizationConstraint]] = None,
    fpas: Optional[List[FPASpec]] = None,
    spectral_band: Optional[SpectralBand] = None,
    only_valid: bool = True,
) -> List[OptimizationResult]:
    """
    Find the best FPA from the database for given optics and requirements.

    Parameters
    ----------
    optics : OpticsConfig
    operating : OperatingParams, optional
    objective : str
    constraints : list of OptimizationConstraint, optional
    fpas : list of FPASpec, optional
        Custom FPA list. If None, uses database.
    spectral_band : SpectralBand, optional
        Filter database by band
    only_valid : bool

    Returns
    -------
    results : list of OptimizationResult, sorted best-first
    """
    if operating is None:
        operating = OperatingParams()

    if fpas is None:
        if spectral_band:
            fpas = search_fpas(spectral_band=spectral_band)
        else:
            fpas = list(get_fpa_database().values())

    results = []
    for fpa in fpas:
        sensor = SensorAssembly(fpa, optics, operating, validate_on_init=True)

        if only_valid and not sensor.is_valid:
            continue

        summary = sensor.performance_summary()
        score = summary.get(objective)
        if score is None:
            continue

        if constraints and not _check_constraints(summary, constraints):
            continue

        results.append(OptimizationResult(sensor=sensor, score=score, summary=summary))

    results.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


def optimize_sensor(
    fpas: Optional[List[FPASpec]] = None,
    spectral_band: Optional[SpectralBand] = None,
    focal_length_range: Optional[Tuple[float, float]] = None,
    f_number_range: Optional[Tuple[float, float]] = None,
    integration_time_range: Optional[Tuple[float, float]] = None,
    frame_rate_range: Optional[Tuple[float, float]] = None,
    steps: int = 8,
    objective: str = 'identification_range_km',
    constraints: Optional[List[OptimizationConstraint]] = None,
    only_valid: bool = True,
    max_results: int = 50,
) -> List[OptimizationResult]:
    """
    Full sensor optimization: sweep FPAs and parameter ranges simultaneously.

    This is the top-level optimization function. It searches across all
    candidate FPAs and parameter combinations to find the best sensor
    configuration.

    Parameters
    ----------
    fpas : list of FPASpec, optional
        Candidate FPAs. If None, uses database.
    spectral_band : SpectralBand, optional
        Filter database by band (only if fpas is None)
    focal_length_range : (min, max) in mm, optional
    f_number_range : (min, max), optional
    integration_time_range : (min, max) in ms, optional
    frame_rate_range : (min, max) in Hz, optional
    steps : int
        Steps per parameter range
    objective : str
        Metric to optimize (from performance_summary keys)
    constraints : list of OptimizationConstraint, optional
    only_valid : bool
    max_results : int

    Returns
    -------
    results : list of OptimizationResult, sorted best-first
    """
    if fpas is None:
        if spectral_band:
            fpas = search_fpas(spectral_band=spectral_band)
        else:
            fpas = list(get_fpa_database().values())

    # Build parameter ranges
    param_ranges = []
    if focal_length_range:
        param_ranges.append(ParameterRange(
            'focal_length_mm', focal_length_range[0], focal_length_range[1], steps))
    if f_number_range:
        param_ranges.append(ParameterRange(
            'f_number', f_number_range[0], f_number_range[1], steps))
    if integration_time_range:
        param_ranges.append(ParameterRange(
            'integration_time_ms', integration_time_range[0], integration_time_range[1], steps))
    if frame_rate_range:
        param_ranges.append(ParameterRange(
            'frame_rate_hz', frame_rate_range[0], frame_rate_range[1], steps))

    all_results = []

    for fpa in fpas:
        if param_ranges:
            fpa_results = sweep_parameters(
                fpa=fpa,
                param_ranges=param_ranges,
                objective=objective,
                constraints=constraints,
                only_valid=only_valid,
                max_results=max_results,
            )
            all_results.extend(fpa_results)
        else:
            # No parameter sweep - just evaluate each FPA at defaults
            optics = OpticsConfig(focal_length_mm=100.0, f_number=2.0)
            operating = OperatingParams()
            sensor = SensorAssembly(fpa, optics, operating, validate_on_init=True)

            if only_valid and not sensor.is_valid:
                continue

            summary = sensor.performance_summary()
            score = summary.get(objective)
            if score is None:
                continue

            if constraints and not _check_constraints(summary, constraints):
                continue

            all_results.append(OptimizationResult(
                sensor=sensor, score=score, summary=summary))

    # Sort and rank
    all_results.sort(key=lambda r: r.score, reverse=True)
    all_results = all_results[:max_results]
    for i, r in enumerate(all_results):
        r.rank = i + 1

    return all_results


# =============================================================================
# Pareto Frontier
# =============================================================================

def pareto_frontier(
    results: List[OptimizationResult],
    objectives: Tuple[str, str],
    maximize: Tuple[bool, bool] = (True, True),
) -> List[OptimizationResult]:
    """
    Extract the Pareto-optimal frontier from optimization results.

    Parameters
    ----------
    results : list of OptimizationResult
    objectives : (metric1, metric2)
        Two metrics from performance_summary to form the 2D Pareto front
    maximize : (bool, bool)
        Whether each objective should be maximized (True) or minimized (False)

    Returns
    -------
    frontier : list of OptimizationResult
        Non-dominated solutions, sorted by first objective
    """
    points = []
    for r in results:
        v1 = r.summary.get(objectives[0])
        v2 = r.summary.get(objectives[1])
        if v1 is None or v2 is None:
            continue
        # Negate values we want to minimize so we always maximize
        s1 = v1 if maximize[0] else -v1
        s2 = v2 if maximize[1] else -v2
        points.append((s1, s2, r))

    # Find non-dominated points
    frontier = []
    for i, (s1i, s2i, ri) in enumerate(points):
        dominated = False
        for j, (s1j, s2j, rj) in enumerate(points):
            if i == j:
                continue
            if s1j >= s1i and s2j >= s2i and (s1j > s1i or s2j > s2i):
                dominated = True
                break
        if not dominated:
            frontier.append(ri)

    # Sort by first objective
    key_idx = objectives[0]
    frontier.sort(key=lambda r: r.summary.get(key_idx, 0),
                  reverse=maximize[0])

    for i, r in enumerate(frontier):
        r.rank = i + 1

    return frontier


# =============================================================================
# Sensitivity Analysis
# =============================================================================

def sensitivity_analysis(
    sensor: SensorAssembly,
    parameter: str,
    variation_pct: float = 20.0,
    steps: int = 11,
    metrics: Optional[List[str]] = None,
) -> Dict[str, List[Tuple[float, Any]]]:
    """
    Analyze how performance metrics change when one parameter varies.

    Parameters
    ----------
    sensor : SensorAssembly
        Baseline configuration
    parameter : str
        Parameter to vary ('focal_length_mm', 'f_number', 'integration_time_ms',
        'frame_rate_hz')
    variation_pct : float
        Percentage variation around baseline (e.g., 20 = +/-20%)
    steps : int
        Number of evaluation points
    metrics : list of str, optional
        Which metrics to track. Default: identification_range_km, ifov_urad,
        estimated_netd_mk, data_rate_mbps

    Returns
    -------
    results : dict
        Keys are metric names, values are lists of (param_value, metric_value) tuples
    """
    if metrics is None:
        metrics = ['identification_range_km', 'ifov_urad',
                   'estimated_netd_mk', 'data_rate_mbps']

    # Get baseline value
    optics_params = {'focal_length_mm', 'f_number', 'transmission'}
    oper_params = {'integration_time_ms', 'frame_rate_hz'}

    if parameter in optics_params:
        base_val = getattr(sensor.optics, parameter)
    elif parameter in oper_params:
        base_val = getattr(sensor.operating, parameter)
    else:
        raise ValueError(f"Unknown parameter: {parameter}")

    # Generate sweep values
    factor = variation_pct / 100.0
    min_val = base_val * (1.0 - factor)
    max_val = base_val * (1.0 + factor)
    sweep_range = ParameterRange(parameter, min_val, max_val, steps)

    # Evaluate
    output = {m: [] for m in metrics}

    for val in sweep_range.values:
        # Clone configs with override
        if parameter in optics_params:
            opt_kwargs = sensor.optics.to_dict()
            opt_kwargs.pop('spectral_filter', None)
            opt_kwargs[parameter] = val
            optics = OpticsConfig(
                spectral_filter=sensor.optics.spectral_filter,
                **opt_kwargs,
            )
            operating = sensor.operating
        else:
            optics = sensor.optics
            op_kwargs = sensor.operating.to_dict()
            op_kwargs[parameter] = val
            if 'window' in op_kwargs and op_kwargs['window']:
                op_kwargs['window'] = tuple(op_kwargs['window'])
            elif 'window' in op_kwargs:
                op_kwargs.pop('window')
            operating = OperatingParams(**op_kwargs)

        test_sensor = SensorAssembly(
            sensor.fpa, optics, operating, validate_on_init=False)
        summary = test_sensor.performance_summary()

        for m in metrics:
            output[m].append((val, summary.get(m)))

    return output


# =============================================================================
# Utility
# =============================================================================

def _check_constraints(summary: Dict, constraints: List[OptimizationConstraint]) -> bool:
    """Check all constraints against a performance summary."""
    for c in constraints:
        val = summary.get(c.metric)
        if not c.is_satisfied(val):
            return False
    return True


def format_results_table(results: List[OptimizationResult],
                         columns: Optional[List[str]] = None) -> str:
    """
    Format optimization results as an ASCII table.

    Parameters
    ----------
    results : list of OptimizationResult
    columns : list of str, optional
        Summary keys to include. Default: rank, FPA, focal length, F#,
        identification range, NETD, data rate.

    Returns
    -------
    table : str
    """
    if not results:
        return "(no results)"

    if columns is None:
        columns = [
            ('rank', 'Rank', '{:d}'),
            ('fpa', 'FPA', '{:s}'),
            ('focal_length_mm', 'FL(mm)', '{:.0f}'),
            ('f_number', 'F/#', '{:.1f}'),
            ('ifov_urad', 'IFOV(urad)', '{:.1f}'),
            ('identification_range_km', 'ID Range(km)', '{:.2f}'),
            ('estimated_netd_mk', 'NETD(mK)', '{:.1f}'),
            ('data_rate_mbps', 'Data(Mbps)', '{:.1f}'),
        ]

    # Build header
    headers = [c[1] for c in columns]
    rows = []
    for r in results:
        row = []
        for key, label, fmt in columns:
            if key == 'rank':
                row.append(fmt.format(r.rank))
            else:
                val = r.summary.get(key)
                if val is None:
                    row.append('N/A')
                else:
                    row.append(fmt.format(val))
        rows.append(row)

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Format
    sep = '  '
    header_line = sep.join(h.rjust(w) for h, w in zip(headers, widths))
    divider = sep.join('-' * w for w in widths)
    lines = [header_line, divider]
    for row in rows:
        lines.append(sep.join(cell.rjust(w) for cell, w in zip(row, widths)))

    return '\n'.join(lines)
