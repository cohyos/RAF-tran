"""
Validation Benchmark Framework
==============================

Provides a framework for comparing RAF-tran outputs against
benchmark data from MODTRAN and other sources.
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime

import numpy as np


@dataclass
class ValidationResult:
    """
    Result from a single validation test.

    Attributes
    ----------
    test_name : str
        Name of the validation test
    passed : bool
        Whether the test passed
    max_error : float
        Maximum absolute error
    mean_error : float
        Mean absolute error
    rms_error : float
        Root mean square error
    relative_error : float
        Maximum relative error (%)
    tolerance : float
        Error tolerance for pass/fail
    benchmark_source : str
        Source of benchmark data
    details : dict
        Additional test details
    """
    test_name: str
    passed: bool
    max_error: float
    mean_error: float
    rms_error: float
    relative_error: float
    tolerance: float
    benchmark_source: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"{self.test_name}: {status}\n"
            f"  Max Error: {self.max_error:.4g} (tolerance: {self.tolerance:.4g})\n"
            f"  Mean Error: {self.mean_error:.4g}\n"
            f"  RMS Error: {self.rms_error:.4g}\n"
            f"  Relative Error: {self.relative_error:.2f}%\n"
            f"  Benchmark: {self.benchmark_source}"
        )


@dataclass
class ValidationSuite:
    """
    Collection of validation results.

    Attributes
    ----------
    results : list of ValidationResult
        Individual test results
    timestamp : str
        When validation was run
    raf_tran_version : str
        RAF-tran version tested
    """
    results: List[ValidationResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    raf_tran_version: str = ""

    def __post_init__(self):
        import raf_tran
        self.raf_tran_version = raf_tran.__version__

    @property
    def n_tests(self) -> int:
        """Total number of tests."""
        return len(self.results)

    @property
    def n_passed(self) -> int:
        """Number of passed tests."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Number of failed tests."""
        return self.n_tests - self.n_passed

    @property
    def all_passed(self) -> bool:
        """True if all tests passed."""
        return self.n_failed == 0

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    def print_summary(self) -> None:
        """Print summary of validation results."""
        print("\n" + "=" * 60)
        print("RAF-tran Validation Summary")
        print("=" * 60)
        print(f"Version: {self.raf_tran_version}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Tests: {self.n_passed}/{self.n_tests} passed")
        print("-" * 60)

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"{status} {result.test_name}: "
                  f"error={result.max_error:.4g} (tol={result.tolerance:.4g})")

        print("=" * 60)
        if self.all_passed:
            print("All validations PASSED")
        else:
            print(f"WARNING: {self.n_failed} validation(s) FAILED")
        print()

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "raf_tran_version": self.raf_tran_version,
            "timestamp": self.timestamp,
            "n_tests": self.n_tests,
            "n_passed": self.n_passed,
            "all_passed": self.all_passed,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "max_error": float(r.max_error),
                    "mean_error": float(r.mean_error),
                    "rms_error": float(r.rms_error),
                    "relative_error": float(r.relative_error),
                    "tolerance": float(r.tolerance),
                    "benchmark_source": r.benchmark_source,
                }
                for r in self.results
            ],
        }

    def save(self, filepath: str) -> None:
        """Save validation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# Benchmark Data
# =============================================================================

# Pre-computed benchmark values (from MODTRAN, literature, analytical)
BENCHMARKS = {
    "rayleigh_optical_depth": {
        "source": "Bodhaine et al. (1999), US Standard Atmosphere",
        "wavelengths_nm": [350, 400, 450, 500, 550, 600, 700, 800, 1000],
        "optical_depth": [0.385, 0.236, 0.155, 0.108, 0.078, 0.058, 0.034, 0.021, 0.010],
        "tolerance": 0.01,  # 1% absolute
    },
    "transmission_550nm": {
        "source": "MODTRAN US Standard Atmosphere, vertical path",
        "sza_deg": [0, 30, 45, 60, 75, 80],
        "transmission": [0.905, 0.896, 0.878, 0.841, 0.730, 0.635],
        "tolerance": 0.02,  # 2% absolute
    },
    "solar_irradiance_toa": {
        "source": "Gueymard (2004), ASTM E490",
        "wavelengths_nm": [300, 400, 500, 600, 700, 800, 1000, 1500, 2000],
        "irradiance_w_m2_nm": [0.52, 1.74, 1.96, 1.81, 1.48, 1.17, 0.75, 0.33, 0.16],
        "tolerance": 0.05,  # 5%
    },
    "atmospheric_profiles": {
        "source": "US Standard Atmosphere 1976",
        "surface_temperature_k": 288.15,
        "surface_pressure_pa": 101325,
        "tropopause_height_km": 11.0,
        "tropopause_temp_k": 216.65,
        "tolerance_temp": 1.0,  # 1 K
        "tolerance_pressure": 100,  # 100 Pa
    },
    "mie_efficiency": {
        "source": "Bohren & Huffman (1983)",
        "size_parameters": [0.1, 1.0, 5.0, 10.0, 20.0],
        "q_ext_water": [0.00011, 0.26, 3.24, 2.08, 2.10],
        "tolerance": 0.1,  # 10%
    },
    "turbulence_r0": {
        "source": "Andrews & Phillips (2005)",
        # HV 5/7 model at 500nm
        "wavelength_nm": 500,
        "r0_cm": 5.0,  # Fried parameter for HV 5/7
        "tolerance": 0.5,  # 0.5 cm
    },
}


def get_benchmark(name: str) -> Dict:
    """
    Get benchmark data by name.

    Parameters
    ----------
    name : str
        Benchmark name

    Returns
    -------
    benchmark : dict
        Benchmark data and metadata

    Raises
    ------
    KeyError
        If benchmark not found
    """
    if name not in BENCHMARKS:
        raise KeyError(
            f"Unknown benchmark: {name}. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    return BENCHMARKS[name]


# =============================================================================
# Validation Runner
# =============================================================================

# Registry of validation tests
_VALIDATION_TESTS: Dict[str, Callable] = {}


def register_validation(name: str):
    """Decorator to register a validation test."""
    def decorator(func: Callable):
        _VALIDATION_TESTS[name] = func
        return func
    return decorator


def list_validations() -> List[str]:
    """List available validation tests."""
    return list(_VALIDATION_TESTS.keys())


def run_validation(name: str) -> ValidationResult:
    """
    Run a single validation test.

    Parameters
    ----------
    name : str
        Name of validation test

    Returns
    -------
    result : ValidationResult
        Validation result
    """
    if name not in _VALIDATION_TESTS:
        raise KeyError(
            f"Unknown validation: {name}. "
            f"Available: {list_validations()}"
        )

    return _VALIDATION_TESTS[name]()


def run_all_validations(verbose: bool = True) -> ValidationSuite:
    """
    Run all validation tests.

    Parameters
    ----------
    verbose : bool
        Print progress during validation

    Returns
    -------
    suite : ValidationSuite
        Complete validation results
    """
    suite = ValidationSuite()

    # Import tests to register them
    from raf_tran.validation import tests

    for name in list_validations():
        if verbose:
            print(f"Running validation: {name}...")

        try:
            result = run_validation(name)
            suite.add_result(result)

            if verbose:
                status = "PASSED" if result.passed else "FAILED"
                print(f"  {status} (error: {result.max_error:.4g})")

        except Exception as e:
            warnings.warn(f"Validation {name} failed with error: {e}")
            suite.add_result(ValidationResult(
                test_name=name,
                passed=False,
                max_error=np.inf,
                mean_error=np.inf,
                rms_error=np.inf,
                relative_error=100.0,
                tolerance=0.0,
                benchmark_source="error",
                details={"error": str(e)},
            ))

    if verbose:
        suite.print_summary()

    return suite


def compute_errors(
    computed: np.ndarray,
    reference: np.ndarray,
) -> Dict[str, float]:
    """
    Compute error metrics between computed and reference values.

    Parameters
    ----------
    computed : ndarray
        Computed values
    reference : ndarray
        Reference/benchmark values

    Returns
    -------
    errors : dict
        Error metrics (max, mean, rms, relative)
    """
    computed = np.asarray(computed)
    reference = np.asarray(reference)

    abs_error = np.abs(computed - reference)

    max_error = float(np.max(abs_error))
    mean_error = float(np.mean(abs_error))
    rms_error = float(np.sqrt(np.mean(abs_error**2)))

    # Relative error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs((computed - reference) / reference) * 100
        rel_error = np.where(reference != 0, rel_error, 0)
        max_rel_error = float(np.max(rel_error))

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "rms_error": rms_error,
        "relative_error": max_rel_error,
    }
