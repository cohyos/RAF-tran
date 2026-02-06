"""
Validation Suite for RAF-tran
=============================

This module provides automated validation tests comparing RAF-tran
results against established benchmarks from:

1. MODTRAN (reference radiative transfer code)
2. US Standard Atmosphere 1976
3. Published literature values
4. Analytical solutions

All validation tests work offline using pre-computed benchmark data.

Usage
-----
>>> from raf_tran.validation import run_all_validations
>>> results = run_all_validations()
>>> results.print_summary()

Or run specific validations:
>>> from raf_tran.validation import validate_rayleigh, validate_transmission
>>> validate_rayleigh()
>>> validate_transmission()

Benchmark Data
--------------
Benchmark data is stored in JSON format and can be updated by running
validation against new MODTRAN outputs (requires MODTRAN license).
"""

from raf_tran.validation.benchmarks import (
    ValidationResult,
    ValidationSuite,
    run_all_validations,
    run_validation,
    list_validations,
)

from raf_tran.validation.tests import (
    validate_rayleigh_optical_depth,
    validate_transmission_spectrum,
    validate_thermal_emission,
    validate_solar_irradiance,
    validate_atmospheric_profiles,
    validate_mie_scattering,
    validate_turbulence_parameters,
)

__all__ = [
    # Main validation interface
    "ValidationResult",
    "ValidationSuite",
    "run_all_validations",
    "run_validation",
    "list_validations",
    # Individual validation tests
    "validate_rayleigh_optical_depth",
    "validate_transmission_spectrum",
    "validate_thermal_emission",
    "validate_solar_irradiance",
    "validate_atmospheric_profiles",
    "validate_mie_scattering",
    "validate_turbulence_parameters",
]
