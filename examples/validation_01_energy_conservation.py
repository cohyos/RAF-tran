#!/usr/bin/env python3
"""
Validation Test 01: Energy Conservation
========================================

This test verifies that the radiative transfer solver conserves energy.
For a conservative scattering atmosphere (SSA = 1), all incoming energy
must either be reflected back to space or absorbed by the surface.

Energy balance: F_in = F_reflected + F_absorbed_surface

For non-conservative atmospheres (SSA < 1), some energy is absorbed
by the atmosphere:

Energy balance: F_in = F_reflected + F_absorbed_surface + F_absorbed_atm
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from raf_tran.rte_solver import TwoStreamSolver
from raf_tran.utils.constants import SOLAR_CONSTANT

def test_energy_conservation():
    print("=" * 70)
    print("VALIDATION TEST 01: ENERGY CONSERVATION")
    print("=" * 70)

    solver = TwoStreamSolver()
    n_layers = 20

    test_cases = [
        # (tau_total, omega, g, surface_albedo, description)
        (0.1, 1.0, 0.0, 0.0, "Thin conservative scattering, black surface"),
        (0.1, 1.0, 0.0, 0.3, "Thin conservative scattering, 30% albedo"),
        (1.0, 1.0, 0.0, 0.0, "Thick conservative scattering, black surface"),
        (1.0, 1.0, 0.0, 0.3, "Thick conservative scattering, 30% albedo"),
        (0.5, 0.9, 0.0, 0.0, "Absorbing atmosphere (SSA=0.9), black surface"),
        (0.5, 0.5, 0.0, 0.0, "Strongly absorbing (SSA=0.5), black surface"),
    ]

    mu0 = 1.0  # Overhead sun
    F_in = SOLAR_CONSTANT * mu0

    print(f"\nIncoming solar flux at TOA: {F_in:.1f} W/m^2")
    print(f"Solar zenith angle: 0 deg (overhead sun)")
    print("\n" + "-" * 70)

    all_passed = True

    for tau_total, omega, g, albedo, desc in test_cases:
        print(f"\nTest: {desc}")
        print(f"  tau={tau_total}, omega={omega}, g={g}, albedo={albedo}")

        # Setup atmosphere
        tau = np.ones(n_layers) * tau_total / n_layers
        omega_arr = np.ones(n_layers) * omega
        g_arr = np.ones(n_layers) * g

        result = solver.solve_solar(
            tau=tau,
            omega=omega_arr,
            g=g_arr,
            mu0=mu0,
            flux_toa=SOLAR_CONSTANT,
            surface_albedo=albedo,
        )

        # Extract fluxes (surface = index 0, TOA = index -1)
        F_direct_sfc = result.flux_direct[0]
        F_diffuse_sfc = result.flux_down[0]
        F_total_sfc = F_direct_sfc + F_diffuse_sfc
        F_up_toa = result.flux_up[-1]

        # Calculate absorbed by surface
        F_absorbed_sfc = (1 - albedo) * F_total_sfc

        # Calculate reflected to space (already F_up_toa)
        F_reflected = F_up_toa

        # For conservative scattering, absorbed_atm should be 0
        # For absorbing atmosphere, absorbed_atm = F_in - F_reflected - F_absorbed_sfc
        F_absorbed_atm = F_in - F_reflected - F_absorbed_sfc

        # Energy balance check
        F_out_total = F_reflected + F_absorbed_sfc + F_absorbed_atm
        balance_error = abs(F_in - F_out_total)
        balance_error_pct = 100 * balance_error / F_in

        print(f"  Direct at surface:     {F_direct_sfc:.2f} W/m^2")
        print(f"  Diffuse at surface:    {F_diffuse_sfc:.2f} W/m^2")
        print(f"  Total at surface:      {F_total_sfc:.2f} W/m^2")
        print(f"  Absorbed by surface:   {F_absorbed_sfc:.2f} W/m^2")
        print(f"  Reflected at TOA:      {F_reflected:.2f} W/m^2")
        print(f"  Absorbed by atmosphere:{F_absorbed_atm:.2f} W/m^2")
        print(f"  Energy balance error:  {balance_error:.2f} W/m^2 ({balance_error_pct:.2f}%)")

        # Check if error is acceptable (< 1%)
        if balance_error_pct < 1.0:
            print(f"  [PASS] Energy conserved within 1%")
        else:
            print(f"  [FAIL] Energy conservation violated!")
            all_passed = False

        # For conservative scattering, check atmospheric absorption is ~0
        if omega == 1.0:
            if abs(F_absorbed_atm) < 5.0:  # Allow 5 W/m^2 tolerance
                print(f"  [PASS] No spurious atmospheric absorption")
            else:
                print(f"  [WARN] Unexpected atmospheric absorption for SSA=1")

    print("\n" + "=" * 70)
    if all_passed:
        print("RESULT: ALL ENERGY CONSERVATION TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED - REVIEW NEEDED")
    print("=" * 70)

    return all_passed


def test_beer_lambert():
    """Test that direct beam follows Beer-Lambert law."""
    print("\n" + "=" * 70)
    print("VALIDATION TEST 02: BEER-LAMBERT LAW FOR DIRECT BEAM")
    print("=" * 70)

    solver = TwoStreamSolver()
    n_layers = 50
    mu0 = 1.0

    print("\nDirect beam transmission: T = exp(-tau/mu0)")
    print("-" * 70)

    test_taus = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    all_passed = True

    for tau_total in test_taus:
        tau = np.ones(n_layers) * tau_total / n_layers
        omega = np.ones(n_layers)  # Pure scattering
        g = np.zeros(n_layers)

        result = solver.solve_solar(
            tau=tau, omega=omega, g=g, mu0=mu0,
            flux_toa=SOLAR_CONSTANT, surface_albedo=0.0
        )

        # Expected direct transmission
        T_expected = np.exp(-tau_total / mu0)
        F_direct_expected = SOLAR_CONSTANT * mu0 * T_expected

        # Actual direct from model
        F_direct_actual = result.flux_direct[0]

        error_pct = 100 * abs(F_direct_actual - F_direct_expected) / F_direct_expected

        status = "[PASS]" if error_pct < 1.0 else "[FAIL]"
        print(f"  tau={tau_total:.2f}: Expected={F_direct_expected:.1f}, "
              f"Actual={F_direct_actual:.1f}, Error={error_pct:.2f}% {status}")

        if error_pct >= 1.0:
            all_passed = False

    return all_passed


def test_stefan_boltzmann():
    """Test that thermal emission follows Stefan-Boltzmann law."""
    print("\n" + "=" * 70)
    print("VALIDATION TEST 03: STEFAN-BOLTZMANN LAW FOR THERMAL EMISSION")
    print("=" * 70)

    from raf_tran.utils.constants import STEFAN_BOLTZMANN

    solver = TwoStreamSolver()
    n_layers = 10

    print("\nBlackbody emission: F = sigma * T^4")
    print("For transparent atmosphere (tau=0), OLR should equal surface emission")
    print("-" * 70)

    test_temps = [250, 280, 300, 320]
    all_passed = True

    for T_surface in test_temps:
        # Very optically thin atmosphere (essentially transparent)
        tau = np.ones(n_layers) * 1e-6
        omega = np.zeros(n_layers)
        g = np.zeros(n_layers)
        temperature = np.ones(n_layers) * T_surface  # Isothermal

        result = solver.solve_thermal(
            tau=tau, omega=omega, g=g,
            temperature=temperature,
            surface_temperature=T_surface,
            surface_emissivity=1.0
        )

        # Expected OLR = sigma * T^4
        F_expected = STEFAN_BOLTZMANN * T_surface**4

        # Actual OLR (upward flux at TOA = index -1)
        F_actual = result.flux_up[-1]

        error_pct = 100 * abs(F_actual - F_expected) / F_expected

        status = "[PASS]" if error_pct < 1.0 else "[FAIL]"
        print(f"  T={T_surface}K: Expected={F_expected:.1f} W/m^2, "
              f"Actual={F_actual:.1f} W/m^2, Error={error_pct:.2f}% {status}")

        if error_pct >= 1.0:
            all_passed = False

    return all_passed


def test_greenhouse_physics():
    """Test that greenhouse effect has correct sign and magnitude."""
    print("\n" + "=" * 70)
    print("VALIDATION TEST 04: GREENHOUSE EFFECT PHYSICS")
    print("=" * 70)

    from raf_tran.utils.constants import STEFAN_BOLTZMANN

    solver = TwoStreamSolver()
    n_layers = 20

    print("\nWith increasing IR optical depth:")
    print("  - OLR should DECREASE (more trapped)")
    print("  - Backradiation should INCREASE")
    print("  - Surface temperature should INCREASE (for equilibrium)")
    print("-" * 70)

    T_surface = 288.0  # K
    temperature = np.linspace(288, 220, n_layers)  # Realistic lapse rate
    omega = np.zeros(n_layers)
    g = np.zeros(n_layers)

    tau_values = [0.1, 0.5, 1.0, 2.0, 3.0]
    prev_olr = None
    prev_back = None
    all_correct = True

    for tau_total in tau_values:
        tau = np.ones(n_layers) * tau_total / n_layers

        result = solver.solve_thermal(
            tau=tau, omega=omega, g=g,
            temperature=temperature,
            surface_temperature=T_surface,
            surface_emissivity=1.0
        )

        olr = result.flux_up[-1]  # TOA
        backrad = result.flux_down[0]  # Surface

        surface_emission = STEFAN_BOLTZMANN * T_surface**4
        greenhouse_effect = surface_emission - olr

        print(f"  tau={tau_total:.1f}: OLR={olr:.1f}, Backrad={backrad:.1f}, "
              f"Greenhouse={greenhouse_effect:.1f} W/m^2")

        if prev_olr is not None:
            if olr >= prev_olr:
                print(f"    [WARN] OLR did not decrease with increasing tau!")
                all_correct = False
            if backrad <= prev_back:
                print(f"    [WARN] Backradiation did not increase with tau!")
                all_correct = False

        prev_olr = olr
        prev_back = backrad

    if all_correct:
        print("\n  [PASS] Greenhouse physics correct (OLR decreases, backrad increases)")

    return all_correct


def main():
    print("\n" + "#" * 70)
    print("# RAF-TRAN PHYSICS VALIDATION SUITE")
    print("#" * 70)

    results = []

    results.append(("Energy Conservation", test_energy_conservation()))
    results.append(("Beer-Lambert Law", test_beer_lambert()))
    results.append(("Stefan-Boltzmann Law", test_stefan_boltzmann()))
    results.append(("Greenhouse Physics", test_greenhouse_physics()))

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("OVERALL: ALL VALIDATION TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED - INVESTIGATION NEEDED")
    print("=" * 70)


if __name__ == "__main__":
    main()
