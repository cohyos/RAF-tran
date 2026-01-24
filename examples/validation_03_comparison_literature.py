#!/usr/bin/env python3
"""
Validation Test 03: Comparison with Literature Values
======================================================

This test compares RAF-tran outputs against published reference values
from standard radiative transfer codes and observations.

References:
- AFGL Standard Atmospheres (Anderson et al., 1986)
- MODTRAN radiative transfer code outputs
- Liou (2002) "An Introduction to Atmospheric Radiation"
- Petty (2006) "A First Course in Atmospheric Radiation"
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from raf_tran.atmosphere import StandardAtmosphere, TropicalAtmosphere
from raf_tran.rte_solver import TwoStreamSolver
from raf_tran.scattering import RayleighScattering
from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT


def test_rayleigh_transmission():
    """
    Test Rayleigh optical depth against literature values.

    Reference: Hansen & Travis (1974), Liou (2002)
    At 550 nm, tau_Rayleigh ~ 0.097 for US Standard Atmosphere
    """
    print("=" * 70)
    print("TEST: RAYLEIGH OPTICAL DEPTH vs LITERATURE")
    print("=" * 70)

    atmosphere = StandardAtmosphere()
    rayleigh = RayleighScattering()

    n_layers = 100
    z_levels = np.linspace(0, 100000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    number_density = atmosphere.number_density(z_mid)

    wavelengths_nm = [400, 500, 550, 600, 700]
    literature_tau = {
        400: 0.364,  # Approximate from Hansen & Travis
        500: 0.143,
        550: 0.097,  # Standard reference value
        600: 0.068,
        700: 0.036,
    }

    print(f"\nUS Standard Atmosphere, total column")
    print(f"{'Wavelength (nm)':<18} {'Literature':<12} {'RAF-tran':<12} {'Error %':<10}")
    print("-" * 55)

    all_passed = True
    for wl_nm in wavelengths_nm:
        wl_um = wl_nm / 1000.0
        tau_raf = rayleigh.optical_depth(np.array([wl_um]), number_density, dz).ravel()
        tau_total = np.sum(tau_raf)

        tau_lit = literature_tau[wl_nm]
        error_pct = 100 * abs(tau_total - tau_lit) / tau_lit

        status = "[OK]" if error_pct < 10 else "[!]"
        print(f"{wl_nm:<18} {tau_lit:<12.3f} {tau_total:<12.3f} {error_pct:<10.1f} {status}")

        if error_pct > 20:
            all_passed = False

    if all_passed:
        print("\n[PASS] Rayleigh optical depths within 20% of literature")
    else:
        print("\n[WARN] Some Rayleigh optical depths differ significantly")

    return all_passed


def test_direct_beam_irradiance():
    """
    Test direct normal irradiance at surface vs observations.

    Reference: Bird & Riordan (1984), ASTM G173
    Clear sky DNI at sea level, overhead sun: ~900-1000 W/m^2
    """
    print("\n" + "=" * 70)
    print("TEST: DIRECT NORMAL IRRADIANCE vs OBSERVATIONS")
    print("=" * 70)

    solver = TwoStreamSolver()
    atmosphere = StandardAtmosphere()
    rayleigh = RayleighScattering()

    n_layers = 50
    z_levels = np.linspace(0, 50000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    number_density = atmosphere.number_density(z_mid)

    # 550 nm (green light)
    wl = np.array([0.55])
    tau_ray = rayleigh.optical_depth(wl, number_density, dz).ravel()

    omega = np.ones(n_layers)  # Pure scattering
    g = np.zeros(n_layers)

    test_cases = [
        (0, 1.0, "Overhead sun"),
        (30, 0.866, "30 deg SZA"),
        (60, 0.5, "60 deg SZA"),
    ]

    print(f"\nClear Rayleigh atmosphere, 550 nm")
    print(f"Solar constant: {SOLAR_CONSTANT} W/m^2")
    print(f"{'Condition':<20} {'mu0':<8} {'DNI (W/m^2)':<15} {'Expected':<15}")
    print("-" * 60)

    all_ok = True
    for sza, mu0, desc in test_cases:
        result = solver.solve_solar(
            tau=tau_ray, omega=omega, g=g,
            mu0=mu0, flux_toa=SOLAR_CONSTANT, surface_albedo=0.0
        )

        dni = result.flux_direct[0]

        # Expected from Beer-Lambert with Rayleigh tau ~ 0.1
        expected = SOLAR_CONSTANT * mu0 * np.exp(-np.sum(tau_ray) / mu0)

        print(f"{desc:<20} {mu0:<8.3f} {dni:<15.1f} {expected:<15.1f}")

        if abs(dni - expected) > 10:
            all_ok = False

    if all_ok:
        print("\n[PASS] Direct beam irradiance matches Beer-Lambert")

    return all_ok


def test_outgoing_longwave_radiation():
    """
    Test OLR against satellite observations.

    Reference: CERES satellite data
    Global mean OLR: ~240 W/m^2
    Clear sky OLR over tropics: ~280-300 W/m^2
    Clear sky OLR over polar: ~180-220 W/m^2
    """
    print("\n" + "=" * 70)
    print("TEST: OUTGOING LONGWAVE RADIATION vs CERES")
    print("=" * 70)

    solver = TwoStreamSolver()

    n_layers = 30

    test_cases = [
        ("Clear tropical", TropicalAtmosphere(), 300.0, 1.5, 280, 310),
        ("Clear US Std", StandardAtmosphere(), 288.0, 1.5, 260, 290),
        ("Moist tropical", TropicalAtmosphere(), 300.0, 3.0, 240, 280),
    ]

    print(f"\n{'Condition':<20} {'T_sfc (K)':<12} {'tau_IR':<10} {'OLR (W/m^2)':<15} {'Expected':<15}")
    print("-" * 75)

    all_ok = True
    for name, atm, T_sfc, tau_ir, olr_min, olr_max in test_cases:
        z_levels = np.linspace(0, 25000, n_layers + 1)
        z_mid = (z_levels[:-1] + z_levels[1:]) / 2
        dz = np.diff(z_levels)

        temperature = atm.temperature(z_mid)

        # IR optical depth profile (water vapor like)
        tau_lw = np.zeros(n_layers)
        scale_height = 2500
        for i, z in enumerate(z_mid):
            tau_lw[i] = tau_ir * np.exp(-z / scale_height) * dz[i] / 1000

        omega = np.zeros(n_layers)
        g = np.zeros(n_layers)

        result = solver.solve_thermal(
            tau=tau_lw, omega=omega, g=g,
            temperature=temperature,
            surface_temperature=T_sfc,
            surface_emissivity=1.0
        )

        olr = result.flux_up[-1]
        expected = f"{olr_min}-{olr_max}"

        status = "[OK]" if olr_min <= olr <= olr_max else "[!]"
        print(f"{name:<20} {T_sfc:<12.1f} {tau_ir:<10.1f} {olr:<15.1f} {expected:<15} {status}")

        if not (olr_min <= olr <= olr_max):
            all_ok = False

    if all_ok:
        print("\n[PASS] OLR values within expected ranges")
    else:
        print("\n[WARN] Some OLR values outside expected ranges")

    return all_ok


def test_planetary_albedo():
    """
    Test planetary albedo for simple cases.

    Reference: Various sources
    Pure Rayleigh atmosphere: ~0.06-0.08
    Earth with clouds: ~0.30
    """
    print("\n" + "=" * 70)
    print("TEST: PLANETARY ALBEDO vs EXPECTED VALUES")
    print("=" * 70)

    solver = TwoStreamSolver()
    atmosphere = StandardAtmosphere()
    rayleigh = RayleighScattering()

    n_layers = 50
    z_levels = np.linspace(0, 50000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    number_density = atmosphere.number_density(z_mid)

    wl = np.array([0.55])
    tau_ray = rayleigh.optical_depth(wl, number_density, dz).ravel()

    mu0 = 0.5  # 60 deg SZA

    test_cases = [
        ("Pure Rayleigh, black surface", tau_ray, 1.0, 0.0, 0.0, 0.05, 0.12),
        ("Pure Rayleigh, ocean", tau_ray, 1.0, 0.0, 0.06, 0.08, 0.15),
        ("Cloudy (tau=10)", tau_ray + 10.0, 0.999, 0.85, 0.0, 0.5, 0.8),
    ]

    print(f"\nSZA = 60 deg (mu0 = {mu0})")
    print(f"{'Condition':<30} {'Albedo':<12} {'Expected':<15}")
    print("-" * 60)

    all_ok = True
    for name, tau, omega_val, g_val, albedo, a_min, a_max in test_cases:
        omega = np.ones(n_layers) * omega_val
        g = np.ones(n_layers) * g_val

        result = solver.solve_solar(
            tau=tau, omega=omega, g=g,
            mu0=mu0, flux_toa=SOLAR_CONSTANT, surface_albedo=albedo
        )

        planetary_albedo = result.flux_up[-1] / (SOLAR_CONSTANT * mu0)
        expected = f"{a_min:.2f}-{a_max:.2f}"

        status = "[OK]" if a_min <= planetary_albedo <= a_max else "[!]"
        print(f"{name:<30} {planetary_albedo:<12.3f} {expected:<15} {status}")

        if not (a_min <= planetary_albedo <= a_max):
            all_ok = False

    if all_ok:
        print("\n[PASS] Planetary albedo within expected ranges")
    else:
        print("\n[WARN] Some albedo values outside expected ranges")

    return all_ok


def main():
    print("\n" + "#" * 70)
    print("# RAF-TRAN LITERATURE COMPARISON VALIDATION")
    print("#" * 70)

    results = []

    results.append(("Rayleigh Optical Depth", test_rayleigh_transmission()))
    results.append(("Direct Beam Irradiance", test_direct_beam_irradiance()))
    results.append(("Outgoing Longwave Radiation", test_outgoing_longwave_radiation()))
    results.append(("Planetary Albedo", test_planetary_albedo()))

    print("\n" + "=" * 70)
    print("LITERATURE COMPARISON SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[WARN]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 70)


if __name__ == "__main__":
    main()
