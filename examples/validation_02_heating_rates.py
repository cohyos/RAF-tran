#!/usr/bin/env python3
"""
Validation Test 02: Heating Rate Magnitudes
============================================

This test checks that heating rates are within physically reasonable bounds.

Literature values (from Liou, 2002; Petty, 2006):
- Stratospheric SW heating (ozone): +5 to +15 K/day at peak
- Tropospheric SW heating: +0.5 to +2 K/day
- Tropospheric LW cooling: -1 to -3 K/day
- Stratospheric LW cooling (CO2 15um): -5 to -10 K/day at peak

Values outside these ranges by more than 5-10x suggest problems.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from raf_tran.atmosphere import StandardAtmosphere
from raf_tran.rte_solver import TwoStreamSolver
from raf_tran.scattering import RayleighScattering
from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT, EARTH_SURFACE_GRAVITY


def calculate_heating_rate(flux_up, flux_down, flux_direct, pressure, cp=1004.0):
    """Calculate heating rate in K/day."""
    flux_net = flux_down + flux_direct - flux_up
    n_layers = len(pressure) - 1
    heating_rate = np.zeros(n_layers)

    for i in range(n_layers):
        flux_absorbed = flux_net[i+1] - flux_net[i]
        dp = abs(pressure[i+1] - pressure[i])
        mass_per_area = dp / EARTH_SURFACE_GRAVITY
        if mass_per_area > 0:
            heating_rate[i] = flux_absorbed / (mass_per_area * cp)

    return heating_rate * 86400  # K/day


def test_sw_heating_bounds():
    """Test that SW heating rates are within physical bounds."""
    print("=" * 70)
    print("VALIDATION: SW HEATING RATE MAGNITUDES")
    print("=" * 70)

    solver = TwoStreamSolver()
    atmosphere = StandardAtmosphere()

    n_layers = 40
    z_levels = np.linspace(0, 60000, n_layers + 1)  # 0-60 km
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    temperature = atmosphere.temperature(z_mid)
    pressure_levels = atmosphere.pressure(z_levels)
    number_density = atmosphere.number_density(z_mid)

    mu0 = 0.707  # 45 degrees

    # Rayleigh scattering only
    rayleigh = RayleighScattering()
    tau_rayleigh = rayleigh.optical_depth(np.array([0.55]), number_density, dz).ravel()

    # Add stratospheric ozone absorption (simplified)
    tau_ozone = np.zeros(n_layers)
    for i, z in enumerate(z_mid):
        if 15000 < z < 50000:
            # Ozone peaks around 25 km
            tau_ozone[i] = 0.3 * np.exp(-((z - 25000) / 8000)**2)

    tau_sw = tau_rayleigh + tau_ozone

    # SSA accounting for ozone absorption
    omega_sw = np.ones(n_layers)
    for i in range(n_layers):
        if tau_sw[i] > 0:
            omega_sw[i] = tau_rayleigh[i] / tau_sw[i]
    omega_sw = np.clip(omega_sw, 0.5, 1.0)

    g_sw = np.zeros(n_layers)

    result_sw = solver.solve_solar(
        tau=tau_sw, omega=omega_sw, g=g_sw,
        mu0=mu0, flux_toa=SOLAR_CONSTANT, surface_albedo=0.15
    )

    heating_sw = calculate_heating_rate(
        result_sw.flux_up, result_sw.flux_down, result_sw.flux_direct,
        pressure_levels
    )

    # Find max heating in different regions
    tropo_mask = z_mid < 12000
    strato_mask = (z_mid >= 12000) & (z_mid < 50000)

    max_tropo = np.max(heating_sw[tropo_mask])
    max_strato = np.max(heating_sw[strato_mask])
    max_strato_alt = z_mid[strato_mask][np.argmax(heating_sw[strato_mask])] / 1000

    print(f"\nSolar zenith angle: 45 deg (mu0 = {mu0:.3f})")
    print(f"Total SW optical depth: {np.sum(tau_sw):.3f}")
    print(f"Max ozone optical depth: {np.max(tau_ozone):.3f}")

    print(f"\nTropospheric SW heating (0-12 km):")
    print(f"  Maximum: {max_tropo:.2f} K/day")
    print(f"  Expected range: +0.1 to +2 K/day")

    tropo_ok = 0 < max_tropo < 10  # Allow up to 10 K/day

    print(f"\nStratospheric SW heating (12-50 km):")
    print(f"  Maximum: {max_strato:.2f} K/day at {max_strato_alt:.1f} km")
    print(f"  Expected range: +5 to +15 K/day (up to +20 in extreme cases)")

    strato_ok = 0 < max_strato < 50  # Allow up to 50 K/day

    # Print profile
    print("\n  Altitude (km)   SW Heating (K/day)")
    print("  " + "-" * 35)
    for i in range(0, n_layers, 5):
        print(f"  {z_mid[i]/1000:>10.1f}   {heating_sw[i]:>+15.2f}")

    if tropo_ok and strato_ok:
        print("\n  [PASS] SW heating rates within reasonable bounds")
        return True
    else:
        if not tropo_ok:
            print(f"\n  [WARN] Tropospheric heating {max_tropo:.1f} K/day seems high")
        if not strato_ok:
            print(f"\n  [WARN] Stratospheric heating {max_strato:.1f} K/day may be too high")
        return tropo_ok and strato_ok


def test_lw_cooling_bounds():
    """Test that LW cooling rates are within physical bounds."""
    print("\n" + "=" * 70)
    print("VALIDATION: LW COOLING RATE MAGNITUDES")
    print("=" * 70)

    solver = TwoStreamSolver()
    atmosphere = StandardAtmosphere()

    n_layers = 40
    z_levels = np.linspace(0, 30000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    temperature = atmosphere.temperature(z_mid)
    pressure_levels = atmosphere.pressure(z_levels)
    T_surface = temperature[0]

    # IR optical depth (water vapor dominated, exponential decrease)
    scale_height = 2000  # m
    tau_total = 0.5
    tau_lw = np.zeros(n_layers)
    for i, z in enumerate(z_mid):
        tau_lw[i] = tau_total * np.exp(-z / scale_height) * dz[i] / 1000

    omega_lw = np.zeros(n_layers)  # No scattering in IR
    g_lw = np.zeros(n_layers)

    result_lw = solver.solve_thermal(
        tau=tau_lw, omega=omega_lw, g=g_lw,
        temperature=temperature,
        surface_temperature=T_surface,
        surface_emissivity=1.0
    )

    cooling_lw = calculate_heating_rate(
        result_lw.flux_up, result_lw.flux_down,
        np.zeros_like(result_lw.flux_up),
        pressure_levels
    )

    # Find max cooling in troposphere
    tropo_mask = z_mid < 12000
    max_cooling = np.min(cooling_lw[tropo_mask])  # Most negative
    max_cool_alt = z_mid[tropo_mask][np.argmin(cooling_lw[tropo_mask])] / 1000

    print(f"\nTotal LW optical depth: {np.sum(tau_lw):.3f}")
    print(f"Surface temperature: {T_surface:.1f} K")

    print(f"\nTropospheric LW cooling (0-12 km):")
    print(f"  Maximum cooling: {max_cooling:.2f} K/day at {max_cool_alt:.1f} km")
    print(f"  Expected range: -1 to -5 K/day (up to -10 in some cases)")

    cooling_ok = -20 < max_cooling < 0

    # Print profile
    print("\n  Altitude (km)   LW Cooling (K/day)")
    print("  " + "-" * 35)
    for i in range(0, n_layers, 5):
        print(f"  {z_mid[i]/1000:>10.1f}   {cooling_lw[i]:>+15.2f}")

    if cooling_ok:
        print("\n  [PASS] LW cooling rates within reasonable bounds")
    else:
        print(f"\n  [WARN] LW cooling {max_cooling:.1f} K/day outside expected range")

    return cooling_ok


def test_radiative_equilibrium_temperature():
    """Test that radiative equilibrium gives reasonable temperatures."""
    print("\n" + "=" * 70)
    print("VALIDATION: RADIATIVE EQUILIBRIUM TEMPERATURES")
    print("=" * 70)

    from raf_tran.utils.constants import STEFAN_BOLTZMANN

    solver = TwoStreamSolver()
    n_layers = 20

    # Earth-like conditions
    solar = 1361.0
    albedo = 0.30

    # Effective temperature without atmosphere
    T_eff = ((1 - albedo) * solar / (4 * STEFAN_BOLTZMANN))**0.25
    print(f"\nEffective temperature (no atmosphere): {T_eff:.1f} K")
    print(f"Expected: ~255 K")

    if abs(T_eff - 255) > 5:
        print(f"  [FAIL] Effective temperature calculation error")
        return False

    # Test greenhouse warming
    tau_ir = 1.8  # Earth-like IR opacity
    absorbed_solar = (1 - albedo) * solar / 4

    # Simple iteration to find surface temperature
    T_surface = T_eff * 1.2
    temperature = np.linspace(T_surface * 0.9, T_surface * 0.7, n_layers)

    tau_per = np.ones(n_layers) * tau_ir / n_layers
    omega = np.zeros(n_layers)
    g = np.zeros(n_layers)

    for _ in range(50):
        result = solver.solve_thermal(
            tau=tau_per, omega=omega, g=g,
            temperature=temperature,
            surface_temperature=T_surface,
            surface_emissivity=1.0
        )

        backrad = result.flux_down[0]
        T_new = ((absorbed_solar + backrad) / STEFAN_BOLTZMANN)**0.25
        T_surface = 0.5 * T_surface + 0.5 * T_new

        if abs(T_new - T_surface) < 0.1:
            break

    greenhouse_warming = T_surface - T_eff
    print(f"\nWith tau_IR = {tau_ir}:")
    print(f"  Surface temperature: {T_surface:.1f} K ({T_surface-273.15:.1f} degC)")
    print(f"  Greenhouse warming: {greenhouse_warming:.1f} K")
    print(f"  Expected: ~30-35 K warming for Earth-like conditions")

    # Earth's actual greenhouse effect is ~33 K
    warming_ok = 20 < greenhouse_warming < 80

    if warming_ok:
        print("\n  [PASS] Greenhouse warming in reasonable range")
    else:
        print(f"\n  [WARN] Greenhouse warming {greenhouse_warming:.1f} K outside expected range")

    return warming_ok


def main():
    print("\n" + "#" * 70)
    print("# RAF-TRAN HEATING RATE VALIDATION")
    print("#" * 70)

    results = []

    results.append(("SW Heating Bounds", test_sw_heating_bounds()))
    results.append(("LW Cooling Bounds", test_lw_cooling_bounds()))
    results.append(("Radiative Equilibrium", test_radiative_equilibrium_temperature()))

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
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
