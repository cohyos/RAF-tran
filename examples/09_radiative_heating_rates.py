#!/usr/bin/env python3
"""
Radiative Heating and Cooling Rates
===================================

This example calculates atmospheric heating rates from radiative
flux divergence. Understanding heating rates is crucial for:

- Climate modeling
- Weather prediction
- Atmospheric dynamics

Shortwave heating (solar absorption) and longwave cooling (thermal emission)
are the primary drivers of atmospheric temperature changes.

Usage:
    python 09_radiative_heating_rates.py
    python 09_radiative_heating_rates.py --sza 45
    python 09_radiative_heating_rates.py --help

Output:
    - Console: Heating rate analysis
    - Graph: radiative_heating_rates.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import StandardAtmosphere, TropicalAtmosphere
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.scattering import RayleighScattering
    from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT, EARTH_SURFACE_GRAVITY
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate radiative heating and cooling rates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Heating rate calculation:
  dT/dt = -g/cp * dF_net/dp

where:
  g = gravitational acceleration
  cp = specific heat at constant pressure
  F_net = net radiative flux (down - up)
  p = pressure

Examples:
  %(prog)s                          # Default calculation
  %(prog)s --sza 60                 # Sun at 60 deg zenith
  %(prog)s --absorber-tau 0.5       # Add absorbing gas
        """
    )
    parser.add_argument(
        "--sza", type=float, default=45,
        help="Solar zenith angle in degrees (default: 45)"
    )
    parser.add_argument(
        "--absorber-tau", type=float, default=0.3,
        help="IR absorber optical depth (default: 0.3)"
    )
    parser.add_argument(
        "--atmosphere", type=str, default="standard",
        choices=["standard", "tropical"],
        help="Atmosphere type (default: standard)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="radiative_heating_rates.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def calculate_heating_rate(flux_up, flux_down, flux_direct, pressure, cp=1004.0):
    """
    Calculate heating rate from flux divergence.

    dT/dt = (absorbed flux) / (layer mass * cp)

    Parameters
    ----------
    flux_up : array
        Upward flux at levels (W/m^2)
    flux_down : array
        Downward diffuse flux at levels (W/m^2)
    flux_direct : array
        Direct (solar) flux at levels (W/m^2)
    pressure : array
        Pressure at levels (Pa)
    cp : float
        Specific heat capacity (J/kg/K)

    Returns
    -------
    heating_rate : array
        Heating rate in K/day
    """
    # Net flux (positive downward)
    flux_net = flux_down + flux_direct - flux_up

    n_layers = len(pressure) - 1
    heating_rate = np.zeros(n_layers)

    for i in range(n_layers):
        # Energy absorbed by layer = flux in minus flux out
        # Levels are ordered from surface (index 0) to TOA (index n)
        # So pressure decreases with index
        # Flux absorbed = (flux from above) - (flux going below)
        # = flux_net[i+1] - flux_net[i]
        flux_absorbed = flux_net[i+1] - flux_net[i]

        # Mass of layer per unit area: dp/g
        dp = abs(pressure[i+1] - pressure[i])
        mass_per_area = dp / EARTH_SURFACE_GRAVITY

        # Heating rate: dT/dt = absorbed_energy / (mass * cp)
        if mass_per_area > 0:
            heating_rate[i] = flux_absorbed / (mass_per_area * cp)

    # Convert K/s to K/day
    heating_rate *= 86400

    return heating_rate


def main():
    args = parse_args()

    print("=" * 80)
    print("RADIATIVE HEATING AND COOLING RATES")
    print("=" * 80)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"IR absorber optical depth: {args.absorber_tau}")
    print(f"Atmosphere type: {args.atmosphere}")

    mu0 = np.cos(np.radians(args.sza))

    # Setup atmosphere
    if args.atmosphere == "tropical":
        atmosphere = TropicalAtmosphere()
    else:
        atmosphere = StandardAtmosphere()

    n_layers = 40
    z_levels = np.linspace(0, 30000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    temperature = atmosphere.temperature(z_mid)
    pressure_levels = atmosphere.pressure(z_levels)
    pressure_mid = atmosphere.pressure(z_mid)
    number_density = atmosphere.number_density(z_mid)
    T_surface = temperature[0]

    print(f"\nSurface temperature: {T_surface:.1f} K")
    print(f"Surface pressure: {pressure_levels[0]/100:.1f} hPa")

    solver = TwoStreamSolver()

    # SHORTWAVE CALCULATION
    print("\n" + "-" * 80)
    print("SHORTWAVE (SOLAR) HEATING")
    print("-" * 80)

    # Rayleigh scattering + some absorption
    rayleigh = RayleighScattering()
    wavelength = np.array([0.55])
    tau_rayleigh = rayleigh.optical_depth(wavelength, number_density, dz).ravel()

    # Add some absorption (e.g., ozone-like)
    tau_sw_abs = np.zeros(n_layers)
    # Absorption layer around 20-30 km (stratospheric ozone)
    for i, z in enumerate(z_mid):
        if 15000 < z < 35000:
            tau_sw_abs[i] = 0.05 * np.exp(-((z - 25000) / 5000)**2)

    tau_sw = tau_rayleigh + tau_sw_abs

    # Single scattering albedo (accounting for absorption)
    omega_sw = np.ones(n_layers)
    for i in range(n_layers):
        if tau_sw[i] > 0:
            omega_sw[i] = tau_rayleigh[i] / tau_sw[i]
    omega_sw = np.clip(omega_sw, 0.5, 1.0)

    g_sw = np.zeros(n_layers)  # Rayleigh (isotropic)

    result_sw = solver.solve_solar(
        tau=tau_sw,
        omega=omega_sw,
        g=g_sw,
        mu0=mu0,
        flux_toa=SOLAR_CONSTANT,
        surface_albedo=0.15,
    )

    heating_sw = calculate_heating_rate(
        result_sw.flux_up,
        result_sw.flux_down,
        result_sw.flux_direct,
        pressure_levels
    )

    # Surface = index 0, TOA = index -1
    sw_absorbed_surface = (1 - 0.15) * (result_sw.flux_direct[0] + result_sw.flux_down[0])
    sw_absorbed_atm = SOLAR_CONSTANT * mu0 - result_sw.flux_up[-1] - sw_absorbed_surface

    print(f"\nSolar constant * u0: {SOLAR_CONSTANT * mu0:.1f} W/m^2")
    print(f"Reflected at TOA: {result_sw.flux_up[-1]:.1f} W/m^2")
    print(f"Absorbed by surface: {sw_absorbed_surface:.1f} W/m^2")
    print(f"Absorbed by atmosphere: {sw_absorbed_atm:.1f} W/m^2")
    print(f"Max SW heating rate: {np.max(heating_sw):.2f} K/day at {z_mid[np.argmax(heating_sw)]/1000:.1f} km")

    # LONGWAVE CALCULATION
    print("\n" + "-" * 80)
    print("LONGWAVE (THERMAL) COOLING")
    print("-" * 80)

    # IR optical depth (water vapor + CO2 like)
    # Decreases exponentially with altitude (water vapor dominated)
    tau_lw = np.zeros(n_layers)
    scale_height = 2000  # meters
    for i, z in enumerate(z_mid):
        tau_lw[i] = args.absorber_tau * np.exp(-z / scale_height) * dz[i] / 1000

    omega_lw = np.zeros(n_layers)  # No scattering in IR
    g_lw = np.zeros(n_layers)

    result_lw = solver.solve_thermal(
        tau=tau_lw,
        omega=omega_lw,
        g=g_lw,
        temperature=temperature,
        surface_temperature=T_surface,
        surface_emissivity=1.0,
    )

    heating_lw = calculate_heating_rate(
        result_lw.flux_up,
        result_lw.flux_down,
        np.zeros_like(result_lw.flux_up),  # No direct beam in thermal
        pressure_levels
    )

    # LW is typically cooling (negative heating rate)
    olr = result_lw.flux_up[-1]  # TOA = index -1
    surface_emission = STEFAN_BOLTZMANN * T_surface**4
    backradiation = result_lw.flux_down[0]  # Surface = index 0

    print(f"\nSurface emission: {surface_emission:.1f} W/m^2")
    print(f"Outgoing LW at TOA: {olr:.1f} W/m^2")
    print(f"Atmospheric backradiation: {backradiation:.1f} W/m^2")
    print(f"Max LW cooling rate: {np.min(heating_lw):.2f} K/day at {z_mid[np.argmin(heating_lw)]/1000:.1f} km")

    # NET HEATING
    print("\n" + "-" * 80)
    print("NET RADIATIVE HEATING")
    print("-" * 80)

    heating_net = heating_sw + heating_lw

    print(f"\nAltitude-averaged heating rates:")
    print(f"  SW heating: {np.mean(heating_sw):+.2f} K/day")
    print(f"  LW cooling: {np.mean(heating_lw):+.2f} K/day")
    print(f"  Net:        {np.mean(heating_net):+.2f} K/day")

    # Profile summary
    print("\n" + "-" * 80)
    print("HEATING RATE PROFILE")
    print("-" * 80)
    print(f"\n{'Altitude':>10} {'Pressure':>10} {'SW Heat':>12} {'LW Cool':>12} {'Net':>12}")
    print(f"{'(km)':>10} {'(hPa)':>10} {'(K/day)':>12} {'(K/day)':>12} {'(K/day)':>12}")
    print("-" * 60)

    for i in range(0, n_layers, 5):
        print(f"{z_mid[i]/1000:>10.1f} {pressure_mid[i]/100:>10.1f} "
              f"{heating_sw[i]:>+12.3f} {heating_lw[i]:>+12.3f} {heating_net[i]:>+12.3f}")

    # Radiative equilibrium discussion
    print("\n" + "=" * 80)
    print("RADIATIVE EQUILIBRIUM")
    print("=" * 80)
    print("""
In radiative equilibrium, heating balances cooling at each level:

  SW_heating + LW_cooling = 0

Key observations:
""")

    # Find where net heating changes sign
    sign_changes = np.where(np.diff(np.sign(heating_net)))[0]
    for idx in sign_changes:
        print(f"  - Net heating changes sign at ~{z_mid[idx]/1000:.1f} km")

    if np.mean(heating_net[:10]) > 0:
        print("  - Lower atmosphere has net heating (driven by solar absorption)")
    else:
        print("  - Lower atmosphere has net cooling (strong IR emission)")

    if np.mean(heating_net[-10:]) < 0:
        print("  - Upper atmosphere has net cooling (IR emission to space)")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            fig.suptitle(f'Radiative Heating Rates (SZA={args.sza} deg, {args.atmosphere} atmosphere)',
                        fontsize=14, fontweight='bold')

            # Plot 1: Temperature profile
            ax1 = axes[0, 0]
            ax1.plot(temperature, z_mid / 1000, 'r-', linewidth=2)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_title('Temperature Profile')
            ax1.grid(True, alpha=0.3)

            # Plot 2: SW flux profiles
            ax2 = axes[0, 1]
            ax2.plot(result_sw.flux_direct, z_levels / 1000, 'orange', linewidth=2, label='Direct')
            ax2.plot(result_sw.flux_down, z_levels / 1000, 'gold', linewidth=2, label='Diffuse down')
            ax2.plot(result_sw.flux_up, z_levels / 1000, 'yellow', linewidth=2, label='Diffuse up')
            ax2.set_xlabel('Flux (W/m^2)')
            ax2.set_ylabel('Altitude (km)')
            ax2.set_title('Shortwave Flux Profiles')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: LW flux profiles
            ax3 = axes[0, 2]
            ax3.plot(result_lw.flux_up, z_levels / 1000, 'r-', linewidth=2, label='Upward')
            ax3.plot(result_lw.flux_down, z_levels / 1000, 'b-', linewidth=2, label='Downward')
            ax3.plot(result_lw.flux_up - result_lw.flux_down, z_levels / 1000, 'k--',
                    linewidth=2, label='Net up')
            ax3.set_xlabel('Flux (W/m^2)')
            ax3.set_ylabel('Altitude (km)')
            ax3.set_title('Longwave Flux Profiles')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: SW heating rate
            ax4 = axes[1, 0]
            ax4.plot(heating_sw, z_mid / 1000, 'orange', linewidth=2)
            ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax4.fill_betweenx(z_mid / 1000, 0, heating_sw, where=heating_sw > 0,
                             color='orange', alpha=0.3)
            ax4.set_xlabel('Heating Rate (K/day)')
            ax4.set_ylabel('Altitude (km)')
            ax4.set_title('Shortwave Heating Rate')
            ax4.grid(True, alpha=0.3)

            # Plot 5: LW cooling rate
            ax5 = axes[1, 1]
            ax5.plot(heating_lw, z_mid / 1000, 'blue', linewidth=2)
            ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax5.fill_betweenx(z_mid / 1000, 0, heating_lw, where=heating_lw < 0,
                             color='blue', alpha=0.3)
            ax5.set_xlabel('Heating Rate (K/day)')
            ax5.set_ylabel('Altitude (km)')
            ax5.set_title('Longwave Cooling Rate')
            ax5.grid(True, alpha=0.3)

            # Plot 6: Net heating rate
            ax6 = axes[1, 2]
            ax6.plot(heating_net, z_mid / 1000, 'green', linewidth=2)
            ax6.axvline(0, color='black', linestyle='-', linewidth=1)
            ax6.fill_betweenx(z_mid / 1000, 0, heating_net, where=heating_net > 0,
                             color='red', alpha=0.3, label='Net heating')
            ax6.fill_betweenx(z_mid / 1000, 0, heating_net, where=heating_net < 0,
                             color='blue', alpha=0.3, label='Net cooling')
            ax6.set_xlabel('Heating Rate (K/day)')
            ax6.set_ylabel('Altitude (km)')
            ax6.set_title('Net Radiative Heating Rate')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
RADIATIVE HEATING IN THE ATMOSPHERE:

1. SHORTWAVE (SOLAR) HEATING:
   - Absorption by ozone (UV), water vapor, aerosols
   - Maximum heating in stratosphere (ozone layer)
   - Also heats troposphere (water vapor, clouds)

2. LONGWAVE (THERMAL) COOLING:
   - Atmosphere emits IR radiation
   - Emission increases with temperature (T^4)
   - Net cooling because emission > absorption from below

3. HEATING RATE FORMULA:
   dT/dt = -g/cp * dF_net/dp
   where F_net = F_down - F_up (positive downward)

4. TYPICAL VALUES:
   - SW heating: +1 to +5 K/day in stratosphere
   - LW cooling: -1 to -3 K/day throughout
   - Net: small residual, balanced by dynamics

5. RADIATIVE-CONVECTIVE EQUILIBRIUM:
   - Radiation alone would create unstable lapse rate
   - Convection redistributes heat vertically
   - Together they set the atmospheric temperature profile
""")


if __name__ == "__main__":
    main()
