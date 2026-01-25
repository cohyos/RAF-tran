#!/usr/bin/env python3
"""
Atmospheric Profiles Comparison
===============================

This example compares different standard atmospheric profiles:
- US Standard Atmosphere 1976
- Tropical
- Midlatitude Summer/Winter
- Subarctic Summer/Winter

These profiles are used in remote sensing and climate models
to represent typical atmospheric conditions.

Usage:
    python 04_atmospheric_profiles.py
    python 04_atmospheric_profiles.py --max-altitude 30000
    python 04_atmospheric_profiles.py --help

Output:
    - Console: Profile statistics
    - Graph: atmospheric_profiles.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import (
        StandardAtmosphere,
        TropicalAtmosphere,
        MidlatitudeSummer,
        MidlatitudeWinter,
        SubarcticSummer,
        SubarcticWinter,
    )
    from raf_tran.utils.constants import STEFAN_BOLTZMANN
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare standard atmospheric profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available atmospheric profiles:
  US Standard  - US Standard Atmosphere 1976
  Tropical     - Mean tropical atmosphere
  MLS          - Midlatitude Summer
  MLW          - Midlatitude Winter
  SAS          - Subarctic Summer
  SAW          - Subarctic Winter

Examples:
  %(prog)s                          # Compare all profiles
  %(prog)s --max-altitude 50000     # Show up to 50 km
        """
    )
    parser.add_argument(
        "--max-altitude", type=float, default=50000,
        help="Maximum altitude in meters (default: 50000)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="atmospheric_profiles.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def calculate_column_amounts(atmosphere, z_levels):
    """Calculate column amounts of gases."""
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    n_air = atmosphere.number_density(z_mid)  # molecules/m^3

    # Column amounts in molecules/m^2
    h2o_col = np.sum(atmosphere.h2o_vmr(z_mid) * n_air * dz)
    o3_col = np.sum(atmosphere.o3_vmr(z_mid) * n_air * dz)
    co2_col = np.sum(atmosphere.co2_vmr(z_mid) * n_air * dz)

    return h2o_col, o3_col, co2_col


def main():
    args = parse_args()

    print("=" * 80)
    print("ATMOSPHERIC PROFILES COMPARISON")
    print("=" * 80)
    print(f"\nAltitude range: 0 - {args.max_altitude/1000:.0f} km")

    # Define profiles
    profiles = {
        "US Standard": StandardAtmosphere(),
        "Tropical": TropicalAtmosphere(),
        "Midlatitude Summer": MidlatitudeSummer(),
        "Midlatitude Winter": MidlatitudeWinter(),
        "Subarctic Summer": SubarcticSummer(),
        "Subarctic Winter": SubarcticWinter(),
    }

    # Altitude grid
    n_points = 200
    z = np.linspace(0, args.max_altitude, n_points)
    z_levels = np.linspace(0, args.max_altitude, 101)

    # Surface conditions
    print("\n" + "-" * 80)
    print("SURFACE CONDITIONS")
    print("-" * 80)
    print(f"\n{'Profile':<22} {'T_surface':>12} {'P_surface':>12} {'rho_surface':>12}")
    print(f"{'':22} {'(K)':>12} {'(hPa)':>12} {'(kg/m^3)':>12}")
    print("-" * 80)

    for name, atm in profiles.items():
        T0 = atm.temperature(np.array([0]))[0]
        P0 = atm.pressure(np.array([0]))[0]
        rho0 = atm.density(np.array([0]))[0]
        print(f"{name:<22} {T0:>12.1f} {P0/100:>12.1f} {rho0:>12.4f}")

    # Tropopause characteristics
    print("\n" + "-" * 80)
    print("TROPOPAUSE CHARACTERISTICS")
    print("-" * 80)
    print(f"\n{'Profile':<22} {'T_tropopause':>14} {'Height':>12} {'Lapse Rate':>12}")
    print(f"{'':22} {'(K)':>14} {'(km)':>12} {'(K/km)':>12}")
    print("-" * 80)

    for name, atm in profiles.items():
        T = atm.temperature(z)
        # Find approximate tropopause (where temperature stops decreasing)
        dT_dz = np.gradient(T, z)
        # Look for where lapse rate becomes small (< 2 K/km)
        tropo_idx = np.argmax(np.abs(dT_dz[10:]) < 0.002) + 10
        tropo_height = z[tropo_idx] / 1000
        T_tropo = T[tropo_idx]
        lapse_rate = (T[0] - T[tropo_idx]) / (z[tropo_idx] / 1000)

        print(f"{name:<22} {T_tropo:>14.1f} {tropo_height:>12.1f} {lapse_rate:>12.2f}")

    # Column water vapor and ozone
    print("\n" + "-" * 80)
    print("COLUMN AMOUNTS")
    print("-" * 80)
    print(f"\n{'Profile':<22} {'H_2O':>14} {'O_3':>14} {'Precipitable':>14}")
    print(f"{'':22} {'(molec/cm^2)':>14} {'(DU)':>14} {'Water (cm)':>14}")
    print("-" * 80)

    for name, atm in profiles.items():
        h2o_col, o3_col, _ = calculate_column_amounts(atm, z_levels)
        # Convert to Dobson Units for ozone (1 DU = 2.687e16 molecules/cm^2)
        o3_DU = o3_col / 1e4 / 2.687e16
        # Precipitable water (cm)
        pw_cm = h2o_col * 18.015 / (6.022e23 * 1e4 * 1.0)  # g/cm^2 ~ cm water
        print(f"{name:<22} {h2o_col/1e4:>14.2e} {o3_DU:>14.1f} {pw_cm:>14.2f}")

    # Thermal radiation characteristics
    print("\n" + "-" * 80)
    print("THERMAL RADIATION")
    print("-" * 80)
    print(f"\n{'Profile':<22} {'Surface Emission':>18} {'Effective T':>14}")
    print(f"{'':22} {'(W/m^2)':>18} {'(K)':>14}")
    print("-" * 80)

    for name, atm in profiles.items():
        T0 = atm.temperature(np.array([0]))[0]
        emission = STEFAN_BOLTZMANN * T0**4
        # Effective temperature (what would give same emission)
        T_eff = (emission / STEFAN_BOLTZMANN)**0.25
        print(f"{name:<22} {emission:>18.1f} {T_eff:>14.1f}")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(16, 12))
            fig.suptitle('Atmospheric Profiles Comparison', fontsize=14, fontweight='bold')

            colors = {
                "US Standard": "black",
                "Tropical": "red",
                "Midlatitude Summer": "orange",
                "Midlatitude Winter": "blue",
                "Subarctic Summer": "green",
                "Subarctic Winter": "purple",
            }

            # Plot 1: Temperature profiles
            ax1 = axes[0, 0]
            for name, atm in profiles.items():
                T = atm.temperature(z)
                ax1.plot(T, z/1000, label=name, color=colors[name], linewidth=2)
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_title('Temperature Profiles')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(180, 320)

            # Plot 2: Pressure profiles (log scale)
            ax2 = axes[0, 1]
            for name, atm in profiles.items():
                P = atm.pressure(z)
                ax2.semilogy(z/1000, P/100, label=name, color=colors[name], linewidth=2)
            ax2.set_xlabel('Altitude (km)')
            ax2.set_ylabel('Pressure (hPa)')
            ax2.set_title('Pressure Profiles')
            ax2.grid(True, alpha=0.3)
            ax2.invert_yaxis()

            # Plot 3: Density profiles
            ax3 = axes[0, 2]
            for name, atm in profiles.items():
                rho = atm.density(z)
                ax3.semilogy(rho, z/1000, label=name, color=colors[name], linewidth=2)
            ax3.set_xlabel('Density (kg/m^3)')
            ax3.set_ylabel('Altitude (km)')
            ax3.set_title('Density Profiles')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Water vapor profiles
            ax4 = axes[1, 0]
            for name, atm in profiles.items():
                h2o = atm.h2o_vmr(z) * 1e6  # ppmv
                ax4.semilogx(h2o, z/1000, label=name, color=colors[name], linewidth=2)
            ax4.set_xlabel('H_2O VMR (ppmv)')
            ax4.set_ylabel('Altitude (km)')
            ax4.set_title('Water Vapor Profiles')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0.1, 100000)

            # Plot 5: Ozone profiles
            ax5 = axes[1, 1]
            for name, atm in profiles.items():
                o3 = atm.o3_vmr(z) * 1e6  # ppmv
                ax5.plot(o3, z/1000, label=name, color=colors[name], linewidth=2)
            ax5.set_xlabel('O_3 VMR (ppmv)')
            ax5.set_ylabel('Altitude (km)')
            ax5.set_title('Ozone Profiles')
            ax5.grid(True, alpha=0.3)

            # Plot 6: Lapse rate profiles
            ax6 = axes[1, 2]
            for name, atm in profiles.items():
                T = atm.temperature(z)
                dT_dz = -np.gradient(T, z) * 1000  # K/km
                ax6.plot(dT_dz, z/1000, label=name, color=colors[name], linewidth=2)
            ax6.axvline(6.5, color='gray', linestyle='--', alpha=0.5, label='Std lapse rate')
            ax6.axvline(0, color='gray', linestyle='-', alpha=0.3)
            ax6.set_xlabel('Lapse Rate (-dT/dz, K/km)')
            ax6.set_ylabel('Altitude (km)')
            ax6.set_title('Temperature Lapse Rate')
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(-5, 15)
            ax6.legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
ATMOSPHERIC STRUCTURE:

1. TROPOSPHERE (0-10 km, varies with latitude):
   - Temperature decreases with altitude (lapse rate ~6.5 K/km)
   - Contains most water vapor and weather
   - Higher in tropics (~18 km), lower at poles (~8 km)

2. STRATOSPHERE (10-50 km):
   - Temperature increases due to ozone absorption of UV
   - Very dry, stable layer
   - Contains the ozone layer

3. LATITUDE VARIATIONS:
   - Tropical: Warmest surface, coldest tropopause, most water vapor
   - Subarctic Winter: Coldest surface, strongest temperature inversion

4. SEASONAL VARIATIONS:
   - Summer: Warmer, more water vapor
   - Winter: Colder, drier, stronger inversions

5. KEY QUANTITIES:
   - Precipitable Water: Total column H_2O if condensed
   - Dobson Units: Column O_3 (1 DU = 2.687e16 molecules/cm^2)
   - Typical ozone: 200-400 DU
""")


if __name__ == "__main__":
    main()
