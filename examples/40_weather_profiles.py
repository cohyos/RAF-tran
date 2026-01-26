#!/usr/bin/env python3
"""
Example 40: Atmospheric Weather Profiles
========================================

This example demonstrates the weather data integration module for
atmospheric profiles including:

1. Built-in standard atmospheres (offline)
2. AFGL reference atmospheres
3. Latitude and seasonal variations
4. Optional online data sources

OFFLINE OPERATION
-----------------
All profiles work FULLY OFFLINE using built-in climatological data.
Online data sources (ECMWF, GFS, MERRA-2) are OPTIONAL enhancements.

Usage:
    python examples/40_weather_profiles.py [--no-plot]
"""

import argparse
import sys
import numpy as np

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RAF-tran imports
from raf_tran.weather import (
    AtmosphericProfile,
    get_atmospheric_profile,
    us_standard_atmosphere,
    tropical_atmosphere,
    midlatitude_summer,
    midlatitude_winter,
    subarctic_summer,
    subarctic_winter,
    ONLINE_AVAILABLE,
    can_fetch_online,
)


def main(args):
    print("=" * 70)
    print("Example 40: Atmospheric Weather Profiles")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # 1. Offline Capability Check
    # ---------------------------------------------------------------------
    print("1. Data Availability")
    print("-" * 40)
    print(f"   Offline operation: Always available")
    print(f"   Online data available: {can_fetch_online()}")
    if not can_fetch_online():
        print("   Note: Install 'requests' for online data access")
    print()

    # ---------------------------------------------------------------------
    # 2. US Standard Atmosphere 1976
    # ---------------------------------------------------------------------
    print("2. US Standard Atmosphere 1976")
    print("-" * 40)

    us_atm = us_standard_atmosphere()

    print(f"   Altitude range: {us_atm.altitudes[0]/1000:.0f} - {us_atm.altitudes[-1]/1000:.0f} km")
    print(f"   Number of levels: {us_atm.n_levels}")
    print(f"   Surface temperature: {us_atm.surface_temperature:.2f} K")
    print(f"   Surface pressure: {us_atm.surface_pressure/100:.1f} hPa")
    print(f"   Scale height: {us_atm.scale_height/1000:.2f} km")

    # Tropopause
    tropo_idx = np.argmin(us_atm.temperature)
    print(f"   Tropopause: {us_atm.altitudes[tropo_idx]/1000:.1f} km, {us_atm.temperature[tropo_idx]:.1f} K")

    print()

    # ---------------------------------------------------------------------
    # 3. AFGL Reference Atmospheres
    # ---------------------------------------------------------------------
    print("3. AFGL Reference Atmospheres")
    print("-" * 40)

    profiles = {
        'Tropical': tropical_atmosphere(),
        'Midlat Summer': midlatitude_summer(),
        'Midlat Winter': midlatitude_winter(),
        'Subarctic Summer': subarctic_summer(),
        'Subarctic Winter': subarctic_winter(),
    }

    print("   Profile           T_surface  P_surface  Scale Height")
    print("   " + "-" * 55)

    for name, prof in profiles.items():
        print(f"   {name:17s}   {prof.surface_temperature:6.1f} K   "
              f"{prof.surface_pressure/100:7.1f} hPa   {prof.scale_height/1000:5.2f} km")

    print()

    # ---------------------------------------------------------------------
    # 4. Latitude/Season Automatic Selection
    # ---------------------------------------------------------------------
    print("4. Automatic Profile Selection by Location")
    print("-" * 40)

    test_cases = [
        ("Equator, July", 0, 7),
        ("Equator, January", 0, 1),
        ("Mid-latitude, July", 45, 7),
        ("Mid-latitude, January", 45, 1),
        ("Arctic, July", 70, 7),
        ("Arctic, January", 70, 1),
    ]

    print("   Location              T_sfc (K)   Profile Type")
    print("   " + "-" * 50)

    for name, lat, month in test_cases:
        prof = get_atmospheric_profile(latitude=lat, month=month)
        print(f"   {name:20s}   {prof.surface_temperature:6.1f}      {prof.source}")

    print()

    # ---------------------------------------------------------------------
    # 5. Profile Properties
    # ---------------------------------------------------------------------
    print("5. Atmospheric Profile Properties")
    print("-" * 40)

    prof = midlatitude_summer()

    # Lapse rate in troposphere
    trop_mask = prof.altitudes < 11000
    if np.sum(trop_mask) > 1:
        dT = np.diff(prof.temperature[trop_mask])
        dz = np.diff(prof.altitudes[trop_mask])
        lapse_rate = -np.mean(dT / dz) * 1000  # K/km
        print(f"   Tropospheric lapse rate: {lapse_rate:.2f} K/km")

    # Precipitable water (if humidity available)
    if prof.humidity is not None:
        # Simple estimate using humidity and density
        pwv = np.trapezoid(prof.humidity * prof.density, prof.altitudes)
        print(f"   Precipitable water vapor: {pwv*1000:.1f} mm (estimate)")

    # Pressure at key altitudes
    print()
    print("   Altitude       Pressure      Temperature    Density")
    print("   " + "-" * 55)
    for alt_km in [0, 5, 10, 20, 30, 50]:
        alt_m = alt_km * 1000
        idx = np.argmin(np.abs(prof.altitudes - alt_m))
        print(f"   {alt_km:3d} km         {prof.pressure[idx]/100:7.2f} hPa   "
              f"{prof.temperature[idx]:6.1f} K      {prof.density[idx]:.4f} kg/m3")

    print()

    # ---------------------------------------------------------------------
    # 6. Profile Interpolation
    # ---------------------------------------------------------------------
    print("6. Profile Interpolation")
    print("-" * 40)

    # Interpolate to custom grid
    custom_alts = np.array([0, 1000, 2000, 5000, 10000, 15000, 20000, 30000])
    interp_prof = prof.interpolate_to(custom_alts)

    print(f"   Original levels: {prof.n_levels}")
    print(f"   Interpolated levels: {interp_prof.n_levels}")
    print()
    print("   Custom Grid Results:")
    print("   Altitude (km)   Temperature (K)   Pressure (hPa)")
    print("   " + "-" * 50)
    for i, alt in enumerate(custom_alts):
        print(f"   {alt/1000:8.1f}          {interp_prof.temperature[i]:6.1f}           "
              f"{interp_prof.pressure[i]/100:7.2f}")

    print()

    # ---------------------------------------------------------------------
    # 7. Online Data Status
    # ---------------------------------------------------------------------
    print("7. Online Data Sources (Optional)")
    print("-" * 40)

    if ONLINE_AVAILABLE:
        print("   Online fetching is available (requests installed)")
        print()
        print("   Available sources:")
        print("     - ECMWF ERA5: Requires cdsapi + credentials")
        print("     - NOAA GFS: Requires pygrib/cfgrib")
        print("     - NASA MERRA-2: Requires netCDF4 + credentials")
        print()
        print("   Note: All sources fall back to offline profiles if unavailable")
    else:
        print("   Online fetching not available (install requests)")
        print("   Simulation runs fully offline using built-in profiles")

    print()

    # ---------------------------------------------------------------------
    # 8. Visualization
    # ---------------------------------------------------------------------
    if not args.no_plot:
        print("8. Creating Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Panel 1: Temperature profiles comparison
        ax1 = axes[0, 0]
        for name, prof in profiles.items():
            ax1.plot(prof.temperature, prof.altitudes/1000, label=name, linewidth=1.5)
        ax1.plot(us_atm.temperature, us_atm.altitudes/1000, 'k--', label='US Std', linewidth=2)
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Temperature Profiles')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(180, 320)
        ax1.set_ylim(0, 50)

        # Panel 2: Pressure profiles (log scale)
        ax2 = axes[0, 1]
        for name, prof in profiles.items():
            ax2.semilogy(prof.altitudes/1000, prof.pressure/100, label=name, linewidth=1.5)
        ax2.set_ylabel('Pressure (hPa)')
        ax2.set_xlabel('Altitude (km)')
        ax2.set_title('Pressure Profiles')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        # Panel 3: Density profiles
        ax3 = axes[0, 2]
        for name, prof in profiles.items():
            ax3.semilogy(prof.density, prof.altitudes/1000, label=name, linewidth=1.5)
        ax3.set_xlabel('Density (kg/m^3)')
        ax3.set_ylabel('Altitude (km)')
        ax3.set_title('Density Profiles')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 50)

        # Panel 4: US Standard Atmosphere details
        ax4 = axes[1, 0]
        ax4_t = ax4.twiny()
        ln1 = ax4.plot(us_atm.temperature, us_atm.altitudes/1000, 'b-', label='T', linewidth=2)
        ln2 = ax4_t.semilogx(us_atm.pressure/100, us_atm.altitudes/1000, 'r--', label='P', linewidth=2)
        ax4.set_xlabel('Temperature (K)', color='blue')
        ax4_t.set_xlabel('Pressure (hPa)', color='red')
        ax4.set_ylabel('Altitude (km)')
        ax4.set_title('US Standard Atmosphere 1976')
        ax4.grid(True, alpha=0.3)
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels)

        # Panel 5: Lapse rate
        ax5 = axes[1, 1]
        dT = np.diff(us_atm.temperature)
        dz = np.diff(us_atm.altitudes)
        lapse = -dT / dz * 1000  # K/km
        alt_mid = (us_atm.altitudes[:-1] + us_atm.altitudes[1:]) / 2
        ax5.plot(lapse, alt_mid/1000, 'g-', linewidth=2)
        ax5.axvline(6.5, color='r', linestyle='--', label='Std lapse (6.5 K/km)')
        ax5.axvline(0, color='gray', linestyle=':', alpha=0.5)
        ax5.set_xlabel('Lapse Rate (K/km)')
        ax5.set_ylabel('Altitude (km)')
        ax5.set_title('Temperature Lapse Rate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-5, 15)
        ax5.set_ylim(0, 50)

        # Panel 6: Humidity profiles (if available)
        ax6 = axes[1, 2]
        for name, prof in profiles.items():
            if prof.humidity is not None:
                ax6.plot(prof.humidity * 100, prof.altitudes/1000, label=name, linewidth=1.5)
        ax6.set_xlabel('Relative Humidity (%)')
        ax6.set_ylabel('Altitude (km)')
        ax6.set_title('Humidity Profiles')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, 100)
        ax6.set_ylim(0, 20)

        plt.tight_layout()
        plt.savefig('outputs/40_weather_profiles.png', dpi=150, bbox_inches='tight')
        print("   Saved: outputs/40_weather_profiles.png")
        plt.close()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Weather Profile Capabilities:

    1. BUILT-IN PROFILES (Always Offline):
       - US Standard Atmosphere 1976
       - AFGL reference atmospheres
       - Tropical, midlatitude, subarctic variants
       - Summer and winter versions

    2. AUTOMATIC SELECTION:
       - By latitude and month
       - Smooth transitions between climate zones
       - Seasonal variations included

    3. PROFILE PROPERTIES:
       - Temperature, pressure, density
       - Humidity (where applicable)
       - Scale height calculation
       - Lapse rate analysis

    4. PROFILE MANIPULATION:
       - Interpolation to custom grids
       - Property calculations
       - Export/import capabilities

    5. OPTIONAL ONLINE DATA:
       - ECMWF ERA5 reanalysis
       - NOAA GFS forecasts
       - NASA MERRA-2 reanalysis
       - All fall back to offline if unavailable

    Key Point: Simulation ALWAYS works offline. Online data is optional.
    """)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Atmospheric Weather Profiles')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    import os
    os.makedirs('outputs', exist_ok=True)

    sys.exit(main(args))
