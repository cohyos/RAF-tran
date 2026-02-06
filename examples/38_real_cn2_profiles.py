#!/usr/bin/env python3
"""
Example 38: Real Cn2 Profile Integration
========================================

This example demonstrates the Cn2 (refractive index structure constant)
profile handling capabilities including:

1. Built-in climatological profiles (offline)
2. Standard models (HV 5/7, SLC)
3. Profile manipulation (interpolation, combination)
4. Weather-based estimation
5. File I/O for measured profiles

OFFLINE OPERATION
-----------------
All profile functions work fully offline using built-in models.
External data sources are optional enhancements.

Usage:
    python examples/38_real_cn2_profiles.py [--no-plot]
"""

import argparse
import sys
import numpy as np

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RAF-tran imports
from raf_tran.turbulence import (
    hufnagel_valley_cn2,
    slc_day_cn2,
    slc_night_cn2,
)
from raf_tran.turbulence.real_cn2_data import (
    Cn2Profile,
    get_climatological_profile,
    hufnagel_valley_57_profile,
    interpolate_profile,
    combine_profiles,
    add_turbulent_layer,
    estimate_cn2_from_weather,
    get_profile,
    save_profile_to_file,
    load_profile_from_file,
    can_run_offline,
)


def main(args):
    print("=" * 70)
    print("Example 38: Real Cn2 Profile Integration")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # 1. Offline Capability Check
    # ---------------------------------------------------------------------
    print("1. Offline Capability")
    print("-" * 40)
    print(f"   Can run offline: {can_run_offline()}")
    print("   All built-in profiles work without internet.")
    print()

    # ---------------------------------------------------------------------
    # 2. Climatological Profiles
    # ---------------------------------------------------------------------
    print("2. Climatological Profiles (Offline)")
    print("-" * 40)

    site_types = ['generic_good', 'generic_median', 'generic_poor',
                  'mountaintop', 'desert', 'coastal']

    profiles = {}
    for site in site_types:
        profile = get_climatological_profile(site=site, season='annual')
        profiles[site] = profile
        print(f"   {site:15s}: r0={profile.r0_500nm*100:5.1f} cm, "
              f"seeing={profile.seeing_arcsec:.2f}\"")

    print()

    # ---------------------------------------------------------------------
    # 3. Standard Models
    # ---------------------------------------------------------------------
    print("3. Standard Cn2 Models")
    print("-" * 40)

    # Hufnagel-Valley 5/7 model
    hv57 = hufnagel_valley_57_profile()
    print(f"   HV 5/7 Model:")
    print(f"     r0 at 500nm: {hv57.r0_500nm*100:.1f} cm")
    print(f"     Seeing: {hv57.seeing_arcsec:.2f} arcsec")
    print(f"     Integrated Cn2: {hv57.integrated_cn2:.2e} m^(1/3)")

    # Compare day vs night (SLC model)
    altitudes = np.linspace(0, 20000, 100)
    cn2_day = np.array([slc_day_cn2(h) for h in altitudes])
    cn2_night = np.array([slc_night_cn2(h) for h in altitudes])

    day_profile = Cn2Profile(altitudes=altitudes, cn2=cn2_day, source="SLC_day")
    night_profile = Cn2Profile(altitudes=altitudes, cn2=cn2_night, source="SLC_night")

    print()
    print(f"   SLC Day Model:")
    print(f"     r0 at 500nm: {day_profile.r0_500nm*100:.1f} cm")
    print(f"     Seeing: {day_profile.seeing_arcsec:.2f} arcsec")

    print()
    print(f"   SLC Night Model:")
    print(f"     r0 at 500nm: {night_profile.r0_500nm*100:.1f} cm")
    print(f"     Seeing: {night_profile.seeing_arcsec:.2f} arcsec")

    print()

    # ---------------------------------------------------------------------
    # 4. Profile Manipulation
    # ---------------------------------------------------------------------
    print("4. Profile Manipulation")
    print("-" * 40)

    # Interpolate to custom grid
    new_altitudes = np.linspace(0, 30000, 50)
    interp_profile = interpolate_profile(hv57, new_altitudes)
    print(f"   Interpolated HV57 to {len(new_altitudes)} levels")
    print(f"     Original: {len(hv57.altitudes)} levels, 0-{hv57.altitudes[-1]/1000:.0f} km")
    print(f"     New: {len(interp_profile.altitudes)} levels, 0-{interp_profile.altitudes[-1]/1000:.0f} km")

    # Combine profiles (ensemble)
    combined = combine_profiles([day_profile, night_profile], weights=[0.5, 0.5])
    print()
    print(f"   Combined Day+Night (50/50):")
    print(f"     r0: {combined.r0_500nm*100:.1f} cm")
    print(f"     Seeing: {combined.seeing_arcsec:.2f} arcsec")

    # Add turbulent layer (jet stream)
    jet_stream = add_turbulent_layer(
        hv57,
        altitude=12000,  # 12 km
        strength=5e-16,  # Strong layer
        width=1000,      # 1 km thick
    )
    print()
    print(f"   Added jet stream layer at 12 km:")
    print(f"     r0: {jet_stream.r0_500nm*100:.1f} cm (was {hv57.r0_500nm*100:.1f} cm)")
    print(f"     Seeing: {jet_stream.seeing_arcsec:.2f}\" (was {hv57.seeing_arcsec:.2f}\")")

    print()

    # ---------------------------------------------------------------------
    # 5. Weather-Based Estimation
    # ---------------------------------------------------------------------
    print("5. Weather-Based Cn2 Estimation")
    print("-" * 40)

    # Create synthetic weather profile
    altitudes_wx = np.linspace(0, 20000, 50)
    temperature = 288 - 6.5e-3 * altitudes_wx  # Simple lapse rate
    temperature = np.maximum(temperature, 216)  # Isothermal stratosphere

    wind_speed = 5 + 20 * np.exp(-((altitudes_wx - 10000) / 5000)**2)  # Jet at 10 km

    wx_profile = estimate_cn2_from_weather(
        temperature_profile=temperature,
        altitude_profile=altitudes_wx,
        wind_speed_profile=wind_speed,
    )

    print(f"   Weather-estimated profile:")
    print(f"     r0: {wx_profile.r0_500nm*100:.1f} cm")
    print(f"     Seeing: {wx_profile.seeing_arcsec:.2f} arcsec")
    print()

    # ---------------------------------------------------------------------
    # 6. File I/O
    # ---------------------------------------------------------------------
    print("6. Profile File I/O")
    print("-" * 40)

    # Save profile
    save_path = 'outputs/cn2_profile_example.json'
    save_profile_to_file(hv57, save_path, format='json')
    print(f"   Saved: {save_path}")

    # Load profile back
    loaded = load_profile_from_file(save_path)
    print(f"   Loaded: {loaded.source}")
    print(f"   r0 match: {abs(loaded.r0_500nm - hv57.r0_500nm) < 1e-10}")

    print()

    # ---------------------------------------------------------------------
    # 7. Wavelength Scaling
    # ---------------------------------------------------------------------
    print("7. Wavelength Scaling")
    print("-" * 40)

    wavelengths = [500, 1000, 1500, 2200, 3500, 5000, 10000]  # nm
    print("   Wavelength (nm)    r0 (cm)    Seeing (arcsec)")
    print("   " + "-" * 45)
    for wl in wavelengths:
        r0_wl = hv57.scale_to_wavelength(wl * 1e-9)
        seeing_wl = 0.98 * wl * 1e-9 / r0_wl * 206265
        print(f"   {wl:8d}         {r0_wl*100:6.1f}      {seeing_wl:.2f}")

    print()

    # ---------------------------------------------------------------------
    # 8. Visualization
    # ---------------------------------------------------------------------
    if not args.no_plot:
        print("8. Creating Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Panel 1: Climatological profiles comparison
        ax1 = axes[0, 0]
        for site, profile in profiles.items():
            ax1.semilogy(profile.cn2, profile.altitudes/1000, label=site, linewidth=1.5)
        ax1.set_xlabel('Cn2 (m^-2/3)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Climatological Profiles')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: HV57 vs SLC models
        ax2 = axes[0, 1]
        ax2.semilogy(hv57.cn2, hv57.altitudes/1000, 'b-', label='HV 5/7', linewidth=2)
        ax2.semilogy(day_profile.cn2, day_profile.altitudes/1000, 'r--', label='SLC Day', linewidth=2)
        ax2.semilogy(night_profile.cn2, night_profile.altitudes/1000, 'g:', label='SLC Night', linewidth=2)
        ax2.set_xlabel('Cn2 (m^-2/3)')
        ax2.set_ylabel('Altitude (km)')
        ax2.set_title('Standard Cn2 Models')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: Jet stream layer effect
        ax3 = axes[0, 2]
        ax3.semilogy(hv57.cn2, hv57.altitudes/1000, 'b-', label='Original', linewidth=2)
        ax3.semilogy(jet_stream.cn2, jet_stream.altitudes/1000, 'r-', label='With Jet Stream', linewidth=2)
        ax3.axhline(12, color='gray', linestyle='--', alpha=0.5, label='Jet altitude')
        ax3.set_xlabel('Cn2 (m^-2/3)')
        ax3.set_ylabel('Altitude (km)')
        ax3.set_title('Added Turbulent Layer')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Weather-based estimation
        ax4 = axes[1, 0]
        ax4_twin = ax4.twiny()
        ln1 = ax4.plot(temperature, altitudes_wx/1000, 'b-', label='Temperature', linewidth=2)
        ln2 = ax4_twin.plot(wind_speed, altitudes_wx/1000, 'r-', label='Wind Speed', linewidth=2)
        ax4.set_xlabel('Temperature (K)', color='blue')
        ax4_twin.set_xlabel('Wind Speed (m/s)', color='red')
        ax4.set_ylabel('Altitude (km)')
        ax4.set_title('Weather Input Profile')
        ax4.grid(True, alpha=0.3)

        # Panel 5: Resulting Cn2 from weather
        ax5 = axes[1, 1]
        ax5.semilogy(wx_profile.cn2, wx_profile.altitudes/1000, 'purple', linewidth=2)
        ax5.set_xlabel('Cn2 (m^-2/3)')
        ax5.set_ylabel('Altitude (km)')
        ax5.set_title('Weather-Estimated Cn2')
        ax5.grid(True, alpha=0.3)

        # Panel 6: Seeing vs wavelength
        ax6 = axes[1, 2]
        wls = np.linspace(400, 5000, 50)
        r0s = [hv57.scale_to_wavelength(w*1e-9)*100 for w in wls]
        seeings = [0.98 * w*1e-9 / (r0/100) * 206265 for w, r0 in zip(wls, r0s)]

        ax6_r0 = ax6.twinx()
        ln1 = ax6.plot(wls, seeings, 'b-', label='Seeing', linewidth=2)
        ln2 = ax6_r0.plot(wls, r0s, 'r--', label='r0', linewidth=2)
        ax6.set_xlabel('Wavelength (nm)')
        ax6.set_ylabel('Seeing (arcsec)', color='blue')
        ax6_r0.set_ylabel('r0 (cm)', color='red')
        ax6.set_title('Wavelength Scaling')
        ax6.grid(True, alpha=0.3)

        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper right')

        plt.tight_layout()
        plt.savefig('outputs/38_real_cn2_profiles.png', dpi=150, bbox_inches='tight')
        print("   Saved: outputs/38_real_cn2_profiles.png")
        plt.close()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Cn2 Profile Handling Capabilities:

    1. BUILT-IN PROFILES (Offline):
       - Climatological profiles (good/median/poor sites)
       - Site-specific: mountaintop, desert, coastal
       - Seasonal variations

    2. STANDARD MODELS:
       - Hufnagel-Valley 5/7 (5" seeing, 7 cm theta0)
       - SLC day and night models
       - Custom ground Cn2 and wind parameters

    3. PROFILE MANIPULATION:
       - Interpolation to custom altitude grids
       - Weighted combination of profiles
       - Add/modify turbulent layers

    4. WEATHER-BASED ESTIMATION:
       - Estimate Cn2 from temperature and wind profiles
       - Uses Tatarskii formula

    5. FILE I/O:
       - Save/load profiles in JSON, CSV, NPZ formats
       - Preserve metadata and source information

    Key Point: All profiles work OFFLINE. External data is optional.
    """)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real Cn2 Profile Integration')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    import os
    os.makedirs('outputs', exist_ok=True)

    sys.exit(main(args))
