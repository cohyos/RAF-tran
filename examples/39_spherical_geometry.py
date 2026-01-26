#!/usr/bin/env python3
"""
Example 39: 3D Spherical Earth Geometry
=======================================

This example demonstrates the 3D geometry module for accurate
radiative transfer calculations beyond plane-parallel approximation.

Features demonstrated:
1. Spherical Earth path calculations
2. Chapman grazing incidence function
3. Solar geometry calculations
4. Coordinate transformations
5. Limb viewing geometry
6. Path integration

OFFLINE OPERATION
-----------------
All geometry calculations work fully offline using built-in models.

Usage:
    python examples/39_spherical_geometry.py [--no-plot]
"""

import argparse
import sys
import numpy as np
from datetime import datetime, timezone

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RAF-tran imports
from raf_tran.geometry import (
    EARTH_RADIUS,
    earth_radius_at_latitude,
    slant_path_length,
    tangent_height,
    line_of_sight_altitudes,
    chapman_function,
    solar_zenith_angle,
    solar_azimuth_angle,
    air_mass_kasten,
    viewing_geometry,
    geodetic_to_ecef,
    ecef_to_geodetic,
)
from raf_tran.geometry.paths import (
    create_vertical_path,
    create_slant_path,
    create_limb_path,
    integrate_along_path,
)


def main(args):
    print("=" * 70)
    print("Example 39: 3D Spherical Earth Geometry")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # 1. Earth Geometry Basics
    # ---------------------------------------------------------------------
    print("1. Earth Geometry Constants")
    print("-" * 40)

    print(f"   Mean Earth radius: {EARTH_RADIUS/1000:.1f} km")

    latitudes = [0, 30, 45, 60, 90]
    print()
    print("   Latitude-dependent radius (WGS84):")
    for lat in latitudes:
        r = earth_radius_at_latitude(lat)
        print(f"     {lat:3d} deg: {r/1000:.2f} km")

    print()

    # ---------------------------------------------------------------------
    # 2. Slant Path Calculations
    # ---------------------------------------------------------------------
    print("2. Slant Path Length vs Zenith Angle")
    print("-" * 40)

    observer_alt = 0  # Sea level
    top_alt = 100000  # 100 km (TOA)

    print(f"   Observer: {observer_alt/1000:.0f} km, TOA: {top_alt/1000:.0f} km")
    print()
    print("   Zenith Angle   Path Length   Air Mass   Plane-Parallel")
    print("   " + "-" * 55)

    for sza in [0, 30, 45, 60, 70, 75, 80, 85, 88]:
        path = slant_path_length(observer_alt, top_alt, sza)
        plane_par = (top_alt - observer_alt) / np.cos(np.radians(sza)) if sza < 90 else np.inf
        air_mass = air_mass_kasten(sza)

        if path < 1e8:
            print(f"   {sza:5.0f} deg       {path/1000:8.1f} km    {air_mass:5.2f}      {plane_par/1000:8.1f} km")
        else:
            print(f"   {sza:5.0f} deg       {'inf':>8s}        {air_mass:5.2f}      {'inf':>8s}")

    print()

    # ---------------------------------------------------------------------
    # 3. Chapman Function for Grazing Incidence
    # ---------------------------------------------------------------------
    print("3. Chapman Grazing Incidence Function")
    print("-" * 40)

    altitude = 30000  # 30 km
    scale_height = 8500  # m

    print(f"   Altitude: {altitude/1000:.0f} km, Scale height: {scale_height/1000:.1f} km")
    print()
    print("   SZA (deg)    sec(SZA)    Chapman     Ratio")
    print("   " + "-" * 50)

    for sza in [0, 30, 60, 75, 85, 89, 90, 91, 95]:
        ch = chapman_function(sza, altitude, scale_height)
        if sza < 90:
            sec_sza = 1.0 / np.cos(np.radians(sza))
        else:
            sec_sza = np.inf

        if sec_sza < 100:
            ratio = ch / sec_sza if sec_sza < np.inf else np.nan
            print(f"   {sza:5.0f}        {sec_sza:8.2f}     {ch:8.2f}    {ratio:6.2f}")
        else:
            print(f"   {sza:5.0f}        {'inf':>8s}     {ch:8.2f}    -")

    print()

    # ---------------------------------------------------------------------
    # 4. Solar Geometry
    # ---------------------------------------------------------------------
    print("4. Solar Position Calculations")
    print("-" * 40)

    # Example location: Boulder, Colorado
    lat, lon = 40.0, -105.25

    # Different times
    times = [
        datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc),  # Summer solstice noon
        datetime(2024, 6, 21, 18, 0, 0, tzinfo=timezone.utc),  # Evening
        datetime(2024, 12, 21, 12, 0, 0, tzinfo=timezone.utc), # Winter solstice noon
        datetime(2024, 3, 21, 12, 0, 0, tzinfo=timezone.utc),  # Equinox noon
    ]

    print(f"   Location: Boulder, CO ({lat:.1f}N, {abs(lon):.2f}W)")
    print()
    print("   Date/Time (UTC)          SZA (deg)   Azimuth (deg)")
    print("   " + "-" * 55)

    for t in times:
        sza = solar_zenith_angle(lat, lon, t)
        azim = solar_azimuth_angle(lat, lon, t)
        print(f"   {t.strftime('%Y-%m-%d %H:%M')}         {sza:6.1f}       {azim:6.1f}")

    print()

    # ---------------------------------------------------------------------
    # 5. Coordinate Transformations
    # ---------------------------------------------------------------------
    print("5. Coordinate Transformations")
    print("-" * 40)

    # Geodetic to ECEF
    locations = [
        ("Equator, Prime Meridian", 0, 0, 0),
        ("North Pole", 90, 0, 0),
        ("Boulder, CO", 40.0, -105.25, 1650),
        ("ISS Orbit", 51.6, 0, 408000),
    ]

    print("   Location                 ECEF (x, y, z) [km]")
    print("   " + "-" * 60)

    for name, lat, lon, alt in locations:
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        print(f"   {name:25s} ({x/1000:9.1f}, {y/1000:9.1f}, {z/1000:9.1f})")

    # Verify round-trip
    print()
    print("   Round-trip verification (ECEF -> Geodetic):")
    x, y, z = geodetic_to_ecef(40.0, -105.25, 1650)
    lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
    print(f"     Original: lat={40.0:.1f}, lon={-105.25:.2f}, alt={1650:.0f} m")
    print(f"     Recovered: lat={lat2:.1f}, lon={lon2:.2f}, alt={alt2:.0f} m")

    print()

    # ---------------------------------------------------------------------
    # 6. Limb Viewing Geometry
    # ---------------------------------------------------------------------
    print("6. Limb Viewing Geometry")
    print("-" * 40)

    satellite_alt = 400000  # 400 km orbit

    print(f"   Satellite altitude: {satellite_alt/1000:.0f} km")
    print()
    print("   Tangent Height   Zenith Angle   Path Length")
    print("   " + "-" * 50)

    for h_tan in [10, 20, 30, 50, 70, 100]:
        h_tan_m = h_tan * 1000

        # Calculate zenith angle for this tangent height
        r_sat = EARTH_RADIUS + satellite_alt
        r_tan = EARTH_RADIUS + h_tan_m

        if r_tan < r_sat:
            sin_zenith = r_tan / r_sat
            zenith = 180 - np.degrees(np.arcsin(sin_zenith))

            # Create limb path
            try:
                limb_path = create_limb_path(
                    observer_altitude=satellite_alt,
                    tangent_alt=h_tan_m,
                    top_altitude=100000,
                )
                path_length = limb_path.total_path_length

                print(f"   {h_tan:6.0f} km        {zenith:6.1f} deg     {path_length/1000:8.1f} km")
            except:
                print(f"   {h_tan:6.0f} km        {zenith:6.1f} deg     (error)")

    print()

    # ---------------------------------------------------------------------
    # 7. Path Integration Example
    # ---------------------------------------------------------------------
    print("7. Path Integration (Extinction)")
    print("-" * 40)

    # Create a slant path
    slant_path = create_slant_path(
        observer_altitude=0,
        zenith_angle_deg=60,
        top_altitude=100000,
        n_layers=50,
    )

    # Define extinction profile (exponential)
    def extinction_profile(altitude):
        # Rayleigh-like extinction
        scale_height = 8500
        beta_0 = 0.01  # km^-1 at surface
        return beta_0 * np.exp(-altitude / scale_height) / 1000  # convert to m^-1

    # Integrate
    tau = integrate_along_path(slant_path, extinction_profile)

    print(f"   Path: Surface to TOA at SZA=60 deg")
    print(f"   Path length: {slant_path.total_path_length/1000:.1f} km")
    print(f"   Integrated optical depth: {tau:.4f}")
    print(f"   Transmission: {np.exp(-tau)*100:.1f}%")

    print()

    # ---------------------------------------------------------------------
    # 8. Visualization
    # ---------------------------------------------------------------------
    if not args.no_plot:
        print("8. Creating Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Panel 1: Slant path length vs SZA
        ax1 = axes[0, 0]
        szas = np.linspace(0, 89, 90)
        paths_spherical = [slant_path_length(0, 100000, sza) for sza in szas]
        paths_plane = (100000) / np.cos(np.radians(szas))

        ax1.semilogy(szas, np.array(paths_spherical)/1000, 'b-', label='Spherical', linewidth=2)
        ax1.semilogy(szas, paths_plane/1000, 'r--', label='Plane-Parallel', linewidth=2)
        ax1.set_xlabel('Solar Zenith Angle (deg)')
        ax1.set_ylabel('Path Length (km)')
        ax1.set_title('Slant Path Length Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 90)

        # Panel 2: Chapman function
        ax2 = axes[0, 1]
        szas_ch = np.linspace(0, 98, 200)
        chapman_30km = [chapman_function(sza, 30000, 8500) for sza in szas_ch]
        chapman_50km = [chapman_function(sza, 50000, 8500) for sza in szas_ch]
        secant = 1 / np.cos(np.radians(np.clip(szas_ch, 0, 89)))

        ax2.semilogy(szas_ch, secant, 'k--', label='sec(SZA)', linewidth=1)
        ax2.semilogy(szas_ch, chapman_30km, 'b-', label='Chapman 30 km', linewidth=2)
        ax2.semilogy(szas_ch, chapman_50km, 'r-', label='Chapman 50 km', linewidth=2)
        ax2.axvline(90, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Solar Zenith Angle (deg)')
        ax2.set_ylabel('Air Mass Factor')
        ax2.set_title('Chapman Grazing Incidence Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(1, 100)

        # Panel 3: Line of sight altitudes
        ax3 = axes[0, 2]
        for sza in [0, 30, 60, 75, 85]:
            distances, alts = line_of_sight_altitudes(0, sza, n_points=100)
            ax3.plot(distances/1000, alts/1000, label=f'SZA={sza} deg')

        ax3.set_xlabel('Path Distance (km)')
        ax3.set_ylabel('Altitude (km)')
        ax3.set_title('Line-of-Sight Altitude Profiles')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Diurnal SZA variation
        ax4 = axes[1, 0]
        hours = np.linspace(0, 24, 48)
        summer_szas = []
        winter_szas = []

        for h in hours:
            t_summer = datetime(2024, 6, 21, int(h), int((h % 1) * 60), tzinfo=timezone.utc)
            t_winter = datetime(2024, 12, 21, int(h), int((h % 1) * 60), tzinfo=timezone.utc)
            summer_szas.append(solar_zenith_angle(40, -105.25, t_summer))
            winter_szas.append(solar_zenith_angle(40, -105.25, t_winter))

        ax4.plot(hours, summer_szas, 'r-', label='Summer Solstice', linewidth=2)
        ax4.plot(hours, winter_szas, 'b-', label='Winter Solstice', linewidth=2)
        ax4.axhline(90, color='gray', linestyle='--', alpha=0.5, label='Horizon')
        ax4.set_xlabel('Hour (UTC)')
        ax4.set_ylabel('Solar Zenith Angle (deg)')
        ax4.set_title('Diurnal SZA Variation (Boulder, CO)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 24)
        ax4.set_ylim(0, 180)

        # Panel 5: Limb path geometry
        ax5 = axes[1, 1]

        # Draw Earth
        theta = np.linspace(0, 2*np.pi, 100)
        earth_x = EARTH_RADIUS * np.cos(theta) / 1000
        earth_y = EARTH_RADIUS * np.sin(theta) / 1000
        ax5.fill(earth_x, earth_y, color='lightblue', alpha=0.5)
        ax5.plot(earth_x, earth_y, 'b-', linewidth=2)

        # Draw atmosphere
        atm_x = (EARTH_RADIUS + 100000) * np.cos(theta) / 1000
        atm_y = (EARTH_RADIUS + 100000) * np.sin(theta) / 1000
        ax5.plot(atm_x, atm_y, 'b--', alpha=0.3)

        # Draw satellite and limb ray
        sat_alt = 400
        sat_x = 0
        sat_y = EARTH_RADIUS/1000 + sat_alt

        # Tangent ray
        h_tan = 30  # km
        r_tan = EARTH_RADIUS/1000 + h_tan
        angle = np.arccos(r_tan / (EARTH_RADIUS/1000 + sat_alt))

        ray_x = np.array([sat_x, -2000 * np.sin(angle)])
        ray_y = np.array([sat_y, r_tan * np.cos(angle)])

        ax5.plot(sat_x, sat_y, 'ro', markersize=10, label='Satellite')
        ax5.plot(ray_x, ray_y, 'r-', linewidth=2, label='Limb ray')
        ax5.set_xlabel('X (km)')
        ax5.set_ylabel('Y (km)')
        ax5.set_title('Limb Viewing Geometry')
        ax5.set_aspect('equal')
        ax5.legend()
        ax5.set_xlim(-1000, 500)
        ax5.set_ylim(6000, 7000)

        # Panel 6: Air mass comparison
        ax6 = axes[1, 2]
        szas_am = np.linspace(0, 89, 90)
        am_secant = 1 / np.cos(np.radians(szas_am))
        am_kasten = [air_mass_kasten(sza) for sza in szas_am]

        ax6.semilogy(szas_am, am_secant, 'b--', label='sec(SZA)', linewidth=1)
        ax6.semilogy(szas_am, am_kasten, 'r-', label='Kasten-Young', linewidth=2)
        ax6.set_xlabel('Solar Zenith Angle (deg)')
        ax6.set_ylabel('Air Mass')
        ax6.set_title('Air Mass Formulas')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('outputs/39_spherical_geometry.png', dpi=150, bbox_inches='tight')
        print("   Saved: outputs/39_spherical_geometry.png")
        plt.close()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    3D Spherical Earth Geometry Capabilities:

    1. PATH CALCULATIONS:
       - Slant path length with Earth curvature
       - Tangent height for limb viewing
       - Line-of-sight altitude profiles

    2. CHAPMAN FUNCTION:
       - Grazing incidence corrections (SZA > 75 deg)
       - Accounts for refraction and curvature
       - Essential for sunset/sunrise calculations

    3. SOLAR GEOMETRY:
       - Solar zenith and azimuth angles
       - Kasten-Young air mass formula
       - Sunrise/sunset time calculations

    4. COORDINATE TRANSFORMS:
       - Geodetic to ECEF (WGS84)
       - ECEF to geodetic
       - Local ENU coordinates

    5. PATH INTEGRATION:
       - Vertical, slant, and limb paths
       - Flexible altitude grids
       - Integration of any quantity along path

    Key Point: All calculations work OFFLINE with built-in Earth models.
    """)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Spherical Earth Geometry')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    import os
    os.makedirs('outputs', exist_ok=True)

    sys.exit(main(args))
