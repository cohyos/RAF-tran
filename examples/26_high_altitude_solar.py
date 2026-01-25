#!/usr/bin/env python3
"""
High Altitude Solar Radiation
=============================

This example compares solar irradiance at different altitudes,
which is important for:

- Aviation (cockpit radiation exposure)
- High-altitude platforms (HAPS, balloons)
- Mountain meteorology
- Solar power at altitude

Key effects with altitude:
- Reduced atmospheric path length
- Less Rayleigh scattering (bluer sky)
- Less ozone absorption (more UV)
- Reduced water vapor absorption

Altitudes analyzed:
- Surface (0 km)
- Mountain (3 km)
- Aircraft cruise (10-12 km)
- Stratosphere (20 km)
- Near space (35 km)

Usage:
    python 26_high_altitude_solar.py
    python 26_high_altitude_solar.py --sza 30
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import StandardAtmosphere
    from raf_tran.scattering.rayleigh import rayleigh_cross_section
    from raf_tran.utils.constants import SOLAR_CONSTANT
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare solar radiation at different altitudes"
    )
    parser.add_argument("--sza", type=float, default=30, help="Solar zenith angle (deg)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="high_altitude_solar.png")
    return parser.parse_args()


def atmospheric_column_above(altitude_m, scale_height=8500):
    """
    Calculate atmospheric column density above a given altitude.

    Parameters
    ----------
    altitude_m : float
        Altitude in meters
    scale_height : float
        Atmospheric scale height in meters (default: 8500 m)

    Returns
    -------
    column_fraction : float
        Fraction of total atmospheric column above this altitude
    """
    return np.exp(-altitude_m / scale_height)


def rayleigh_transmission(wavelength_um, altitude_m, sza_deg):
    """
    Calculate Rayleigh transmission from space to given altitude.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    altitude_m : float
        Observer altitude in meters
    sza_deg : float
        Solar zenith angle in degrees

    Returns
    -------
    T : float
        Transmission (0 to 1)
    """
    # Column density at sea level
    N_sea_level = 2.55e25  # molecules/m^2

    # Column above observer
    N_above = N_sea_level * atmospheric_column_above(altitude_m)

    # Rayleigh cross section
    sigma = rayleigh_cross_section(wavelength_um)

    # Air mass
    mu0 = np.cos(np.radians(sza_deg))
    air_mass = 1.0 / mu0 if mu0 > 0 else 100

    # Optical depth
    tau = sigma * N_above * air_mass

    return np.exp(-tau)


def ozone_transmission(wavelength_um, altitude_m, sza_deg, ozone_du=300):
    """
    Calculate ozone transmission for UV-B wavelengths.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    altitude_m : float
        Observer altitude in meters
    sza_deg : float
        Solar zenith angle in degrees
    ozone_du : float
        Total ozone column in Dobson Units

    Returns
    -------
    T : float
        Transmission (0 to 1)
    """
    wl = wavelength_um

    # Ozone cross section (simplified)
    if wl < 0.29:
        sigma_o3 = 1e-20  # Very strong absorption
    elif wl < 0.32:
        sigma_o3 = 5e-21 * np.exp(-(wl - 0.29) / 0.01)  # Hartley band
    elif wl < 0.36:
        sigma_o3 = 1e-21 * np.exp(-(wl - 0.32) / 0.02)  # Huggins band
    else:
        sigma_o3 = 1e-25  # Negligible

    # Column density (1 DU = 2.687e20 molecules/m^2)
    N_o3 = ozone_du * 2.687e20

    # Ozone layer is mainly at 20-30 km, so altitude effect is complex
    # Simplified: above 25 km, most ozone is below
    if altitude_m > 25000:
        altitude_factor = 0.1
    elif altitude_m > 15000:
        altitude_factor = 0.5
    else:
        altitude_factor = 1.0

    # Air mass
    mu0 = np.cos(np.radians(sza_deg))
    air_mass = 1.0 / mu0 if mu0 > 0 else 100

    tau = sigma_o3 * N_o3 * altitude_factor * air_mass

    return np.exp(-tau)


def total_transmission(wavelength_um, altitude_m, sza_deg):
    """Calculate total transmission including Rayleigh, ozone, and aerosols."""
    T_ray = rayleigh_transmission(wavelength_um, altitude_m, sza_deg)
    T_o3 = ozone_transmission(wavelength_um, altitude_m, sza_deg)

    # Simple aerosol model (decreases with altitude)
    aod_surface = 0.1  # AOD at surface
    aod = aod_surface * atmospheric_column_above(altitude_m, scale_height=2000)
    mu0 = np.cos(np.radians(sza_deg))
    T_aer = np.exp(-aod / mu0) if mu0 > 0 else 0

    return T_ray * T_o3 * T_aer


def main():
    args = parse_args()

    print("=" * 70)
    print("HIGH ALTITUDE SOLAR RADIATION")
    print("=" * 70)
    print(f"\nSolar zenith angle: {args.sza} deg")

    # Altitude levels
    altitudes = [
        (0, "Sea level"),
        (1500, "Denver (1.5 km)"),
        (3000, "Mountain (3 km)"),
        (5500, "Mt Everest summit"),
        (10000, "Aircraft cruise (10 km)"),
        (20000, "Stratosphere (20 km)"),
        (35000, "Near space (35 km)"),
    ]

    # Atmospheric column analysis
    print("\n" + "-" * 70)
    print("ATMOSPHERIC COLUMN ABOVE ALTITUDE")
    print("-" * 70)

    print(f"\n{'Altitude':>20} {'Column %':>12} {'Pressure (mbar)':>18}")
    print("-" * 55)

    for alt, name in altitudes:
        col_frac = atmospheric_column_above(alt) * 100
        P = 1013.25 * atmospheric_column_above(alt)
        print(f"{name:>20} {col_frac:>11.1f}% {P:>17.1f}")

    # Solar irradiance at different wavelengths
    print("\n" + "-" * 70)
    print(f"DIRECT BEAM TRANSMISSION (SZA = {args.sza} deg)")
    print("-" * 70)

    wavelengths = [
        (0.31, "UV-B (310 nm)"),
        (0.40, "Violet (400 nm)"),
        (0.50, "Green (500 nm)"),
        (0.70, "Red (700 nm)"),
        (1.00, "Near-IR (1000 nm)"),
    ]

    # Header
    print(f"\n{'Altitude':>14}", end="")
    for wl, name in wavelengths:
        print(f"{wl} um".rjust(12), end="")
    print()
    print("-" * 75)

    for alt, alt_name in altitudes:
        print(f"{alt_name:>14}", end="")
        for wl, wl_name in wavelengths:
            T = total_transmission(wl, alt, args.sza) * 100
            print(f"{T:>11.1f}%", end="")
        print()

    # UV index analysis
    print("\n" + "-" * 70)
    print("UV INDEX AT ALTITUDE")
    print("-" * 70)
    print("""
The UV Index scales with altitude due to:
1. Reduced Rayleigh scattering (minor effect)
2. Less ozone above observer (major for high altitude)
3. Reduced aerosol extinction

Rule of thumb: UV increases ~10-15% per 1000 m altitude
""")

    # Calculate relative UV-B at different altitudes
    uv_b_wl = 0.305
    uv_ref = total_transmission(uv_b_wl, 0, args.sza)

    print(f"{'Altitude':>20} {'Relative UV-B':>15} {'UV Index Factor':>18}")
    print("-" * 55)

    for alt, name in altitudes:
        T = total_transmission(uv_b_wl, alt, args.sza)
        relative = T / uv_ref if uv_ref > 0 else 0
        uv_factor = relative if relative > 0 else 0
        print(f"{name:>20} {relative:>14.1f}x {uv_factor:>17.1f}x")

    # Total solar irradiance
    print("\n" + "-" * 70)
    print("TOTAL SOLAR IRRADIANCE")
    print("-" * 70)

    # Simplified broadband calculation
    print(f"\n{'Altitude':>20} {'Direct Normal':>15} {'Relative':>12}")
    print("-" * 50)

    for alt, name in altitudes:
        # Average over visible spectrum
        wl_array = np.linspace(0.3, 2.0, 50)
        T_array = [total_transmission(wl, alt, args.sza) for wl in wl_array]
        T_avg = np.mean(T_array)

        DNI = SOLAR_CONSTANT * T_avg * np.cos(np.radians(args.sza))
        relative = DNI / (SOLAR_CONSTANT * np.cos(np.radians(args.sza)) * 0.7)  # vs ~70% at surface

        print(f"{name:>20} {DNI:>12.0f} W/m^2 {relative:>10.1f}x")

    # Sky color with altitude
    print("\n" + "-" * 70)
    print("SKY COLOR VS ALTITUDE")
    print("-" * 70)
    print("""
Sky color changes with altitude due to reduced Rayleigh scattering:

Altitude        Sky Description
--------        ---------------
Surface         Blue sky (strong Rayleigh)
3 km            Deeper blue
10 km           Very dark blue
20 km           Dark blue/violet
35 km           Near black with visible stars

The blue/red scattering ratio:
""")

    print(f"{'Altitude':>20} {'Blue T (450nm)':>16} {'Red T (650nm)':>15} {'Blue/Red ratio':>16}")
    print("-" * 70)

    for alt, name in altitudes[:6]:
        T_blue = rayleigh_transmission(0.45, alt, args.sza)
        T_red = rayleigh_transmission(0.65, alt, args.sza)
        ratio = (1 - T_blue) / (1 - T_red) if (1 - T_red) > 0.001 else 0
        print(f"{name:>20} {T_blue:>15.1%} {T_red:>14.1%} {ratio:>15.1f}")

    # Applications
    print("\n" + "-" * 70)
    print("APPLICATIONS")
    print("-" * 70)
    print("""
1. AVIATION
   - Cockpit UV exposure at cruise altitude is 2-3x surface
   - Pilots need UV protection
   - Solar power for aircraft systems is more efficient

2. HIGH ALTITUDE PLATFORMS (HAPS)
   - Solar panels at 20 km receive ~40% more power
   - Longer daylight hours at high latitude
   - Less weather interference

3. MOUNTAINEERING
   - UV protection critical above 3000 m
   - Snow reflection doubles UV exposure
   - Risk of snow blindness and sunburn

4. SOLAR ENERGY
   - Mountain sites have higher solar resource
   - Reduced aerosol impact
   - But temperature effects on panel efficiency

5. ASTRONOMY
   - High altitude observatories (Mauna Kea, Atacama)
   - Reduced atmospheric absorption
   - Better seeing conditions
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Solar radiation analysis for SZA = {args.sza} deg:

Key findings:
- Atmospheric column at 10 km: {atmospheric_column_above(10000) * 100:.0f}% of sea level
- Atmospheric column at 35 km: {atmospheric_column_above(35000) * 100:.1f}% of sea level

UV-B increase with altitude:
- At 3 km: ~{(total_transmission(0.305, 3000, args.sza) / total_transmission(0.305, 0, args.sza) - 1) * 100:.0f}% more
- At 10 km: ~{(total_transmission(0.305, 10000, args.sza) / total_transmission(0.305, 0, args.sza) - 1) * 100:.0f}% more

Total solar irradiance increase:
- Significant for solar power applications
- Important for thermal management at altitude

The combination of reduced path length and atmospheric absorption
makes high altitude an excellent location for solar applications.
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('High Altitude Solar Radiation', fontsize=14, fontweight='bold')

            # Plot 1: Transmission vs altitude for different wavelengths
            ax1 = axes[0, 0]
            alt_range = np.linspace(0, 40000, 100)

            for wl, name in [(0.31, '310 nm (UV-B)'), (0.45, '450 nm (blue)'),
                             (0.55, '550 nm (green)'), (0.70, '700 nm (red)')]:
                T = [total_transmission(wl, alt, args.sza) * 100 for alt in alt_range]
                ax1.plot(alt_range / 1000, T, linewidth=2, label=name)

            ax1.set_xlabel('Altitude (km)')
            ax1.set_ylabel('Transmission (%)')
            ax1.set_title(f'Transmission vs Altitude (SZA = {args.sza} deg)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 40)
            ax1.set_ylim(0, 100)

            # Plot 2: UV index factor
            ax2 = axes[0, 1]
            uv_ref = total_transmission(0.305, 0, args.sza)
            uv_factor = [total_transmission(0.305, alt, args.sza) / uv_ref for alt in alt_range]

            ax2.plot(alt_range / 1000, uv_factor, 'r-', linewidth=2)
            ax2.set_xlabel('Altitude (km)')
            ax2.set_ylabel('Relative UV-B Intensity')
            ax2.set_title('UV-B Enhancement with Altitude')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 40)

            # Mark typical altitudes
            for alt, name in [(3, 'Mountain'), (10, 'Aircraft'), (20, 'Stratosphere')]:
                ax2.axvline(alt, color='gray', linestyle='--', alpha=0.5)
                ax2.text(alt + 0.5, 0.9 * ax2.get_ylim()[1], name, fontsize=9)

            # Plot 3: Atmospheric column and pressure
            ax3 = axes[1, 0]
            column = [atmospheric_column_above(alt) * 100 for alt in alt_range]
            ax3.plot(alt_range / 1000, column, 'b-', linewidth=2)
            ax3.set_xlabel('Altitude (km)')
            ax3.set_ylabel('Atmospheric Column (%)')
            ax3.set_title('Atmospheric Column Above Altitude')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 40)

            # Add pressure axis
            ax3b = ax3.twinx()
            pressure = [1013.25 * atmospheric_column_above(alt) for alt in alt_range]
            ax3b.plot(alt_range / 1000, pressure, 'r--', linewidth=1, alpha=0.5)
            ax3b.set_ylabel('Pressure (mbar)', color='red')
            ax3b.tick_params(axis='y', labelcolor='red')

            # Plot 4: Direct Normal Irradiance
            ax4 = axes[1, 1]
            wl_array = np.linspace(0.3, 2.0, 50)

            for alt, name in [(0, 'Surface'), (3000, '3 km'), (10000, '10 km'), (35000, '35 km')]:
                T_array = [total_transmission(wl, alt, args.sza) for wl in wl_array]
                ax4.plot(wl_array * 1000, T_array, linewidth=2, label=name)

            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Transmission')
            ax4.set_title('Spectral Transmission at Different Altitudes')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(300, 2000)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
