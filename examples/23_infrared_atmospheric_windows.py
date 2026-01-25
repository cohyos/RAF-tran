#!/usr/bin/env python3
"""
Infrared Atmospheric Windows Analysis
=====================================

This example analyzes transmission in key infrared atmospheric windows,
comparing transparent regions with strong absorption bands.

Key IR Windows:
- SWIR window (1.5-1.8 um, 2.0-2.5 um)
- MWIR window (3-5 um)
- LWIR window (8-13 um)

Strong Absorption Bands:
- H2O: 1.4, 1.9, 2.7, 6.3 um
- CO2: 2.7, 4.3, 15 um
- O3: 9.6 um (within LWIR window)

Applications:
- Thermal imaging (MWIR, LWIR)
- Gas sensing (specific absorption lines)
- Remote sensing and surveillance
- Infrared astronomy

Usage:
    python 23_infrared_atmospheric_windows.py
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.utils.constants import SPEED_OF_LIGHT, PLANCK_CONSTANT, BOLTZMANN_CONSTANT
    from raf_tran.utils.spectral import planck_function
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze infrared atmospheric windows"
    )
    parser.add_argument("--path-km", type=float, default=1.0, help="Path length (km)")
    parser.add_argument("--humidity", type=float, default=50, help="Relative humidity (percent)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="ir_windows.png")
    return parser.parse_args()


def h2o_absorption_model(wavelength_um, precipitable_water_cm=1.0):
    """
    Simplified H2O absorption model for IR wavelengths.

    Parameters
    ----------
    wavelength_um : array_like
        Wavelength in micrometers
    precipitable_water_cm : float
        Precipitable water vapor in cm (typical: 0.5-5 cm)

    Returns
    -------
    tau : ndarray
        Optical depth due to water vapor
    """
    wl = np.asarray(wavelength_um)
    tau = np.zeros_like(wl)

    # H2O absorption bands (simplified Gaussian profiles)
    bands = [
        (0.94, 0.05, 0.5),   # 0.94 um band
        (1.14, 0.08, 0.8),   # 1.14 um band
        (1.38, 0.15, 3.0),   # 1.38 um band (strong)
        (1.87, 0.20, 4.0),   # 1.87 um band (strong)
        (2.70, 0.40, 8.0),   # 2.7 um band (very strong)
        (6.30, 1.50, 10.0),  # 6.3 um band (bending mode)
    ]

    for center, width, strength in bands:
        tau += strength * np.exp(-0.5 * ((wl - center) / width)**2)

    # Continuum absorption in far-IR
    tau += 0.1 * precipitable_water_cm * (wl / 10)**2 * (wl > 5)

    return tau * precipitable_water_cm


def co2_absorption_model(wavelength_um, co2_ppm=420):
    """
    Simplified CO2 absorption model for IR wavelengths.

    Parameters
    ----------
    wavelength_um : array_like
        Wavelength in micrometers
    co2_ppm : float
        CO2 concentration in ppm

    Returns
    -------
    tau : ndarray
        Optical depth due to CO2
    """
    wl = np.asarray(wavelength_um)
    tau = np.zeros_like(wl)

    # CO2 absorption bands
    bands = [
        (2.0, 0.05, 0.3),    # 2.0 um combination band
        (2.7, 0.15, 1.5),    # 2.7 um band (overlaps H2O)
        (4.26, 0.25, 5.0),   # 4.26 um band (strong asymmetric stretch)
        (15.0, 2.0, 8.0),    # 15 um band (bending mode, very strong)
    ]

    for center, width, strength in bands:
        tau += strength * np.exp(-0.5 * ((wl - center) / width)**2)

    return tau * (co2_ppm / 400)


def o3_absorption_model(wavelength_um, ozone_du=300):
    """
    Simplified O3 absorption model for IR wavelengths.

    Parameters
    ----------
    wavelength_um : array_like
        Wavelength in micrometers
    ozone_du : float
        Ozone column in Dobson Units

    Returns
    -------
    tau : ndarray
        Optical depth due to ozone
    """
    wl = np.asarray(wavelength_um)
    tau = np.zeros_like(wl)

    # O3 absorption band at 9.6 um
    tau += 2.0 * np.exp(-0.5 * ((wl - 9.6) / 0.8)**2)

    return tau * (ozone_du / 300)


def atmospheric_transmission(wavelength_um, path_length_km=1.0,
                            precipitable_water_cm=1.0, co2_ppm=420, ozone_du=300):
    """
    Calculate total atmospheric transmission in IR.

    Parameters
    ----------
    wavelength_um : array_like
        Wavelength in micrometers
    path_length_km : float
        Horizontal path length in km
    precipitable_water_cm : float
        Precipitable water vapor in cm
    co2_ppm : float
        CO2 concentration in ppm
    ozone_du : float
        Ozone column in Dobson Units

    Returns
    -------
    T : ndarray
        Transmission (0 to 1)
    """
    # Scale water vapor by path length (rough approximation)
    pw_scaled = precipitable_water_cm * path_length_km / 2

    # Total optical depth
    tau_h2o = h2o_absorption_model(wavelength_um, pw_scaled)
    tau_co2 = co2_absorption_model(wavelength_um, co2_ppm) * path_length_km
    tau_o3 = o3_absorption_model(wavelength_um, ozone_du) * path_length_km / 10

    tau_total = tau_h2o + tau_co2 + tau_o3

    return np.exp(-tau_total)


def main():
    args = parse_args()

    print("=" * 70)
    print("INFRARED ATMOSPHERIC WINDOWS ANALYSIS")
    print("=" * 70)
    print(f"\nPath length: {args.path_km} km (horizontal)")
    print(f"Relative humidity: {args.humidity}%")

    # Convert humidity to precipitable water (rough approximation)
    pw_cm = args.humidity / 50  # ~1 cm at 50% RH

    # Wavelength grid
    wavelengths = np.linspace(0.8, 20, 1000)

    # Calculate transmission
    T = atmospheric_transmission(wavelengths, args.path_km, pw_cm)

    # IR windows overview
    print("\n" + "-" * 70)
    print("INFRARED ATMOSPHERIC WINDOWS")
    print("-" * 70)
    print("""
The atmosphere has several transmission windows in the infrared:

Window          Wavelength      Primary Application
------          ----------      -------------------
Near-IR         0.7-1.0 um      Night vision, vegetation
SWIR-1          1.0-1.35 um     InGaAs detectors
SWIR-2          1.5-1.8 um      Eye-safe lasers, moisture
SWIR-3          2.0-2.5 um      Hyperspectral, SWIR imaging
MWIR            3.0-5.0 um      Thermal imaging, missile warning
LWIR            8.0-14.0 um     Thermal imaging, night vision
""")

    # Calculate average transmission in each window
    windows = [
        ("Near-IR", 0.7, 1.0),
        ("SWIR-1", 1.0, 1.35),
        ("SWIR-2", 1.5, 1.8),
        ("SWIR-3", 2.0, 2.5),
        ("MWIR", 3.0, 5.0),
        ("LWIR", 8.0, 14.0),
    ]

    print("\n" + "-" * 70)
    print(f"WINDOW TRANSMISSION ({args.path_km} km path, {args.humidity}% RH)")
    print("-" * 70)
    print(f"\n{'Window':>12} {'Range (um)':>15} {'Avg Trans':>12} {'Min Trans':>12}")
    print("-" * 55)

    for name, wl_min, wl_max in windows:
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        T_window = T[mask]
        avg_T = np.mean(T_window)
        min_T = np.min(T_window)
        print(f"{name:>12} {wl_min:>6.1f}-{wl_max:<5.1f} um {avg_T:>11.1%} {min_T:>11.1%}")

    # Absorption bands
    print("\n" + "-" * 70)
    print("MAJOR ABSORPTION BANDS")
    print("-" * 70)
    print("""
Molecule    Wavelength      Band Type           Notes
--------    ----------      ---------           -----
H2O         1.38 um         Combination         Strong, limits SWIR
H2O         1.87 um         Combination         Strong, between SWIR windows
H2O         2.7 um          Fundamental         Very strong, with CO2
H2O         6.3 um          Bending             Strong, between windows
CO2         4.26 um         Asymm. stretch      Strong, MWIR edge
CO2         15 um           Bending             Very strong, LWIR limit
O3          9.6 um          Asymm. stretch      Within LWIR window
""")

    # Thermal imaging analysis
    print("\n" + "-" * 70)
    print("THERMAL IMAGING BANDS COMPARISON")
    print("-" * 70)

    # Calculate blackbody radiance at different temperatures
    temps = [253, 273, 293, 313, 333]  # -20C to +60C
    temp_names = ["-20C", "0C", "20C", "40C", "60C"]

    print("\nBlackbody radiance comparison (W/m^2/sr/um):")
    print(f"{'Temp':>8}", end="")
    for name in temp_names:
        print(f"{name:>12}", end="")
    print()
    print("-" * 70)

    for band_name, wl_center in [("MWIR (4um)", 4e-6), ("LWIR (10um)", 10e-6)]:
        print(f"{band_name:>8}", end="")
        for T_k in temps:
            B = planck_function(wl_center, T_k)
            print(f"{B:>12.2e}", end="")
        print()

    # MWIR vs LWIR advantages
    print("\n" + "-" * 70)
    print("MWIR vs LWIR COMPARISON")
    print("-" * 70)
    print("""
                    MWIR (3-5 um)           LWIR (8-14 um)
                    -------------           --------------
Peak emission       ~750K (hot targets)     ~300K (ambient)
Contrast            High for hot objects    High for ambient
Solar glint         Can be issue            Minimal
Cooling needed      Moderate (200K)         Yes (77K typical)
Detector cost       Moderate                Higher
Smoke/dust          Better penetration      More scattering
Water vapor         Moderate absorption     Less absorption
Applications        Missiles, fires         Surveillance, medical
""")

    # Specific wavelength analysis
    print("\n" + "-" * 70)
    print("SPECIFIC WAVELENGTH TRANSMISSION")
    print("-" * 70)

    specific_wl = [1.06, 1.55, 2.0, 3.39, 4.0, 4.6, 8.0, 10.6, 12.0]
    applications = [
        "Nd:YAG laser",
        "Telecom laser",
        "SWIR imaging",
        "HeNe laser",
        "MWIR peak",
        "CO2 band edge",
        "LWIR start",
        "CO2 laser",
        "LWIR peak",
    ]

    print(f"\n{'Wavelength':>12} {'Transmission':>14} {'Application':>25}")
    print("-" * 55)

    for wl, app in zip(specific_wl, applications):
        T_wl = atmospheric_transmission(np.array([wl]), args.path_km, pw_cm)[0]
        print(f"{wl:>10.2f} um {T_wl:>13.1%} {app:>25}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
For {args.path_km} km horizontal path at {args.humidity}% relative humidity:

Best transmission windows:
- SWIR (1.5-1.8 um): Good for eye-safe imaging
- MWIR (3-5 um): Excellent for hot target detection
- LWIR (8-14 um): Best for ambient temperature thermal imaging

Key absorbers:
- H2O: Dominates SWIR absorption, major in MWIR
- CO2: Strong 4.26 um band affects MWIR, 15 um limits LWIR
- O3: 9.6 um band within LWIR (but weak for horizontal path)

Recommendations:
- Short range thermal: LWIR (8-12 um) for best contrast
- Long range / high humidity: MWIR (3-5 um)
- Laser applications: 1.55 um or 10.6 um depending on power
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Infrared Atmospheric Windows', fontsize=14, fontweight='bold')

            # Plot 1: Full IR transmission spectrum
            ax1 = axes[0, 0]
            ax1.plot(wavelengths, T * 100, 'b-', linewidth=1)

            # Shade windows
            for name, wl_min, wl_max in windows:
                ax1.axvspan(wl_min, wl_max, alpha=0.2, color='green')

            ax1.set_xlabel('Wavelength (um)')
            ax1.set_ylabel('Transmission (%)')
            ax1.set_title(f'IR Transmission ({args.path_km} km path)')
            ax1.set_xlim(0.8, 20)
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)

            # Mark absorption bands
            for wl, label in [(1.38, 'H2O'), (1.87, 'H2O'), (2.7, 'H2O+CO2'),
                             (4.26, 'CO2'), (6.3, 'H2O'), (9.6, 'O3'), (15, 'CO2')]:
                if wl < 20:
                    ax1.axvline(wl, color='red', alpha=0.3, linestyle='--')

            # Plot 2: MWIR window detail
            ax2 = axes[0, 1]
            mask_mwir = (wavelengths >= 2.5) & (wavelengths <= 5.5)
            ax2.plot(wavelengths[mask_mwir], T[mask_mwir] * 100, 'b-', linewidth=2)
            ax2.axvspan(3.0, 5.0, alpha=0.2, color='green', label='MWIR window')
            ax2.axvline(4.26, color='red', linestyle='--', label='CO2 4.26 um')
            ax2.set_xlabel('Wavelength (um)')
            ax2.set_ylabel('Transmission (%)')
            ax2.set_title('MWIR Window (3-5 um)')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: LWIR window detail
            ax3 = axes[1, 0]
            mask_lwir = (wavelengths >= 7) & (wavelengths <= 15)
            ax3.plot(wavelengths[mask_lwir], T[mask_lwir] * 100, 'b-', linewidth=2)
            ax3.axvspan(8.0, 14.0, alpha=0.2, color='green', label='LWIR window')
            ax3.axvline(9.6, color='orange', linestyle='--', label='O3 9.6 um')
            ax3.set_xlabel('Wavelength (um)')
            ax3.set_ylabel('Transmission (%)')
            ax3.set_title('LWIR Window (8-14 um)')
            ax3.set_ylim(0, 100)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Blackbody curves with atmospheric windows
            ax4 = axes[1, 1]
            wl_bb = np.linspace(1, 20, 200) * 1e-6

            for T_k, label in [(300, '300K (27C)'), (500, '500K (227C)'),
                               (1000, '1000K (727C)')]:
                B = planck_function(wl_bb, T_k)
                ax4.plot(wl_bb * 1e6, B / 1e6, label=label, linewidth=2)

            ax4.axvspan(3, 5, alpha=0.15, color='blue', label='MWIR')
            ax4.axvspan(8, 14, alpha=0.15, color='red', label='LWIR')
            ax4.set_xlabel('Wavelength (um)')
            ax4.set_ylabel('Radiance (MW/m^2/sr/um)')
            ax4.set_title('Blackbody Curves and IR Windows')
            ax4.set_xlim(1, 20)
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
