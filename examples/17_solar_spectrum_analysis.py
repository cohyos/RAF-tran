#!/usr/bin/env python3
"""
Solar Spectrum Analysis
=======================

This example analyzes the solar spectrum and its interaction with
Earth's atmosphere, including:

- Extraterrestrial solar irradiance
- Atmospheric absorption windows
- Surface irradiance calculations
- UV, visible, and IR band analysis

References:
- Gueymard (2004). The sun's total and spectral irradiance.
- ASTM E-490: Standard Solar Constant and Zero Air Mass Solar Spectral Irradiance.

Usage:
    python 17_solar_spectrum_analysis.py
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.utils.constants import SOLAR_CONSTANT, PLANCK_CONSTANT, SPEED_OF_LIGHT, BOLTZMANN_CONSTANT
    from raf_tran.scattering.rayleigh import rayleigh_optical_depth
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze solar spectrum and atmospheric transmission"
    )
    parser.add_argument("--sza", type=float, default=30, help="Solar zenith angle (deg)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="solar_spectrum.png")
    return parser.parse_args()


def planck_spectral_irradiance(wavelength_m, temperature_k, solid_angle):
    """
    Planck function integrated over solid angle for spectral irradiance.
    Returns W/m^2/nm at Earth orbit.
    """
    h = PLANCK_CONSTANT
    c = SPEED_OF_LIGHT
    k = BOLTZMANN_CONSTANT
    T = temperature_k

    wl = wavelength_m
    c1 = 2 * np.pi * h * c**2  # Factor of pi for hemispheric
    c2 = h * c / k

    B = c1 / (wl**5 * (np.exp(c2 / (wl * T)) - 1))

    # Convert to W/m^2/nm at Earth
    # Sun subtends 6.8e-5 sr from Earth
    return B * solid_angle * 1e-9  # Convert m to nm


def approximate_solar_spectrum(wavelength_nm):
    """
    Approximate extraterrestrial solar spectral irradiance.
    Based on 5778 K blackbody scaled to solar constant.
    Returns W/m^2/nm.
    """
    wl_m = wavelength_nm * 1e-9
    T_sun = 5778  # K
    sun_solid_angle = 6.8e-5  # sr (as seen from Earth)

    # Planck function
    h = PLANCK_CONSTANT
    c = SPEED_OF_LIGHT
    k = BOLTZMANN_CONSTANT

    c1 = 2 * np.pi * h * c**2
    c2 = h * c / k

    with np.errstate(over='ignore'):
        exp_term = np.exp(c2 / (wl_m * T_sun))
        B = c1 / (wl_m**5 * (exp_term - 1))

    # Spectral irradiance at Earth (W/m^2/m)
    E_spectral = B * sun_solid_angle

    # Convert to W/m^2/nm
    return E_spectral * 1e-9


def ozone_optical_depth(wavelength_nm, column_DU=300):
    """Approximate ozone optical depth based on Hartley/Huggins bands."""
    wl = wavelength_nm

    # Simplified cross-section model
    if wl < 200:
        sigma = 0
    elif wl < 310:
        # Hartley band
        sigma = 1.1e-17 * np.exp(-((wl - 255) / 30)**2)
    elif wl < 350:
        # Huggins band
        sigma = 5e-19 * np.exp(-((wl - 310) / 20)**2)
    else:
        sigma = 0

    # Column ozone in molecules/cm^2
    DU_to_molec = 2.687e16
    column = column_DU * DU_to_molec

    return sigma * column


def main():
    args = parse_args()

    print("=" * 70)
    print("SOLAR SPECTRUM ANALYSIS")
    print("=" * 70)
    print(f"\nSolar zenith angle: {args.sza} deg")

    mu0 = np.cos(np.radians(args.sza))
    air_mass = 1.0 / mu0

    print(f"Air mass: {air_mass:.3f}")
    print(f"Solar constant: {SOLAR_CONSTANT} W/m^2")

    # Spectral bands
    print("\n" + "-" * 70)
    print("SPECTRAL BAND DEFINITIONS")
    print("-" * 70)

    bands = {
        "UV-C": (100, 280, "Germicidal, absorbed by O2/O3"),
        "UV-B": (280, 315, "Causes sunburn, mostly absorbed"),
        "UV-A": (315, 400, "Black light, reaches surface"),
        "Visible": (400, 700, "Human vision"),
        "Near-IR": (700, 1400, "Heating, water absorption"),
        "SWIR": (1400, 3000, "Water vapor absorption"),
        "MWIR": (3000, 8000, "Thermal emission begins"),
        "LWIR": (8000, 14000, "Atmospheric window"),
    }

    print(f"\n{'Band':<12} {'Range (nm)':<16} {'Description':<30}")
    print("-" * 60)
    for name, (wl_min, wl_max, desc) in bands.items():
        print(f"{name:<12} {wl_min:>5}-{wl_max:<8} {desc:<30}")

    # Calculate spectral irradiance (vectorized)
    wavelengths = np.linspace(200, 2500, 500)  # nm
    E_toa = approximate_solar_spectrum(wavelengths)

    # Atmospheric transmission
    print("\n" + "-" * 70)
    print("ATMOSPHERIC TRANSMISSION AT KEY WAVELENGTHS")
    print("-" * 70)

    key_wavelengths = [300, 400, 500, 550, 600, 700, 1000, 1500, 2000]

    print(f"\n{'Wavelength':>12} {'E_TOA':>12} {'tau_Ray':>10} {'tau_O3':>10} {'T_atm':>10} {'E_sfc':>12}")
    print(f"{'(nm)':>12} {'(W/m2/nm)':>12} {'':>10} {'':>10} {'':>10} {'(W/m2/nm)':>12}")
    print("-" * 75)

    for wl in key_wavelengths:
        E = approximate_solar_spectrum(wl)

        # Rayleigh optical depth
        tau_ray = rayleigh_optical_depth(wl / 1000, 2.152e25)

        # Ozone optical depth
        tau_o3 = ozone_optical_depth(wl)

        # Total transmission
        tau_total = tau_ray + tau_o3
        T_atm = np.exp(-tau_total * air_mass)

        E_surface = E * T_atm

        print(f"{wl:>12} {E:>12.4f} {tau_ray:>10.4f} {tau_o3:>10.4f} {T_atm:>10.4f} {E_surface:>12.4f}")

    # Band-integrated irradiances
    print("\n" + "-" * 70)
    print("BAND-INTEGRATED IRRADIANCE")
    print("-" * 70)

    print(f"\n{'Band':<12} {'TOA (W/m2)':>14} {'Surface (W/m2)':>16} {'Absorbed (%)':>14}")
    print("-" * 60)

    total_toa = 0
    total_sfc = 0

    for name, (wl_min, wl_max, _) in bands.items():
        if wl_max > 2500:
            continue  # Skip bands outside our range

        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        if not np.any(mask):
            continue

        wl_band = wavelengths[mask]
        E_band_toa = E_toa[mask]

        # Calculate transmission for each wavelength
        T_band = np.zeros_like(wl_band)
        for i, wl in enumerate(wl_band):
            tau_ray = rayleigh_optical_depth(wl / 1000, 2.152e25)
            tau_o3 = ozone_optical_depth(wl)
            T_band[i] = np.exp(-(tau_ray + tau_o3) * air_mass)

        E_band_sfc = E_band_toa * T_band

        # Integrate
        irrad_toa = np.trapezoid(E_band_toa, wl_band)
        irrad_sfc = np.trapezoid(E_band_sfc, wl_band)

        absorbed_pct = (1 - irrad_sfc / irrad_toa) * 100 if irrad_toa > 0 else 0

        total_toa += irrad_toa
        total_sfc += irrad_sfc

        print(f"{name:<12} {irrad_toa:>14.1f} {irrad_sfc:>16.1f} {absorbed_pct:>13.1f}%")

    print("-" * 60)
    print(f"{'TOTAL':<12} {total_toa:>14.1f} {total_sfc:>16.1f} {(1-total_sfc/total_toa)*100:>13.1f}%")

    # Comparison with actual solar constant
    print(f"\n(Note: Blackbody approximation, actual solar constant = {SOLAR_CONSTANT} W/m^2)")

    # UV analysis
    print("\n" + "-" * 70)
    print("UV RADIATION ANALYSIS")
    print("-" * 70)

    uv_wavelengths = np.linspace(280, 400, 100)
    E_uv_toa = np.array([approximate_solar_spectrum(wl) for wl in uv_wavelengths])

    # UV with and without ozone
    T_with_o3 = []
    T_without_o3 = []

    for wl in uv_wavelengths:
        tau_ray = rayleigh_optical_depth(wl / 1000, 2.152e25)
        tau_o3 = ozone_optical_depth(wl)

        T_with_o3.append(np.exp(-(tau_ray + tau_o3) * air_mass))
        T_without_o3.append(np.exp(-tau_ray * air_mass))

    E_uv_with_o3 = E_uv_toa * np.array(T_with_o3)
    E_uv_no_o3 = E_uv_toa * np.array(T_without_o3)

    uv_with = np.trapezoid(E_uv_with_o3, uv_wavelengths)
    uv_without = np.trapezoid(E_uv_no_o3, uv_wavelengths)

    print(f"\nUV irradiance (280-400 nm):")
    print(f"  With ozone layer:    {uv_with:.2f} W/m^2")
    print(f"  Without ozone layer: {uv_without:.2f} W/m^2")
    print(f"  Ozone protection:    {(1 - uv_with/uv_without)*100:.1f}% reduction")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Solar spectrum characteristics:
- Peak wavelength (Wien): {2898/5778*1000:.0f} nm (visible green-yellow)
- ~50% of energy in visible (400-700 nm)
- ~45% in near-IR (700-3000 nm)
- ~5% in UV (< 400 nm)

Atmospheric effects at SZA = {args.sza} deg:
- Rayleigh scattering: Blue attenuated more than red
- Ozone absorption: Removes UV-C and most UV-B
- Total clear-sky transmission: {total_sfc/total_toa*100:.1f}%
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Solar Spectrum Analysis (SZA = {args.sza} deg)',
                        fontsize=14, fontweight='bold')

            # Plot 1: Solar spectrum at TOA
            ax1 = axes[0, 0]
            ax1.plot(wavelengths, E_toa, 'b-', linewidth=2, label='TOA (5778K blackbody)')

            # Add band colors
            ax1.axvspan(280, 400, alpha=0.2, color='purple', label='UV')
            ax1.axvspan(400, 700, alpha=0.2, color='green', label='Visible')
            ax1.axvspan(700, 1400, alpha=0.2, color='red', label='Near-IR')

            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Spectral Irradiance (W/m^2/nm)')
            ax1.set_title('Extraterrestrial Solar Spectrum')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(200, 2500)

            # Plot 2: Atmospheric transmission
            ax2 = axes[0, 1]

            T_total = []
            T_ray = []
            T_o3 = []

            for wl in wavelengths:
                tau_r = rayleigh_optical_depth(wl / 1000, 2.152e25)
                tau_oz = ozone_optical_depth(wl)

                T_ray.append(np.exp(-tau_r * air_mass))
                T_o3.append(np.exp(-tau_oz * air_mass))
                T_total.append(np.exp(-(tau_r + tau_oz) * air_mass))

            ax2.plot(wavelengths, T_ray, 'b--', linewidth=1.5, label='Rayleigh only')
            ax2.plot(wavelengths, T_o3, 'g--', linewidth=1.5, label='Ozone only')
            ax2.plot(wavelengths, T_total, 'r-', linewidth=2, label='Total')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Transmission')
            ax2.set_title('Atmospheric Transmission')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(200, 2500)
            ax2.set_ylim(0, 1.1)

            # Plot 3: TOA vs Surface spectrum
            ax3 = axes[1, 0]
            E_surface = E_toa * np.array(T_total)

            ax3.fill_between(wavelengths, E_toa, alpha=0.3, color='orange', label='Absorbed')
            ax3.plot(wavelengths, E_toa, 'r-', linewidth=2, label='TOA')
            ax3.plot(wavelengths, E_surface, 'b-', linewidth=2, label='Surface')
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Spectral Irradiance (W/m^2/nm)')
            ax3.set_title('TOA vs Surface Irradiance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(200, 2500)

            # Plot 4: UV detail
            ax4 = axes[1, 1]
            ax4.plot(uv_wavelengths, E_uv_toa, 'r-', linewidth=2, label='TOA')
            ax4.plot(uv_wavelengths, E_uv_no_o3, 'g--', linewidth=2, label='No ozone')
            ax4.plot(uv_wavelengths, E_uv_with_o3, 'b-', linewidth=2, label='With ozone')
            ax4.axvspan(280, 315, alpha=0.2, color='red', label='UV-B')
            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Spectral Irradiance (W/m^2/nm)')
            ax4.set_title('UV Radiation Detail')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
