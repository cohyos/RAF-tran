#!/usr/bin/env python3
"""
Spectral Transmission Through the Atmosphere
=============================================

This example calculates the atmospheric transmission spectrum
from UV to near-infrared wavelengths, showing how Rayleigh
scattering causes the blue color of the sky.

The strong wavelength dependence (lambda⁻^4) of Rayleigh scattering
means shorter (blue) wavelengths are scattered much more than
longer (red) wavelengths.

Usage:
    python 02_spectral_transmission.py
    python 02_spectral_transmission.py --sza 30
    python 02_spectral_transmission.py --altitude 3000 --help

Output:
    - Console: Transmission values at key wavelengths
    - Graph: spectral_transmission.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import StandardAtmosphere
    from raf_tran.scattering import RayleighScattering
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate spectral transmission through the atmosphere",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default settings
  %(prog)s --sza 60                 # Sun at 60 deg zenith angle
  %(prog)s --altitude 3000          # Observer at 3000m elevation
  %(prog)s --wl-min 0.3 --wl-max 1.0  # Custom wavelength range
        """
    )
    parser.add_argument(
        "--sza", type=float, default=0,
        help="Solar zenith angle in degrees (default: 0 = overhead)"
    )
    parser.add_argument(
        "--altitude", type=float, default=0,
        help="Observer altitude in meters (default: 0 = sea level)"
    )
    parser.add_argument(
        "--wl-min", type=float, default=0.3,
        help="Minimum wavelength in um (default: 0.3)"
    )
    parser.add_argument(
        "--wl-max", type=float, default=1.0,
        help="Maximum wavelength in um (default: 1.0)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="spectral_transmission.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def wavelength_to_color(wavelength_nm):
    """Convert wavelength to approximate RGB color."""
    wl = wavelength_nm
    if wl < 380:
        return (0.5, 0, 0.5)  # UV - violet
    elif wl < 440:
        return ((440 - wl) / 60, 0, 1)  # Violet
    elif wl < 490:
        return (0, (wl - 440) / 50, 1)  # Blue
    elif wl < 510:
        return (0, 1, (510 - wl) / 20)  # Cyan
    elif wl < 580:
        return ((wl - 510) / 70, 1, 0)  # Green-Yellow
    elif wl < 645:
        return (1, (645 - wl) / 65, 0)  # Orange
    elif wl < 780:
        return (1, 0, 0)  # Red
    else:
        return (0.5, 0, 0)  # IR - dark red


def main():
    args = parse_args()

    print("=" * 70)
    print("SPECTRAL TRANSMISSION THROUGH THE ATMOSPHERE")
    print("=" * 70)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"Observer altitude: {args.altitude} m")
    print(f"Wavelength range: {args.wl_min} - {args.wl_max} um")

    # Calculate air mass
    mu0 = np.cos(np.radians(args.sza))
    if mu0 <= 0:
        print("Error: Sun is below the horizon!")
        sys.exit(1)
    air_mass = 1 / mu0
    print(f"Air mass: {air_mass:.2f}")

    # Setup atmosphere from observer altitude to TOA
    atmosphere = StandardAtmosphere()
    n_layers = 100
    z_levels = np.linspace(args.altitude, 80000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    # Atmospheric properties
    number_density = atmosphere.number_density(z_mid)

    # Wavelength grid
    wavelengths = np.linspace(args.wl_min, args.wl_max, 200)

    # Calculate Rayleigh optical depth and transmission
    rayleigh = RayleighScattering()

    print("\n" + "-" * 70)
    print("RAYLEIGH SCATTERING ANALYSIS")
    print("-" * 70)

    # Calculate optical depth for all wavelengths
    tau_vertical = np.zeros(len(wavelengths))
    transmission = np.zeros(len(wavelengths))

    for i, wl in enumerate(wavelengths):
        tau = rayleigh.optical_depth(np.array([wl]), number_density, dz)
        tau_vertical[i] = np.sum(tau)
        # Direct beam transmission: T = exp(-tau / u0)
        transmission[i] = np.exp(-tau_vertical[i] / mu0)

    # Print key wavelength values
    key_wavelengths = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.00]
    print(f"\n{'Wavelength':>12} {'Color':>10} {'tau_vertical':>12} {'tau_slant':>12} {'Transmission':>14}")
    print("-" * 70)

    color_names = {
        0.35: "UV",
        0.40: "Violet",
        0.45: "Blue",
        0.50: "Cyan",
        0.55: "Green",
        0.60: "Yellow",
        0.70: "Red",
        0.80: "Near-IR",
        1.00: "IR"
    }

    for wl in key_wavelengths:
        if args.wl_min <= wl <= args.wl_max:
            idx = np.argmin(np.abs(wavelengths - wl))
            tau_v = tau_vertical[idx]
            tau_s = tau_v / mu0
            trans = transmission[idx]
            print(f"{wl*1000:>10.0f} nm {color_names.get(wl, ''):>10} "
                  f"{tau_v:>12.4f} {tau_s:>12.4f} {trans:>14.4f}")

    # Calculate blue-to-red ratio (explains sky color)
    idx_blue = np.argmin(np.abs(wavelengths - 0.45))
    idx_red = np.argmin(np.abs(wavelengths - 0.65))

    tau_ratio = tau_vertical[idx_blue] / tau_vertical[idx_red]
    scatter_ratio = (0.65 / 0.45) ** 4  # lambda⁻^4 dependence

    print("\n" + "-" * 70)
    print("WHY IS THE SKY BLUE?")
    print("-" * 70)
    print(f"\nRayleigh scattering cross-section ∝ lambda⁻^4")
    print(f"\nBlue (450nm) vs Red (650nm):")
    print(f"  Theoretical ratio: (650/450)^4 = {scatter_ratio:.2f}")
    print(f"  Calculated tau ratio: {tau_ratio:.2f}")
    print(f"\nBlue light is scattered {tau_ratio:.1f}x more than red light!")
    print("This scattered blue light is what we see as the blue sky.")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Spectral Transmission (SZA={args.sza} deg, Alt={args.altitude}m)',
                        fontsize=14, fontweight='bold')

            # Create spectrum colorbar
            colors = [wavelength_to_color(wl * 1000) for wl in wavelengths]

            # Plot 1: Transmission spectrum
            ax1 = axes[0, 0]
            for i in range(len(wavelengths) - 1):
                ax1.fill_between(
                    [wavelengths[i] * 1000, wavelengths[i + 1] * 1000],
                    [transmission[i], transmission[i + 1]],
                    alpha=0.7,
                    color=colors[i]
                )
            ax1.plot(wavelengths * 1000, transmission, 'k-', linewidth=1.5)
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Direct Beam Transmission')
            ax1.set_title('Atmospheric Transmission Spectrum')
            ax1.set_xlim(args.wl_min * 1000, args.wl_max * 1000)
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Optical depth spectrum
            ax2 = axes[0, 1]
            ax2.semilogy(wavelengths * 1000, tau_vertical, 'b-', linewidth=2, label='Vertical tau')
            ax2.semilogy(wavelengths * 1000, tau_vertical / mu0, 'r--', linewidth=2,
                        label=f'Slant tau (AM={air_mass:.1f})')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Optical Depth')
            ax2.set_title('Rayleigh Optical Depth vs Wavelength')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Verify lambda⁻^4 dependence
            wl_ref = 0.55
            tau_ref = tau_vertical[np.argmin(np.abs(wavelengths - wl_ref))]
            tau_theory = tau_ref * (wl_ref / wavelengths) ** 4
            ax2.semilogy(wavelengths * 1000, tau_theory, 'g:', linewidth=1,
                        label='lambda⁻^4 fit', alpha=0.7)
            ax2.legend()

            # Plot 3: Scattering coefficient vs wavelength
            ax3 = axes[1, 0]
            sigma = rayleigh.cross_section(wavelengths)
            ax3.loglog(wavelengths * 1000, sigma, 'b-', linewidth=2)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Cross Section (m^2)')
            ax3.set_title('Rayleigh Scattering Cross Section')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Solar spectrum modification
            ax4 = axes[1, 1]
            # Approximate solar spectrum (blackbody at 5778K)
            from raf_tran.utils.spectral import planck_function
            solar_temp = 5778
            wl_m = wavelengths * 1e-6  # Convert to meters
            B_sun = planck_function(wl_m, solar_temp)
            B_sun_norm = B_sun / np.max(B_sun)
            B_transmitted = B_sun_norm * transmission

            ax4.fill_between(wavelengths * 1000, B_sun_norm, alpha=0.3, color='orange',
                            label='TOA solar spectrum')
            ax4.fill_between(wavelengths * 1000, B_transmitted, alpha=0.5, color='blue',
                            label='Surface spectrum')
            ax4.plot(wavelengths * 1000, B_sun_norm, 'orange', linewidth=2)
            ax4.plot(wavelengths * 1000, B_transmitted, 'b-', linewidth=2)
            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Relative Intensity')
            ax4.set_title('Solar Spectrum: TOA vs Surface')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(args.wl_min * 1000, args.wl_max * 1000)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 70)
    print("PHYSICAL EXPLANATION")
    print("=" * 70)
    print("""
RAYLEIGH SCATTERING AND SKY COLOR:

1. WAVELENGTH DEPENDENCE: Rayleigh scattering cross-section sigma ∝ lambda⁻^4
   - Blue light (450nm) scatters ~5x more than red light (650nm)
   - This selective scattering creates the blue sky

2. WHY SUNSETS ARE RED:
   - At sunset, sunlight travels through much more atmosphere
   - Most blue light is scattered away before reaching you
   - The remaining direct light appears red/orange

3. TRANSMISSION FORMULA:
   - Direct beam: T = exp(-tau/u0) where u0 = cos(SZA)
   - At high SZA, path length increases as 1/u0

4. ALTITUDE EFFECTS:
   - Higher altitude = less atmosphere above = higher transmission
   - Mountain observatories benefit from this effect
""")


if __name__ == "__main__":
    main()
