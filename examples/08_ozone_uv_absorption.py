#!/usr/bin/env python3
"""
Ozone Layer and UV Absorption
=============================

This example demonstrates the crucial role of stratospheric ozone
in absorbing harmful ultraviolet (UV) radiation from the sun.

We examine:
- UV-C (100-280 nm): Completely absorbed, lethal to life
- UV-B (280-315 nm): Mostly absorbed, causes sunburn
- UV-A (315-400 nm): Mostly transmitted, less harmful

Usage:
    python 08_ozone_uv_absorption.py
    python 08_ozone_uv_absorption.py --ozone-column 300
    python 08_ozone_uv_absorption.py --help

Output:
    - Console: UV transmission analysis
    - Graph: ozone_uv_absorption.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import StandardAtmosphere
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


# Ozone absorption cross sections (approximate, in cm^2)
# Data from scientific literature (simplified)
def ozone_cross_section(wavelength_nm):
    """
    Ozone absorption cross section (Huggins and Hartley bands).

    Parameters
    ----------
    wavelength_nm : array_like
        Wavelength in nanometers

    Returns
    -------
    sigma : ndarray
        Absorption cross section in cm^2
    """
    wl = np.asarray(wavelength_nm)
    sigma = np.zeros_like(wl, dtype=float)

    # Hartley band (200-310 nm) - very strong absorption
    # Peak around 255 nm with sigma ~ 1.1e-17 cm^2
    hartley_mask = (wl >= 200) & (wl <= 310)
    if np.any(hartley_mask):
        # Gaussian approximation
        sigma[hartley_mask] = 1.1e-17 * np.exp(-((wl[hartley_mask] - 255) / 30)**2)

    # Huggins band (310-350 nm) - weaker, structured
    huggins_mask = (wl > 310) & (wl <= 350)
    if np.any(huggins_mask):
        sigma[huggins_mask] = 5e-19 * np.exp(-((wl[huggins_mask] - 310) / 20)**2)

    # Chappuis band (400-700 nm) - very weak, visible
    chappuis_mask = (wl > 400) & (wl <= 700)
    if np.any(chappuis_mask):
        sigma[chappuis_mask] = 5e-21 * np.exp(-((wl[chappuis_mask] - 600) / 100)**2)

    return sigma


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study ozone absorption of UV radiation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UV radiation bands:
  UV-C (100-280 nm): Germicidal, completely absorbed by ozone
  UV-B (280-315 nm): Causes sunburn, mostly absorbed
  UV-A (315-400 nm): Reaches surface, less harmful

Ozone column is measured in Dobson Units (DU):
  1 DU = 2.687e16 molecules/cm^2 = 0.01 mm at STP
  Global average: ~300 DU
  Antarctic ozone hole: can drop below 100 DU

Examples:
  %(prog)s                          # Standard atmosphere
  %(prog)s --ozone-column 200       # Depleted ozone (like ozone hole)
  %(prog)s --sza 60                 # Sun at 60 deg zenith
        """
    )
    parser.add_argument(
        "--ozone-column", type=float, default=None,
        help="Total ozone column in Dobson Units (default: from atmosphere)"
    )
    parser.add_argument(
        "--sza", type=float, default=30,
        help="Solar zenith angle in degrees (default: 30)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="ozone_uv_absorption.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("OZONE LAYER AND UV ABSORPTION")
    print("=" * 80)
    print(f"\nSolar zenith angle: {args.sza} deg")

    mu0 = np.cos(np.radians(args.sza))
    if mu0 <= 0:
        print("Error: Sun below horizon!")
        sys.exit(1)

    air_mass = 1 / mu0
    print(f"Air mass: {air_mass:.2f}")

    # Calculate ozone column from atmosphere
    atmosphere = StandardAtmosphere()
    z_levels = np.linspace(0, 60000, 121)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    n_air = atmosphere.number_density(z_mid)
    o3_vmr = atmosphere.o3_vmr(z_mid)

    # Column ozone in molecules/cm^2
    o3_column_calc = np.sum(o3_vmr * n_air * dz) / 1e4  # molecules/cm^2

    # Convert to Dobson Units (1 DU = 2.687e16 molecules/cm^2)
    DU_factor = 2.687e16
    o3_DU_calc = o3_column_calc / DU_factor

    if args.ozone_column is not None:
        o3_DU = args.ozone_column
        o3_column = o3_DU * DU_factor
        print(f"Ozone column: {o3_DU:.0f} DU (user specified)")
    else:
        o3_DU = o3_DU_calc
        o3_column = o3_column_calc
        print(f"Ozone column: {o3_DU:.0f} DU (from standard atmosphere)")

    # Wavelength grid
    wavelengths = np.linspace(200, 400, 201)  # nm

    # Calculate ozone optical depth and transmission
    sigma_o3 = ozone_cross_section(wavelengths)  # cm^2
    tau_o3_vertical = sigma_o3 * o3_column  # optical depth
    tau_o3_slant = tau_o3_vertical / mu0  # slant path

    transmission = np.exp(-tau_o3_slant)

    print("\n" + "-" * 80)
    print("UV TRANSMISSION THROUGH OZONE LAYER")
    print("-" * 80)

    # UV band analysis
    uv_bands = {
        "UV-C": (200, 280, "Germicidal - completely absorbed"),
        "UV-B": (280, 315, "Sunburn - partially absorbed"),
        "UV-A": (315, 400, "Tanning - mostly transmitted"),
    }

    print(f"\n{'UV Band':<8} {'Wavelength':>12} {'Mean tau':>10} {'Mean Trans':>12} {'Status':<25}")
    print("-" * 80)

    band_results = {}
    for band_name, (wl_min, wl_max, status) in uv_bands.items():
        mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
        mean_tau = np.mean(tau_o3_slant[mask])
        mean_trans = np.mean(transmission[mask])

        band_results[band_name] = {
            "wl_min": wl_min,
            "wl_max": wl_max,
            "mean_tau": mean_tau,
            "mean_trans": mean_trans,
        }

        print(f"{band_name:<8} {wl_min:>4}-{wl_max:<4} nm {mean_tau:>10.1f} "
              f"{mean_trans*100:>11.2f}% {status:<25}")

    # Specific wavelengths
    print("\n" + "-" * 80)
    print("TRANSMISSION AT KEY WAVELENGTHS")
    print("-" * 80)

    key_wavelengths = [254, 280, 300, 310, 320, 340, 380]
    print(f"\n{'Wavelength':>12} {'sigma_O3 (cm^2)':>14} {'tau (vertical)':>14} "
          f"{'tau (slant)':>12} {'Transmission':>14}")
    print("-" * 80)

    for wl in key_wavelengths:
        idx = np.argmin(np.abs(wavelengths - wl))
        sigma = sigma_o3[idx]
        tau_v = tau_o3_vertical[idx]
        tau_s = tau_o3_slant[idx]
        trans = transmission[idx]

        print(f"{wl:>10} nm {sigma:>14.2e} {tau_v:>14.2f} {tau_s:>12.2f} {trans*100:>13.4f}%")

    # Ozone hole scenario
    print("\n" + "=" * 80)
    print("OZONE DEPLETION IMPACT")
    print("=" * 80)

    scenarios = {
        "Normal (300 DU)": 300,
        "Moderate depletion (200 DU)": 200,
        "Severe (ozone hole, 100 DU)": 100,
    }

    print(f"\nUV-B transmission at 300 nm:")
    print("-" * 60)

    wl_300_idx = np.argmin(np.abs(wavelengths - 300))
    sigma_300 = sigma_o3[wl_300_idx]

    for scenario_name, o3_du in scenarios.items():
        o3_col = o3_du * DU_factor
        tau = sigma_300 * o3_col / mu0
        trans = np.exp(-tau) * 100
        print(f"  {scenario_name:<35}: {trans:>8.4f}%")

    # Calculate increase in UV-B
    print("\n" + "-" * 60)
    print("RELATIVE UV-B INCREASE WITH OZONE DEPLETION:")
    print("-" * 60)

    # Mean UV-B transmission (280-315 nm)
    # IMPORTANT: Compute transmission at each wavelength FIRST, then average
    # (NOT: compute mean cross-section then transmission - that's wrong!)
    uvb_mask = (wavelengths >= 280) & (wavelengths <= 315)
    uvb_sigma = sigma_o3[uvb_mask]

    # Transmission at each UV-B wavelength for different ozone columns
    trans_normal = np.mean(np.exp(-uvb_sigma * 300 * DU_factor / mu0))
    trans_depleted = np.mean(np.exp(-uvb_sigma * 200 * DU_factor / mu0))
    trans_hole = np.mean(np.exp(-uvb_sigma * 100 * DU_factor / mu0))

    print(f"  300 DU -> 200 DU: UV-B increases by {(trans_depleted/trans_normal - 1)*100:+.0f}%")
    print(f"  300 DU -> 100 DU: UV-B increases by {(trans_hole/trans_normal - 1)*100:+.0f}%")

    print("""
HEALTH IMPLICATIONS:
  - Each 1% decrease in ozone -> ~2% increase in UV-B at surface
  - Each 1% increase in UV-B -> ~2% increase in skin cancer risk
  - Antarctic ozone hole (2/3 depletion) -> UV-B doubles -> cancer risk quadruples
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Ozone Layer and UV Absorption (O_3 = {o3_DU:.0f} DU, SZA = {args.sza} deg)',
                        fontsize=14, fontweight='bold')

            # Plot 1: Ozone absorption cross section
            ax1 = axes[0, 0]
            ax1.semilogy(wavelengths, sigma_o3, 'b-', linewidth=2)
            ax1.axvspan(200, 280, alpha=0.3, color='purple', label='UV-C')
            ax1.axvspan(280, 315, alpha=0.3, color='blue', label='UV-B')
            ax1.axvspan(315, 400, alpha=0.3, color='cyan', label='UV-A')
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Absorption Cross Section (cm^2)')
            ax1.set_title('Ozone Absorption Cross Section')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(200, 400)

            # Plot 2: Transmission spectrum
            ax2 = axes[0, 1]
            ax2.semilogy(wavelengths, transmission * 100, 'g-', linewidth=2)
            ax2.axvspan(200, 280, alpha=0.3, color='purple')
            ax2.axvspan(280, 315, alpha=0.3, color='blue')
            ax2.axvspan(315, 400, alpha=0.3, color='cyan')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Transmission (%)')
            ax2.set_title('UV Transmission Through Ozone Layer')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(200, 400)
            ax2.set_ylim(1e-10, 200)

            # Add horizontal lines for reference
            ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(1, color='gray', linestyle='--', alpha=0.5)

            # Plot 3: Ozone profile
            ax3 = axes[1, 0]
            o3_conc = o3_vmr * n_air  # molecules/m^3
            o3_ppb = o3_vmr * 1e9

            ax3.plot(o3_ppb, z_mid / 1000, 'b-', linewidth=2)
            ax3.axhspan(15, 35, alpha=0.2, color='blue', label='Ozone layer')
            ax3.set_xlabel('Ozone Mixing Ratio (ppb)')
            ax3.set_ylabel('Altitude (km)')
            ax3.set_title('Ozone Vertical Profile')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, max(o3_ppb) * 1.1)

            # Mark peak
            peak_idx = np.argmax(o3_ppb)
            ax3.plot(o3_ppb[peak_idx], z_mid[peak_idx] / 1000, 'ro', markersize=10)
            ax3.annotate(f'Peak: {z_mid[peak_idx]/1000:.0f} km',
                        (o3_ppb[peak_idx], z_mid[peak_idx] / 1000),
                        textcoords="offset points", xytext=(10, 0))

            # Plot 4: Comparison of ozone depletion scenarios
            ax4 = axes[1, 1]

            ozone_levels = [300, 250, 200, 150, 100]
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(ozone_levels)))

            for o3_du, color in zip(ozone_levels, colors):
                o3_col = o3_du * DU_factor
                tau = sigma_o3 * o3_col / mu0
                trans = np.exp(-tau) * 100
                ax4.semilogy(wavelengths, trans, color=color, linewidth=2,
                            label=f'{o3_du} DU')

            ax4.axvspan(280, 315, alpha=0.2, color='blue')
            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Transmission (%)')
            ax4.set_title('Effect of Ozone Depletion on UV Transmission')
            ax4.legend(title='Ozone Column')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(200, 400)
            ax4.set_ylim(1e-6, 200)

            # Add annotation
            ax4.annotate('UV-B band\n(280-315 nm)', xy=(297, 1e-3), ha='center',
                        fontsize=9, color='blue')

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
THE OZONE LAYER AND UV PROTECTION:

1. OZONE FORMATION (Chapman Cycle):
   - UV-C splits O_2 -> 2O
   - O + O_2 -> O_3 (ozone)
   - Occurs mainly at 15-35 km altitude

2. UV ABSORPTION:
   - O_3 + UV -> O_2 + O
   - Strongest in Hartley band (200-310 nm)
   - Absorbs essentially all UV-C and most UV-B

3. BIOLOGICAL IMPORTANCE:
   - UV-C (< 280 nm): DNA damage, sterilization
   - UV-B (280-315 nm): Sunburn, skin cancer, cataracts
   - UV-A (315-400 nm): Tanning, aging, less carcinogenic

4. OZONE DEPLETION:
   - CFCs release chlorine in stratosphere
   - Cl + O_3 -> ClO + O_2 (catalytic destruction)
   - Montreal Protocol (1987) phased out CFCs
   - Ozone layer is slowly recovering

5. UNITS:
   - Dobson Unit (DU): Column ozone measurement
   - 1 DU = 0.01 mm of pure O_3 at STP
   - 300 DU = 3 mm compressed ozone
""")


if __name__ == "__main__":
    main()
