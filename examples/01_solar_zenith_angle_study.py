#!/usr/bin/env python3
"""
Solar Zenith Angle Study
========================

This example demonstrates how solar zenith angle affects:
- Direct beam transmission
- Diffuse radiation
- Total downwelling flux at the surface

The solar zenith angle (SZA) is the angle between the sun and the vertical.
At SZA=0 deg, the sun is directly overhead; at SZA=90 deg, it's at the horizon.

Usage:
    python 01_solar_zenith_angle_study.py
    python 01_solar_zenith_angle_study.py --wavelength 0.55 --tau 0.3
    python 01_solar_zenith_angle_study.py --help

Output:
    - Console: Table of flux values at different SZA
    - Graph: solar_zenith_angle_study.png
"""

import argparse
import numpy as np
import sys

# Add parent directory to path for running without installation
sys.path.insert(0, '..')

try:
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.atmosphere import StandardAtmosphere
    from raf_tran.scattering import RayleighScattering
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study the effect of solar zenith angle on atmospheric radiation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default: 550nm wavelength
  %(prog)s --wavelength 0.4         # Blue light (400nm)
  %(prog)s --tau 0.5                # Higher optical depth
  %(prog)s --albedo 0.8             # Snow-covered surface
        """
    )
    parser.add_argument(
        "--wavelength", type=float, default=0.55,
        help="Wavelength in micrometers (default: 0.55 = green light)"
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Total optical depth (default: calculated from Rayleigh)"
    )
    parser.add_argument(
        "--albedo", type=float, default=0.1,
        help="Surface albedo (default: 0.1)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting (text output only)"
    )
    parser.add_argument(
        "--output", type=str, default="solar_zenith_angle_study.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("SOLAR ZENITH ANGLE STUDY")
    print("=" * 70)
    print(f"\nWavelength: {args.wavelength * 1000:.0f} nm")
    print(f"Surface albedo: {args.albedo}")

    # Setup atmosphere
    atmosphere = StandardAtmosphere()
    n_layers = 50
    z_levels = np.linspace(0, 50000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    # Calculate Rayleigh optical depth
    rayleigh = RayleighScattering()
    number_density = atmosphere.number_density(z_mid)
    wavelength = np.array([args.wavelength])

    if args.tau is not None:
        # Use specified optical depth, distributed exponentially
        tau_total = args.tau
        scale_height = 8000  # meters
        tau_per_layer = tau_total * np.exp(-z_mid / scale_height)
        tau_per_layer /= tau_per_layer.sum() / tau_total
        print(f"Optical depth: {args.tau} (user specified)")
    else:
        tau_rayleigh = rayleigh.optical_depth(wavelength, number_density, dz)
        tau_per_layer = tau_rayleigh.ravel()
        tau_total = np.sum(tau_per_layer)
        print(f"Rayleigh optical depth: {tau_total:.4f}")

    # Solar angles to study
    sza_values = np.arange(0, 90, 5)  # 0 deg to 85 deg in 5 deg steps

    # Results storage
    direct_flux = []
    diffuse_flux = []
    total_flux = []
    upward_flux = []

    # Two-stream solver
    solver = TwoStreamSolver()
    omega = np.ones(n_layers)  # Pure scattering (Rayleigh)
    g = np.zeros(n_layers)     # Isotropic scattering

    print("\n" + "-" * 70)
    print(f"{'SZA ( deg)':>8} {'u0':>8} {'Direct':>12} {'Diffuse':>12} {'Total':>12} {'Reflected':>12}")
    print(f"{'':>8} {'':>8} {'(W/m^2)':>12} {'(W/m^2)':>12} {'(W/m^2)':>12} {'(W/m^2)':>12}")
    print("-" * 70)

    solar_flux = 1361.0  # W/m^2 (solar constant)

    for sza in sza_values:
        mu0 = np.cos(np.radians(sza))

        if mu0 <= 0.01:  # Sun below horizon
            direct_flux.append(0)
            diffuse_flux.append(0)
            total_flux.append(0)
            upward_flux.append(0)
            continue

        result = solver.solve_solar(
            tau=tau_per_layer,
            omega=omega,
            g=g,
            mu0=mu0,
            flux_toa=solar_flux,
            surface_albedo=args.albedo,
        )

        # Extract fluxes - with levels_surface_to_toa=True:
        # index 0 = surface, index -1 = TOA
        F_direct = result.flux_direct[0]   # Direct flux at surface
        F_diffuse = result.flux_down[0]    # Diffuse flux at surface
        F_total = F_direct + F_diffuse
        F_up = result.flux_up[-1]          # Reflected at TOA

        direct_flux.append(F_direct)
        diffuse_flux.append(F_diffuse)
        total_flux.append(F_total)
        upward_flux.append(F_up)

        print(f"{sza:>8.0f} {mu0:>8.3f} {F_direct:>12.2f} {F_diffuse:>12.2f} "
              f"{F_total:>12.2f} {F_up:>12.2f}")

    print("-" * 70)

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    max_total_idx = np.argmax(total_flux)
    print(f"\n1. Maximum surface flux: {total_flux[max_total_idx]:.1f} W/m^2 at SZA={sza_values[max_total_idx]} deg")

    if len(total_flux) > 0 and total_flux[0] > 0:
        ratio_60 = total_flux[12] / total_flux[0] if total_flux[0] > 0 else 0  # SZA=60 deg
        print(f"2. Flux at SZA=60 deg is {ratio_60*100:.1f}% of overhead sun flux")

    diffuse_fraction = [d/(d+f) if (d+f) > 0 else 0
                        for d, f in zip(diffuse_flux, direct_flux)]
    print(f"3. Diffuse fraction at SZA=0 deg: {diffuse_fraction[0]*100:.1f}%")
    print(f"4. Diffuse fraction at SZA=80 deg: {diffuse_fraction[16]*100:.1f}%")
    print(f"\n   (Higher diffuse fraction at large SZA due to longer path length)")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Solar Zenith Angle Study (lambda={args.wavelength*1000:.0f}nm, tau={tau_total:.3f})',
                        fontsize=14, fontweight='bold')

            # Plot 1: Flux components vs SZA
            ax1 = axes[0, 0]
            ax1.plot(sza_values, direct_flux, 'b-', linewidth=2, label='Direct')
            ax1.plot(sza_values, diffuse_flux, 'g--', linewidth=2, label='Diffuse')
            ax1.plot(sza_values, total_flux, 'r-', linewidth=2, label='Total')
            ax1.set_xlabel('Solar Zenith Angle ( deg)')
            ax1.set_ylabel('Surface Flux (W/m^2)')
            ax1.set_title('Surface Irradiance vs Solar Zenith Angle')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 90)

            # Plot 2: Flux vs cos(SZA)
            ax2 = axes[0, 1]
            mu0_values = np.cos(np.radians(sza_values))
            ax2.plot(mu0_values, total_flux, 'ro-', linewidth=2, markersize=4)
            ax2.set_xlabel('cos(SZA) = u0')
            ax2.set_ylabel('Total Surface Flux (W/m^2)')
            ax2.set_title('Surface Flux vs cos(SZA)')
            ax2.grid(True, alpha=0.3)

            # Add reference line for F = F_0 * u0 (no atmosphere)
            F_no_atm = solar_flux * mu0_values
            ax2.plot(mu0_values, F_no_atm, 'k--', alpha=0.5, label='No atmosphere')
            ax2.legend()

            # Plot 3: Diffuse fraction
            ax3 = axes[1, 0]
            ax3.plot(sza_values, [f*100 for f in diffuse_fraction], 'g-', linewidth=2)
            ax3.set_xlabel('Solar Zenith Angle ( deg)')
            ax3.set_ylabel('Diffuse Fraction (%)')
            ax3.set_title('Fraction of Diffuse Radiation')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 90)
            ax3.set_ylim(0, 100)

            # Plot 4: Reflected flux
            ax4 = axes[1, 1]
            ax4.plot(sza_values, upward_flux, 'm-', linewidth=2)
            ax4.set_xlabel('Solar Zenith Angle ( deg)')
            ax4.set_ylabel('TOA Upward Flux (W/m^2)')
            ax4.set_title('Reflected Radiation at Top of Atmosphere')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 90)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 70)
    print("PHYSICAL EXPLANATION")
    print("=" * 70)
    print("""
As the solar zenith angle increases:

1. PATH LENGTH EFFECT: The direct beam travels through more atmosphere
   (path length = 1/cos(SZA)), causing more scattering and absorption.

2. DIRECT vs DIFFUSE: At low SZA, most radiation reaches the surface as
   direct beam. At high SZA, the longer path length increases scattering,
   converting direct to diffuse radiation.

3. COSINE EFFECT: Even without atmosphere, surface irradiance decreases
   as cos(SZA) because the same solar flux is spread over a larger area.

4. AIR MASS: Optical air mass ~ 1/cos(SZA). At SZA=60 deg, air mass=2;
   at SZA=80 deg, air mass~6. This dramatically increases attenuation.
""")


if __name__ == "__main__":
    main()
