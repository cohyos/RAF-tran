#!/usr/bin/env python3
"""
Atmospheric Polarization (Skylight Polarization)
=================================================

This example demonstrates Rayleigh and Mie scattering polarization patterns
across the sky, which is important for:

- Polarimetric remote sensing
- Navigation using sky polarization (insects, Vikings)
- Atmospheric composition retrieval
- Glare reduction in imaging systems

The degree of polarization depends on:
- Scattering angle (maximum at 90 deg from sun)
- Wavelength (stronger at shorter wavelengths)
- Aerosol content (depolarizes the signal)

References:
- Coulson, K.L. (1988). Polarization and Intensity of Light in the Atmosphere.
- Horváth, G. & Varjú, D. (2004). Polarized Light in Animal Vision.

Usage:
    python 22_atmospheric_polarization.py
    python 22_atmospheric_polarization.py --sza 45 --wavelength 0.45
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering.rayleigh import rayleigh_cross_section
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate atmospheric polarization patterns"
    )
    parser.add_argument("--sza", type=float, default=45, help="Solar zenith angle (deg)")
    parser.add_argument("--wavelength", type=float, default=0.45, help="Wavelength (um)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="atmospheric_polarization.png")
    return parser.parse_args()


def rayleigh_polarization(scattering_angle_deg):
    """
    Calculate degree of polarization for Rayleigh scattering.

    For unpolarized incident light, the scattered light polarization is:
        P = sin^2(theta) / (1 + cos^2(theta))

    Maximum polarization (100%) occurs at theta = 90 degrees.

    Parameters
    ----------
    scattering_angle_deg : float or array_like
        Scattering angle in degrees

    Returns
    -------
    P : float or ndarray
        Degree of linear polarization (0 to 1)
    """
    theta = np.radians(scattering_angle_deg)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rayleigh polarization formula
    P = sin_theta**2 / (1 + cos_theta**2)

    return P


def mie_depolarization_factor(size_parameter):
    """
    Estimate depolarization factor for aerosols using Mie theory approximation.

    Larger particles (larger size parameter) cause more depolarization.

    Parameters
    ----------
    size_parameter : float
        Mie size parameter x = 2*pi*r/wavelength

    Returns
    -------
    depol : float
        Depolarization factor (0 = no effect, 1 = complete depolarization)
    """
    # Empirical approximation
    if size_parameter < 0.1:
        return 0.0  # Rayleigh regime
    elif size_parameter < 1:
        return 0.1 * size_parameter
    elif size_parameter < 10:
        return 0.3 + 0.05 * size_parameter
    else:
        return 0.8  # Geometric optics regime


def scattering_angle_from_geometry(sza_deg, view_zenith_deg, relative_azimuth_deg):
    """
    Calculate scattering angle from viewing geometry.

    Parameters
    ----------
    sza_deg : float
        Solar zenith angle in degrees
    view_zenith_deg : float
        Viewing zenith angle in degrees
    relative_azimuth_deg : float
        Relative azimuth between sun and view direction (deg)

    Returns
    -------
    scattering_angle : float
        Scattering angle in degrees
    """
    sza = np.radians(sza_deg)
    vza = np.radians(view_zenith_deg)
    phi = np.radians(relative_azimuth_deg)

    # Scattering angle from spherical geometry
    cos_theta = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(phi)

    # Clamp to valid range
    cos_theta = np.clip(cos_theta, -1, 1)

    return np.degrees(np.arccos(cos_theta))


def main():
    args = parse_args()

    print("=" * 70)
    print("ATMOSPHERIC POLARIZATION (SKYLIGHT POLARIZATION)")
    print("=" * 70)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"Wavelength: {args.wavelength} um")

    # Rayleigh polarization basics
    print("\n" + "-" * 70)
    print("RAYLEIGH SCATTERING POLARIZATION")
    print("-" * 70)
    print("""
Rayleigh scattering produces linearly polarized light. The degree of
polarization depends on the scattering angle:

    P = sin^2(theta) / (1 + cos^2(theta))

where theta is the angle between incident and scattered light.
""")

    # Calculate polarization vs scattering angle
    angles = np.linspace(0, 180, 37)
    polarization = rayleigh_polarization(angles)

    print(f"{'Scattering Angle':>18} {'Polarization':>15}")
    print("-" * 40)
    for angle, pol in zip([0, 30, 60, 90, 120, 150, 180],
                          rayleigh_polarization([0, 30, 60, 90, 120, 150, 180])):
        print(f"{angle:>15} deg {pol:>14.1%}")

    # Sky polarization pattern
    print("\n" + "-" * 70)
    print("SKY POLARIZATION PATTERN")
    print("-" * 70)
    print(f"""
For a sun at SZA = {args.sza} deg, the sky polarization pattern shows:
- Maximum polarization in a band 90 deg from the sun
- Zero polarization looking directly at or away from sun
- E-vector perpendicular to scattering plane
""")

    # Calculate sky polarization at various view angles
    view_zeniths = [0, 30, 60, 90]
    azimuths = [0, 45, 90, 135, 180]

    print(f"\nSky polarization at various viewing directions (SZA = {args.sza} deg):")
    print(f"{'VZA':>6} {'Azimuth':>8} {'Scatter Angle':>14} {'Polarization':>14}")
    print("-" * 50)

    for vza in [30, 60, 90]:
        for phi in [0, 90, 180]:
            scatter = scattering_angle_from_geometry(args.sza, vza, phi)
            pol = rayleigh_polarization(scatter)
            print(f"{vza:>4} deg {phi:>6} deg {scatter:>12.1f} deg {pol:>13.1%}")

    # Wavelength dependence
    print("\n" + "-" * 70)
    print("WAVELENGTH DEPENDENCE")
    print("-" * 70)

    wavelengths_um = [0.35, 0.45, 0.55, 0.65, 0.85]
    scatter_angle = 90  # Maximum polarization angle

    print(f"\nPolarization at 90 deg scattering angle:")
    print("(Rayleigh polarization is wavelength-independent, but scattering")
    print("intensity follows lambda^-4, so shorter wavelengths dominate sky color)")

    print(f"\n{'Wavelength':>12} {'Cross Section':>18} {'Relative Intensity':>20}")
    print("-" * 55)

    sigma_ref = rayleigh_cross_section(0.55)
    for wl in wavelengths_um:
        sigma = rayleigh_cross_section(wl)
        relative = sigma / sigma_ref
        print(f"{wl:>10.2f} um {sigma:>15.2e} m^2 {relative:>18.2f}")

    # Aerosol effects
    print("\n" + "-" * 70)
    print("AEROSOL DEPOLARIZATION EFFECTS")
    print("-" * 70)
    print("""
Aerosols reduce the degree of polarization because:
1. Multiple scattering randomizes polarization
2. Large particles have complex phase functions
3. Non-spherical particles cause additional depolarization
""")

    aerosol_radii = [0.01, 0.1, 0.5, 1.0, 5.0]  # um
    wavelength = args.wavelength

    print(f"\nDepolarization factor for different aerosol sizes (at {wavelength} um):")
    print(f"{'Radius (um)':>12} {'Size Param':>12} {'Depol Factor':>14} {'Net P at 90deg':>16}")
    print("-" * 60)

    P_rayleigh = rayleigh_polarization(90)
    for r in aerosol_radii:
        x = 2 * np.pi * r / wavelength
        depol = mie_depolarization_factor(x)
        P_net = P_rayleigh * (1 - depol)
        print(f"{r:>12.2f} {x:>12.2f} {depol:>14.2f} {P_net:>15.1%}")

    # Polarization band across sky
    print("\n" + "-" * 70)
    print("POLARIZATION BAND (NEUTRAL POINTS)")
    print("-" * 70)
    print("""
The sky shows special "neutral points" where polarization is zero:

- Babinet point: Above the sun (depolarized by multiple scattering)
- Brewster point: Below the sun
- Arago point: Opposite the sun (antisolar point region)

These points are important for polarimetric navigation and sensing.
""")

    # Applications
    print("\n" + "-" * 70)
    print("APPLICATIONS")
    print("-" * 70)
    print("""
1. POLARIMETRIC REMOTE SENSING
   - Aerosol characterization (depolarization ratio)
   - Cloud phase detection (ice vs water)
   - Surface BRDF measurement

2. NAVIGATION
   - Viking sun compass using sky polarization
   - Insect navigation (bees, ants)
   - Backup aircraft navigation

3. IMAGING SYSTEMS
   - Glare reduction using polarizing filters
   - Enhanced contrast through atmosphere
   - Underwater visibility improvement

4. ATMOSPHERIC SCIENCE
   - Aerosol optical depth retrieval
   - Cloud microphysics
   - Multiple scattering studies
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Key findings for SZA = {args.sza} deg, wavelength = {args.wavelength} um:

- Maximum Rayleigh polarization: {rayleigh_polarization(90):.1%} at 90 deg scatter
- Polarization pattern: Band of max polarization 90 deg from sun
- Wavelength effect: Shorter wavelengths scatter more (lambda^-4)
- Aerosol effect: Larger particles cause depolarization

The degree of polarization is a powerful remote sensing tool for
characterizing atmospheric aerosols and surface properties.
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Atmospheric Polarization Patterns', fontsize=14, fontweight='bold')

            # Plot 1: Polarization vs scattering angle
            ax1 = axes[0, 0]
            angles_plot = np.linspace(0, 180, 181)
            pol_plot = rayleigh_polarization(angles_plot)

            ax1.plot(angles_plot, pol_plot * 100, 'b-', linewidth=2)
            ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
            ax1.axvline(90, color='red', linestyle='--', alpha=0.5, label='Max at 90 deg')
            ax1.set_xlabel('Scattering Angle (deg)')
            ax1.set_ylabel('Degree of Polarization (%)')
            ax1.set_title('Rayleigh Polarization vs Scattering Angle')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 180)
            ax1.set_ylim(0, 105)
            ax1.legend()

            # Plot 2: All-sky polarization pattern
            ax2 = axes[0, 1]
            ax2 = plt.subplot(222, projection='polar')

            # Create polar grid
            theta_grid = np.linspace(0, 2 * np.pi, 72)
            r_grid = np.linspace(0, 90, 19)
            THETA, R = np.meshgrid(theta_grid, r_grid)

            # Calculate polarization for each sky position
            POL = np.zeros_like(THETA)
            for i, vza in enumerate(r_grid):
                for j, phi in enumerate(np.degrees(theta_grid)):
                    scatter = scattering_angle_from_geometry(args.sza, vza, phi)
                    POL[i, j] = rayleigh_polarization(scatter) * 100

            c = ax2.pcolormesh(THETA, R, POL, cmap='viridis', shading='auto')
            plt.colorbar(c, ax=ax2, label='Polarization (%)')
            ax2.set_title(f'Sky Polarization (SZA={args.sza} deg)')
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)

            # Plot 3: Wavelength dependence
            ax3 = axes[1, 0]
            wl_range = np.linspace(0.3, 1.0, 50)
            sigma_range = [rayleigh_cross_section(wl) for wl in wl_range]

            ax3.semilogy(wl_range * 1000, sigma_range, 'b-', linewidth=2)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Cross Section (m^2)')
            ax3.set_title('Rayleigh Scattering Intensity (lambda^-4)')
            ax3.grid(True, alpha=0.3)

            # Mark visible colors
            colors = ['violet', 'blue', 'green', 'orange', 'red']
            wl_marks = [400, 450, 550, 600, 700]
            for wl, c in zip(wl_marks, colors):
                ax3.axvline(wl, color=c, alpha=0.3, linewidth=5)

            # Plot 4: Aerosol depolarization
            ax4 = axes[1, 1]
            radii = np.logspace(-2, 1, 50)
            size_params = 2 * np.pi * radii / args.wavelength
            depol_factors = [mie_depolarization_factor(x) for x in size_params]
            net_pol = [rayleigh_polarization(90) * (1 - d) * 100 for d in depol_factors]

            ax4.semilogx(radii, net_pol, 'b-', linewidth=2, label='Net polarization')
            ax4.semilogx(radii, [rayleigh_polarization(90) * 100] * len(radii),
                        'g--', linewidth=1, label='Pure Rayleigh')
            ax4.set_xlabel('Aerosol Radius (um)')
            ax4.set_ylabel('Net Polarization at 90 deg (%)')
            ax4.set_title('Aerosol Depolarization Effect')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.set_ylim(0, 105)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
