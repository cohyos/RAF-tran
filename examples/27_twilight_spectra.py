#!/usr/bin/env python3
"""
Twilight and Sunrise/Sunset Spectra
===================================

This example shows the spectral changes during twilight, including:
- Strong reddening at sunset/sunrise
- Enhanced path length effects
- Ozone Chappuis band absorption
- Belt of Venus and Earth shadow
- Green flash phenomenon

Twilight phases:
- Civil twilight: SZA 90-96 deg (can read newspaper)
- Nautical twilight: SZA 96-102 deg (horizon visible)
- Astronomical twilight: SZA 102-108 deg (sky glow)

Applications:
- Atmospheric remote sensing
- Twilight airglow studies
- Photography golden hour
- Aviation operations

Usage:
    python 27_twilight_spectra.py
    python 27_twilight_spectra.py --sza 92
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering.rayleigh import rayleigh_cross_section
    from raf_tran.utils.air_mass import kasten_young_air_mass
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze twilight spectral effects"
    )
    parser.add_argument("--sza", type=float, default=92, help="Solar zenith angle (deg)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="twilight_spectra.png")
    return parser.parse_args()


def twilight_air_mass(sza_deg):
    """
    Calculate effective air mass during twilight.

    Uses spherical geometry for SZA > 90 deg.

    Parameters
    ----------
    sza_deg : float
        Solar zenith angle in degrees

    Returns
    -------
    m : float
        Effective air mass
    """
    if sza_deg < 89:
        return kasten_young_air_mass(sza_deg)
    elif sza_deg < 90:
        # Transition region
        return kasten_young_air_mass(89) + (sza_deg - 89) * 20
    else:
        # Twilight - sun below horizon
        # Air mass increases rapidly
        # Simplified model based on refracted path through atmosphere
        m_90 = 40  # Air mass at horizon
        # Increases ~10 per degree below horizon
        return m_90 + (sza_deg - 90) * 15


def rayleigh_extinction_coefficient(wavelength_um):
    """Calculate Rayleigh extinction coefficient at sea level."""
    sigma = rayleigh_cross_section(wavelength_um)
    N = 2.55e25  # molecules/m^2 per atmosphere
    H = 8500  # scale height in m
    return sigma * N / H  # per meter


def twilight_transmission(wavelength_um, sza_deg):
    """
    Calculate transmission during twilight.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    sza_deg : float
        Solar zenith angle in degrees

    Returns
    -------
    T : float
        Transmission (0 to 1)
    """
    # Rayleigh optical depth at zenith
    sigma = rayleigh_cross_section(wavelength_um)
    N_column = 2.55e25  # molecules/m^2
    tau_zenith = sigma * N_column

    # Effective air mass
    m = twilight_air_mass(sza_deg)

    # Total optical depth
    tau = tau_zenith * m

    return np.exp(-tau)


def ozone_chappuis_absorption(wavelength_um, sza_deg, ozone_du=300):
    """
    Calculate Chappuis band absorption (visible ozone).

    The Chappuis bands (440-740 nm) are responsible for the
    blue hour sky color during twilight.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    sza_deg : float
        Solar zenith angle in degrees
    ozone_du : float
        Total ozone column in Dobson Units

    Returns
    -------
    T : float
        Transmission due to Chappuis band
    """
    wl = wavelength_um

    # Chappuis band cross section (simplified Gaussian)
    # Peak around 600 nm
    sigma_max = 5e-25  # m^2 at peak
    center = 0.60  # um
    width = 0.10  # um

    sigma = sigma_max * np.exp(-0.5 * ((wl - center) / width)**2)

    # Ozone column (1 DU = 2.687e20 molecules/m^2)
    N_o3 = ozone_du * 2.687e20

    # Air mass
    m = twilight_air_mass(sza_deg)

    tau = sigma * N_o3 * m

    return np.exp(-tau)


def sky_color_during_twilight(sza_deg):
    """
    Describe the dominant sky color at different twilight stages.

    Parameters
    ----------
    sza_deg : float
        Solar zenith angle in degrees

    Returns
    -------
    description : str
        Description of sky color
    """
    if sza_deg < 85:
        return "Yellow/white near sun, blue overhead"
    elif sza_deg < 90:
        return "Orange/red near sun (Golden Hour)"
    elif sza_deg < 93:
        return "Deep orange/red horizon, purple-blue above"
    elif sza_deg < 96:
        return "Red horizon fading to purple (Belt of Venus)"
    elif sza_deg < 102:
        return "Blue hour - deep blue sky with pink/purple horizon"
    else:
        return "Dark blue to black, stars visible"


def main():
    args = parse_args()

    print("=" * 70)
    print("TWILIGHT AND SUNRISE/SUNSET SPECTRA")
    print("=" * 70)
    print(f"\nSolar zenith angle: {args.sza} deg")

    # Determine twilight phase
    if args.sza < 90:
        phase = "Daytime / Sunset"
    elif args.sza < 96:
        phase = "Civil Twilight"
    elif args.sza < 102:
        phase = "Nautical Twilight"
    elif args.sza < 108:
        phase = "Astronomical Twilight"
    else:
        phase = "Night"

    print(f"Twilight phase: {phase}")

    # Twilight phases explanation
    print("\n" + "-" * 70)
    print("TWILIGHT PHASES")
    print("-" * 70)
    print("""
Phase               SZA Range       Characteristics
-----               ---------       ---------------
Day                 < 90 deg        Sun above horizon
Civil twilight      90-96 deg       Outdoor activities possible
Nautical twilight   96-102 deg      Horizon barely visible
Astronomical        102-108 deg     Sky glow from sun
Night               > 108 deg       Full darkness
""")

    # Air mass during twilight
    print("\n" + "-" * 70)
    print("AIR MASS VS SOLAR ZENITH ANGLE")
    print("-" * 70)

    sza_values = [0, 30, 60, 80, 85, 88, 90, 92, 95, 100]
    print(f"{'SZA (deg)':>12} {'Air Mass':>12} {'Description':>25}")
    print("-" * 55)

    for sza in sza_values:
        m = twilight_air_mass(sza)
        desc = "Overhead" if sza == 0 else "Low sun" if sza < 80 else "Horizon" if sza < 92 else "Below horizon"
        print(f"{sza:>12} {m:>12.1f} {desc:>25}")

    # Spectral transmission
    print("\n" + "-" * 70)
    print(f"SPECTRAL TRANSMISSION AT SZA = {args.sza} deg")
    print("-" * 70)

    wavelengths = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
    colors = ['UV', 'Violet', 'Blue', 'Cyan', 'Green', 'Yellow', 'Orange', 'Red', 'NIR']

    print(f"\n{'Wavelength':>12} {'Color':>10} {'Rayleigh T':>12} {'Chappuis T':>12} {'Total T':>10}")
    print("-" * 65)

    for wl, color in zip(wavelengths, colors):
        T_ray = twilight_transmission(wl, args.sza)
        T_chap = ozone_chappuis_absorption(wl, args.sza)
        T_total = T_ray * T_chap

        print(f"{wl:>10.2f} um {color:>10} {T_ray:>11.1e} {T_chap:>11.2%} {T_total:>9.1e}")

    # Blue/Red ratio (reddening)
    print("\n" + "-" * 70)
    print("REDDENING DURING TWILIGHT")
    print("-" * 70)

    T_blue = twilight_transmission(0.45, args.sza)
    T_red = twilight_transmission(0.65, args.sza)
    reddening = T_red / T_blue if T_blue > 0 else float('inf')

    print(f"""
Reddening analysis at SZA = {args.sza} deg:

  Blue (450 nm) transmission: {T_blue:.2e}
  Red (650 nm) transmission: {T_red:.2e}
  Red/Blue ratio: {reddening:.1f}

For comparison:
  Overhead sun (SZA=0):   Red/Blue ~ 1.3
  Low sun (SZA=80):       Red/Blue ~ 5
  Horizon (SZA=90):       Red/Blue ~ 20
  Civil twilight (SZA=95): Red/Blue ~ 100+

This explains the orange/red colors at sunset!
""")

    # Sky color description
    print("\n" + "-" * 70)
    print("SKY COLOR PROGRESSION")
    print("-" * 70)

    sza_progression = [80, 85, 88, 90, 92, 95, 100, 105]
    print(f"{'SZA (deg)':>10} {'Sky Color':>45}")
    print("-" * 60)

    for sza in sza_progression:
        color = sky_color_during_twilight(sza)
        print(f"{sza:>10} {color:>45}")

    # Special phenomena
    print("\n" + "-" * 70)
    print("SPECIAL TWILIGHT PHENOMENA")
    print("-" * 70)
    print("""
1. GREEN FLASH
   - Brief green/blue flash as sun sets
   - Caused by atmospheric refraction and dispersion
   - Green light refracted more than red
   - Requires clear horizon (ocean view best)

2. BELT OF VENUS
   - Pink/purple band above Earth's shadow
   - Caused by backscattered reddened sunlight
   - Visible opposite the sun during twilight
   - Earth's shadow appears blue-gray below

3. PURPLE LIGHT
   - Purple/violet sky 15-20 min after sunset
   - Caused by mix of:
     - Red light from below horizon sun
     - Blue/violet scattered from upper atmosphere
   - Enhanced after volcanic eruptions

4. CREPUSCULAR RAYS
   - Rays of light through clouds
   - Made visible by scattering off particles
   - Appear to radiate from sun position
   - "God rays" or "Jacob's ladder"

5. ANTI-CREPUSCULAR RAYS
   - Same rays converging at antisolar point
   - Optical perspective effect
   - Best seen during clear twilight
""")

    # Chappuis band effect
    print("\n" + "-" * 70)
    print("CHAPPUIS BAND (OZONE) EFFECT")
    print("-" * 70)
    print("""
The Chappuis bands (440-740 nm) cause subtle absorption in the visible.
During twilight, the long path enhances this effect:

- Weak absorption centered at 600 nm (orange)
- Creates slight blue bias in twilight sky
- Contributes to "blue hour" color
- More pronounced at higher ozone columns

Effect at different SZA:
""")

    for sza in [60, 80, 90, 95]:
        T_green = ozone_chappuis_absorption(0.55, sza)
        T_orange = ozone_chappuis_absorption(0.60, sza)
        print(f"  SZA {sza} deg: Green T = {T_green:.1%}, Orange T = {T_orange:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Twilight analysis at SZA = {args.sza} deg ({phase}):

Atmospheric effects:
- Effective air mass: {twilight_air_mass(args.sza):.0f}
- Reddening (Red/Blue ratio): {reddening:.0f}
- Sky color: {sky_color_during_twilight(args.sza)}

Key physical processes:
1. Enhanced Rayleigh scattering preferentially removes blue light
2. Long path through atmosphere creates reddening
3. Ozone Chappuis bands add subtle blue absorption
4. Atmospheric refraction extends visible twilight
5. Multiple scattering creates complex color gradients

The beautiful colors of twilight result from the combination of
these processes acting over the very long atmospheric path.
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Twilight Spectral Effects', fontsize=14, fontweight='bold')

            # Plot 1: Transmission spectrum at different SZA
            ax1 = axes[0, 0]
            wl_range = np.linspace(0.35, 0.85, 100)

            for sza in [60, 80, 88, 92, 95]:
                T = [twilight_transmission(wl, sza) for wl in wl_range]
                ax1.semilogy(wl_range * 1000, T, linewidth=2, label=f'SZA={sza} deg')

            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Transmission')
            ax1.set_title('Spectral Transmission vs SZA')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(350, 850)

            # Plot 2: Air mass vs SZA
            ax2 = axes[0, 1]
            sza_range = np.linspace(0, 105, 100)
            air_mass = [twilight_air_mass(sza) for sza in sza_range]

            ax2.semilogy(sza_range, air_mass, 'b-', linewidth=2)
            ax2.axvline(90, color='red', linestyle='--', label='Horizon')
            ax2.axvline(args.sza, color='green', linestyle='--', label=f'Current ({args.sza} deg)')
            ax2.set_xlabel('Solar Zenith Angle (deg)')
            ax2.set_ylabel('Air Mass')
            ax2.set_title('Air Mass vs Solar Zenith Angle')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 105)

            # Plot 3: Red/Blue ratio (reddening)
            ax3 = axes[1, 0]
            reddening_arr = []
            for sza in sza_range:
                T_b = twilight_transmission(0.45, sza)
                T_r = twilight_transmission(0.65, sza)
                reddening_arr.append(T_r / T_b if T_b > 1e-20 else 1000)

            ax3.semilogy(sza_range, reddening_arr, 'r-', linewidth=2)
            ax3.axvline(90, color='gray', linestyle='--')
            ax3.axvline(args.sza, color='green', linestyle='--')
            ax3.set_xlabel('Solar Zenith Angle (deg)')
            ax3.set_ylabel('Red/Blue Ratio')
            ax3.set_title('Reddening During Twilight')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 105)
            ax3.set_ylim(1, 1000)

            # Plot 4: Simulated sunset colors
            ax4 = axes[1, 1]

            # Create color gradient representing sky colors
            n_colors = 100
            colors_array = []

            for i, sza in enumerate(np.linspace(80, 100, n_colors)):
                # Calculate relative intensities
                T_r = twilight_transmission(0.65, sza)
                T_g = twilight_transmission(0.55, sza)
                T_b = twilight_transmission(0.45, sza)

                # Normalize and clip
                max_T = max(T_r, T_g, T_b) + 1e-10
                r = min(1, T_r / max_T)
                g = min(1, T_g / max_T)
                b = min(1, T_b / max_T)

                colors_array.append([r, g, b])

            colors_array = np.array(colors_array)

            # Display as image
            ax4.imshow(colors_array.reshape(1, n_colors, 3), aspect='auto',
                      extent=[80, 100, 0, 1])
            ax4.set_xlabel('Solar Zenith Angle (deg)')
            ax4.set_yticks([])
            ax4.set_title('Simulated Sky Color vs SZA')

            # Add twilight phase labels
            ax4.axvline(90, color='white', linestyle='--', alpha=0.5)
            ax4.axvline(96, color='white', linestyle='--', alpha=0.5)
            ax4.text(85, 0.5, 'Sunset', ha='center', color='white', fontsize=10)
            ax4.text(93, 0.5, 'Civil', ha='center', color='white', fontsize=10)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
