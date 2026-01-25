#!/usr/bin/env python3
"""
Atmospheric Optical Turbulence
==============================

This example demonstrates atmospheric turbulence effects on optical
beam propagation - critical for electro-optics applications.

We examine:
- Cn2 profiles (refractive index structure constant)
- Fried parameter (atmospheric coherence length)
- Scintillation index (intensity fluctuations)
- Beam wander and spreading

Applications:
- Free-space optical communications
- Laser radar (LIDAR)
- Astronomical adaptive optics
- Remote sensing

Usage:
    python 11_atmospheric_turbulence.py
    python 11_atmospheric_turbulence.py --wavelength 1.55
    python 11_atmospheric_turbulence.py --path-length 5000
    python 11_atmospheric_turbulence.py --help

Output:
    - Console: Turbulence analysis results
    - Graph: atmospheric_turbulence.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.turbulence import (
        hufnagel_valley_cn2, slc_day_cn2, slc_night_cn2,
        fried_parameter, isoplanatic_angle, scintillation_index,
        rytov_variance, beam_wander_variance, strehl_ratio,
        greenwood_frequency, coherence_time,
        kolmogorov_spectrum, von_karman_spectrum
    )
    from raf_tran.turbulence.cn2_profiles import integrated_cn2
except ImportError:
    print("Error: raf_tran turbulence module not found.")
    print("Please install raf_tran first: pip install -e . (from the project root)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze atmospheric optical turbulence for beam propagation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical scenarios:
  Horizontal path (1-10 km): Strong boundary layer turbulence
  Slant path to aircraft: Mixed turbulence
  Vertical astronomical: Integrated column through all layers

Common wavelengths:
  0.55 um  - Visible (green)
  1.064 um - Nd:YAG laser
  1.55 um  - Telecom, eye-safe
  10.6 um  - CO2 laser

Examples:
  %(prog)s                              # Default 1.55 um, 10 km
  %(prog)s --wavelength 0.55            # Visible wavelength
  %(prog)s --path-length 1000           # Short 1 km path
  %(prog)s --scenario astronomical      # Vertical path to space
        """
    )
    parser.add_argument(
        "--wavelength", type=float, default=1.55,
        help="Wavelength in micrometers (default: 1.55 um)"
    )
    parser.add_argument(
        "--path-length", type=float, default=10000,
        help="Path length in meters (default: 10000 m = 10 km)"
    )
    parser.add_argument(
        "--aperture", type=float, default=0.1,
        help="Receiver aperture diameter in meters (default: 0.1 m)"
    )
    parser.add_argument(
        "--beam-diameter", type=float, default=0.05,
        help="Transmitter beam diameter in meters (default: 0.05 m)"
    )
    parser.add_argument(
        "--scenario", type=str, default="horizontal",
        choices=["horizontal", "slant", "astronomical"],
        help="Propagation scenario (default: horizontal)"
    )
    parser.add_argument(
        "--time-of-day", type=str, default="day",
        choices=["day", "night"],
        help="Time of day for SLC model (default: day)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="atmospheric_turbulence.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    wavelength_m = args.wavelength * 1e-6
    path_length_m = args.path_length

    print("=" * 80)
    print("ATMOSPHERIC OPTICAL TURBULENCE ANALYSIS")
    print("=" * 80)
    print(f"\nWavelength: {args.wavelength} um")
    print(f"Path length: {args.path_length/1000:.1f} km")
    print(f"Scenario: {args.scenario}")
    print(f"Receiver aperture: {args.aperture*100:.0f} cm")
    print(f"Transmitter beam: {args.beam_diameter*100:.0f} cm")

    # Set up altitude profile based on scenario
    if args.scenario == "horizontal":
        # Horizontal path at 10 m altitude
        altitudes = np.ones(100) * 10  # meters
        zenith_angle = 90  # horizontal
        description = "Horizontal path at 10 m altitude"
    elif args.scenario == "slant":
        # Slant path from ground to 10 km
        altitudes = np.linspace(0, 10000, 100)
        zenith_angle = 45
        description = "45-degree slant path to 10 km"
    else:  # astronomical
        # Vertical path through entire atmosphere
        altitudes = np.linspace(0, 30000, 100)
        zenith_angle = 0
        description = "Vertical path through atmosphere"

    print(f"Description: {description}")

    # Compute Cn2 profile
    print("\n" + "-" * 80)
    print("Cn2 PROFILE COMPARISON")
    print("-" * 80)

    print(f"\n{'Altitude (km)':<15} {'HV 5/7':<12} {'SLC Day':<12} {'SLC Night':<12}")
    print("-" * 60)

    sample_altitudes = [0, 0.1, 0.5, 1, 2, 5, 10, 20]
    for h_km in sample_altitudes:
        h_m = h_km * 1000
        hv = hufnagel_valley_cn2(h_m)
        day = slc_day_cn2(h_m)
        night = slc_night_cn2(h_m)
        print(f"{h_km:<15.1f} {hv:<12.2e} {day:<12.2e} {night:<12.2e}")

    # Use appropriate model based on scenario and time
    if args.scenario == "horizontal":
        # For horizontal, use average Cn2 at path altitude
        if args.time_of_day == "day":
            cn2_avg = slc_day_cn2(10)  # At 10 m
        else:
            cn2_avg = slc_night_cn2(10)
        cn2_integrated = cn2_avg * path_length_m
    else:
        # For slant/vertical, integrate Cn2 profile
        if args.time_of_day == "day":
            cn2_profile = slc_day_cn2(altitudes)
        else:
            cn2_profile = slc_night_cn2(altitudes)
        cn2_integrated = integrated_cn2(altitudes, cn2_profile, zenith_angle)
        cn2_avg = cn2_integrated / path_length_m

    print(f"\nPath-averaged Cn2: {cn2_avg:.2e} m^(-2/3)")
    print(f"Integrated Cn2: {cn2_integrated:.2e} m^(1/3)")

    # Propagation parameters
    print("\n" + "-" * 80)
    print("BEAM PROPAGATION PARAMETERS")
    print("-" * 80)

    # Fried parameter
    r0 = fried_parameter(wavelength_m, cn2_integrated)
    print(f"\nFried parameter r0: {r0*100:.2f} cm")
    print(f"  (Atmospheric coherence diameter)")
    print(f"  At 500 nm: {r0 * (0.5/args.wavelength)**(6/5) * 100:.2f} cm")

    # Rytov variance
    sigma_r2 = rytov_variance(wavelength_m, cn2_avg, path_length_m)
    print(f"\nRytov variance sigma_R^2: {sigma_r2:.3f}")
    if sigma_r2 < 0.3:
        regime = "WEAK (Rytov theory valid)"
    elif sigma_r2 < 5:
        regime = "MODERATE (extended theory needed)"
    else:
        regime = "STRONG/SATURATED"
    print(f"  Fluctuation regime: {regime}")

    # Scintillation index
    si_point = scintillation_index(wavelength_m, cn2_avg, path_length_m)
    si_aperture = scintillation_index(wavelength_m, cn2_avg, path_length_m,
                                       aperture_diameter_m=args.aperture)
    print(f"\nScintillation index sigma_I^2:")
    print(f"  Point receiver: {si_point:.4f}")
    print(f"  {args.aperture*100:.0f} cm aperture: {si_aperture:.4f}")
    print(f"  Aperture averaging factor: {si_aperture/si_point:.3f}")

    # Beam wander
    sigma_bw2 = beam_wander_variance(wavelength_m, cn2_avg, path_length_m,
                                      args.beam_diameter)
    sigma_bw = np.sqrt(sigma_bw2)
    print(f"\nBeam wander RMS: {sigma_bw*1e6:.1f} um")
    print(f"  As fraction of beam: {sigma_bw/args.beam_diameter*100:.2f}%")

    # Coherence time
    wind_speed = 10.0  # m/s typical
    tau0 = coherence_time(wavelength_m, cn2_integrated, wind_speed)
    print(f"\nCoherence time tau_0: {tau0*1000:.2f} ms (at {wind_speed} m/s wind)")

    # Strehl ratio
    S = strehl_ratio(wavelength_m, r0=r0, aperture_diameter_m=args.aperture)
    print(f"\nStrehl ratio (D={args.aperture*100:.0f} cm): {S:.4f}")

    # System performance implications
    print("\n" + "-" * 80)
    print("SYSTEM PERFORMANCE IMPLICATIONS")
    print("-" * 80)

    # Bit error rate estimate for OOK
    if si_aperture > 0:
        # Simplified fade probability
        fade_margin_db = 3  # dB
        fade_prob = 0.5 * (1 - np.exp(-si_aperture / (2 * 10**(fade_margin_db/10))))
        print(f"\nFree-space optical comm (OOK modulation):")
        print(f"  Scintillation-induced BER floor: ~{fade_prob:.2e}")
        print(f"  Required fade margin: >{-10*np.log10(1-2*si_aperture):.1f} dB")

    # Imaging resolution
    diffraction_limit = wavelength_m / args.aperture  # radians
    turbulence_limit = wavelength_m / r0
    effective_resolution = max(diffraction_limit, turbulence_limit)
    print(f"\nImaging system ({args.aperture*100:.0f} cm aperture):")
    print(f"  Diffraction limit: {diffraction_limit*1e6:.2f} urad")
    print(f"  Turbulence limit: {turbulence_limit*1e6:.2f} urad")
    print(f"  Effective resolution: {effective_resolution*1e6:.2f} urad")
    if r0 < args.aperture:
        print(f"  -> TURBULENCE LIMITED (r0 < D)")
    else:
        print(f"  -> DIFFRACTION LIMITED (r0 > D)")

    # Adaptive optics requirements
    print(f"\nAdaptive optics requirements:")
    print(f"  Minimum actuators: ~(D/r0)^2 = {(args.aperture/r0)**2:.0f}")
    print(f"  Update rate: >{1/tau0:.0f} Hz")

    # Wavelength comparison
    print("\n" + "-" * 80)
    print("WAVELENGTH COMPARISON")
    print("-" * 80)

    wavelengths_um = [0.55, 1.064, 1.55, 3.8, 10.6]
    print(f"\n{'Wavelength (um)':<18} {'r0 (cm)':<12} {'sigma_I^2':<12} {'Strehl':<10}")
    print("-" * 60)

    for wl_um in wavelengths_um:
        wl_m = wl_um * 1e-6
        r0_wl = fried_parameter(wl_m, cn2_integrated)
        si_wl = scintillation_index(wl_m, cn2_avg, path_length_m,
                                     aperture_diameter_m=args.aperture)
        S_wl = strehl_ratio(wl_m, r0=r0_wl, aperture_diameter_m=args.aperture)
        print(f"{wl_um:<18.2f} {r0_wl*100:<12.2f} {si_wl:<12.4f} {S_wl:<10.4f}")

    print("""
Longer wavelengths -> larger r0 -> less turbulence effect
Scaling: r0 ~ wavelength^(6/5), sigma_I^2 ~ wavelength^(-7/6)
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Atmospheric Turbulence Analysis ({args.wavelength} um, {args.path_length/1000:.0f} km)',
                        fontsize=14, fontweight='bold')

            # Plot 1: Cn2 profiles
            ax1 = axes[0, 0]
            h_plot = np.logspace(0, 4.5, 100)  # 1 m to 30 km
            cn2_hv = hufnagel_valley_cn2(h_plot)
            cn2_day = slc_day_cn2(h_plot)
            cn2_night = slc_night_cn2(h_plot)

            ax1.loglog(cn2_hv, h_plot/1000, 'b-', linewidth=2, label='Hufnagel-Valley')
            ax1.loglog(cn2_day, h_plot/1000, 'r-', linewidth=2, label='SLC Day')
            ax1.loglog(cn2_night, h_plot/1000, 'g-', linewidth=2, label='SLC Night')
            ax1.set_xlabel('Cn2 (m^(-2/3))')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_title('Cn2 Vertical Profiles')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(1e-18, 1e-12)
            ax1.set_ylim(0.001, 30)

            # Plot 2: Scintillation vs path length
            ax2 = axes[0, 1]
            path_lengths = np.logspace(2, 5, 50)  # 100 m to 100 km
            si_vs_path = [scintillation_index(wavelength_m, cn2_avg, L) for L in path_lengths]

            ax2.loglog(path_lengths/1000, si_vs_path, 'b-', linewidth=2)
            ax2.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='Weak/moderate boundary')
            ax2.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Saturation')
            ax2.axvline(args.path_length/1000, color='green', linestyle=':', alpha=0.7,
                       label=f'Current: {args.path_length/1000:.0f} km')
            ax2.set_xlabel('Path Length (km)')
            ax2.set_ylabel('Scintillation Index')
            ax2.set_title('Scintillation vs Path Length')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Fried parameter vs wavelength (vectorized)
            ax3 = axes[1, 0]
            wavelengths_plot = np.linspace(0.4, 12, 100)
            r0_vs_wl = fried_parameter(wavelengths_plot * 1e-6, cn2_integrated) * 100

            ax3.plot(wavelengths_plot, r0_vs_wl, 'b-', linewidth=2)
            ax3.axhline(args.aperture*100, color='red', linestyle='--',
                       label=f'Aperture D = {args.aperture*100:.0f} cm')
            ax3.axvline(args.wavelength, color='green', linestyle=':',
                       label=f'Current: {args.wavelength} um')
            ax3.set_xlabel('Wavelength (um)')
            ax3.set_ylabel('Fried Parameter r0 (cm)')
            ax3.set_title('Fried Parameter vs Wavelength')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0.4, 12)

            # Mark regions
            ax3.fill_between([0.4, 0.7], 0, 200, alpha=0.1, color='blue', label='Visible')
            ax3.fill_between([0.7, 2.5], 0, 200, alpha=0.1, color='red', label='NIR')
            ax3.fill_between([3, 5], 0, 200, alpha=0.1, color='orange', label='MWIR')
            ax3.fill_between([8, 12], 0, 200, alpha=0.1, color='purple', label='LWIR')

            # Plot 4: Turbulence spectrum
            ax4 = axes[1, 1]
            kappa = np.logspace(-2, 4, 200)  # rad/m
            L0, l0 = 100, 0.001  # Outer/inner scale

            phi_k = kolmogorov_spectrum(kappa, cn2_avg)
            phi_vk = von_karman_spectrum(kappa, cn2_avg, L0=L0, l0=l0)

            ax4.loglog(kappa, phi_k, 'b--', linewidth=2, label='Kolmogorov', alpha=0.7)
            ax4.loglog(kappa, phi_vk, 'r-', linewidth=2, label=f'von Karman (L0={L0}m, l0={l0*1000}mm)')
            ax4.axvline(2*np.pi/L0, color='gray', linestyle=':', alpha=0.5)
            ax4.axvline(2*np.pi/l0, color='gray', linestyle=':', alpha=0.5)
            ax4.text(2*np.pi/L0, 1e-20, 'L0', ha='center')
            ax4.text(2*np.pi/l0, 1e-20, 'l0', ha='center')
            ax4.set_xlabel('Spatial Frequency kappa (rad/m)')
            ax4.set_ylabel('Power Spectrum Phi_n (m^3)')
            ax4.set_title('Turbulence Power Spectrum')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
ATMOSPHERIC OPTICAL TURBULENCE:

1. ORIGIN OF TURBULENCE:
   - Temperature fluctuations cause refractive index fluctuations
   - Cn2 = refractive index structure constant (m^(-2/3))
   - Stronger near surface (heating), weaker at altitude
   - Day: Convective turbulence (strong near ground)
   - Night: Mechanical turbulence (wind shear dominated)

2. KEY PARAMETERS:
   - Fried parameter r0: Coherence diameter (larger = less turbulence)
   - Rytov variance: Turbulence strength metric
   - Scintillation index: Intensity fluctuation variance
   - Isoplanatic angle: Angular coherence extent

3. EFFECTS ON OPTICAL SYSTEMS:
   - Beam spreading: Beyond diffraction limit
   - Beam wander: Random centroid motion
   - Scintillation: Intensity "twinkling"
   - Wavefront distortion: Phase aberrations

4. MITIGATION STRATEGIES:
   - Adaptive optics: Real-time wavefront correction
   - Aperture averaging: Large receivers reduce scintillation
   - Longer wavelengths: r0 ~ wavelength^(6/5)
   - Path selection: Avoid boundary layer when possible

5. FLUCTUATION REGIMES:
   - Weak (sigma_R^2 < 0.3): Rytov theory valid, log-normal stats
   - Moderate (0.3 < sigma_R^2 < 5): Extended theory needed
   - Strong (sigma_R^2 > 5): Saturation, intensity decorrelates
""")


if __name__ == "__main__":
    main()
