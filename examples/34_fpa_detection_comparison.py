#!/usr/bin/env python3
"""
FPA Detection Range Comparison
==============================

This example compares the detection range of fighter aircraft using
two different Focal Plane Array (FPA) sensor technologies:

1. InSb (Indium Antimonide) - MWIR (3-5 um)
   - High sensitivity in mid-wave infrared
   - Excellent for hot target detection (exhaust plumes)
   - Requires cooling to 77K

2. MCT (HgCdTe) - LWIR (8-12 um)
   - Peak sensitivity at ambient temperatures
   - Better for detecting cooler skin temperatures
   - Also requires cryogenic cooling

The comparison considers:
- Target signature (fighter aircraft with/without afterburner)
- Atmospheric transmission in each band
- Detector sensitivity (D*, NETD)
- SNR vs range curves

Applications:
- Missile warning systems
- IRST (Infrared Search and Track)
- Air defense sensors
- Target acquisition systems

Usage:
    python 34_fpa_detection_comparison.py
    python 34_fpa_detection_comparison.py --afterburner
    python 34_fpa_detection_comparison.py --visibility 10 --humidity 80
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.detectors import InSbDetector, MCTDetector
    from raf_tran.targets import generic_fighter
    from raf_tran.detection import (
        calculate_detection_range,
        calculate_snr_vs_range,
        atmospheric_transmission_ir,
    )
except ImportError as e:
    print(f"Error: Could not import raf_tran modules: {e}")
    print("Make sure raf_tran is installed: pip install -e .")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FPA detection ranges for fighter aircraft"
    )
    parser.add_argument(
        "--afterburner", action="store_true",
        help="Fighter with afterburner engaged"
    )
    parser.add_argument(
        "--aspect", type=str, default="rear",
        choices=["rear", "side", "front"],
        help="Viewing aspect (default: rear)"
    )
    parser.add_argument(
        "--visibility", type=float, default=23.0,
        help="Meteorological visibility in km (default: 23)"
    )
    parser.add_argument(
        "--humidity", type=float, default=50.0,
        help="Relative humidity in percent (default: 50)"
    )
    parser.add_argument(
        "--altitude", type=float, default=5.0,
        help="Mean path altitude in km (default: 5)"
    )
    parser.add_argument(
        "--snr-threshold", type=float, default=5.0,
        help="SNR threshold for detection (default: 5)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="fpa_detection.png",
        help="Output plot filename"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("FPA DETECTION RANGE COMPARISON")
    print("=" * 70)

    # Create detectors
    insb = InSbDetector(
        name="InSb MWIR",
        pixel_pitch=15.0,
        f_number=2.0,
        integration_time=10.0,
    )

    mct = MCTDetector(
        name="MCT LWIR",
        spectral_band=(8.0, 12.0),
        pixel_pitch=20.0,
        f_number=2.0,
        integration_time=10.0,
    )

    # Create target
    fighter = generic_fighter(
        aspect=args.aspect,
        afterburner=args.afterburner,
        mach=0.9,
    )

    # Print configuration
    print("\n" + "-" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    print(f"\nTarget: {fighter.name}")
    print(f"  Aspect: {args.aspect}")
    print(f"  Afterburner: {'Yes' if args.afterburner else 'No'}")
    print(f"  Exhaust temp: {fighter.exhaust_temp:.0f} K")
    print(f"  Exhaust area: {fighter.exhaust_area:.2f} m^2")
    print(f"  Skin temp: {fighter.skin_temp:.0f} K")
    print(f"  Skin area: {fighter.skin_area:.1f} m^2")

    print(f"\nAtmospheric conditions:")
    print(f"  Visibility: {args.visibility} km")
    print(f"  Humidity: {args.humidity}%")
    print(f"  Altitude: {args.altitude} km")

    print(f"\nDetection threshold: SNR = {args.snr_threshold}")

    # Print detector specs
    print("\n" + "-" * 70)
    print("DETECTOR SPECIFICATIONS")
    print("-" * 70)

    for det in [insb, mct]:
        print(f"\n{det.name}:")
        print(f"  Spectral band: {det.spectral_band[0]}-{det.spectral_band[1]} um")
        print(f"  D*: {det.d_star:.1e} cm*sqrt(Hz)/W")
        print(f"  NETD: {det.netd} mK")
        print(f"  Pixel pitch: {det.pixel_pitch} um")
        print(f"  f-number: f/{det.f_number}")
        print(f"  Integration time: {det.integration_time} ms")
        print(f"  NEI: {det.noise_equivalent_irradiance():.2e} W/cm^2")
        print(f"  IFOV: {det.ifov:.2f} mrad")

    # Calculate target signatures
    print("\n" + "-" * 70)
    print("TARGET SIGNATURE")
    print("-" * 70)

    mwir_intensity = fighter.radiant_intensity_mwir()
    lwir_intensity = fighter.radiant_intensity_lwir()

    print(f"\nRadiant intensity:")
    print(f"  MWIR (3-5 um): {mwir_intensity:.1f} W/sr")
    print(f"  LWIR (8-12 um): {lwir_intensity:.1f} W/sr")
    print(f"  MWIR/LWIR ratio: {mwir_intensity/lwir_intensity:.2f}")

    # Atmospheric transmission comparison
    print("\n" + "-" * 70)
    print("ATMOSPHERIC TRANSMISSION")
    print("-" * 70)

    ranges_test = [1, 5, 10, 20, 50]
    print(f"\n{'Range (km)':>12} {'MWIR (3-5um)':>15} {'LWIR (8-12um)':>15}")
    print("-" * 45)

    for r in ranges_test:
        trans_mwir = atmospheric_transmission_ir(
            r, 3.0, 5.0, args.visibility, args.humidity, args.altitude
        )
        trans_lwir = atmospheric_transmission_ir(
            r, 8.0, 12.0, args.visibility, args.humidity, args.altitude
        )
        print(f"{r:>12} {trans_mwir:>14.1%} {trans_lwir:>14.1%}")

    # Calculate detection ranges
    print("\n" + "-" * 70)
    print("DETECTION RANGE CALCULATION")
    print("-" * 70)

    result_insb = calculate_detection_range(
        insb, fighter, args.snr_threshold,
        args.visibility, args.humidity, args.altitude,
    )

    result_mct = calculate_detection_range(
        mct, fighter, args.snr_threshold,
        args.visibility, args.humidity, args.altitude,
    )

    print(f"\nInSb MWIR:")
    print(f"  Detection range: {result_insb.detection_range_km:.1f} km")
    print(f"  Atmospheric transmission: {result_insb.atmospheric_transmission:.1%}")
    print(f"  Target irradiance: {result_insb.target_irradiance:.2e} W/cm^2")

    print(f"\nMCT LWIR:")
    print(f"  Detection range: {result_mct.detection_range_km:.1f} km")
    print(f"  Atmospheric transmission: {result_mct.atmospheric_transmission:.1%}")
    print(f"  Target irradiance: {result_mct.target_irradiance:.2e} W/cm^2")

    # Determine winner
    if result_insb.detection_range_km > result_mct.detection_range_km:
        winner = "InSb MWIR"
        advantage = result_insb.detection_range_km / max(result_mct.detection_range_km, 0.1)
    else:
        winner = "MCT LWIR"
        advantage = result_mct.detection_range_km / max(result_insb.detection_range_km, 0.1)

    print(f"\n*** {winner} provides {advantage:.1f}x longer detection range ***")

    # SNR vs range analysis
    print("\n" + "-" * 70)
    print("SNR vs RANGE ANALYSIS")
    print("-" * 70)

    ranges = np.linspace(0.5, 50, 100)

    snr_insb, irrad_insb, trans_insb = calculate_snr_vs_range(
        insb, fighter, ranges,
        args.visibility, args.humidity, args.altitude,
    )

    snr_mct, irrad_mct, trans_mct = calculate_snr_vs_range(
        mct, fighter, ranges,
        args.visibility, args.humidity, args.altitude,
    )

    print(f"\n{'Range (km)':>12} {'SNR MWIR':>12} {'SNR LWIR':>12} {'Better':>12}")
    print("-" * 50)

    for r in [1, 2, 5, 10, 20, 30, 40, 50]:
        idx = np.argmin(np.abs(ranges - r))
        s_mwir = snr_insb[idx]
        s_lwir = snr_mct[idx]
        better = "MWIR" if s_mwir > s_lwir else "LWIR"
        print(f"{r:>12} {s_mwir:>12.1f} {s_lwir:>12.1f} {better:>12}")

    # Physical interpretation
    print("\n" + "-" * 70)
    print("PHYSICAL INTERPRETATION")
    print("-" * 70)
    print("""
MWIR (3-5 um) Advantages:
- Hot exhaust plumes emit strongly in MWIR (Wien's law peak ~4 um at 700K)
- Better penetration through smoke and some aerosols
- Lower detector cooling requirements than LWIR

LWIR (8-12 um) Advantages:
- Stronger emission from ambient temperature surfaces (skin)
- Less affected by solar glint
- Better performance against low-signature targets
- Wider atmospheric window

For This Scenario:""")

    if args.afterburner:
        print("  - Afterburner greatly increases exhaust temperature (1800K)")
        print("  - MWIR strongly favored due to hot plume emission")
        print("  - Peak blackbody emission shifts toward MWIR")
    else:
        print("  - Military power exhaust at ~700K")
        print("  - Both bands can detect, MWIR slightly favored for hot exhaust")
        print("  - LWIR better for cooler skin signature")

    if args.aspect == "front":
        print("  - Front aspect minimizes exhaust visibility")
        print("  - LWIR may perform better (skin dominates)")
    elif args.aspect == "rear":
        print("  - Rear aspect maximizes exhaust visibility")
        print("  - MWIR strongly favored")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Target: {fighter.name}
Conditions: {args.visibility} km visibility, {args.humidity}% humidity

Detection Ranges (SNR > {args.snr_threshold}):
  InSb MWIR (3-5 um):  {result_insb.detection_range_km:>6.1f} km
  MCT LWIR (8-12 um):  {result_mct.detection_range_km:>6.1f} km

Winner: {winner} by {advantage:.1f}x
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                f'FPA Detection Comparison: {fighter.name}\n'
                f'Visibility={args.visibility}km, Humidity={args.humidity}%',
                fontsize=14, fontweight='bold'
            )

            # Plot 1: SNR vs Range
            ax1 = axes[0, 0]
            ax1.semilogy(ranges, snr_insb, 'b-', linewidth=2, label='InSb MWIR (3-5 um)')
            ax1.semilogy(ranges, snr_mct, 'r-', linewidth=2, label='MCT LWIR (8-12 um)')
            ax1.axhline(y=args.snr_threshold, color='k', linestyle='--',
                       label=f'Detection threshold (SNR={args.snr_threshold})')
            ax1.axvline(x=result_insb.detection_range_km, color='b', linestyle=':',
                       alpha=0.7)
            ax1.axvline(x=result_mct.detection_range_km, color='r', linestyle=':',
                       alpha=0.7)
            ax1.set_xlabel('Range (km)')
            ax1.set_ylabel('Signal-to-Noise Ratio')
            ax1.set_title('SNR vs Range')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 50)
            ax1.set_ylim(0.1, 1000)

            # Plot 2: Atmospheric Transmission
            ax2 = axes[0, 1]
            ax2.plot(ranges, trans_insb * 100, 'b-', linewidth=2,
                    label='MWIR (3-5 um)')
            ax2.plot(ranges, trans_mct * 100, 'r-', linewidth=2,
                    label='LWIR (8-12 um)')
            ax2.set_xlabel('Range (km)')
            ax2.set_ylabel('Transmission (%)')
            ax2.set_title('Atmospheric Transmission')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 50)
            ax2.set_ylim(0, 100)

            # Plot 3: Target Irradiance vs Range
            ax3 = axes[1, 0]
            ax3.semilogy(ranges, irrad_insb, 'b-', linewidth=2,
                        label='MWIR irradiance')
            ax3.semilogy(ranges, irrad_mct, 'r-', linewidth=2,
                        label='LWIR irradiance')
            ax3.axhline(y=insb.noise_equivalent_irradiance(), color='b',
                       linestyle='--', alpha=0.7, label='MWIR NEI')
            ax3.axhline(y=mct.noise_equivalent_irradiance(), color='r',
                       linestyle='--', alpha=0.7, label='LWIR NEI')
            ax3.set_xlabel('Range (km)')
            ax3.set_ylabel('Irradiance (W/cm^2)')
            ax3.set_title('Target Irradiance at Detector')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 50)

            # Plot 4: Detection Range Comparison Bar Chart
            ax4 = axes[1, 1]
            detectors = ['InSb\nMWIR', 'MCT\nLWIR']
            det_ranges = [result_insb.detection_range_km, result_mct.detection_range_km]
            colors = ['blue', 'red']
            bars = ax4.bar(detectors, det_ranges, color=colors, alpha=0.7, edgecolor='black')
            ax4.set_ylabel('Detection Range (km)')
            ax4.set_title(f'Detection Range Comparison (SNR > {args.snr_threshold})')
            ax4.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, val in zip(bars, det_ranges):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f} km', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available for plotting")


if __name__ == "__main__":
    main()
