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
- Atmospheric transmission in each band (with slant path geometry)
- Detector sensitivity (D*, NETD)
- SNR vs range curves
- Sensor and target altitude effects

Applications:
- Missile warning systems
- IRST (Infrared Search and Track)
- Air defense sensors
- Target acquisition systems

Usage:
    python 34_fpa_detection_comparison.py
    python 34_fpa_detection_comparison.py --afterburner
    python 34_fpa_detection_comparison.py --sensor-altitude 0 --target-altitude 10
    python 34_fpa_detection_comparison.py --altitude-scan
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
        calculate_detection_range_slant,
        scan_altitude_performance,
        elevation_angle,
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
        "--sensor-altitude", type=float, default=0.0,
        help="Sensor platform altitude in km (default: 0 = ground)"
    )
    parser.add_argument(
        "--target-altitude", type=float, default=10.0,
        help="Target aircraft altitude in km (default: 10)"
    )
    parser.add_argument(
        "--snr-threshold", type=float, default=5.0,
        help="SNR threshold for detection (default: 5)"
    )
    parser.add_argument(
        "--altitude-scan", action="store_true",
        help="Run altitude scan and generate comparison heatmap"
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

    print(f"\nGeometry:")
    print(f"  Sensor altitude: {args.sensor_altitude} km")
    print(f"  Target altitude: {args.target_altitude} km")
    elev = elevation_angle(args.sensor_altitude, args.target_altitude, 20.0)
    print(f"  Elevation angle (at 20km range): {elev:.1f} deg")

    print(f"\nAtmospheric conditions:")
    print(f"  Visibility: {args.visibility} km")
    print(f"  Humidity: {args.humidity}%")

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

    # Atmospheric transmission comparison (using slant path)
    print("\n" + "-" * 70)
    print("ATMOSPHERIC TRANSMISSION (Slant Path)")
    print("-" * 70)
    print(f"  Sensor at {args.sensor_altitude} km, Target at {args.target_altitude} km")

    from raf_tran.detection import atmospheric_transmission_slant

    ranges_test = [1, 5, 10, 20, 50]
    print(f"\n{'Slant Range':>12} {'MWIR (3-5um)':>15} {'LWIR (8-12um)':>15}")
    print("-" * 45)

    for r in ranges_test:
        trans_mwir = atmospheric_transmission_slant(
            args.sensor_altitude, args.target_altitude, r,
            3.0, 5.0, args.visibility, args.humidity
        )
        trans_lwir = atmospheric_transmission_slant(
            args.sensor_altitude, args.target_altitude, r,
            8.0, 12.0, args.visibility, args.humidity
        )
        print(f"{r:>12} {trans_mwir:>14.1%} {trans_lwir:>14.1%}")

    # Calculate detection ranges (with slant path geometry)
    print("\n" + "-" * 70)
    print("DETECTION RANGE CALCULATION (Slant Path)")
    print("-" * 70)

    result_insb = calculate_detection_range_slant(
        insb, fighter,
        args.sensor_altitude, args.target_altitude,
        args.snr_threshold, args.visibility, args.humidity,
    )

    result_mct = calculate_detection_range_slant(
        mct, fighter,
        args.sensor_altitude, args.target_altitude,
        args.snr_threshold, args.visibility, args.humidity,
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
    mean_alt = (args.sensor_altitude + args.target_altitude) / 2

    snr_insb, irrad_insb, trans_insb = calculate_snr_vs_range(
        insb, fighter, ranges,
        args.visibility, args.humidity, mean_alt,
    )

    snr_mct, irrad_mct, trans_mct = calculate_snr_vs_range(
        mct, fighter, ranges,
        args.visibility, args.humidity, mean_alt,
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
Geometry: Sensor at {args.sensor_altitude} km, Target at {args.target_altitude} km
Conditions: {args.visibility} km visibility, {args.humidity}% humidity

Detection Ranges (SNR > {args.snr_threshold}):
  InSb MWIR (3-5 um):  {result_insb.detection_range_km:>6.1f} km
  MCT LWIR (8-12 um):  {result_mct.detection_range_km:>6.1f} km

Winner: {winner} by {advantage:.1f}x
""")

    # Altitude scan analysis
    altitude_scan_data = None
    if args.altitude_scan:
        print("\n" + "=" * 70)
        print("ALTITUDE SCAN ANALYSIS")
        print("=" * 70)
        print("\nScanning detection range vs sensor/target altitude combinations...")

        sensor_alts = np.array([0, 2, 5, 8, 10, 12, 15])  # km
        target_alts = np.array([2, 5, 8, 10, 12, 15, 20])  # km

        print(f"\nSensor altitudes: {sensor_alts} km")
        print(f"Target altitudes: {target_alts} km")

        # Scan for MWIR
        print("\nScanning InSb MWIR...")
        ranges_mwir = scan_altitude_performance(
            insb, fighter, sensor_alts, target_alts,
            args.snr_threshold, args.visibility, args.humidity
        )

        # Scan for LWIR
        print("Scanning MCT LWIR...")
        ranges_lwir = scan_altitude_performance(
            mct, fighter, sensor_alts, target_alts,
            args.snr_threshold, args.visibility, args.humidity
        )

        # Print results table
        header = "Sensor\\Target"
        print("\n" + "-" * 70)
        print("MWIR Detection Range (km) vs Altitude")
        print("-" * 70)
        print(f"{header:>12}", end="")
        for t in target_alts:
            print(f"{t:>8.0f}", end="")
        print(" km")
        print("-" * (12 + 8 * len(target_alts)))
        for i, s in enumerate(sensor_alts):
            print(f"{s:>10.0f} km", end="")
            for j in range(len(target_alts)):
                print(f"{ranges_mwir[i, j]:>8.1f}", end="")
            print()

        print("\n" + "-" * 70)
        print("LWIR Detection Range (km) vs Altitude")
        print("-" * 70)
        print(f"{header:>12}", end="")
        for t in target_alts:
            print(f"{t:>8.0f}", end="")
        print(" km")
        print("-" * (12 + 8 * len(target_alts)))
        for i, s in enumerate(sensor_alts):
            print(f"{s:>10.0f} km", end="")
            for j in range(len(target_alts)):
                print(f"{ranges_lwir[i, j]:>8.1f}", end="")
            print()

        # Find best configurations
        best_mwir_idx = np.unravel_index(np.argmax(ranges_mwir), ranges_mwir.shape)
        best_lwir_idx = np.unravel_index(np.argmax(ranges_lwir), ranges_lwir.shape)

        print(f"\nBest MWIR: {ranges_mwir[best_mwir_idx]:.1f} km at "
              f"sensor={sensor_alts[best_mwir_idx[0]]:.0f} km, "
              f"target={target_alts[best_mwir_idx[1]]:.0f} km")
        print(f"Best LWIR: {ranges_lwir[best_lwir_idx]:.1f} km at "
              f"sensor={sensor_alts[best_lwir_idx[0]]:.0f} km, "
              f"target={target_alts[best_lwir_idx[1]]:.0f} km")

        altitude_scan_data = {
            'sensor_alts': sensor_alts,
            'target_alts': target_alts,
            'ranges_mwir': ranges_mwir,
            'ranges_lwir': ranges_lwir,
        }

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            if altitude_scan_data is not None:
                # Create 3x2 figure with altitude scan heatmaps
                fig = plt.figure(figsize=(16, 14))
                fig.suptitle(
                    f'FPA Detection Comparison: {fighter.name}\n'
                    f'Sensor={args.sensor_altitude}km, Target={args.target_altitude}km, '
                    f'Visibility={args.visibility}km',
                    fontsize=14, fontweight='bold'
                )

                # Plot 1: SNR vs Range
                ax1 = fig.add_subplot(3, 2, 1)
                ax1.semilogy(ranges, snr_insb, 'b-', linewidth=2, label='InSb MWIR')
                ax1.semilogy(ranges, snr_mct, 'r-', linewidth=2, label='MCT LWIR')
                ax1.axhline(y=args.snr_threshold, color='k', linestyle='--',
                           label=f'Threshold (SNR={args.snr_threshold})')
                ax1.set_xlabel('Range (km)')
                ax1.set_ylabel('SNR')
                ax1.set_title('SNR vs Range')
                ax1.legend(fontsize=9)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 50)

                # Plot 2: Detection Range Bar Chart
                ax2 = fig.add_subplot(3, 2, 2)
                detectors = ['InSb\nMWIR', 'MCT\nLWIR']
                det_ranges = [result_insb.detection_range_km, result_mct.detection_range_km]
                colors = ['blue', 'red']
                bars = ax2.bar(detectors, det_ranges, color=colors, alpha=0.7, edgecolor='black')
                ax2.set_ylabel('Detection Range (km)')
                ax2.set_title('Detection Range Comparison')
                ax2.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars, det_ranges):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

                # Plot 3: MWIR Altitude Heatmap
                ax3 = fig.add_subplot(3, 2, 3)
                sensor_alts = altitude_scan_data['sensor_alts']
                target_alts = altitude_scan_data['target_alts']
                ranges_mwir = altitude_scan_data['ranges_mwir']
                im3 = ax3.imshow(ranges_mwir, cmap='Blues', aspect='auto',
                                origin='lower', vmin=0)
                ax3.set_xticks(range(len(target_alts)))
                ax3.set_xticklabels([f'{t:.0f}' for t in target_alts])
                ax3.set_yticks(range(len(sensor_alts)))
                ax3.set_yticklabels([f'{s:.0f}' for s in sensor_alts])
                ax3.set_xlabel('Target Altitude (km)')
                ax3.set_ylabel('Sensor Altitude (km)')
                ax3.set_title('MWIR Detection Range (km)')
                cbar3 = plt.colorbar(im3, ax=ax3)
                cbar3.set_label('Range (km)')
                # Add text annotations
                for i in range(len(sensor_alts)):
                    for j in range(len(target_alts)):
                        ax3.text(j, i, f'{ranges_mwir[i, j]:.0f}',
                                ha='center', va='center', fontsize=8,
                                color='white' if ranges_mwir[i, j] > ranges_mwir.max()/2 else 'black')

                # Plot 4: LWIR Altitude Heatmap
                ax4 = fig.add_subplot(3, 2, 4)
                ranges_lwir = altitude_scan_data['ranges_lwir']
                im4 = ax4.imshow(ranges_lwir, cmap='Reds', aspect='auto',
                                origin='lower', vmin=0)
                ax4.set_xticks(range(len(target_alts)))
                ax4.set_xticklabels([f'{t:.0f}' for t in target_alts])
                ax4.set_yticks(range(len(sensor_alts)))
                ax4.set_yticklabels([f'{s:.0f}' for s in sensor_alts])
                ax4.set_xlabel('Target Altitude (km)')
                ax4.set_ylabel('Sensor Altitude (km)')
                ax4.set_title('LWIR Detection Range (km)')
                cbar4 = plt.colorbar(im4, ax=ax4)
                cbar4.set_label('Range (km)')
                for i in range(len(sensor_alts)):
                    for j in range(len(target_alts)):
                        ax4.text(j, i, f'{ranges_lwir[i, j]:.0f}',
                                ha='center', va='center', fontsize=8,
                                color='white' if ranges_lwir[i, j] > ranges_lwir.max()/2 else 'black')

                # Plot 5: MWIR vs LWIR Difference
                ax5 = fig.add_subplot(3, 2, 5)
                diff = ranges_mwir - ranges_lwir
                max_abs = max(abs(diff.min()), abs(diff.max()))
                im5 = ax5.imshow(diff, cmap='RdBu', aspect='auto',
                                origin='lower', vmin=-max_abs, vmax=max_abs)
                ax5.set_xticks(range(len(target_alts)))
                ax5.set_xticklabels([f'{t:.0f}' for t in target_alts])
                ax5.set_yticks(range(len(sensor_alts)))
                ax5.set_yticklabels([f'{s:.0f}' for s in sensor_alts])
                ax5.set_xlabel('Target Altitude (km)')
                ax5.set_ylabel('Sensor Altitude (km)')
                ax5.set_title('MWIR - LWIR Range (km)\nBlue=MWIR better, Red=LWIR better')
                cbar5 = plt.colorbar(im5, ax=ax5)
                cbar5.set_label('Difference (km)')

                # Plot 6: MWIR/LWIR Ratio
                ax6 = fig.add_subplot(3, 2, 6)
                ratio = ranges_mwir / np.maximum(ranges_lwir, 0.1)
                im6 = ax6.imshow(ratio, cmap='PRGn', aspect='auto',
                                origin='lower', vmin=0.5, vmax=2.0)
                ax6.set_xticks(range(len(target_alts)))
                ax6.set_xticklabels([f'{t:.0f}' for t in target_alts])
                ax6.set_yticks(range(len(sensor_alts)))
                ax6.set_yticklabels([f'{s:.0f}' for s in sensor_alts])
                ax6.set_xlabel('Target Altitude (km)')
                ax6.set_ylabel('Sensor Altitude (km)')
                ax6.set_title('MWIR/LWIR Range Ratio\n>1 = MWIR better')
                cbar6 = plt.colorbar(im6, ax=ax6)
                cbar6.set_label('Ratio')

                plt.tight_layout()
                plt.savefig(args.output, dpi=150, bbox_inches='tight')
                print(f"\nAltitude scan plot saved to: {args.output}")

            else:
                # Standard 2x2 plot without altitude scan
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(
                    f'FPA Detection Comparison: {fighter.name}\n'
                    f'Sensor={args.sensor_altitude}km, Target={args.target_altitude}km, '
                    f'Visibility={args.visibility}km',
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
                print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available for plotting")


if __name__ == "__main__":
    main()
