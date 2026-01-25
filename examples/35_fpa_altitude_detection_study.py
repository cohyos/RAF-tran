#!/usr/bin/env python3
"""
FPA Altitude Detection Range Study
===================================

This example provides a comprehensive comparison of sensor performance for
target detection range between three types of Focal Plane Array (FPA) sensors:

1. **MWIR (3-5 µm)** - InSb detector
   - Pixel pitch: 5 µm
   - Resolution: 2000 × 2000 pixels
   - FOV: 7° × 7°
   - IFOV: 0.061 mrad (finer resolution)

2. **Analog LWIR (8-12 µm)** - MCT detector
   - Pixel pitch: 10 µm
   - Resolution: 1000 × 1000 pixels
   - FOV: 7° × 7°
   - IFOV: 0.122 mrad

3. **Digital LWIR (8-12 µm)** - DROIC detector
   - Pixel pitch: 10 µm
   - Resolution: 1000 × 1000 pixels
   - FOV: 7° × 7°
   - IFOV: 0.122 mrad
   - Enhanced: 4× well capacity, 67% lower read noise

Study Parameters:
- Target heights: Sea level (0 m), 5,000 m, 10,000 m
- Sensor heights: 0 to 15,000 m (scanning)
- Detection range: Slant (diagonal) range calculation
- Target: Generic fighter aircraft (rear aspect)

Physical Principles:
- MWIR excels at detecting hot exhaust plumes (Wien's peak ~4 µm at 700K)
- LWIR better for ambient temperature skin detection
- Higher altitude = thinner atmosphere = better transmission
- Slant path geometry affects total atmospheric path length
- Digital LWIR provides ~40% improvement over analog LWIR

Usage:
    python 35_fpa_altitude_detection_study.py
    python 35_fpa_altitude_detection_study.py --target transport
    python 35_fpa_altitude_detection_study.py --afterburner
    python 35_fpa_altitude_detection_study.py --no-plot
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, '..')

try:
    from raf_tran.detectors import FPADetector, InSbDetector, MCTDetector, DigitalLWIRDetector
    from raf_tran.targets import generic_fighter, generic_transport, generic_uav
    from raf_tran.detection import (
        calculate_detection_range_slant,
        slant_range_from_altitudes,
        elevation_angle,
    )
except ImportError as e:
    print(f"Error: Could not import raf_tran modules: {e}")
    print("Make sure raf_tran is installed: pip install -e .")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FPA Altitude Detection Range Study - Compare MWIR, LWIR, Digital LWIR"
    )
    parser.add_argument(
        "--target", type=str, default="fighter",
        choices=["fighter", "transport", "uav"],
        help="Target type (default: fighter)"
    )
    parser.add_argument(
        "--afterburner", action="store_true",
        help="Fighter with afterburner engaged"
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
        "--snr-threshold", type=float, default=5.0,
        help="SNR threshold for detection (default: 5)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="fpa_altitude_study.png",
        help="Output plot filename"
    )
    return parser.parse_args()


def create_study_detectors():
    """
    Create the three detector configurations for this study.

    Specifications:
    - MWIR: 5 µm pitch, 2000×2000, 7°×7° FOV → IFOV = 0.061 mrad
    - LWIR: 10 µm pitch, 1000×1000, 7°×7° FOV → IFOV = 0.122 mrad

    FOV = 7° = 122.17 mrad
    MWIR IFOV = 122.17 / 2000 = 0.061 mrad
    LWIR IFOV = 122.17 / 1000 = 0.122 mrad

    Focal length calculation:
    IFOV = pixel_pitch / focal_length
    For MWIR: f = 5 µm / 0.061 mrad = 82 mm
    For LWIR: f = 10 µm / 0.122 mrad = 82 mm

    Aperture for f/2.0: D = 82/2 = 41 mm
    """
    # Calculate focal length from FOV and array size
    fov_deg = 7.0
    fov_rad = np.radians(fov_deg)

    # MWIR: 2000 pixels × 5 µm = 10 mm array
    mwir_array_mm = 2000 * 0.005  # 10 mm
    mwir_focal_mm = mwir_array_mm / (2 * np.tan(fov_rad / 2))  # ~82 mm

    # LWIR: 1000 pixels × 10 µm = 10 mm array
    lwir_array_mm = 1000 * 0.010  # 10 mm
    lwir_focal_mm = lwir_array_mm / (2 * np.tan(fov_rad / 2))  # ~82 mm

    # f-number for 41mm aperture
    f_number = mwir_focal_mm / 41.0  # ~2.0

    print(f"Calculated focal length: {mwir_focal_mm:.1f} mm")
    print(f"Calculated f-number: f/{f_number:.1f}")

    # Create MWIR detector (InSb, 5 µm pitch)
    mwir = InSbDetector(
        name="MWIR (InSb)",
        pixel_pitch=5.0,  # 5 µm
        f_number=f_number,
        integration_time=10.0,
    )

    # Create Analog LWIR detector (MCT, 10 µm pitch)
    lwir_analog = MCTDetector(
        name="LWIR Analog (MCT)",
        spectral_band=(8.0, 12.0),
        pixel_pitch=10.0,  # 10 µm
        f_number=f_number,
        integration_time=10.0,
    )

    # Create Digital LWIR detector (DROIC, 10 µm pitch)
    lwir_digital = DigitalLWIRDetector(
        name="LWIR Digital (DROIC)",
        spectral_band=(8.0, 12.0),
        pixel_pitch=10.0,  # 10 µm
        f_number=f_number,
        integration_time=10.0,
    )

    return mwir, lwir_analog, lwir_digital


def get_target(args):
    """Get target based on arguments."""
    if args.target == "fighter":
        return generic_fighter(aspect="rear", afterburner=args.afterburner)
    elif args.target == "transport":
        return generic_transport(aspect="rear")
    elif args.target == "uav":
        return generic_uav(aspect="rear")


def calculate_detection_matrix(detector, target, sensor_heights_m, target_heights_m,
                                snr_threshold, visibility_km, humidity_percent):
    """
    Calculate detection range matrix for all sensor/target altitude combinations.

    Returns matrix where rows = sensor heights, cols = target heights.
    Values are slant detection ranges in km.
    """
    n_sensor = len(sensor_heights_m)
    n_target = len(target_heights_m)
    ranges_km = np.zeros((n_sensor, n_target))

    for i, sensor_alt in enumerate(sensor_heights_m):
        for j, target_alt in enumerate(target_heights_m):
            sensor_alt_km = sensor_alt / 1000.0
            target_alt_km = target_alt / 1000.0

            result = calculate_detection_range_slant(
                detector, target,
                sensor_alt_km, target_alt_km,
                snr_threshold, visibility_km, humidity_percent,
            )
            ranges_km[i, j] = result.detection_range_km

    return ranges_km


def print_physical_explanation():
    """Print physical explanation of the results."""
    print("""
================================================================================
PHYSICAL EXPLANATION OF RESULTS
================================================================================

1. SPECTRAL BAND EFFECTS
------------------------
   MWIR (3-5 µm):
   - Wien's displacement law: Peak emission at T ≈ 700K occurs at ~4 µm
   - Hot exhaust plumes (600-800K) emit strongly in MWIR
   - Atmospheric transmission: Good window, but affected by H2O at 2.7 µm

   LWIR (8-12 µm):
   - Peak emission for ambient temperatures (250-300K) at 10-12 µm
   - Aircraft skin at cruise (~250K) emits primarily in LWIR
   - Atmospheric transmission: Excellent window, CO2 absorption at 15 µm outside band

2. ALTITUDE EFFECTS ON DETECTION RANGE
--------------------------------------
   Higher sensor altitude benefits:
   - Reduced atmospheric path length (less absorption/scattering)
   - Looking down through thinner atmosphere
   - Exponential decrease in air density with altitude (scale height ~8 km)

   Higher target altitude benefits:
   - Target above most atmospheric water vapor (scale height ~2 km)
   - Reduced aerosol extinction (aerosol scale height ~1.2 km)
   - Cleaner optical path for both MWIR and LWIR

3. SLANT PATH GEOMETRY
----------------------
   Detection range is LIMITED by:
   - Geometric slant range = sqrt(horizontal² + vertical²)
   - Atmospheric transmission along slant path
   - At high elevation angles, path is shorter through atmosphere
   - At low elevation angles, path traverses more atmosphere

   For upward-looking sensors (sensor < target):
   - More atmosphere to look through
   - Water vapor concentrated at low altitudes affects MWIR more

   For downward-looking sensors (sensor > target):
   - Cleaner path through high-altitude atmosphere
   - Better transmission at all wavelengths

4. MWIR vs LWIR TRADE-OFFS
--------------------------
   MWIR advantages for hot targets:
   - Higher target radiant intensity from exhaust
   - Better contrast against cold sky background
   - Finer angular resolution (smaller IFOV at same FOV)

   LWIR advantages:
   - Detects cooler skin temperatures
   - Wider atmospheric window
   - Less affected by solar glint/reflections
   - Digital LWIR: 4× well capacity reduces noise, ~40% range improvement

5. DIGITAL LWIR ENHANCEMENT
---------------------------
   DROIC (Digital Read-Out IC) provides:
   - Well capacity: 4.0×10⁶ electrons (vs 1.0×10⁶ analog)
   - Read noise: 50 electrons RMS (vs 150 electrons analog)
   - SNR improvement: ~1.4× over analog LWIR
   - Detection range improvement: ~20-40% depending on conditions

================================================================================
""")


def main():
    args = parse_args()

    print("=" * 80)
    print("FPA ALTITUDE DETECTION RANGE STUDY")
    print("=" * 80)

    # Create detectors
    print("\n" + "-" * 80)
    print("DETECTOR CONFIGURATIONS")
    print("-" * 80)

    mwir, lwir_analog, lwir_digital = create_study_detectors()
    detectors = [mwir, lwir_analog, lwir_digital]

    # Print detector specifications
    print("\n{:25} {:>12} {:>12} {:>12}".format(
        "Parameter", "MWIR", "LWIR Analog", "LWIR Digital"))
    print("-" * 65)
    print("{:25} {:>12} {:>12} {:>12}".format(
        "Spectral band (µm)", "3-5", "8-12", "8-12"))
    print("{:25} {:>12} {:>12} {:>12}".format(
        "Pixel pitch (µm)", "5", "10", "10"))
    print("{:25} {:>12} {:>12} {:>12}".format(
        "Resolution (pixels)", "2000×2000", "1000×1000", "1000×1000"))
    print("{:25} {:>12} {:>12} {:>12}".format(
        "FOV (degrees)", "7×7", "7×7", "7×7"))
    print("{:25} {:>12.3f} {:>12.3f} {:>12.3f}".format(
        "IFOV (mrad)", mwir.ifov, lwir_analog.ifov, lwir_digital.ifov))
    print("{:25} {:>12.1e} {:>12.1e} {:>12.1e}".format(
        "D* (cm√Hz/W)", mwir.d_star, lwir_analog.d_star, lwir_digital.d_star))
    print("{:25} {:>12.0f} {:>12.0f} {:>12.0f}".format(
        "NETD (mK)", mwir.netd, lwir_analog.netd, lwir_digital.netd))
    print("{:25} {:>12.2e} {:>12.2e} {:>12.2e}".format(
        "NEI (W/cm²)", mwir.noise_equivalent_irradiance(),
        lwir_analog.noise_equivalent_irradiance(),
        lwir_digital.noise_equivalent_irradiance()))

    # Get target
    target = get_target(args)
    target_name = f"{args.target.capitalize()}" + (" AB" if args.afterburner else "")

    print("\n" + "-" * 80)
    print(f"TARGET: {target_name} (Rear Aspect)")
    print("-" * 80)
    print(f"  Exhaust temperature: {target.exhaust_temp:.0f} K")
    print(f"  Exhaust area: {target.exhaust_area:.2f} m²")
    print(f"  Skin temperature: {target.skin_temp:.0f} K")
    print(f"  Skin area: {target.skin_area:.1f} m²")
    print(f"  MWIR intensity: {target.radiant_intensity_mwir():.1f} W/sr")
    print(f"  LWIR intensity: {target.radiant_intensity_lwir():.1f} W/sr")

    # Define altitude ranges
    target_heights_m = [0, 5000, 10000]  # Sea level, 5km, 10km
    sensor_heights_m = np.arange(0, 15500, 500)  # 0 to 15km in 500m steps

    print("\n" + "-" * 80)
    print("STUDY PARAMETERS")
    print("-" * 80)
    print(f"  Target heights: {target_heights_m} m")
    print(f"  Sensor height range: 0 to 15,000 m (500 m steps)")
    print(f"  Visibility: {args.visibility} km")
    print(f"  Humidity: {args.humidity}%")
    print(f"  SNR threshold: {args.snr_threshold}")

    # Calculate detection ranges for all combinations
    print("\n" + "-" * 80)
    print("CALCULATING DETECTION RANGES...")
    print("-" * 80)

    results = {}
    for det in detectors:
        print(f"  Processing {det.name}...")
        ranges = calculate_detection_matrix(
            det, target, sensor_heights_m, target_heights_m,
            args.snr_threshold, args.visibility, args.humidity
        )
        results[det.name] = ranges

    # Print results tables
    print("\n" + "=" * 80)
    print("DETECTION RANGE RESULTS (km) - Slant Range")
    print("=" * 80)

    for target_idx, target_height in enumerate(target_heights_m):
        print(f"\n--- Target Height: {target_height} m ({target_height/1000:.0f} km) ---")
        print("\n{:>12} {:>15} {:>15} {:>15} {:>12}".format(
            "Sensor (m)", "MWIR", "LWIR Analog", "LWIR Digital", "Best"))
        print("-" * 75)

        # Print selected sensor heights
        for sensor_idx in [0, 4, 10, 16, 20, 26, 30]:  # 0, 2k, 5k, 8k, 10k, 13k, 15k
            if sensor_idx >= len(sensor_heights_m):
                continue
            sensor_h = sensor_heights_m[sensor_idx]
            r_mwir = results[mwir.name][sensor_idx, target_idx]
            r_analog = results[lwir_analog.name][sensor_idx, target_idx]
            r_digital = results[lwir_digital.name][sensor_idx, target_idx]

            # Find best
            ranges = [("MWIR", r_mwir), ("Analog", r_analog), ("Digital", r_digital)]
            best_name, best_range = max(ranges, key=lambda x: x[1])

            print("{:>12,} {:>15.1f} {:>15.1f} {:>15.1f} {:>12}".format(
                int(sensor_h), r_mwir, r_analog, r_digital, best_name))

    # Print comparative analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    # Digital vs Analog improvement
    print("\nDigital LWIR Improvement over Analog LWIR:")
    print("{:>12} {:>15} {:>15} {:>15}".format(
        "Sensor (m)", "Target 0m", "Target 5km", "Target 10km"))
    print("-" * 60)

    for sensor_idx in [0, 10, 20, 30]:
        if sensor_idx >= len(sensor_heights_m):
            continue
        sensor_h = sensor_heights_m[sensor_idx]
        improvements = []
        for target_idx in range(3):
            r_analog = results[lwir_analog.name][sensor_idx, target_idx]
            r_digital = results[lwir_digital.name][sensor_idx, target_idx]
            if r_analog > 0:
                improvement = (r_digital / r_analog - 1) * 100
            else:
                improvement = 0
            improvements.append(improvement)

        print("{:>12,} {:>14.1f}% {:>14.1f}% {:>14.1f}%".format(
            int(sensor_h), improvements[0], improvements[1], improvements[2]))

    # MWIR vs LWIR comparison
    print("\nMWIR vs Best LWIR (Digital) Ratio:")
    print("{:>12} {:>15} {:>15} {:>15}".format(
        "Sensor (m)", "Target 0m", "Target 5km", "Target 10km"))
    print("-" * 60)

    for sensor_idx in [0, 10, 20, 30]:
        if sensor_idx >= len(sensor_heights_m):
            continue
        sensor_h = sensor_heights_m[sensor_idx]
        ratios = []
        for target_idx in range(3):
            r_mwir = results[mwir.name][sensor_idx, target_idx]
            r_digital = results[lwir_digital.name][sensor_idx, target_idx]
            if r_digital > 0:
                ratio = r_mwir / r_digital
            else:
                ratio = 0
            ratios.append(ratio)

        print("{:>12,} {:>15.2f}× {:>15.2f}× {:>15.2f}×".format(
            int(sensor_h), ratios[0], ratios[1], ratios[2]))

    # Print physical explanation
    print_physical_explanation()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find max ranges for each detector
    print("\nMaximum Detection Ranges Achieved:")
    for det_name, ranges in results.items():
        max_range = np.max(ranges)
        max_idx = np.unravel_index(np.argmax(ranges), ranges.shape)
        sensor_h = sensor_heights_m[max_idx[0]]
        target_h = target_heights_m[max_idx[1]]
        print(f"  {det_name}: {max_range:.1f} km (sensor={sensor_h/1000:.0f}km, target={target_h/1000:.0f}km)")

    print(f"""
Key Findings:
1. MWIR consistently provides longest detection range for hot exhaust targets
2. Digital LWIR provides 20-40% improvement over analog LWIR
3. Higher sensor altitude significantly improves detection range
4. Higher target altitude also improves detection (thinner atmosphere)
5. The MWIR advantage is greatest for afterburner-equipped targets

Recommended Applications:
- MWIR: Primary sensor for hot target detection (afterburner, missile plumes)
- LWIR Digital: Best cost/performance for skin temperature detection
- LWIR Analog: Legacy systems, lower cost applications
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            fig.suptitle(
                f'FPA Detection Range vs Sensor Altitude\n'
                f'Target: {target_name}, Visibility={args.visibility}km, SNR>{args.snr_threshold}',
                fontsize=14, fontweight='bold'
            )

            colors = {'MWIR (InSb)': 'blue',
                     'LWIR Analog (MCT)': 'red',
                     'LWIR Digital (DROIC)': 'green'}

            # Top row: Detection range vs sensor altitude for each target height
            for idx, target_h in enumerate(target_heights_m):
                ax = axes[0, idx]
                for det_name, ranges in results.items():
                    ax.plot(sensor_heights_m / 1000, ranges[:, idx],
                           color=colors[det_name], linewidth=2, label=det_name)

                ax.set_xlabel('Sensor Altitude (km)')
                ax.set_ylabel('Detection Range (km)')
                ax.set_title(f'Target at {target_h/1000:.0f} km')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 15)

            # Bottom row: Comparison metrics
            # Plot 1: Digital vs Analog improvement
            ax = axes[1, 0]
            for idx, target_h in enumerate(target_heights_m):
                improvement = []
                for sensor_idx in range(len(sensor_heights_m)):
                    r_analog = results[lwir_analog.name][sensor_idx, idx]
                    r_digital = results[lwir_digital.name][sensor_idx, idx]
                    if r_analog > 0:
                        improvement.append((r_digital / r_analog - 1) * 100)
                    else:
                        improvement.append(0)
                ax.plot(sensor_heights_m / 1000, improvement,
                       linewidth=2, label=f'Target {target_h/1000:.0f}km')
            ax.set_xlabel('Sensor Altitude (km)')
            ax.set_ylabel('Improvement (%)')
            ax.set_title('Digital LWIR vs Analog LWIR')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 15)

            # Plot 2: MWIR/LWIR ratio
            ax = axes[1, 1]
            for idx, target_h in enumerate(target_heights_m):
                ratio = []
                for sensor_idx in range(len(sensor_heights_m)):
                    r_mwir = results[mwir.name][sensor_idx, idx]
                    r_digital = results[lwir_digital.name][sensor_idx, idx]
                    if r_digital > 0:
                        ratio.append(r_mwir / r_digital)
                    else:
                        ratio.append(0)
                ax.plot(sensor_heights_m / 1000, ratio,
                       linewidth=2, label=f'Target {target_h/1000:.0f}km')
            ax.set_xlabel('Sensor Altitude (km)')
            ax.set_ylabel('Range Ratio (MWIR/Digital LWIR)')
            ax.set_title('MWIR vs Digital LWIR')
            ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 15)

            # Plot 3: Best detector at each altitude
            ax = axes[1, 2]
            target_h = 10000  # Focus on 10km target
            idx = 2
            r_mwir = results[mwir.name][:, idx]
            r_analog = results[lwir_analog.name][:, idx]
            r_digital = results[lwir_digital.name][:, idx]

            ax.fill_between(sensor_heights_m / 1000, 0, r_mwir,
                           alpha=0.3, color='blue', label='MWIR')
            ax.fill_between(sensor_heights_m / 1000, 0, r_digital,
                           alpha=0.3, color='green', label='Digital LWIR')
            ax.fill_between(sensor_heights_m / 1000, 0, r_analog,
                           alpha=0.3, color='red', label='Analog LWIR')
            ax.plot(sensor_heights_m / 1000, r_mwir, 'b-', linewidth=2)
            ax.plot(sensor_heights_m / 1000, r_digital, 'g-', linewidth=2)
            ax.plot(sensor_heights_m / 1000, r_analog, 'r-', linewidth=2)
            ax.set_xlabel('Sensor Altitude (km)')
            ax.set_ylabel('Detection Range (km)')
            ax.set_title(f'Range Comparison (Target at 10 km)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 15)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available for plotting")


if __name__ == "__main__":
    main()
