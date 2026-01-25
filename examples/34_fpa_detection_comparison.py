#!/usr/bin/env python3
"""
FPA Detection Range Comparison
==============================

This example compares the detection range of aircraft using different
Focal Plane Array (FPA) sensor technologies with Johnson criteria analysis.

Sensor Technologies:
1. InSb (Indium Antimonide) - MWIR (3-5 um)
   - High sensitivity in mid-wave infrared
   - Excellent for hot target detection (exhaust plumes)
   - Requires cooling to 77K

2. MCT (HgCdTe) - LWIR (8-12 um) Analog
   - Peak sensitivity at ambient temperatures
   - Better for detecting cooler skin temperatures
   - Also requires cryogenic cooling

3. Digital LWIR (DROIC) - LWIR (8-12 um) Digital (NEW)
   - Digital Read-Out IC with 4x well capacity (4.0e6 electrons)
   - 50e RMS read noise (vs 150e analog) - 67% reduction
   - ~1.7-1.9x SNR improvement over analog LWIR

Features:
- SNR-based detection range calculations
- Johnson criteria recognition ranges (detection/orientation/recognition/identification)
- Multi-target comparison (fighter, transport, UAV)
- YAML configuration file support
- Slant path atmospheric transmission
- Altitude scan analysis
- Monte Carlo uncertainty analysis with confidence intervals (NEW)
- 3-way detector comparison: InSb MWIR, MCT LWIR (analog), Digital LWIR (DROIC) (NEW)

Usage:
    python 34_fpa_detection_comparison.py
    python 34_fpa_detection_comparison.py --afterburner
    python 34_fpa_detection_comparison.py --target all --johnson
    python 34_fpa_detection_comparison.py --config configs/detection_scenario.yaml
    python 34_fpa_detection_comparison.py --altitude-scan
    python 34_fpa_detection_comparison.py --detector all  # 3-way comparison
    python 34_fpa_detection_comparison.py --monte-carlo 1000 --seed 42  # Monte Carlo
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, '..')

try:
    from raf_tran.detectors import InSbDetector, MCTDetector, DigitalLWIRDetector
    from raf_tran.targets import generic_fighter, generic_transport, generic_uav
    from raf_tran.detection import (
        calculate_detection_range,
        calculate_snr_vs_range,
        atmospheric_transmission_ir,
        calculate_detection_range_slant,
        scan_altitude_performance,
        elevation_angle,
        # Johnson criteria
        RecognitionTask,
        JOHNSON_CYCLES,
        detection_probability,
        cycles_on_target,
        range_for_probability,
        calculate_recognition_ranges,
        calculate_probability_vs_range,
        johnson_analysis,
        # Scenario loading
        load_scenario,
        create_detector,
        create_target,
        # Monte Carlo
        MonteCarloConfig,
        MonteCarloResult,
        monte_carlo_detection_range,
        monte_carlo_multi_detector,
        default_monte_carlo_config,
    )
except ImportError as e:
    print(f"Error: Could not import raf_tran modules: {e}")
    print("Make sure raf_tran is installed: pip install -e .")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FPA detection ranges for aircraft with Johnson criteria"
    )
    parser.add_argument(
        "--config", type=str,
        help="YAML scenario configuration file"
    )
    parser.add_argument(
        "--target", type=str, default="fighter",
        choices=["fighter", "transport", "uav", "all"],
        help="Target type to analyze (default: fighter)"
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
        "--johnson", action="store_true",
        help="Calculate Johnson criteria recognition ranges"
    )
    parser.add_argument(
        "--probability", type=float, default=0.5,
        help="Target probability for Johnson criteria (default: 0.5)"
    )
    parser.add_argument(
        "--altitude-scan", action="store_true",
        help="Run altitude scan and generate comparison heatmap"
    )
    parser.add_argument(
        "--detector", type=str, default="all",
        choices=["insb", "mct_lwir", "digital_lwir", "all"],
        help="Detector type(s) to compare (default: all)"
    )
    parser.add_argument(
        "--monte-carlo", type=int, default=0,
        metavar="N",
        help="Number of Monte Carlo samples (0=deterministic, default: 0)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for Monte Carlo reproducibility"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.90,
        help="Confidence interval for Monte Carlo (default: 0.90)"
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


def get_detectors(args):
    """Get list of detectors based on arguments."""
    detectors = []

    if args.detector == "all":
        detectors.append(InSbDetector(
            name="InSb MWIR (Analog)",
            pixel_pitch=15.0,
            f_number=2.0,
            integration_time=10.0,
        ))
        detectors.append(MCTDetector(
            name="MCT LWIR (Analog)",
            spectral_band=(8.0, 12.0),
            pixel_pitch=20.0,
            f_number=2.0,
            integration_time=10.0,
        ))
        detectors.append(DigitalLWIRDetector(
            name="Digital LWIR (DROIC)",
            spectral_band=(8.0, 12.0),
            pixel_pitch=15.0,
            f_number=2.0,
            integration_time=10.0,
        ))
    elif args.detector == "insb":
        detectors.append(InSbDetector(
            name="InSb MWIR (Analog)",
            pixel_pitch=15.0,
            f_number=2.0,
            integration_time=10.0,
        ))
    elif args.detector == "mct_lwir":
        detectors.append(MCTDetector(
            name="MCT LWIR (Analog)",
            spectral_band=(8.0, 12.0),
            pixel_pitch=20.0,
            f_number=2.0,
            integration_time=10.0,
        ))
    elif args.detector == "digital_lwir":
        detectors.append(DigitalLWIRDetector(
            name="Digital LWIR (DROIC)",
            spectral_band=(8.0, 12.0),
            pixel_pitch=15.0,
            f_number=2.0,
            integration_time=10.0,
        ))

    return detectors


def get_targets(args):
    """Get list of targets based on arguments."""
    targets = []

    if args.target == "all":
        # All target types
        targets.append(("Fighter (Rear)", generic_fighter(aspect="rear", afterburner=False)))
        if args.afterburner:
            targets.append(("Fighter AB (Rear)", generic_fighter(aspect="rear", afterburner=True)))
        targets.append(("Fighter (Side)", generic_fighter(aspect="side", afterburner=False)))
        targets.append(("Transport (Rear)", generic_transport(aspect="rear")))
        targets.append(("UAV (Rear)", generic_uav(aspect="rear")))
    elif args.target == "fighter":
        name = "Fighter AB" if args.afterburner else "Fighter"
        targets.append((f"{name} ({args.aspect})",
                       generic_fighter(aspect=args.aspect, afterburner=args.afterburner)))
    elif args.target == "transport":
        targets.append((f"Transport ({args.aspect})",
                       generic_transport(aspect=args.aspect)))
    elif args.target == "uav":
        targets.append((f"UAV ({args.aspect})",
                       generic_uav(aspect=args.aspect)))

    return targets


def print_johnson_analysis(insb, mct, targets, args):
    """Print Johnson criteria recognition ranges for all targets."""
    print("\n" + "=" * 80)
    print("JOHNSON CRITERIA RECOGNITION RANGES (P = {:.0%})".format(args.probability))
    print("=" * 80)

    print("\nJohnson criteria relate resolution cycles on target to recognition tasks:")
    print("  Detection:      {:.1f} cycles - Is something there?".format(JOHNSON_CYCLES[RecognitionTask.DETECTION]))
    print("  Orientation:    {:.1f} cycles - Which way is it facing?".format(JOHNSON_CYCLES[RecognitionTask.ORIENTATION]))
    print("  Recognition:    {:.1f} cycles - What type/class is it?".format(JOHNSON_CYCLES[RecognitionTask.RECOGNITION]))
    print("  Identification: {:.1f} cycles - What specific model?".format(JOHNSON_CYCLES[RecognitionTask.IDENTIFICATION]))

    # Header
    print("\n" + "-" * 80)
    print("{:20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Target", "Dim (m)", "IFOV", "Det", "Rec", "ID"))
    print("{:20} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "", "", "(mrad)", "(km)", "(km)", "(km)"))
    print("-" * 80)

    for name, target in targets:
        dim = target.characteristic_dimension_m

        # MWIR ranges
        mwir_ranges = calculate_recognition_ranges(dim, insb.ifov, args.probability)
        det_mwir = mwir_ranges[RecognitionTask.DETECTION] / 1000
        rec_mwir = mwir_ranges[RecognitionTask.RECOGNITION] / 1000
        id_mwir = mwir_ranges[RecognitionTask.IDENTIFICATION] / 1000

        print("{:20} {:>10.1f} {:>10.2f} {:>10.1f} {:>10.1f} {:>10.1f}  MWIR".format(
            name[:20], dim, insb.ifov, det_mwir, rec_mwir, id_mwir))

        # LWIR ranges
        lwir_ranges = calculate_recognition_ranges(dim, mct.ifov, args.probability)
        det_lwir = lwir_ranges[RecognitionTask.DETECTION] / 1000
        rec_lwir = lwir_ranges[RecognitionTask.RECOGNITION] / 1000
        id_lwir = lwir_ranges[RecognitionTask.IDENTIFICATION] / 1000

        print("{:20} {:>10} {:>10.2f} {:>10.1f} {:>10.1f} {:>10.1f}  LWIR".format(
            "", "", mct.ifov, det_lwir, rec_lwir, id_lwir))

    print("-" * 80)

    # Note about SNR vs resolution limits
    print("\nNote: Actual detection ranges are limited by BOTH:")
    print("  1. SNR (signal strength vs noise)")
    print("  2. Resolution (Johnson criteria)")
    print("  Effective range = min(SNR range, Resolution range)")


def print_monte_carlo_results(mc_results, targets, args):
    """Print Monte Carlo simulation results."""
    print("\n" + "=" * 90)
    print(f"MONTE CARLO DETECTION RANGE (N={args.monte_carlo}, {args.confidence:.0%} CI)")
    print("=" * 90)

    # Build header based on number of detectors
    detector_names = list(mc_results.keys())
    header = "{:25}".format("Target")
    for det_name in detector_names:
        header += " {:>20}".format(det_name[:20])
    print("\n" + header)
    print("-" * (25 + 21 * len(detector_names)))

    for name, target in targets:
        row = "{:25}".format(name[:25])
        for det_name in detector_names:
            result = mc_results[det_name]
            if args.confidence == 0.90:
                low, high = result.confidence_interval_90
            else:
                low, high = result.confidence_interval_50
            row += " {:>6.1f} [{:>5.1f}-{:>5.1f}]".format(
                result.mean_range_km, low, high
            )
        print(row)

    print("-" * (25 + 21 * len(detector_names)))
    if args.confidence == 0.90:
        print("All ranges in km with 90% confidence intervals [p5-p95]")
    else:
        print("All ranges in km with 50% confidence intervals [p25-p75]")

    # Print statistics summary
    print("\n" + "-" * 90)
    print("MONTE CARLO STATISTICS")
    print("-" * 90)
    print("{:25} {:>12} {:>12} {:>12} {:>12}".format(
        "Detector", "Mean (km)", "Std (km)", "Median (km)", "CoV (%)"))
    print("-" * 75)
    for det_name in detector_names:
        result = mc_results[det_name]
        cov = 100 * result.std_range_km / result.mean_range_km if result.mean_range_km > 0 else 0
        print("{:25} {:>12.1f} {:>12.1f} {:>12.1f} {:>12.1f}".format(
            det_name[:25], result.mean_range_km, result.std_range_km,
            result.median_range_km, cov))


def print_combined_analysis(insb, mct, targets, args):
    """Print combined SNR and Johnson analysis."""
    print("\n" + "=" * 80)
    print("COMBINED SNR + JOHNSON ANALYSIS")
    print("=" * 80)

    print("\n{:20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Target", "SNR Det", "SNR Det", "Rec", "Rec", "ID", "ID"))
    print("{:20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "", "MWIR", "LWIR", "MWIR", "LWIR", "MWIR", "LWIR"))
    print("-" * 80)

    for name, target in targets:
        dim = target.characteristic_dimension_m

        # SNR-limited detection ranges
        snr_mwir = calculate_detection_range_slant(
            insb, target, args.sensor_altitude, args.target_altitude,
            args.snr_threshold, args.visibility, args.humidity
        ).detection_range_km

        snr_lwir = calculate_detection_range_slant(
            mct, target, args.sensor_altitude, args.target_altitude,
            args.snr_threshold, args.visibility, args.humidity
        ).detection_range_km

        # Johnson recognition ranges
        rec_mwir_raw = range_for_probability(dim, insb.ifov, RecognitionTask.RECOGNITION, args.probability) / 1000
        rec_lwir_raw = range_for_probability(dim, mct.ifov, RecognitionTask.RECOGNITION, args.probability) / 1000
        id_mwir_raw = range_for_probability(dim, insb.ifov, RecognitionTask.IDENTIFICATION, args.probability) / 1000
        id_lwir_raw = range_for_probability(dim, mct.ifov, RecognitionTask.IDENTIFICATION, args.probability) / 1000

        # Effective ranges (minimum of SNR and resolution)
        rec_mwir = min(snr_mwir, rec_mwir_raw)
        rec_lwir = min(snr_lwir, rec_lwir_raw)
        id_mwir = min(snr_mwir, id_mwir_raw)
        id_lwir = min(snr_lwir, id_lwir_raw)

        print("{:20} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f} {:>8.1f}".format(
            name[:20], snr_mwir, snr_lwir, rec_mwir, rec_lwir, id_mwir, id_lwir))

    print("-" * 80)
    print("All ranges in km. Recognition requires 4 cycles, ID requires 6.4 cycles.")


def main():
    args = parse_args()

    # Check for YAML config
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        scenario = load_scenario(args.config)
        print(f"Loaded scenario: {scenario.name}")
        print(f"Description: {scenario.description}")

        # Create detectors and targets from config
        detectors = [create_detector(d) for d in scenario.detectors]
        targets = [(t.name, create_target(t)) for t in scenario.targets]

        # Use scenario parameters
        args.sensor_altitude = scenario.scenario.sensor_altitude_km
        args.target_altitude = scenario.scenario.target_altitude_km
        args.visibility = scenario.scenario.visibility_km
        args.humidity = scenario.scenario.humidity_percent
        args.snr_threshold = scenario.scenario.snr_threshold
        args.probability = scenario.johnson_probability

        # Check if Johnson analysis requested
        if 'recognition_range' in scenario.analysis_types or 'johnson_probability' in scenario.analysis_types:
            args.johnson = True

        # Assign to detectors list for unified handling
        all_detectors = detectors
        # For backward compatibility with Johnson analysis
        insb = detectors[0] if len(detectors) > 0 else InSbDetector()
        mct = detectors[1] if len(detectors) > 1 else MCTDetector()
    else:
        # Create detectors based on --detector argument
        all_detectors = get_detectors(args)

        # For backward compatibility with existing Johnson analysis
        insb = next((d for d in all_detectors if 'MWIR' in d.name), InSbDetector())
        mct = next((d for d in all_detectors if 'LWIR' in d.name and 'Digital' not in d.name), MCTDetector())

        targets = get_targets(args)

    print("=" * 70)
    print("FPA DETECTION RANGE COMPARISON")
    print("=" * 70)

    # Print configuration
    print("\n" + "-" * 70)
    print("CONFIGURATION")
    print("-" * 70)

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

    for det in all_detectors:
        print(f"\n{det.name}:")
        print(f"  Spectral band: {det.spectral_band[0]}-{det.spectral_band[1]} um")
        print(f"  D*: {det.d_star:.1e} cm*sqrt(Hz)/W")
        print(f"  NETD: {det.netd} mK")
        print(f"  Pixel pitch: {det.pixel_pitch} um")
        print(f"  f-number: f/{det.f_number}")
        print(f"  Integration time: {det.integration_time} ms")
        print(f"  NEI: {det.noise_equivalent_irradiance():.2e} W/cm^2")
        print(f"  IFOV: {det.ifov:.2f} mrad")

    # Print target info
    print("\n" + "-" * 70)
    print("TARGET SIGNATURES")
    print("-" * 70)

    for name, target in targets:
        print(f"\n{name}:")
        print(f"  Exhaust temp: {target.exhaust_temp:.0f} K")
        print(f"  Exhaust area: {target.exhaust_area:.2f} m^2")
        print(f"  Skin temp: {target.skin_temp:.0f} K")
        print(f"  Skin area: {target.skin_area:.1f} m^2")
        print(f"  Characteristic dimension: {target.characteristic_dimension_m:.1f} m")
        mwir_intensity = target.radiant_intensity_mwir()
        lwir_intensity = target.radiant_intensity_lwir()
        print(f"  MWIR intensity: {mwir_intensity:.1f} W/sr")
        print(f"  LWIR intensity: {lwir_intensity:.1f} W/sr")

    # Detection range comparison
    print("\n" + "-" * 70)
    print("DETECTION RANGE COMPARISON (SNR-Limited)")
    print("-" * 70)

    primary_target = targets[0][1]  # For later plots

    # Monte Carlo simulation
    mc_results = {}
    if args.monte_carlo > 0:
        print(f"\nRunning Monte Carlo simulation with {args.monte_carlo} samples...")
        if args.seed is not None:
            print(f"Random seed: {args.seed}")

        mc_config = default_monte_carlo_config(args.monte_carlo, args.seed)

        # Run MC for each target with correlated sampling across detectors
        for name, target in targets:
            target_mc_results = monte_carlo_multi_detector(
                all_detectors, target, mc_config,
                snr_threshold=args.snr_threshold,
                visibility_km=args.visibility,
                humidity_percent=args.humidity,
                sensor_altitude_km=args.sensor_altitude,
                target_altitude_km=args.target_altitude,
            )
            mc_results[name] = target_mc_results

        # Print MC results for first target
        print_monte_carlo_results(mc_results[targets[0][0]], targets, args)

    # Deterministic comparison
    print("\n" + "-" * 70)
    print("DETERMINISTIC DETECTION RANGE COMPARISON")
    print("-" * 70)

    # Build dynamic header based on number of detectors
    header = "\n{:25}".format("Target")
    for det in all_detectors:
        short_name = det.name.split()[0] if len(det.name.split()) > 1 else det.name[:10]
        header += " {:>12}".format(short_name[:12])
    header += " {:>12}".format("Winner")
    print(header)
    print("-" * (25 + 13 * (len(all_detectors) + 1)))

    det_results = {}  # Store results for later use
    for name, target in targets:
        row = "{:25}".format(name[:25])
        results = []
        for det in all_detectors:
            result = calculate_detection_range_slant(
                det, target,
                args.sensor_altitude, args.target_altitude,
                args.snr_threshold, args.visibility, args.humidity,
            )
            results.append((det.name, result.detection_range_km))
            row += " {:>12.1f}".format(result.detection_range_km)

        # Find winner
        winner_name, winner_range = max(results, key=lambda x: x[1])
        winner_short = winner_name.split()[0] if len(winner_name.split()) > 1 else winner_name[:8]
        row += " {:>12}".format(winner_short)
        print(row)

        det_results[name] = {r[0]: r[1] for r in results}

    # Store first detector results for backward compatibility
    result_insb = calculate_detection_range_slant(
        insb, primary_target,
        args.sensor_altitude, args.target_altitude,
        args.snr_threshold, args.visibility, args.humidity,
    )
    result_mct = calculate_detection_range_slant(
        mct, primary_target,
        args.sensor_altitude, args.target_altitude,
        args.snr_threshold, args.visibility, args.humidity,
    )

    # Johnson criteria analysis
    if args.johnson:
        print_johnson_analysis(insb, mct, targets, args)
        print_combined_analysis(insb, mct, targets, args)

    # SNR vs range analysis for primary target
    print("\n" + "-" * 70)
    print(f"SNR vs RANGE ANALYSIS: {targets[0][0]}")
    print("-" * 70)

    ranges = np.linspace(0.5, 50, 100)
    mean_alt = (args.sensor_altitude + args.target_altitude) / 2

    snr_insb, irrad_insb, trans_insb = calculate_snr_vs_range(
        insb, primary_target, ranges,
        args.visibility, args.humidity, mean_alt,
    )

    snr_mct, irrad_mct, trans_mct = calculate_snr_vs_range(
        mct, primary_target, ranges,
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
            insb, primary_target, sensor_alts, target_alts,
            args.snr_threshold, args.visibility, args.humidity
        )

        # Scan for LWIR
        print("Scanning MCT LWIR...")
        ranges_lwir = scan_altitude_performance(
            mct, primary_target, sensor_alts, target_alts,
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

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Get results for primary target
    result_insb = calculate_detection_range_slant(
        insb, primary_target,
        args.sensor_altitude, args.target_altitude,
        args.snr_threshold, args.visibility, args.humidity,
    )
    result_mct = calculate_detection_range_slant(
        mct, primary_target,
        args.sensor_altitude, args.target_altitude,
        args.snr_threshold, args.visibility, args.humidity,
    )

    if result_insb.detection_range_km > result_mct.detection_range_km:
        winner = "InSb MWIR"
        advantage = result_insb.detection_range_km / max(result_mct.detection_range_km, 0.1)
    else:
        winner = "MCT LWIR"
        advantage = result_mct.detection_range_km / max(result_insb.detection_range_km, 0.1)

    print(f"""
Primary Target: {targets[0][0]}
Geometry: Sensor at {args.sensor_altitude} km, Target at {args.target_altitude} km
Conditions: {args.visibility} km visibility, {args.humidity}% humidity

SNR Detection Ranges (SNR > {args.snr_threshold}):
  InSb MWIR (3-5 um):  {result_insb.detection_range_km:>6.1f} km
  MCT LWIR (8-12 um):  {result_mct.detection_range_km:>6.1f} km

Winner: {winner} by {advantage:.1f}x
""")

    if args.johnson:
        dim = primary_target.characteristic_dimension_m
        rec_mwir = min(result_insb.detection_range_km,
                       range_for_probability(dim, insb.ifov, RecognitionTask.RECOGNITION, args.probability) / 1000)
        rec_lwir = min(result_mct.detection_range_km,
                       range_for_probability(dim, mct.ifov, RecognitionTask.RECOGNITION, args.probability) / 1000)

        print(f"Recognition Ranges (P={args.probability:.0%}, min of SNR + Johnson):")
        print(f"  MWIR Recognition: {rec_mwir:>6.1f} km")
        print(f"  LWIR Recognition: {rec_lwir:>6.1f} km")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            if args.johnson and not args.altitude_scan:
                # Create figure with Johnson probability curves
                fig = plt.figure(figsize=(16, 12))
                fig.suptitle(
                    f'FPA Detection Comparison: {targets[0][0]}\n'
                    f'Sensor={args.sensor_altitude}km, Target={args.target_altitude}km, '
                    f'Visibility={args.visibility}km',
                    fontsize=14, fontweight='bold'
                )

                # Plot 1: SNR vs Range
                ax1 = fig.add_subplot(2, 2, 1)
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
                ax2 = fig.add_subplot(2, 2, 2)
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

                # Plot 3: Johnson Probability Curves (MWIR)
                ax3 = fig.add_subplot(2, 2, 3)
                ranges_m = ranges * 1000
                dim = primary_target.characteristic_dimension_m

                for task in [RecognitionTask.DETECTION, RecognitionTask.RECOGNITION, RecognitionTask.IDENTIFICATION]:
                    prob = calculate_probability_vs_range(dim, insb.ifov, ranges_m, task)
                    ax3.plot(ranges, prob * 100, linewidth=2, label=f'{task.value.capitalize()}')

                ax3.axhline(y=args.probability * 100, color='k', linestyle='--', alpha=0.5)
                ax3.axvline(x=result_insb.detection_range_km, color='gray', linestyle=':', alpha=0.5,
                           label=f'SNR limit ({result_insb.detection_range_km:.1f}km)')
                ax3.set_xlabel('Range (km)')
                ax3.set_ylabel('Probability (%)')
                ax3.set_title(f'MWIR Johnson Probability ({dim:.0f}m target)')
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(0, 50)
                ax3.set_ylim(0, 100)

                # Plot 4: Johnson Probability Curves (LWIR)
                ax4 = fig.add_subplot(2, 2, 4)
                for task in [RecognitionTask.DETECTION, RecognitionTask.RECOGNITION, RecognitionTask.IDENTIFICATION]:
                    prob = calculate_probability_vs_range(dim, mct.ifov, ranges_m, task)
                    ax4.plot(ranges, prob * 100, linewidth=2, label=f'{task.value.capitalize()}')

                ax4.axhline(y=args.probability * 100, color='k', linestyle='--', alpha=0.5)
                ax4.axvline(x=result_mct.detection_range_km, color='gray', linestyle=':', alpha=0.5,
                           label=f'SNR limit ({result_mct.detection_range_km:.1f}km)')
                ax4.set_xlabel('Range (km)')
                ax4.set_ylabel('Probability (%)')
                ax4.set_title(f'LWIR Johnson Probability ({dim:.0f}m target)')
                ax4.legend(fontsize=9)
                ax4.grid(True, alpha=0.3)
                ax4.set_xlim(0, 50)
                ax4.set_ylim(0, 100)

                plt.tight_layout()
                plt.savefig(args.output, dpi=150, bbox_inches='tight')
                print(f"\nPlot saved to: {args.output}")

            elif altitude_scan_data is not None:
                # Create 3x2 figure with altitude scan heatmaps
                fig = plt.figure(figsize=(16, 14))
                fig.suptitle(
                    f'FPA Detection Comparison: {targets[0][0]}\n'
                    f'Visibility={args.visibility}km, SNR threshold={args.snr_threshold}',
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
                detectors_names = ['InSb\nMWIR', 'MCT\nLWIR']
                det_ranges = [result_insb.detection_range_km, result_mct.detection_range_km]
                colors = ['blue', 'red']
                bars = ax2.bar(detectors_names, det_ranges, color=colors, alpha=0.7, edgecolor='black')
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
                # Include all detectors in the comparison
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(
                    f'FPA Detection Comparison: {targets[0][0]}\n'
                    f'Sensor={args.sensor_altitude}km, Target={args.target_altitude}km, '
                    f'Visibility={args.visibility}km',
                    fontsize=14, fontweight='bold'
                )

                # Calculate SNR for all detectors
                det_colors = {'InSb': 'blue', 'MCT': 'red', 'Digital': 'green'}
                det_snr_data = {}
                det_trans_data = {}
                det_irrad_data = {}
                det_range_results = {}

                for det in all_detectors:
                    snr_det, irrad_det, trans_det = calculate_snr_vs_range(
                        det, primary_target, ranges,
                        args.visibility, args.humidity, mean_alt,
                    )
                    det_snr_data[det.name] = snr_det
                    det_trans_data[det.name] = trans_det
                    det_irrad_data[det.name] = irrad_det

                    result_det = calculate_detection_range_slant(
                        det, primary_target,
                        args.sensor_altitude, args.target_altitude,
                        args.snr_threshold, args.visibility, args.humidity,
                    )
                    det_range_results[det.name] = result_det

                # Plot 1: SNR vs Range for all detectors
                ax1 = axes[0, 0]
                for det in all_detectors:
                    # Determine color based on detector type
                    if 'MWIR' in det.name or 'InSb' in det.name:
                        color = 'blue'
                    elif 'Digital' in det.name or 'DROIC' in det.name:
                        color = 'green'
                    else:
                        color = 'red'
                    ax1.semilogy(ranges, det_snr_data[det.name], color=color, linewidth=2, label=det.name)

                ax1.axhline(y=args.snr_threshold, color='k', linestyle='--',
                           label=f'Detection threshold (SNR={args.snr_threshold})')
                ax1.set_xlabel('Range (km)')
                ax1.set_ylabel('Signal-to-Noise Ratio')
                ax1.set_title('SNR vs Range')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 50)
                ax1.set_ylim(0.1, 1000)

                # Plot 2: Atmospheric Transmission
                ax2 = axes[0, 1]
                # Just show MWIR and LWIR bands (transmission is the same for analog/digital LWIR)
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

                # Plot 3: Target Irradiance vs Range for all detectors
                ax3 = axes[1, 0]
                for det in all_detectors:
                    if 'MWIR' in det.name or 'InSb' in det.name:
                        color = 'blue'
                    elif 'Digital' in det.name or 'DROIC' in det.name:
                        color = 'green'
                    else:
                        color = 'red'
                    ax3.semilogy(ranges, det_irrad_data[det.name], color=color, linewidth=2,
                                label=f'{det.name} irradiance')
                    ax3.axhline(y=det.noise_equivalent_irradiance(), color=color,
                               linestyle='--', alpha=0.7, label=f'{det.name.split()[0]} NEI')

                ax3.set_xlabel('Range (km)')
                ax3.set_ylabel('Irradiance (W/cm^2)')
                ax3.set_title('Target Irradiance at Detector')
                ax3.legend(fontsize=7)
                ax3.grid(True, alpha=0.3)
                ax3.set_xlim(0, 50)

                # Plot 4: Detection Range Comparison Bar Chart for ALL detectors
                ax4 = axes[1, 1]
                det_names_short = []
                det_ranges_vals = []
                det_colors_list = []
                for det in all_detectors:
                    short_name = det.name.replace(' (', '\n(')
                    det_names_short.append(short_name)
                    det_ranges_vals.append(det_range_results[det.name].detection_range_km)
                    if 'MWIR' in det.name or 'InSb' in det.name:
                        det_colors_list.append('blue')
                    elif 'Digital' in det.name or 'DROIC' in det.name:
                        det_colors_list.append('green')
                    else:
                        det_colors_list.append('red')

                bars = ax4.bar(det_names_short, det_ranges_vals, color=det_colors_list, alpha=0.7, edgecolor='black')
                ax4.set_ylabel('Detection Range (km)')
                ax4.set_title(f'Detection Range Comparison (SNR > {args.snr_threshold})')
                ax4.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar, val in zip(bars, det_ranges_vals):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{val:.1f} km', ha='center', va='bottom', fontweight='bold', fontsize=9)

                plt.tight_layout()
                plt.savefig(args.output, dpi=150, bbox_inches='tight')
                print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available for plotting")


if __name__ == "__main__":
    main()
