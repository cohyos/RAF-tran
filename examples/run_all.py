#!/usr/bin/env python3
"""
Run All Examples
================

This script runs all RAF-tran examples sequentially.

Usage:
    python run_all.py              # Run all examples
    python run_all.py --no-plot    # Run without generating plots
    python run_all.py --list       # List available examples
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


EXAMPLES = [
    # Core Examples (Demonstration)
    ("01_solar_zenith_angle_study.py", "Solar Zenith Angle Effects"),
    ("02_spectral_transmission.py", "Spectral Transmission (Sky Color)"),
    ("03_aerosol_types_comparison.py", "Aerosol Types Comparison"),
    ("04_atmospheric_profiles.py", "Atmospheric Profiles"),
    ("05_greenhouse_effect.py", "Greenhouse Effect"),
    ("06_surface_albedo_effects.py", "Surface Albedo Effects"),
    ("07_cloud_radiative_effects.py", "Cloud Radiative Effects"),
    ("08_ozone_uv_absorption.py", "Ozone UV Absorption"),
    ("09_radiative_heating_rates.py", "Radiative Heating Rates"),
    ("10_satellite_observation.py", "Satellite Observation Simulation"),
    ("11_atmospheric_turbulence.py", "Atmospheric Turbulence (Cn2, Fried)"),
    # Validation Examples (Physics Verification)
    ("12_beer_lambert_validation.py", "Beer-Lambert Law Validation"),
    ("13_planck_blackbody_validation.py", "Planck Blackbody Validation"),
    ("14_rayleigh_scattering_validation.py", "Rayleigh Scattering Validation"),
    ("15_mie_scattering_validation.py", "Mie Scattering Validation"),
    ("16_two_stream_benchmarks.py", "Two-Stream Solver Benchmarks"),
    ("17_solar_spectrum_analysis.py", "Solar Spectrum Analysis"),
    ("18_thermal_emission_validation.py", "Thermal Emission Validation"),
    ("19_path_radiance_remote_sensing.py", "Path Radiance Remote Sensing"),
    ("20_visibility_contrast.py", "Visibility and Contrast"),
    ("21_laser_propagation.py", "Laser Propagation"),
    # Advanced Applications
    ("22_atmospheric_polarization.py", "Atmospheric Polarization"),
    ("23_infrared_atmospheric_windows.py", "IR Atmospheric Windows"),
    ("24_volcanic_aerosol_forcing.py", "Volcanic Aerosol Forcing"),
    ("25_water_vapor_feedback.py", "Water Vapor Feedback"),
    ("26_high_altitude_solar.py", "High Altitude Solar Radiation"),
    ("27_twilight_spectra.py", "Twilight Spectra"),
    ("28_multi_layer_cloud.py", "Multi-Layer Cloud Overlap"),
    ("29_aod_retrieval_visibility.py", "AOD Retrieval and Visibility"),
    ("30_spectral_surface_albedo.py", "Spectral Surface Albedo"),
    ("31_limb_viewing_geometry.py", "Limb Viewing Geometry"),
    ("32_config_file_demo.py", "Configuration File Usage"),
    ("33_validation_visualization.py", "Physics Validation Visualization"),
    # Detection Applications
    ("34_fpa_detection_comparison.py", "FPA Detection Range Comparison"),
    ("35_fpa_altitude_detection_study.py", "FPA Altitude Detection Study"),
    # New Feature Demonstrations
    ("36_hitran_gas_absorption.py", "HITRAN Gas Absorption (Optional)"),
    ("37_adaptive_optics_simulation.py", "Adaptive Optics Simulation"),
    ("38_real_cn2_profiles.py", "Real Cn2 Profile Integration"),
    ("39_spherical_geometry.py", "3D Spherical Earth Geometry"),
    ("40_weather_profiles.py", "Atmospheric Weather Profiles"),
    ("41_fpa_library_comparison.py", "FPA Library Sensor Comparison"),
    ("42_sensor_assembly_optimization.py", "Sensor Assembly and Optimization"),
    ("43_air_to_air_detection_optimization.py", "Air-to-Air F-16 Detection MC Optimization"),
    ("44_air_tractor_detection_mc.py", "Air Tractor AT-802 Detection MC Optimization"),
    ("45_ballistic_missile_detection_mc.py", "Ballistic Missile Post-Boost Detection MC"),
    # SCD-Only Variants
    ("46_scd_fpa_library_comparison.py", "SCD FPA Library Comparison"),
    ("47_scd_air_to_air_detection_mc.py", "SCD Air-to-Air F-16 Detection MC"),
    ("48_scd_air_tractor_detection_mc.py", "SCD Air Tractor AT-802 Detection MC"),
    ("49_scd_ballistic_missile_detection_mc.py", "SCD Ballistic Missile Detection MC"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all RAF-tran examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation for all examples"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available examples without running them"
    )
    parser.add_argument(
        "--example", type=int, nargs="+",
        help="Run specific example(s) by number (e.g., --example 1 5 10)"
    )
    parser.add_argument(
        "--pause", action="store_true",
        help="Pause between examples (press Enter to continue)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Change to examples directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    if args.list:
        print("=" * 60)
        print("AVAILABLE RAF-TRAN EXAMPLES")
        print("=" * 60)
        for filename, description in EXAMPLES:
            num = filename.split("_")[0]
            print(f"  {num}: {description}")
            print(f"       {filename}")
        print("=" * 60)
        return

    # Filter examples if specific ones requested
    if args.example:
        selected = []
        for num in args.example:
            if 1 <= num <= len(EXAMPLES):
                selected.append(EXAMPLES[num - 1])
            else:
                print(f"Warning: Example {num} not found (valid: 1-{len(EXAMPLES)})")
        examples_to_run = selected
    else:
        examples_to_run = EXAMPLES

    if not examples_to_run:
        print("No examples to run!")
        return

    print("=" * 70)
    print("RAF-TRAN EXAMPLES RUNNER")
    print("=" * 70)
    print(f"Running {len(examples_to_run)} example(s)...")
    if args.no_plot:
        print("(Plot generation disabled)")
    print()

    results = []

    for i, (filename, description) in enumerate(examples_to_run, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(examples_to_run)}] {description}")
        print(f"    Running: {filename}")
        print("=" * 70 + "\n")

        # Build command
        cmd = [sys.executable, filename]
        if args.no_plot:
            cmd.append("--no-plot")

        try:
            result = subprocess.run(cmd, check=False)
            success = result.returncode == 0
            results.append((filename, success))

            if success:
                print(f"\n[OK] {filename} completed successfully")
            else:
                print(f"\n[X] {filename} exited with code {result.returncode}")

        except Exception as e:
            print(f"\n[X] {filename} failed: {e}")
            results.append((filename, False))

        if args.pause and i < len(examples_to_run):
            input("\nPress Enter to continue to next example...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed

    for filename, success in results:
        status = "[OK] PASS" if success else "[X] FAIL"
        print(f"  {status}  {filename}")

    print("-" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {len(results)}")
    print("=" * 70)

    # List generated plots
    plots = list(Path(".").glob("*.png"))
    if plots:
        print(f"\nGenerated {len(plots)} plot(s):")
        for p in sorted(plots):
            print(f"  - {p}")


if __name__ == "__main__":
    main()
