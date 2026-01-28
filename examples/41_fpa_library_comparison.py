#!/usr/bin/env python3
"""
Example 41: FPA Library - Sensor Comparison and Analysis
========================================================

This example demonstrates the FPA (Focal Plane Array) Library for
comparing infrared sensors from multiple vendors:

1. Database browsing and search
2. Vendor portfolio analysis
3. Johnson criteria DRI range calculations
4. SWaP-C trade study
5. Technology landscape analysis
6. Configuration save/load
7. Spectral coverage visualization

The FPA Library is modular and can be attached to other projects
independently of RAF-tran.

Vendors covered: SCD, Teledyne FLIR, L3Harris, Raytheon, DRS, Axiom

Usage:
    python examples/41_fpa_library_comparison.py [--no-plot]
"""

import argparse
import sys
import os

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# FPA Library imports
from raf_tran.fpa_library import (
    get_fpa_database,
    get_roic_database,
    search_fpas,
    get_vendor_portfolio,
    get_band_options,
    list_fpas,
    list_roics,
    compare_fpas,
    rank_fpas,
    compute_dri_ranges,
    compute_swap_score,
    compute_sensitivity_score,
    pitch_miniaturization_factor,
    hot_reliability_gain,
    spectral_band_comparison,
    save_fpa_config,
    export_database_json,
    create_comparison_config,
    save_comparison_config,
)
from raf_tran.fpa_library.models import (
    FPASpec, ArrayFormat, SpectralRange,
    SpectralBand, DetectorType, CoolingType, Vendor, ApplicationDomain,
)
from raf_tran.fpa_library.visualization import (
    plot_resolution_vs_pitch,
    plot_netd_comparison,
    plot_dri_ranges,
    plot_swap_analysis,
    plot_spectral_coverage,
    plot_technology_landscape,
    plot_vendor_portfolio_summary,
)


def main(args):
    print("=" * 70)
    print("Example 41: FPA Library - Sensor Comparison and Analysis")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------
    # 1. Database Overview
    # -----------------------------------------------------------------
    print("1. FPA Database Overview")
    print("-" * 40)

    db = get_fpa_database()
    roic_db = get_roic_database()

    print(f"   Total FPAs in database: {len(db)}")
    print(f"   Total ROICs in database: {len(roic_db)}")
    print()

    # Count by vendor
    vendor_counts = {}
    for fpa in db.values():
        v = fpa.vendor.value.split('(')[0].strip()
        vendor_counts[v] = vendor_counts.get(v, 0) + 1

    print("   FPAs by Vendor:")
    for v, count in sorted(vendor_counts.items(), key=lambda x: -x[1]):
        print(f"     {v:30s}: {count}")
    print()

    # Count by band
    band_counts = {}
    for fpa in db.values():
        band_counts[fpa.spectral_band.value] = band_counts.get(fpa.spectral_band.value, 0) + 1

    print("   FPAs by Spectral Band:")
    for b, count in sorted(band_counts.items(), key=lambda x: -x[1]):
        print(f"     {b:15s}: {count}")
    print()

    # -----------------------------------------------------------------
    # 2. Search and Filter
    # -----------------------------------------------------------------
    print("2. Search and Filter Examples")
    print("-" * 40)

    # MWIR cooled sensors
    mwir_cooled = search_fpas(
        spectral_band=SpectralBand.MWIR,
        cooling=CoolingType.COOLED_STIRLING,
    )
    print(f"   MWIR Cooled sensors: {len(mwir_cooled)}")
    for fpa in mwir_cooled:
        print(f"     - {fpa.name} ({fpa.vendor.value.split('(')[0].strip()}) "
              f"{fpa.resolution_str} @ {fpa.pixel_pitch_um}um")
    print()

    # Uncooled LWIR sensors
    uncooled_lwir = search_fpas(
        spectral_band=SpectralBand.LWIR,
        cooling=CoolingType.UNCOOLED,
    )
    print(f"   Uncooled LWIR sensors: {len(uncooled_lwir)}")
    for fpa in uncooled_lwir:
        netd_str = f"NETD={fpa.netd_mk}mK" if fpa.netd_mk else ""
        print(f"     - {fpa.name} ({fpa.vendor.value.split('(')[0].strip()}) "
              f"{fpa.resolution_str} @ {fpa.pixel_pitch_um}um {netd_str}")
    print()

    # Small pixel pitch (<=10um)
    small_pitch = search_fpas(max_pitch_um=10.0)
    print(f"   Small pitch (<=10um): {len(small_pitch)}")
    for fpa in small_pitch:
        print(f"     - {fpa.name}: {fpa.pixel_pitch_um}um pitch, "
              f"{fpa.resolution_str}")
    print()

    # -----------------------------------------------------------------
    # 3. Detailed Comparison Table
    # -----------------------------------------------------------------
    print("3. MWIR Cooled Sensor Comparison")
    print("-" * 40)

    comparison = compare_fpas(mwir_cooled, focal_length_mm=100.0)

    print(f"   {'Name':<25s} {'Resolution':<15s} {'Pitch':>6s} {'NETD':>8s} "
          f"{'Det km':>8s} {'Rec km':>8s} {'ID km':>8s}")
    print("   " + "-" * 85)

    for row in comparison:
        netd = f"{row['netd_mk']:.0f}" if row['netd_mk'] else "N/A"
        print(f"   {row['name']:<25s} {row['resolution']:<15s} "
              f"{row['pitch_um']:>5.0f}u {netd:>7s}mK "
              f"{row['detection_km']:>7.1f} {row['recognition_km']:>7.1f} "
              f"{row['identification_km']:>7.1f}")
    print()

    # -----------------------------------------------------------------
    # 4. Ranking
    # -----------------------------------------------------------------
    print("4. FPA Rankings")
    print("-" * 40)

    all_fpas = list(db.values())

    # Rank by sensitivity
    print("   Top 5 by Sensitivity Score:")
    ranked = rank_fpas(all_fpas, metric='sensitivity_score')
    for i, (fpa, score) in enumerate(ranked[:5]):
        print(f"     {i+1}. {fpa.name} ({fpa.vendor.value.split('(')[0].strip()}) "
              f"- Score: {score:.1f}")
    print()

    # Rank by megapixels
    print("   Top 5 by Resolution:")
    ranked_mp = rank_fpas(all_fpas, metric='megapixels')
    for i, (fpa, mp) in enumerate(ranked_mp[:5]):
        print(f"     {i+1}. {fpa.name} ({fpa.vendor.value.split('(')[0].strip()}) "
              f"- {mp:.2f} MP")
    print()

    # Rank by smallest pitch
    print("   Top 5 by Smallest Pixel Pitch:")
    ranked_pitch = rank_fpas(all_fpas, metric='pixel_pitch_um', ascending=True)
    for i, (fpa, pitch) in enumerate(ranked_pitch[:5]):
        print(f"     {i+1}. {fpa.name} ({fpa.vendor.value.split('(')[0].strip()}) "
              f"- {pitch:.0f} um")
    print()

    # -----------------------------------------------------------------
    # 5. Technology Trends
    # -----------------------------------------------------------------
    print("5. Technology Trends Analysis")
    print("-" * 40)

    # Pitch miniaturization
    print("   Pixel Pitch Miniaturization (vs 15um reference):")
    for pitch in [15, 12, 10, 8, 5]:
        factor = pitch_miniaturization_factor(float(pitch))
        print(f"     {pitch}um: {factor:.2f}x optics size "
              f"({(1-factor)*100:.0f}% reduction)")
    print()

    # HOT reliability gains
    print("   HOT Technology Reliability Gains (vs 77K InSb):")
    for temp in [100, 120, 150, 180]:
        gains = hot_reliability_gain(float(temp))
        print(f"     {temp}K: MTTF x{gains['mttf_multiplier']:.1f}, "
              f"Cooldown {gains['cooldown_reduction']*100:.0f}%, "
              f"Power {gains['power_reduction']*100:.0f}%")
    print()

    # -----------------------------------------------------------------
    # 6. Configuration Save/Load Demo
    # -----------------------------------------------------------------
    print("6. Configuration Save/Load")
    print("-" * 40)

    # Export database
    os.makedirs('outputs', exist_ok=True)
    export_database_json('outputs/fpa_database.json')
    print("   Exported full database to outputs/fpa_database.json")

    # Save comparison config
    config = create_comparison_config(
        fpa_keys=['SCD_Pelican_D_LW', 'FLIR_A6781_SLS'],
        focal_length_mm=150.0,
        target_size_m=2.3,
    )
    save_comparison_config(config, 'outputs/lwir_comparison.json')
    print("   Saved LWIR comparison config to outputs/lwir_comparison.json")

    # Custom FPA definition
    custom_fpa = FPASpec(
        name='Custom Sensor X',
        vendor=Vendor.SCD,
        product_family='Custom',
        detector_type=DetectorType.T2SL,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(7.5, 10.0),
        cooling=CoolingType.COOLED_STIRLING,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=10.0,
        netd_mk=12.0,
        notes="User-defined custom sensor for trade study.",
    )
    save_fpa_config(custom_fpa, 'outputs/custom_fpa.json')
    print("   Saved custom FPA definition to outputs/custom_fpa.json")
    print()

    # -----------------------------------------------------------------
    # 7. ROIC Database
    # -----------------------------------------------------------------
    print("7. ROIC Database")
    print("-" * 40)

    print(f"   {'ROIC':<20s} {'Vendor':<20s} {'Format':<15s} {'Pitch':>6s} "
          f"{'Well':>10s} {'ADC':>5s}")
    print("   " + "-" * 80)

    for name, roic in roic_db.items():
        well = f"{roic.well_capacity_Me:.1f}Me" if roic.well_capacity_Me else "N/A"
        adc = f"{roic.adc_bits}b" if roic.adc_bits else "N/A"
        vendor = roic.vendor.value.split('(')[0].strip()
        print(f"   {roic.name:<20s} {vendor:<20s} {str(roic.array_format):<15s} "
              f"{roic.pixel_pitch_um:>5.0f}u {well:>9s} {adc:>5s}")
    print()

    # -----------------------------------------------------------------
    # 8. Spectral Band Reference
    # -----------------------------------------------------------------
    print("8. Spectral Band Reference")
    print("-" * 40)

    bands = spectral_band_comparison()
    for band_name, props in bands.items():
        print(f"   {band_name}:")
        for key, val in props.items():
            print(f"     {key:30s}: {val}")
        print()

    # -----------------------------------------------------------------
    # 9. Visualization
    # -----------------------------------------------------------------
    if not args.no_plot:
        print("9. Creating Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Panel 1: Resolution vs Pitch
        plot_resolution_vs_pitch(all_fpas, ax=axes[0, 0])

        # Panel 2: NETD comparison
        plot_netd_comparison(all_fpas, ax=axes[0, 1])

        # Panel 3: DRI ranges for MWIR cooled
        plot_dri_ranges(mwir_cooled, focal_length_mm=100.0, ax=axes[0, 2])

        # Panel 4: Technology landscape
        plot_technology_landscape(all_fpas, ax=axes[1, 0])

        # Panel 5: Spectral coverage
        plot_spectral_coverage(all_fpas, ax=axes[1, 1])

        # Panel 6: Vendor portfolio
        plot_vendor_portfolio_summary(all_fpas, ax=axes[1, 2])

        plt.suptitle('FPA Library - Comprehensive Sensor Analysis',
                      fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig('outputs/41_fpa_library_comparison.png',
                     dpi=150, bbox_inches='tight')
        print("   Saved: outputs/41_fpa_library_comparison.png")
        plt.close()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    FPA Library Capabilities:

    1. DATABASE: {n_fpa} FPAs and {n_roic} ROICs from {n_vendors} vendors
    2. SEARCH: Filter by vendor, band, cooling, resolution, pitch, NETD
    3. ANALYSIS: Johnson criteria DRI, SWaP scores, sensitivity metrics
    4. CONFIG: Save/load JSON configurations for reproducible analysis
    5. VISUALIZATION: 7 chart types for trade studies and presentations
    6. MODULAR: Can be used independently of RAF-tran

    Vendors: SCD, Teledyne FLIR, L3Harris, Raytheon, DRS, Axiom
    Bands: SWIR, MWIR, LWIR, Dual MW/LW
    Technologies: InSb, XBn, HFM, MCT, T2SL, SLS, VOx, Dual-Band
    """.format(n_fpa=len(db), n_roic=len(roic_db),
               n_vendors=len(vendor_counts)))

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FPA Library Comparison')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)

    sys.exit(main(args))
