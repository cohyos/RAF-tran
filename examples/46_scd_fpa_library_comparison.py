#!/usr/bin/env python3
"""
Example 46: SCD FPA Library - Sensor Comparison and Analysis
=============================================================

This example demonstrates the FPA Library focused exclusively on
Semi Conductor Devices (SCD) FPA products:

1. SCD product portfolio overview
2. MWIR vs LWIR cooled/uncooled comparison
3. Johnson criteria DRI range calculations
4. SWaP-C trade study across SCD product lines
5. Technology trends (XBn, InSb, T2SL, VOx)
6. Visualization of SCD sensor landscape

SCD FPA families covered:
- Crane (2560x2048, 5um XBn MWIR)
- Blackbird 1920 (1920x1536, 10um InSb MWIR)
- Sparrow-HD (1280x1024, 5um XBn MWIR)
- Mini Blackbird 1280 (1280x1024, 10um XBn MWIR)
- Hercules 1280 (1280x1024, 15um InSb MWIR)
- Sundra (1280x1024, 15um InSb MWIR)
- Pelican-D LW (640x512, 15um T2SL LWIR cooled)
- Bird XGA (1024x768, 17um VOx LWIR uncooled)
- Bird 640 (640x480, 17um VOx LWIR uncooled)

Usage:
    python examples/46_scd_fpa_library_comparison.py [--no-plot]
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
    compare_fpas,
    rank_fpas,
    compute_dri_ranges,
    compute_swap_score,
    compute_sensitivity_score,
    pitch_miniaturization_factor,
    hot_reliability_gain,
    export_database_json,
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
)


def get_scd_fpas():
    """Get all SCD FPAs from the database."""
    db = get_fpa_database()
    scd_fpas = {k: v for k, v in db.items()
                if 'SCD' in k or 'Semi Conductor' in v.vendor.value}
    return scd_fpas


def main(args):
    print("=" * 70)
    print("Example 46: SCD FPA Library - Sensor Comparison and Analysis")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------
    # 1. SCD Portfolio Overview
    # -----------------------------------------------------------------
    print("1. SCD FPA Portfolio Overview")
    print("-" * 40)

    scd_db = get_scd_fpas()
    print(f"   Total SCD FPAs in database: {len(scd_db)}")
    print()

    # Count by band
    band_counts = {}
    for fpa in scd_db.values():
        band_counts[fpa.spectral_band.value] = band_counts.get(fpa.spectral_band.value, 0) + 1

    print("   SCD FPAs by Spectral Band:")
    for b, count in sorted(band_counts.items(), key=lambda x: -x[1]):
        print(f"     {b:15s}: {count}")
    print()

    # Count by detector type
    det_counts = {}
    for fpa in scd_db.values():
        det_counts[fpa.detector_type.value] = det_counts.get(fpa.detector_type.value, 0) + 1

    print("   SCD FPAs by Detector Technology:")
    for d, count in sorted(det_counts.items(), key=lambda x: -x[1]):
        print(f"     {d:15s}: {count}")
    print()

    # Count by cooling
    cool_counts = {}
    for fpa in scd_db.values():
        cool_counts[fpa.cooling.value] = cool_counts.get(fpa.cooling.value, 0) + 1

    print("   SCD FPAs by Cooling Type:")
    for c, count in sorted(cool_counts.items(), key=lambda x: -x[1]):
        print(f"     {c:25s}: {count}")
    print()

    # -----------------------------------------------------------------
    # 2. Complete SCD Product Listing
    # -----------------------------------------------------------------
    print("2. Complete SCD Product Listing")
    print("-" * 40)

    scd_list = list(scd_db.values())

    print(f"   {'Name':<25s} {'Det Type':<8s} {'Band':<6s} {'Format':<15s} "
          f"{'Pitch':>6s} {'NETD':>8s} {'Cooling':<20s}")
    print("   " + "-" * 95)

    for fpa in sorted(scd_list, key=lambda f: f.name):
        netd = f"{fpa.netd_mk:.0f}mK" if fpa.netd_mk else "N/A"
        cooling = fpa.cooling.value[:20]
        print(f"   {fpa.name:<25s} {fpa.detector_type.value:<8s} "
              f"{fpa.spectral_band.value:<6s} {fpa.resolution_str:<15s} "
              f"{fpa.pixel_pitch_um:>5.0f}u {netd:>7s} {cooling:<20s}")
    print()

    # -----------------------------------------------------------------
    # 3. MWIR Cooled Comparison
    # -----------------------------------------------------------------
    print("3. SCD MWIR Cooled Sensors - DRI Analysis")
    print("-" * 40)

    mwir_scd = [fpa for fpa in scd_list
                if fpa.spectral_band == SpectralBand.MWIR]

    comparison = compare_fpas(mwir_scd, focal_length_mm=100.0)

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
    # 4. LWIR Comparison (cooled + uncooled)
    # -----------------------------------------------------------------
    print("4. SCD LWIR Sensors - DRI Analysis")
    print("-" * 40)

    lwir_scd = [fpa for fpa in scd_list
                if fpa.spectral_band == SpectralBand.LWIR]

    if lwir_scd:
        comparison_lwir = compare_fpas(lwir_scd, focal_length_mm=100.0)

        print(f"   {'Name':<25s} {'Resolution':<15s} {'Pitch':>6s} {'NETD':>8s} "
              f"{'Det km':>8s} {'Rec km':>8s} {'ID km':>8s} {'Cooling':<15s}")
        print("   " + "-" * 105)

        for row in comparison_lwir:
            netd = f"{row['netd_mk']:.0f}" if row['netd_mk'] else "N/A"
            # Find original FPA for cooling info
            fpa_obj = next((f for f in lwir_scd if f.name == row['name']), None)
            cooling = fpa_obj.cooling.value[:15] if fpa_obj else ""
            print(f"   {row['name']:<25s} {row['resolution']:<15s} "
                  f"{row['pitch_um']:>5.0f}u {netd:>7s}mK "
                  f"{row['detection_km']:>7.1f} {row['recognition_km']:>7.1f} "
                  f"{row['identification_km']:>7.1f} {cooling:<15s}")
    print()

    # -----------------------------------------------------------------
    # 5. Rankings
    # -----------------------------------------------------------------
    print("5. SCD FPA Rankings")
    print("-" * 40)

    # Rank by sensitivity
    print("   Sensitivity Score Ranking (all SCD):")
    ranked = rank_fpas(scd_list, metric='sensitivity_score')
    for i, (fpa, score) in enumerate(ranked):
        score_str = f"{score:.1f}" if score is not None else "N/A"
        print(f"     {i+1}. {fpa.name:<25s} - Score: {score_str}  "
              f"({fpa.spectral_band.value}, {fpa.detector_type.value})")
    print()

    # Rank by megapixels
    print("   Resolution Ranking (megapixels):")
    ranked_mp = rank_fpas(scd_list, metric='megapixels')
    for i, (fpa, mp) in enumerate(ranked_mp):
        print(f"     {i+1}. {fpa.name:<25s} - {mp:.2f} MP  "
              f"({fpa.resolution_str})")
    print()

    # Rank by smallest pitch
    print("   Pixel Pitch Ranking (smallest first):")
    ranked_pitch = rank_fpas(scd_list, metric='pixel_pitch_um', ascending=True)
    for i, (fpa, pitch) in enumerate(ranked_pitch):
        print(f"     {i+1}. {fpa.name:<25s} - {pitch:.0f} um  "
              f"({fpa.detector_type.value})")
    print()

    # -----------------------------------------------------------------
    # 6. SCD Technology Analysis
    # -----------------------------------------------------------------
    print("6. SCD Technology Analysis")
    print("-" * 40)

    # XBn vs InSb comparison
    xbn_fpas = [f for f in scd_list if f.detector_type == DetectorType.XBn]
    insb_fpas = [f for f in scd_list if f.detector_type == DetectorType.InSb]
    t2sl_fpas = [f for f in scd_list if f.detector_type == DetectorType.T2SL]
    vox_fpas = [f for f in scd_list if f.detector_type == DetectorType.VOx]

    print("   XBn (HOT technology):")
    for f in xbn_fpas:
        print(f"     - {f.name}: {f.resolution_str}, {f.pixel_pitch_um}um pitch")
    print()

    print("   InSb (traditional cooled):")
    for f in insb_fpas:
        print(f"     - {f.name}: {f.resolution_str}, {f.pixel_pitch_um}um pitch")
    print()

    print("   T2SL (Type-II Superlattice, LWIR cooled):")
    for f in t2sl_fpas:
        print(f"     - {f.name}: {f.resolution_str}, {f.pixel_pitch_um}um pitch")
    print()

    print("   VOx (microbolometer, uncooled):")
    for f in vox_fpas:
        print(f"     - {f.name}: {f.resolution_str}, {f.pixel_pitch_um}um pitch")
    print()

    # XBn advantage: smaller pitch, HOT
    if xbn_fpas:
        avg_xbn_pitch = sum(f.pixel_pitch_um for f in xbn_fpas) / len(xbn_fpas)
        print(f"   XBn average pitch: {avg_xbn_pitch:.1f} um")
    if insb_fpas:
        avg_insb_pitch = sum(f.pixel_pitch_um for f in insb_fpas) / len(insb_fpas)
        print(f"   InSb average pitch: {avg_insb_pitch:.1f} um")
    print()

    # Miniaturization factors for SCD pitches
    scd_pitches = sorted(set(f.pixel_pitch_um for f in scd_list))
    print("   SCD Pixel Pitch Miniaturization (vs 15um reference):")
    for pitch in scd_pitches:
        factor = pitch_miniaturization_factor(pitch)
        print(f"     {pitch:5.0f}um: {factor:.2f}x optics size "
              f"({(1-factor)*100:.0f}% reduction)")
    print()

    # -----------------------------------------------------------------
    # 7. SCD ROIC Database
    # -----------------------------------------------------------------
    print("7. SCD ROIC Database")
    print("-" * 40)

    roic_db = get_roic_database()
    scd_roics = {k: v for k, v in roic_db.items()
                 if 'SCD' in v.vendor.value or 'Semi Conductor' in v.vendor.value
                 or 'Pelican' in k}

    if scd_roics:
        for name, roic in scd_roics.items():
            print(f"   {roic.name}:")
            well = f"{roic.well_capacity_Me:.1f}Me" if roic.well_capacity_Me else "N/A"
            adc = f"{roic.adc_bits}b" if roic.adc_bits else "N/A"
            print(f"     Format: {roic.array_format}")
            print(f"     Pitch: {roic.pixel_pitch_um} um")
            print(f"     Well capacity: {well}")
            print(f"     ADC: {adc}")
            if roic.process_node_um:
                print(f"     Process: {roic.process_node_um} um")
            print()
    else:
        print("   No SCD-specific ROICs found")
        print()

    # -----------------------------------------------------------------
    # 8. DRI at Multiple Focal Lengths
    # -----------------------------------------------------------------
    print("8. DRI Ranges at Multiple Focal Lengths")
    print("-" * 40)

    focal_lengths = [50, 100, 150, 200, 300]
    print(f"   {'FPA':<25s}", end="")
    for fl in focal_lengths:
        print(f" | f={fl:3d}mm ID(km)", end="")
    print()
    print("   " + "-" * (25 + len(focal_lengths) * 18))

    for fpa in sorted(scd_list, key=lambda f: f.name):
        print(f"   {fpa.name:<25s}", end="")
        for fl in focal_lengths:
            dri = compute_dri_ranges(fpa, fl)
            if dri:
                print(f" | {dri['identification_m'] / 1000:>14.1f}", end="")
            else:
                print(f" | {'N/A':>14s}", end="")
        print()
    print()

    # -----------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------
    if not args.no_plot:
        print("9. Creating SCD Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Panel 1: Resolution vs Pitch (SCD only)
        plot_resolution_vs_pitch(scd_list, ax=axes[0, 0])
        axes[0, 0].set_title('SCD: Resolution vs Pitch')

        # Panel 2: NETD comparison (SCD only)
        plot_netd_comparison(scd_list, ax=axes[0, 1])
        axes[0, 1].set_title('SCD: NETD Comparison')

        # Panel 3: DRI ranges for MWIR
        if mwir_scd:
            plot_dri_ranges(mwir_scd, focal_length_mm=100.0, ax=axes[0, 2])
            axes[0, 2].set_title('SCD MWIR: DRI Ranges (f=100mm)')

        # Panel 4: DRI comparison across all SCD at f=150mm
        ax = axes[1, 0]
        names = []
        det_ranges = []
        id_ranges = []
        colors = []
        band_color_map = {
            SpectralBand.MWIR: '#2196F3',
            SpectralBand.LWIR: '#F44336',
        }
        for fpa in sorted(scd_list, key=lambda f: f.name):
            dri = compute_dri_ranges(fpa, 150.0)
            if dri:
                names.append(fpa.name)
                det_ranges.append(dri['detection_m'] / 1000)
                id_ranges.append(dri['identification_m'] / 1000)
                colors.append(band_color_map.get(fpa.spectral_band, '#888'))

        import numpy as np
        y = np.arange(len(names))
        ax.barh(y - 0.15, det_ranges, 0.3, label='Detection', color=colors, alpha=0.8,
                edgecolor='black', linewidth=0.5)
        ax.barh(y + 0.15, id_ranges, 0.3, label='Identification', color=colors, alpha=0.4,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Range (km)')
        ax.set_title('SCD All: DRI at f=150mm')
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        # Panel 5: Technology landscape (SCD only)
        plot_technology_landscape(scd_list, ax=axes[1, 1])
        axes[1, 1].set_title('SCD: Technology Landscape')

        # Panel 6: Spectral coverage (SCD only)
        plot_spectral_coverage(scd_list, ax=axes[1, 2])
        axes[1, 2].set_title('SCD: Spectral Coverage')

        plt.suptitle('SCD FPA Library - Comprehensive Sensor Analysis',
                      fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, '46_scd_fpa_library_comparison.png'),
                     dpi=150, bbox_inches='tight')
        print("   Saved: outputs/46_scd_fpa_library_comparison.png")
        plt.close()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY - SCD FPA Portfolio")
    print("=" * 70)
    print(f"""
    SCD Product Lines:

    MWIR Cooled ({len(mwir_scd)} products):
      - XBn technology: Crane (5MP), Sparrow-HD (1.3MP), Mini Blackbird (1.3MP)
      - InSb technology: Blackbird 1920 (3MP), Hercules 1280 (1.3MP), Sundra (1.3MP)

    LWIR ({len(lwir_scd)} products):
      - T2SL cooled: Pelican-D LW (640x512, 15mK NETD)
      - VOx uncooled: Bird XGA (1024x768), Bird 640 (640x480)

    Key Differentiators:
      - XBn: Barrier detector technology enables HOT (Higher Operating Temperature)
        operation with smaller pixels (5um) for size/weight reduction
      - Crane: Highest resolution MWIR (2560x2048, 5MP) with 5um pitch
      - Pelican-D LW: Best cooled LWIR NETD (15mK) for thermal sensitivity
      - Bird series: Cost-effective uncooled microbolometers
    """)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCD FPA Library Comparison')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plot generation')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)
    sys.exit(main(args))
