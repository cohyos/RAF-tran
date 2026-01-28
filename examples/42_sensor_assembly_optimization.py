#!/usr/bin/env python3
"""
Example 42: Sensor Assembly Definition and Optimization
=======================================================

This example demonstrates the sensor assembly and optimizer capabilities:

1. Define a sensor by combining an FPA with optics and operating parameters
2. Validate parameter conflicts (integration time, F-number, well fill, etc.)
3. Sweep focal length to find optimal DRI ranges
4. Multi-parameter optimization across FPA + optics space
5. Find the best FPA for a given mission from the entire database
6. Pareto frontier analysis (range vs SWaP tradeoff)
7. Sensitivity analysis of key parameters
8. Visualization of optimization results

Usage:
    python examples/42_sensor_assembly_optimization.py [--no-plot]
"""

import argparse
import sys
import os

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# FPA Library imports
from raf_tran.fpa_library import (
    get_fpa_database,
    get_fpa,
    search_fpas,
)
from raf_tran.fpa_library.models import SpectralBand, Vendor
from raf_tran.fpa_library.sensor import (
    SensorAssembly,
    OpticsConfig,
    OperatingParams,
)
from raf_tran.fpa_library.optimizer import (
    optimize_sensor,
    find_best_fpa,
    sweep_focal_length,
    sweep_parameters,
    pareto_frontier,
    sensitivity_analysis,
    format_results_table,
    ParameterRange,
    OptimizationConstraint,
    OptimizationResult,
)


def main(args):
    print("=" * 70)
    print("Example 42: Sensor Assembly Definition and Optimization")
    print("=" * 70)
    print()

    # =================================================================
    # 1. Define a Sensor Assembly
    # =================================================================
    print("1. Sensor Assembly Definition")
    print("-" * 40)

    fpa = get_fpa('SCD_Pelican_D_LW')
    optics = OpticsConfig(
        focal_length_mm=100.0,
        f_number=4.0,
        transmission=0.85,
    )
    operating = OperatingParams(
        integration_time_ms=0.5,
        frame_rate_hz=30.0,
    )

    sensor = SensorAssembly(fpa, optics, operating, name="LWIR Surveillance")
    print(sensor)
    print()

    # =================================================================
    # 2. Conflict Validation Examples
    # =================================================================
    print("2. Parameter Conflict Validation")
    print("-" * 40)

    # Good config
    print("   a) Valid configuration:")
    print(f"      {sensor.validation}")
    print()

    # Bad config: integration time > frame period
    bad_op = OperatingParams(integration_time_ms=50.0, frame_rate_hz=60.0)
    bad_sensor = SensorAssembly(fpa, OpticsConfig(100, 4.0), bad_op, name="Bad Config")
    print("   b) Integration time conflict (50ms at 60Hz):")
    for issue in bad_sensor.validation.issues:
        print(f"      {issue}")
    print()

    # Unrealistic F-number
    bad_optics = OpticsConfig(focal_length_mm=100.0, f_number=0.5)
    bad_sensor2 = SensorAssembly(fpa, bad_optics, operating, name="Bad Optics")
    print("   c) Unrealistic F-number (F/0.5):")
    for issue in bad_sensor2.validation.issues:
        print(f"      {issue}")
    print()

    # =================================================================
    # 3. Performance Summary
    # =================================================================
    print("3. Performance Summary")
    print("-" * 40)

    summary = sensor.performance_summary()
    for key in ['ifov_urad', 'ifov_mrad', 'fov_h_deg', 'fov_v_deg',
                'detection_range_km', 'recognition_range_km',
                'identification_range_km', 'estimated_netd_mk',
                'data_rate_mbps', 'duty_cycle']:
        val = summary.get(key)
        if val is not None:
            print(f"   {key:30s}: {val:.2f}")
    print()

    # Pixels on target
    pot = sensor.pixels_on_target
    print("   Pixels on 2.3m target:")
    for rng, pix in pot.items():
        print(f"     {rng:6s}: {pix:.1f} pixels")
    print()

    # =================================================================
    # 4. Focal Length Sweep
    # =================================================================
    print("4. Focal Length Sweep (50-300mm)")
    print("-" * 40)

    # Use MWIR FPA for focal length sweep (lower background flux)
    sweep_fpa = get_fpa('FLIR_Neutrino_SX8')
    sweep_op = OperatingParams(integration_time_ms=8.0, frame_rate_hz=60.0)
    focal_lengths = list(np.linspace(50, 300, 20))
    fl_results = sweep_focal_length(
        fpa=sweep_fpa,
        f_number=2.5,
        focal_lengths_mm=focal_lengths,
        operating=sweep_op,
        objective='identification_range_km',
    )
    print(f"   Evaluated {len(fl_results)} valid configurations")
    if fl_results:
        best = fl_results[0]
        print(f"   Best ID range: {best.score:.2f} km "
              f"at f={best.sensor.optics.focal_length_mm:.0f}mm")
        worst = fl_results[-1]
        print(f"   Worst ID range: {worst.score:.2f} km "
              f"at f={worst.sensor.optics.focal_length_mm:.0f}mm")
    print()

    # =================================================================
    # 5. Multi-Parameter Sweep
    # =================================================================
    print("5. Multi-Parameter Optimization (FL x F/#)")
    print("-" * 40)

    param_results = sweep_parameters(
        fpa=fpa,
        param_ranges=[
            ParameterRange('focal_length_mm', 50, 300, steps=10),
            ParameterRange('f_number', 2.0, 5.6, steps=8),
        ],
        base_operating=operating,
        objective='identification_range_km',
        max_results=80,
    )
    print(format_results_table(param_results[:10]))
    print()

    # =================================================================
    # 6. Find Best FPA for MWIR Mission
    # =================================================================
    print("6. Best FPA Search (MWIR, f=150mm, F/2.5)")
    print("-" * 40)

    mwir_optics = OpticsConfig(focal_length_mm=150.0, f_number=2.5)
    mwir_results = find_best_fpa(
        optics=mwir_optics,
        objective='identification_range_km',
        spectral_band=SpectralBand.MWIR,
    )
    if mwir_results:
        print(format_results_table(mwir_results[:8]))
    else:
        print("   No valid MWIR configurations found")
    print()

    # =================================================================
    # 7. Constrained Optimization
    # =================================================================
    print("7. Constrained Optimization (LWIR, ID>2km, data<500Mbps)")
    print("-" * 40)

    constrained = optimize_sensor(
        spectral_band=SpectralBand.LWIR,
        focal_length_range=(80, 250),
        f_number_range=(2.0, 5.6),
        steps=6,
        objective='identification_range_km',
        constraints=[
            OptimizationConstraint('identification_range_km', min_val=2.0),
            OptimizationConstraint('data_rate_mbps', max_val=500.0),
        ],
        max_results=15,
    )
    print(f"   Found {len(constrained)} valid configurations")
    if constrained:
        print(format_results_table(constrained[:8]))
    print()

    # =================================================================
    # 8. Sensitivity Analysis
    # =================================================================
    print("8. Sensitivity Analysis (focal length +/-30%)")
    print("-" * 40)

    sa = sensitivity_analysis(
        sensor=sensor,
        parameter='focal_length_mm',
        variation_pct=30.0,
        steps=11,
    )
    print(f"   Parameter: focal_length_mm (70-130mm)")
    for metric, points in sa.items():
        vals = [v for _, v in points if v is not None]
        if vals:
            print(f"   {metric:30s}: {min(vals):.2f} - {max(vals):.2f}")
    print()

    # =================================================================
    # Visualization
    # =================================================================
    if args.no_plot:
        print("[Plots skipped (--no-plot)]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Sensor Assembly Optimization Results', fontsize=14, fontweight='bold')

    # --- Plot 1: Focal Length vs ID Range ---
    ax = axes[0, 0]
    if fl_results:
        fls = [r.sensor.optics.focal_length_mm for r in fl_results]
        id_ranges = [r.score for r in fl_results]
        ax.plot(fls, id_ranges, 'o-', color='#1f77b4', markersize=5)
        ax.set_xlabel('Focal Length (mm)')
        ax.set_ylabel('ID Range (km)')
        ax.set_title('Focal Length vs ID Range')
        ax.grid(True, alpha=0.3)

    # --- Plot 2: Multi-param heatmap (FL vs F/# -> ID range) ---
    ax = axes[0, 1]
    if param_results:
        fl_vals = sorted(set(r.sensor.optics.focal_length_mm for r in param_results))
        fn_vals = sorted(set(r.sensor.optics.f_number for r in param_results))
        grid = np.full((len(fn_vals), len(fl_vals)), np.nan)
        for r in param_results:
            fi = fl_vals.index(r.sensor.optics.focal_length_mm)
            ni = fn_vals.index(r.sensor.optics.f_number)
            grid[ni, fi] = r.score
        im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis',
                       extent=[min(fl_vals), max(fl_vals), min(fn_vals), max(fn_vals)])
        fig.colorbar(im, ax=ax, label='ID Range (km)')
        ax.set_xlabel('Focal Length (mm)')
        ax.set_ylabel('F-Number')
        ax.set_title('ID Range: FL vs F/#')

    # --- Plot 3: Best FPA ranking ---
    ax = axes[0, 2]
    if mwir_results:
        names = [r.summary['fpa'] for r in mwir_results[:8]]
        scores = [r.score for r in mwir_results[:8]]
        colors = ['#2196F3' if i == 0 else '#90CAF9' for i in range(len(names))]
        bars = ax.barh(range(len(names)), scores, color=colors,
                       edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('ID Range (km)')
        ax.set_title('Best MWIR FPAs (f=150mm, F/2.5)')
        ax.invert_yaxis()
        for bar, val in zip(bars, scores):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', fontsize=8)
        ax.grid(True, axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No MWIR FPAs found', ha='center', va='center',
                transform=ax.transAxes)

    # --- Plot 4: Constrained results comparison ---
    ax = axes[1, 0]
    if constrained:
        top = constrained[:10]
        names = [r.summary['fpa'][:15] for r in top]
        id_r = [r.summary['identification_range_km'] for r in top]
        dr = [r.summary.get('data_rate_mbps', 0) or 0 for r in top]
        x = np.arange(len(names))
        ax.bar(x - 0.2, id_r, 0.35, label='ID Range (km)', color='#2196F3', alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, dr, 0.35, label='Data Rate (Mbps)', color='#FF9800', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('ID Range (km)', color='#2196F3')
        ax2.set_ylabel('Data Rate (Mbps)', color='#FF9800')
        ax.set_title('Constrained LWIR Configs')
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No constrained results', ha='center', va='center',
                transform=ax.transAxes)

    # --- Plot 5: Sensitivity analysis ---
    ax = axes[1, 1]
    colors_sa = {'identification_range_km': '#F44336', 'ifov_urad': '#2196F3',
                 'estimated_netd_mk': '#4CAF50', 'data_rate_mbps': '#FF9800'}
    for metric, points in sa.items():
        x_vals = [x for x, v in points if v is not None]
        y_vals = [v for _, v in points if v is not None]
        if y_vals:
            # Normalize to percentage change from center
            center = y_vals[len(y_vals) // 2] if y_vals else 1
            if center and center != 0:
                y_norm = [(v / center - 1) * 100 for v in y_vals]
            else:
                y_norm = y_vals
            ax.plot(x_vals, y_norm, 'o-', label=metric.replace('_', ' '),
                    color=colors_sa.get(metric, '#333'), markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Focal Length (mm)')
    ax.set_ylabel('Change from Baseline (%)')
    ax.set_title('Sensitivity: Focal Length')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Validation summary ---
    ax = axes[1, 2]
    # Show validation pass/fail for several configs
    configs = [
        ("Valid (f=100, F/4)", OpticsConfig(100, 4.0), OperatingParams(0.5, 30)),
        ("Long tint (50ms@60Hz)", OpticsConfig(100, 4.0), OperatingParams(50, 60)),
        ("Fast F/1.0", OpticsConfig(100, 1.0), OperatingParams(0.05, 30)),
        ("Slow F/5.6", OpticsConfig(100, 5.6), OperatingParams(0.5, 30)),
        ("High rate 120Hz", OpticsConfig(100, 4.0), OperatingParams(0.5, 120)),
        ("TDI x4", OpticsConfig(100, 4.0), OperatingParams(0.5, 30, tdi_stages=4)),
    ]
    labels = []
    errors = []
    warnings = []
    for label, opt, op in configs:
        s = SensorAssembly(fpa, opt, op, validate_on_init=True)
        labels.append(label)
        errors.append(s.validation.error_count)
        warnings.append(s.validation.warning_count)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos - 0.15, errors, 0.3, label='Errors', color='#F44336', alpha=0.8)
    ax.barh(y_pos + 0.15, warnings, 0.3, label='Warnings', color='#FFC107', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Issue Count')
    ax.set_title('Validation Results')
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '42_sensor_assembly_optimization.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n   Plot saved to: {output_path}")

    print()
    print("=" * 70)
    print("Example 42 complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Assembly Optimization")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()
    main(args)
