#!/usr/bin/env python3
"""
Example 36: HITRAN Gas Absorption (Optional High-Fidelity Mode)
===============================================================

This example demonstrates the optional HITRAN/HAPI integration for
high-fidelity line-by-line gas absorption calculations.

OFFLINE OPERATION
-----------------
RAF-tran works fully offline using the built-in correlated-k (CKD) method.
HITRAN integration is OPTIONAL and only needed when ~1% accuracy is required
(vs ~5% for the default CKD method).

Features demonstrated:
1. Checking HITRAN availability
2. Fallback to CKD method when HAPI not installed
3. Comparing CKD vs HITRAN (when available)
4. Offline operation guarantee

Usage:
    python examples/36_hitran_gas_absorption.py [--no-plot]
"""

import argparse
import sys
import numpy as np

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RAF-tran imports
from raf_tran.gas_optics import (
    GasOptics,
    CKDTable,
    create_simple_ckd_table,
    HAPI_AVAILABLE,
    HITRANGasOptics,
    can_run_offline,
)
from raf_tran.gas_optics.ckd import compute_optical_depth


def main(args):
    print("=" * 70)
    print("Example 36: HITRAN Gas Absorption (Optional High-Fidelity)")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # 1. Check Offline Capability
    # ---------------------------------------------------------------------
    print("1. Offline Capability Check")
    print("-" * 40)

    print(f"   Can run offline: {can_run_offline()}")
    print(f"   HAPI available: {HAPI_AVAILABLE}")

    if HAPI_AVAILABLE:
        print("   Mode: HITRAN line-by-line available (~1% accuracy)")
    else:
        print("   Mode: Correlated-k method (~5% accuracy)")
        print("   Note: Install HAPI for higher accuracy: pip install hitran-api")

    print()

    # ---------------------------------------------------------------------
    # 2. Create CKD Gas Optics (Always Available)
    # ---------------------------------------------------------------------
    print("2. Correlated-k (CKD) Method - Always Available")
    print("-" * 40)

    # Create simple CKD tables for common atmospheric gases
    molecules = ['H2O', 'CO2', 'O3']

    ckd_optics = GasOptics()
    for mol in molecules:
        table = create_simple_ckd_table(
            gas_name=mol,
            wavenumber_bounds=(500, 2500),  # cm^-1 (4-20 um)
            n_g_points=8,
        )
        ckd_optics.add_gas(table)
        print(f"   Created CKD table for {mol}: {table.n_g_points} g-points")

    print()

    # ---------------------------------------------------------------------
    # 3. Demonstrate CKD Optical Depth Calculation
    # ---------------------------------------------------------------------
    print("3. CKD Optical Depth Calculation")
    print("-" * 40)

    # Example atmospheric layer
    n_layers = 20
    pressure = np.linspace(101325, 10000, n_layers)  # Pa
    temperature = np.linspace(288, 220, n_layers)  # K
    dz = np.ones(n_layers) * 1000  # 1 km layers

    # Number density from ideal gas law
    k_B = 1.38e-23  # Boltzmann constant
    number_density = pressure / (k_B * temperature)

    # Volume mixing ratios
    vmr = {
        'H2O': 0.01 * np.exp(-np.linspace(0, 10, n_layers)),  # Decreasing with altitude
        'CO2': np.ones(n_layers) * 420e-6,  # 420 ppmv
        'O3': 5e-6 * np.exp(-((np.linspace(0, 50, n_layers) - 25)**2) / 100),  # Peak at 25 km
    }

    # Compute optical depth
    tau, g_weights = ckd_optics.compute_optical_depth(
        pressure=pressure,
        temperature=temperature,
        vmr=vmr,
        dz=dz,
        number_density=number_density,
    )

    print(f"   Computed optical depth shape: {tau.shape}")
    print(f"   Total column optical depth: {tau.sum():.4f}")
    print(f"   G-point weights: {g_weights}")
    print()

    # ---------------------------------------------------------------------
    # 4. HITRAN Integration (If Available)
    # ---------------------------------------------------------------------
    print("4. HITRAN Integration Status")
    print("-" * 40)

    if HAPI_AVAILABLE:
        print("   HITRAN/HAPI is available!")
        print("   Creating HITRANGasOptics with fallback enabled...")

        hitran_optics = HITRANGasOptics(
            molecules=['H2O', 'CO2'],
            fallback_to_ckd=True,
        )

        print(f"   Using HITRAN: {hitran_optics.using_hitran}")

        # Note: Actual HITRAN calculations require downloaded data
        print("   Note: HITRAN data must be downloaded before use.")
        print("   Download with: hitran.download_hitran_data('H2O', 1000, 2000)")
    else:
        print("   HITRAN/HAPI not installed.")
        print("   The simulation will use CKD method (works offline).")
        print()
        print("   To enable HITRAN:")
        print("   1. pip install hitran-api")
        print("   2. Download data for your spectral region")
        print("   3. Data is cached locally for offline use")

    print()

    # ---------------------------------------------------------------------
    # 5. Comparison: CKD vs Analytical
    # ---------------------------------------------------------------------
    print("5. CKD Method Validation")
    print("-" * 40)

    # Compare with simple Beer-Lambert for validation
    # Using H2O as example
    wavelength_um = 10.0  # LWIR
    wavenumber = 10000 / wavelength_um  # cm^-1

    # Simple absorption coefficient estimate
    # Real values would come from HITRAN
    k_simple = 1e-22  # m^2/molecule (order of magnitude)

    # Column amount for H2O
    h2o_column = np.sum(vmr['H2O'] * number_density * dz)  # molecules/m^2
    tau_simple = k_simple * h2o_column

    print(f"   Wavelength: {wavelength_um} um ({wavenumber:.0f} cm^-1)")
    print(f"   H2O column: {h2o_column:.2e} molecules/m^2")
    print(f"   Simple tau estimate: {tau_simple:.4f}")
    print(f"   CKD total tau: {tau.sum():.4f}")
    print()
    print("   Note: CKD method averages over spectral band,")
    print("   while simple estimate is for single wavelength.")

    print()

    # ---------------------------------------------------------------------
    # 6. Visualization
    # ---------------------------------------------------------------------
    if not args.no_plot:
        print("6. Creating Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: VMR profiles
        ax1 = axes[0, 0]
        altitudes = np.linspace(0, 50, n_layers)
        ax1.semilogy(vmr['H2O'] * 1e6, altitudes, 'b-', label='H2O', linewidth=2)
        ax1.semilogy(vmr['CO2'] * 1e6, altitudes, 'r-', label='CO2', linewidth=2)
        ax1.semilogy(vmr['O3'] * 1e6, altitudes, 'g-', label='O3', linewidth=2)
        ax1.set_xlabel('Volume Mixing Ratio (ppmv)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Gas Concentration Profiles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Optical depth by g-point
        ax2 = axes[0, 1]
        for i in range(tau.shape[1]):
            ax2.plot(tau[:, i], altitudes, label=f'g={i+1}', alpha=0.7)
        ax2.set_xlabel('Layer Optical Depth')
        ax2.set_ylabel('Altitude (km)')
        ax2.set_title('Optical Depth by G-point')
        ax2.legend(ncol=2, fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Cumulative optical depth
        ax3 = axes[1, 0]
        tau_cumulative = np.cumsum(tau.sum(axis=1))
        ax3.plot(tau_cumulative, altitudes, 'k-', linewidth=2)
        ax3.set_xlabel('Cumulative Optical Depth')
        ax3.set_ylabel('Altitude (km)')
        ax3.set_title('Cumulative Optical Depth (All Gases)')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Method comparison
        ax4 = axes[1, 1]
        methods = ['CKD Method\n(Default)', 'HITRAN LBL\n(Optional)']
        accuracy = [5, 1]  # % error
        colors = ['#2ecc71', '#3498db']
        availability = ['Always\n(Offline)', 'Requires\nHAPI']

        bars = ax4.bar(methods, accuracy, color=colors, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Typical Error (%)')
        ax4.set_title('Absorption Method Comparison')
        ax4.set_ylim(0, 7)

        # Add availability labels
        for bar, avail in zip(bars, availability):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    avail, ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('outputs/36_hitran_gas_absorption.png', dpi=150, bbox_inches='tight')
        print("   Saved: outputs/36_hitran_gas_absorption.png")
        plt.close()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    RAF-tran Gas Optics:

    1. DEFAULT MODE (CKD - Always Available):
       - Works completely offline
       - ~5% accuracy for broadband calculations
       - Fast computation using g-point quadrature
       - Suitable for most applications

    2. OPTIONAL MODE (HITRAN - Requires HAPI):
       - ~1% accuracy with line-by-line calculations
       - Requires: pip install hitran-api
       - Data cached locally after first download
       - Best for high-resolution spectroscopy

    Key Point: The simulation ALWAYS works offline using CKD method.
    HITRAN is an optional enhancement for specialized applications.
    """)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HITRAN Gas Absorption Example')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    # Ensure output directory exists
    import os
    os.makedirs('outputs', exist_ok=True)

    sys.exit(main(args))
