#!/usr/bin/env python3
"""
Beer-Lambert Law Validation
============================

This example validates the fundamental Beer-Lambert law of absorption:
    I = I_0 * exp(-tau * m)

where tau is optical depth and m is air mass.

We compare RAF-tran solver results against analytical Beer-Lambert predictions
to verify the core physics implementation.

Validation criteria:
- Direct beam transmission must match exp(-tau/mu0) exactly
- Error should be < 0.01% for optically thin cases
- Error should be < 1% for optically thick cases

Usage:
    python 12_beer_lambert_validation.py
    python 12_beer_lambert_validation.py --help
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.rte_solver import TwoStreamSolver
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Beer-Lambert law implementation"
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="beer_lambert_validation.png")
    return parser.parse_args()


def beer_lambert_analytical(tau, mu0):
    """Analytical Beer-Lambert transmission."""
    return np.exp(-tau / mu0)


def main():
    args = parse_args()

    print("=" * 70)
    print("BEER-LAMBERT LAW VALIDATION")
    print("=" * 70)

    solver = TwoStreamSolver()

    # Test parameters
    tau_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    sza_values = [0, 30, 45, 60, 75]

    print("\n" + "-" * 70)
    print("TEST 1: Direct Beam Transmission vs Optical Depth")
    print("-" * 70)

    all_passed = True
    results = []

    for sza in sza_values:
        mu0 = np.cos(np.radians(sza))
        print(f"\nSZA = {sza} deg (mu0 = {mu0:.4f}):")
        print(f"  {'tau':>8} {'Analytical':>12} {'RAF-tran':>12} {'Error %':>10} {'Status':>8}")
        print("  " + "-" * 55)

        for tau in tau_values:
            # Analytical prediction
            T_analytical = beer_lambert_analytical(tau, mu0)

            # RAF-tran solver (pure absorption, no scattering)
            result = solver.solve_solar(
                tau=np.array([tau]),
                omega=np.array([0.0]),  # No scattering
                g=np.array([0.0]),
                mu0=mu0,
                surface_albedo=0.0,
                flux_toa=1.0,
                levels_surface_to_toa=True,
            )

            # Direct transmission
            T_solver = result.flux_direct[0]  # At surface

            # Calculate error
            if T_analytical > 1e-10:
                error_pct = abs(T_solver - T_analytical) / T_analytical * 100
            else:
                error_pct = abs(T_solver - T_analytical) * 100

            # Check tolerance
            if tau < 1:
                tolerance = 0.01  # 0.01% for thin
            else:
                tolerance = 1.0   # 1% for thick

            passed = error_pct < tolerance
            status = "[OK]" if passed else "[FAIL]"
            if not passed:
                all_passed = False

            results.append((sza, tau, T_analytical, T_solver, error_pct, passed))

            print(f"  {tau:>8.2f} {T_analytical:>12.6e} {T_solver:>12.6e} "
                  f"{error_pct:>9.4f}% {status:>8}")

    # Test 2: Air mass scaling
    print("\n" + "-" * 70)
    print("TEST 2: Air Mass Scaling (T should scale as exp(-tau*m))")
    print("-" * 70)

    tau_fixed = 0.5
    print(f"\nFixed tau = {tau_fixed}")
    print(f"  {'SZA':>6} {'Air Mass':>10} {'Expected T':>12} {'Actual T':>12} {'Error %':>10}")
    print("  " + "-" * 55)

    for sza in [0, 20, 40, 60, 70, 80]:
        mu0 = np.cos(np.radians(sza))
        air_mass = 1.0 / mu0

        T_expected = np.exp(-tau_fixed * air_mass)

        result = solver.solve_solar(
            tau=np.array([tau_fixed]),
            omega=np.array([0.0]),
            g=np.array([0.0]),
            mu0=mu0,
            surface_albedo=0.0,
            flux_toa=1.0,
            levels_surface_to_toa=True,
        )
        T_actual = result.flux_direct[0]

        error_pct = abs(T_actual - T_expected) / T_expected * 100
        print(f"  {sza:>6} {air_mass:>10.3f} {T_expected:>12.6e} {T_actual:>12.6e} {error_pct:>9.4f}%")

    # Test 3: Multi-layer consistency
    print("\n" + "-" * 70)
    print("TEST 3: Multi-Layer Consistency")
    print("-" * 70)
    print("(Same total tau split into N layers should give same transmission)")

    tau_total = 2.0
    mu0 = 0.5
    n_layers_list = [1, 2, 5, 10, 20, 50]

    T_expected = np.exp(-tau_total / mu0)
    print(f"\nTotal tau = {tau_total}, mu0 = {mu0}")
    print(f"Expected transmission: {T_expected:.6e}")
    print(f"\n  {'N layers':>10} {'T (direct)':>14} {'Error %':>10}")
    print("  " + "-" * 40)

    for n_layers in n_layers_list:
        tau_per_layer = np.ones(n_layers) * tau_total / n_layers

        result = solver.solve_solar(
            tau=tau_per_layer,
            omega=np.zeros(n_layers),
            g=np.zeros(n_layers),
            mu0=mu0,
            surface_albedo=0.0,
            flux_toa=1.0,
            levels_surface_to_toa=True,
        )
        T_actual = result.flux_direct[0]
        error_pct = abs(T_actual - T_expected) / T_expected * 100

        print(f"  {n_layers:>10} {T_actual:>14.6e} {error_pct:>9.4f}%")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    n_passed = sum(1 for r in results if r[5])
    n_total = len(results)

    if all_passed:
        print(f"\n[PASS] All {n_total} tests passed!")
        print("Beer-Lambert law is correctly implemented.")
    else:
        print(f"\n[WARN] {n_passed}/{n_total} tests passed")
        print("Some tests exceeded tolerance - check implementation.")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Beer-Lambert Law Validation', fontsize=14, fontweight='bold')

            # Plot 1: Transmission vs optical depth
            ax1 = axes[0]
            tau_plot = np.linspace(0.01, 10, 100)

            for sza in [0, 30, 60]:
                mu0 = np.cos(np.radians(sza))
                T = np.exp(-tau_plot / mu0)
                ax1.semilogy(tau_plot, T, '-', linewidth=2, label=f'SZA={sza} deg')

            ax1.set_xlabel('Optical Depth (tau)')
            ax1.set_ylabel('Transmission')
            ax1.set_title('Beer-Lambert: T = exp(-tau/mu0)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 10)

            # Plot 2: Error analysis
            ax2 = axes[1]
            for sza in sza_values:
                sza_results = [r for r in results if r[0] == sza]
                taus = [r[1] for r in sza_results]
                errors = [r[4] for r in sza_results]
                ax2.semilogy(taus, errors, 'o-', label=f'SZA={sza} deg')

            ax2.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='0.01% tolerance')
            ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.5, label='1% tolerance')
            ax2.set_xlabel('Optical Depth (tau)')
            ax2.set_ylabel('Relative Error (%)')
            ax2.set_title('Solver Error vs Analytical')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
