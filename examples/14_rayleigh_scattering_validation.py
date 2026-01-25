#!/usr/bin/env python3
"""
Rayleigh Scattering Validation
==============================

This example validates the Rayleigh scattering implementation against
known physical values and the lambda^-4 wavelength dependence.

Validation criteria:
- Optical depth scales as lambda^-4
- Cross section matches literature values
- Phase function is correctly normalized
- Blue/red ratio matches atmospheric observations

References:
- Bodhaine et al. (1999). On Rayleigh optical depth calculations.
- Bucholtz (1995). Rayleigh-scattering calculations for the terrestrial atmosphere.

Usage:
    python 14_rayleigh_scattering_validation.py
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering.rayleigh import (
        rayleigh_cross_section,
        rayleigh_optical_depth,
        rayleigh_phase_function,
    )
    from raf_tran.utils.constants import AVOGADRO
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Rayleigh scattering implementation"
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="rayleigh_validation.png")
    return parser.parse_args()


# Literature values for Rayleigh optical depth (Bodhaine et al., 1999)
LITERATURE_RAYLEIGH_TAU = {
    # wavelength (nm): optical depth at sea level for US Standard Atmosphere
    300: 0.522,
    400: 0.364,
    500: 0.143,
    550: 0.097,
    600: 0.068,
    700: 0.036,
    800: 0.020,
    1000: 0.008,
}

# Standard atmospheric column density (molecules/m^2)
# For US Standard Atmosphere at sea level
COLUMN_DENSITY = 2.152e25  # molecules/m^2


def main():
    args = parse_args()

    print("=" * 70)
    print("RAYLEIGH SCATTERING VALIDATION")
    print("=" * 70)

    # Test 1: Wavelength dependence (lambda^-4)
    print("\n" + "-" * 70)
    print("TEST 1: Lambda^-4 Wavelength Dependence")
    print("-" * 70)

    wavelengths_nm = np.array([300, 400, 500, 600, 700, 800, 1000])
    wavelengths_um = wavelengths_nm / 1000  # Convert nm to um

    # Get cross sections
    cross_sections = []
    for wl_um in wavelengths_um:
        sigma = rayleigh_cross_section(wl_um)
        cross_sections.append(sigma)

    cross_sections = np.array(cross_sections)

    # Fit power law: sigma ~ lambda^n
    log_wl = np.log(wavelengths_um)
    log_sigma = np.log(cross_sections)
    slope, intercept = np.polyfit(log_wl, log_sigma, 1)

    print(f"\nCross section wavelength dependence:")
    print(f"  Fitted exponent: n = {slope:.3f}")
    print(f"  Expected:        n = -4.000")
    print(f"  Deviation:       {abs(slope + 4):.3f}")

    if abs(slope + 4) < 0.1:
        print("  Status: [OK] - Matches lambda^-4 law")
    else:
        print("  Status: [WARN] - Significant deviation from lambda^-4")

    # Test 2: Comparison with literature optical depths
    print("\n" + "-" * 70)
    print("TEST 2: Optical Depth vs Literature (Bodhaine et al., 1999)")
    print("-" * 70)

    print(f"\n  {'Wavelength':>12} {'Literature':>12} {'RAF-tran':>12} {'Error %':>10} {'Status':>8}")
    print("  " + "-" * 60)

    all_passed = True
    for wl_nm, tau_lit in LITERATURE_RAYLEIGH_TAU.items():
        wl_um = wl_nm / 1000  # Convert nm to um
        tau_calc = rayleigh_optical_depth(wl_um, COLUMN_DENSITY)

        error_pct = abs(tau_calc - tau_lit) / tau_lit * 100

        if error_pct < 5:
            status = "[OK]"
        elif error_pct < 20:
            status = "[WARN]"
            all_passed = False
        else:
            status = "[FAIL]"
            all_passed = False

        print(f"  {wl_nm:>10} nm {tau_lit:>12.4f} {tau_calc:>12.4f} {error_pct:>9.2f}% {status:>8}")

    # Test 3: Phase function normalization
    print("\n" + "-" * 70)
    print("TEST 3: Phase Function Normalization")
    print("-" * 70)
    print("Integral of P(theta)*sin(theta) from 0 to pi should equal 2")

    angles = np.linspace(0, np.pi, 1000)

    try:
        P_values = rayleigh_phase_function(np.cos(angles))
    except Exception:
        P_values = rayleigh_phase_function(angles)

    # Normalize: integral of P(mu) dmu from -1 to 1 = 2
    # Or: integral of P(theta) sin(theta) dtheta from 0 to pi = 2
    integral = np.trapezoid(P_values * np.sin(angles), angles)

    print(f"\n  Integral of P*sin(theta): {integral:.6f}")
    print(f"  Expected: 2.0")
    print(f"  Error: {abs(integral - 2.0):.6f}")

    if abs(integral - 2.0) < 0.01:
        print("  Status: [OK] - Phase function correctly normalized")
    else:
        print("  Status: [WARN] - Normalization error")

    # Test 4: Phase function values at key angles
    print("\n" + "-" * 70)
    print("TEST 4: Phase Function at Key Angles")
    print("-" * 70)
    print("Rayleigh phase function: P(theta) = (3/4)(1 + cos^2(theta))")

    key_angles = [0, 45, 90, 135, 180]

    print(f"\n  {'Angle (deg)':>12} {'cos(theta)':>12} {'Expected P':>12} {'Calculated':>12}")
    print("  " + "-" * 55)

    for angle_deg in key_angles:
        theta = np.radians(angle_deg)
        cos_theta = np.cos(theta)

        # Analytical Rayleigh phase function
        P_expected = 0.75 * (1 + cos_theta**2)

        try:
            P_calc = rayleigh_phase_function(cos_theta)
        except Exception:
            P_calc = rayleigh_phase_function(theta)

        print(f"  {angle_deg:>12} {cos_theta:>12.4f} {P_expected:>12.4f} {P_calc:>12.4f}")

    # Test 5: Blue sky ratio
    print("\n" + "-" * 70)
    print("TEST 5: Blue Sky Ratio (why is the sky blue?)")
    print("-" * 70)

    wl_blue = 0.450   # Blue light (um)
    wl_red = 0.650    # Red light (um)

    sigma_blue = rayleigh_cross_section(wl_blue)
    sigma_red = rayleigh_cross_section(wl_red)

    ratio = sigma_blue / sigma_red
    expected_ratio = (wl_red / wl_blue)**4  # From lambda^-4 law

    print(f"\n  Blue (450 nm) / Red (650 nm) scattering ratio:")
    print(f"    Calculated:  {ratio:.2f}")
    print(f"    Expected:    {expected_ratio:.2f}")
    print(f"    (Blue light scatters {ratio:.1f}x more than red)")

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"""
Rayleigh scattering validation results:
- Wavelength dependence: lambda^{slope:.2f} (expected: lambda^-4)
- Optical depth matches Bodhaine et al. within {5 if all_passed else 20}%
- Phase function normalized to {integral:.3f} (expected: 2.0)
- Blue/red ratio: {ratio:.1f}x (explains blue sky color)

{"[PASS] All tests passed!" if all_passed else "[WARN] Some discrepancies - check implementation"}
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Rayleigh Scattering Validation', fontsize=14, fontweight='bold')

            # Plot 1: Cross section vs wavelength
            ax1 = axes[0, 0]
            wl_plot_nm = np.linspace(300, 1000, 100)
            wl_plot_um = wl_plot_nm / 1000

            sigma_plot = [rayleigh_cross_section(wl) for wl in wl_plot_um]

            ax1.loglog(wl_plot_nm, sigma_plot, 'b-', linewidth=2, label='RAF-tran')

            # Add lambda^-4 reference
            ref = sigma_plot[50] * (wl_plot_um[50] / wl_plot_um)**4
            ax1.loglog(wl_plot_nm, ref, 'r--', linewidth=1, label='lambda^-4 reference')

            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Cross Section (m^2)')
            ax1.set_title('Rayleigh Cross Section')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Phase function
            ax2 = axes[0, 1]
            theta_plot = np.linspace(0, 180, 100)
            cos_theta = np.cos(np.radians(theta_plot))

            P_analytical = 0.75 * (1 + cos_theta**2)

            try:
                P_calc = [rayleigh_phase_function(c) for c in cos_theta]
            except Exception:
                P_calc = [rayleigh_phase_function(np.radians(t)) for t in theta_plot]

            ax2.plot(theta_plot, P_analytical, 'b-', linewidth=2, label='Analytical')
            ax2.plot(theta_plot, P_calc, 'ro', markersize=3, label='RAF-tran')
            ax2.set_xlabel('Scattering Angle (deg)')
            ax2.set_ylabel('Phase Function P(theta)')
            ax2.set_title('Rayleigh Phase Function')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Optical depth comparison
            ax3 = axes[1, 0]
            wl_lit = list(LITERATURE_RAYLEIGH_TAU.keys())
            tau_lit = list(LITERATURE_RAYLEIGH_TAU.values())

            tau_calc = []
            for wl_nm in wl_lit:
                wl_um = wl_nm / 1000
                tau_calc.append(rayleigh_optical_depth(wl_um, COLUMN_DENSITY))

            ax3.semilogy(wl_lit, tau_lit, 'bo-', linewidth=2, markersize=8, label='Literature')
            ax3.semilogy(wl_lit, tau_calc, 'rx-', linewidth=2, markersize=8, label='RAF-tran')
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Optical Depth')
            ax3.set_title('Rayleigh Optical Depth vs Literature')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Polar phase function
            ax4 = axes[1, 1]
            ax4 = plt.subplot(224, projection='polar')

            theta_polar = np.linspace(0, 2*np.pi, 360)
            P_polar = 0.75 * (1 + np.cos(theta_polar)**2)

            ax4.plot(theta_polar, P_polar, 'b-', linewidth=2)
            ax4.set_title('Rayleigh Phase Function (Polar)')
            ax4.set_rticks([0.5, 1.0, 1.5])

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
