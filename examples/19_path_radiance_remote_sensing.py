#!/usr/bin/env python3
"""
Path Radiance and Remote Sensing
================================

This example demonstrates path radiance calculations for remote
sensing applications, including:

- Upwelling radiance from surface + atmosphere
- Atmospheric correction for satellite imagery
- Apparent reflectance vs true reflectance
- Adjacency effects

Applications:
- Landsat/Sentinel atmospheric correction
- Ocean color remote sensing
- Thermal infrared remote sensing

Usage:
    python 19_path_radiance_remote_sensing.py
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.utils.constants import SOLAR_CONSTANT
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate path radiance for remote sensing"
    )
    parser.add_argument("--sza", type=float, default=30, help="Solar zenith angle (deg)")
    parser.add_argument("--vza", type=float, default=0, help="View zenith angle (deg)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="path_radiance.png")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PATH RADIANCE AND REMOTE SENSING")
    print("=" * 70)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"View zenith angle: {args.vza} deg")

    mu0 = np.cos(np.radians(args.sza))
    mu_v = np.cos(np.radians(args.vza))

    solver = TwoStreamSolver()

    # Remote sensing equation components
    print("\n" + "-" * 70)
    print("REMOTE SENSING RADIATIVE TRANSFER")
    print("-" * 70)
    print("""
At-sensor radiance L_sensor consists of:
  L_sensor = L_path + T * L_surface
           = L_path + T * (rho/pi) * E_surface

where:
  L_path    = atmospheric path radiance (scattering)
  T         = atmospheric transmittance (view path)
  L_surface = surface-leaving radiance
  rho       = surface reflectance
  E_surface = downwelling irradiance at surface
""")

    # Scenario: Different atmospheric conditions
    print("\n" + "-" * 70)
    print("SCENARIO COMPARISON")
    print("-" * 70)

    scenarios = [
        ("Clear", 0.1, 0.99, 0.0, "Molecular scattering only"),
        ("Hazy", 0.3, 0.95, 0.7, "Light aerosol loading"),
        ("Moderate", 0.5, 0.90, 0.7, "Typical conditions"),
        ("Turbid", 1.0, 0.80, 0.7, "Heavy aerosol/haze"),
    ]

    surface_reflectances = [0.05, 0.1, 0.2, 0.3, 0.5]  # Different surface types

    print(f"\n{'Condition':<12} {'tau':>6} {'T_atm':>8} {'E_sfc/E0':>10} {'Path L':>10}")
    print("-" * 55)

    for name, tau, omega, g, desc in scenarios:
        result = solver.solve_solar(
            tau=np.array([tau]),
            omega=np.array([omega]),
            g=np.array([g]),
            mu0=mu0,
            surface_albedo=0.0,  # Black surface for path radiance
            flux_toa=1.0,
            levels_surface_to_toa=True,
        )

        # Transmittance (direct + diffuse reaching surface)
        T_down = result.flux_direct[0] + result.flux_down[0]

        # Path radiance (upwelling from atmosphere over black surface)
        L_path = result.flux_up[-1]

        print(f"{name:<12} {tau:>6.2f} {omega:>8.2f} {T_down:>10.4f} {L_path:>10.4f}")

    # Atmospheric correction example
    print("\n" + "-" * 70)
    print("ATMOSPHERIC CORRECTION EXAMPLE")
    print("-" * 70)

    tau = 0.3
    omega = 0.95
    g = 0.7
    E0 = SOLAR_CONSTANT  # W/m^2

    # Calculate atmospheric parameters
    result_black = solver.solve_solar(
        tau=np.array([tau]),
        omega=np.array([omega]),
        g=np.array([g]),
        mu0=mu0,
        surface_albedo=0.0,
        flux_toa=1.0,
        levels_surface_to_toa=True,
    )

    result_white = solver.solve_solar(
        tau=np.array([tau]),
        omega=np.array([omega]),
        g=np.array([g]),
        mu0=mu0,
        surface_albedo=1.0,
        flux_toa=1.0,
        levels_surface_to_toa=True,
    )

    # Path radiance (from black surface case)
    L_path = result_black.flux_up[-1] * E0 * mu0

    # Total downwelling at surface
    E_surface = (result_black.flux_direct[0] + result_black.flux_down[0]) * E0 * mu0

    # Atmospheric transmittance (upward)
    T_up = np.exp(-tau / mu_v)

    # Spherical albedo of atmosphere
    S_atm = (result_white.flux_up[-1] - result_black.flux_up[-1]) / 1.0

    print(f"\nAtmospheric parameters (tau={tau}, omega={omega}, g={g}):")
    print(f"  Path radiance: {L_path:.2f} W/m^2/sr")
    print(f"  Surface irradiance: {E_surface:.2f} W/m^2")
    print(f"  Upward transmittance: {T_up:.4f}")
    print(f"  Spherical albedo: {S_atm:.4f}")

    # Calculate apparent vs true reflectance
    print(f"\n{'True rho':>10} {'L_surface':>12} {'L_sensor':>12} {'Apparent rho':>14} {'Error':>10}")
    print("-" * 65)

    for rho_true in surface_reflectances:
        # Surface-leaving radiance (Lambertian)
        L_surface = rho_true / np.pi * E_surface

        # At-sensor radiance
        L_sensor = L_path + T_up * L_surface / (1 - S_atm * rho_true)

        # Apparent reflectance (if no atm correction)
        rho_apparent = np.pi * L_sensor / (E0 * mu0)

        # Error
        error_pct = (rho_apparent / rho_true - 1) * 100 if rho_true > 0 else 0

        print(f"{rho_true:>10.2f} {L_surface:>12.2f} {L_sensor:>12.2f} "
              f"{rho_apparent:>14.4f} {error_pct:>9.1f}%")

    # Dark object subtraction
    print("\n" + "-" * 70)
    print("DARK OBJECT SUBTRACTION (DOS) ATMOSPHERIC CORRECTION")
    print("-" * 70)
    print("""
Simple atmospheric correction assuming:
  L_sensor = L_path + T * (rho/pi) * E_surface
  rho_corrected = (L_sensor - L_dark) * pi / (T * E_surface)
""")

    L_dark = L_path  # Assuming we identify a dark pixel
    print(f"\nDark object radiance (L_path estimate): {L_dark:.2f} W/m^2/sr")
    print(f"\n{'L_sensor':>12} {'rho_corrected':>14} {'True rho':>10} {'Error':>10}")
    print("-" * 55)

    for rho_true in [0.05, 0.1, 0.2, 0.3, 0.5]:
        L_surface = rho_true / np.pi * E_surface
        L_sensor = L_path + T_up * L_surface

        # DOS correction
        rho_corrected = (L_sensor - L_dark) * np.pi / (T_up * E_surface)
        error_pct = (rho_corrected / rho_true - 1) * 100 if rho_true > 0 else 0

        print(f"{L_sensor:>12.2f} {rho_corrected:>14.4f} {rho_true:>10.2f} {error_pct:>9.1f}%")

    # Multiple wavelength analysis
    print("\n" + "-" * 70)
    print("MULTI-SPECTRAL ANALYSIS")
    print("-" * 70)

    # Simulate different bands with different atmospheric properties
    bands = [
        ("Blue (450nm)", 0.5, 0.999, 0.0),   # Strong Rayleigh
        ("Green (550nm)", 0.3, 0.998, 0.0),  # Moderate Rayleigh
        ("Red (650nm)", 0.2, 0.995, 0.0),    # Weak Rayleigh
        ("NIR (850nm)", 0.1, 0.99, 0.0),     # Very weak
        ("SWIR (1600nm)", 0.05, 0.95, 0.0),  # Minimal
    ]

    rho_veg = 0.05  # Vegetation reflectance at blue

    print(f"\nVegetation target (low blue, high NIR):")
    print(f"{'Band':<16} {'tau':>6} {'Path L':>10} {'True rho':>10} {'Apparent':>10}")
    print("-" * 60)

    veg_reflectances = [0.03, 0.08, 0.05, 0.40, 0.25]  # Typical vegetation

    for (name, tau, omega, g), rho in zip(bands, veg_reflectances):
        result = solver.solve_solar(
            tau=np.array([tau]),
            omega=np.array([omega]),
            g=np.array([g]),
            mu0=mu0,
            surface_albedo=0.0,
            flux_toa=1.0,
            levels_surface_to_toa=True,
        )

        L_path = result.flux_up[-1]
        E_sfc = result.flux_direct[0] + result.flux_down[0]
        L_sfc = rho / np.pi * E_sfc
        L_tot = L_path + L_sfc * np.exp(-tau / mu_v)
        rho_app = np.pi * L_tot / (mu0 * 1.0)

        print(f"{name:<16} {tau:>6.2f} {L_path:>10.4f} {rho:>10.2f} {rho_app:>10.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("REMOTE SENSING SUMMARY")
    print("=" * 70)
    print("""
Key concepts for atmospheric correction:
1. Path radiance adds to signal (increases apparent reflectance)
2. Atmospheric absorption reduces signal (decreases apparent reflectance)
3. Multiple scattering couples surface and atmosphere
4. Blue band most affected by Rayleigh scattering
5. NIR/SWIR bands have minimal atmospheric effects

Correction methods:
- Dark Object Subtraction (DOS): Simple, assumes L_path = L_dark
- 6S/MODTRAN LUT: Precomputed lookup tables
- FLAASH/ATCOR: Full radiative transfer inversion
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Path Radiance and Remote Sensing',
                        fontsize=14, fontweight='bold')

            # Plot 1: At-sensor radiance components
            ax1 = axes[0, 0]
            rho_range = np.linspace(0, 0.5, 50)

            L_path_val = L_path
            L_surface_arr = rho_range / np.pi * E_surface
            L_sensor_arr = L_path_val + T_up * L_surface_arr

            ax1.plot(rho_range, [L_path_val]*len(rho_range), 'b--',
                    linewidth=2, label='Path radiance')
            ax1.plot(rho_range, T_up * L_surface_arr, 'g--',
                    linewidth=2, label='Surface contribution')
            ax1.plot(rho_range, L_sensor_arr, 'r-',
                    linewidth=2, label='At-sensor radiance')
            ax1.set_xlabel('Surface Reflectance')
            ax1.set_ylabel('Radiance (W/m^2/sr)')
            ax1.set_title('Radiance Components')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Apparent vs true reflectance
            ax2 = axes[0, 1]
            rho_apparent_arr = np.pi * L_sensor_arr / (E0 * mu0)

            ax2.plot(rho_range, rho_range, 'k--', linewidth=1, label='1:1 line')
            ax2.plot(rho_range, rho_apparent_arr, 'r-', linewidth=2, label='Apparent')
            ax2.set_xlabel('True Reflectance')
            ax2.set_ylabel('Apparent Reflectance')
            ax2.set_title('Atmospheric Effect on Reflectance')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Spectral path radiance
            ax3 = axes[1, 0]
            band_names = [b[0] for b in bands]
            tau_vals = [b[1] for b in bands]

            path_radiances = []
            for name, tau, omega, g in bands:
                result = solver.solve_solar(
                    tau=np.array([tau]),
                    omega=np.array([omega]),
                    g=np.array([g]),
                    mu0=mu0,
                    surface_albedo=0.0,
                    flux_toa=1.0,
                    levels_surface_to_toa=True,
                )
                path_radiances.append(result.flux_up[-1])

            ax3.bar(range(len(bands)), path_radiances, color='blue', alpha=0.7)
            ax3.set_xticks(range(len(bands)))
            ax3.set_xticklabels([b[0].split()[0] for b in bands])
            ax3.set_ylabel('Relative Path Radiance')
            ax3.set_title('Spectral Path Radiance')
            ax3.grid(True, alpha=0.3, axis='y')

            # Plot 4: Vegetation spectrum
            ax4 = axes[1, 1]
            wavelengths = [450, 550, 650, 850, 1600]

            ax4.plot(wavelengths, veg_reflectances, 'go-',
                    linewidth=2, markersize=8, label='True')

            apparent = []
            for (name, tau, omega, g), rho in zip(bands, veg_reflectances):
                result = solver.solve_solar(
                    tau=np.array([tau]),
                    omega=np.array([omega]),
                    g=np.array([g]),
                    mu0=mu0,
                    surface_albedo=0.0,
                    flux_toa=1.0,
                    levels_surface_to_toa=True,
                )
                L_path = result.flux_up[-1]
                E_sfc = result.flux_direct[0] + result.flux_down[0]
                L_sfc = rho / np.pi * E_sfc
                L_tot = L_path + L_sfc * np.exp(-tau / mu_v)
                apparent.append(np.pi * L_tot / (mu0 * 1.0))

            ax4.plot(wavelengths, apparent, 'ro-',
                    linewidth=2, markersize=8, label='Apparent')
            ax4.set_xlabel('Wavelength (nm)')
            ax4.set_ylabel('Reflectance')
            ax4.set_title('Vegetation: True vs Apparent Spectrum')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
