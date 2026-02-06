#!/usr/bin/env python3
"""
Atmospheric Visibility and Contrast
====================================

This example calculates atmospheric visibility and contrast reduction
due to scattering and absorption.

Key concepts:
- Visibility (meteorological range)
- Contrast reduction (Koschmieder's law)
- Target detection range
- Color shift with distance

Applications:
- Aviation visibility
- Target detection
- Photography through atmosphere
- Air quality assessment

Usage:
    python 20_visibility_contrast.py
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering.rayleigh import rayleigh_optical_depth
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate visibility and contrast reduction"
    )
    parser.add_argument("--visibility", type=float, default=23,
                       help="Meteorological visibility in km (default: 23 km)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="visibility_contrast.png")
    return parser.parse_args()


def extinction_coefficient(visibility_km, wavelength_nm=550):
    """
    Calculate extinction coefficient from visibility.

    Koschmieder formula: V = 3.912 / beta
    where V is visibility and beta is extinction coefficient.
    """
    return 3.912 / visibility_km  # km^-1


def contrast_transmission(distance_km, beta):
    """
    Contrast transmission (Koschmieder's law).

    C(r) / C(0) = exp(-beta * r)
    """
    return np.exp(-beta * distance_km)


def apparent_contrast(inherent_contrast, distance_km, beta):
    """
    Apparent contrast of target against background.

    C_apparent = C_inherent * exp(-beta * r)
    """
    return inherent_contrast * contrast_transmission(distance_km, beta)


def minimum_resolvable_contrast():
    """
    Minimum contrast detectable by human eye.
    Typically 0.02-0.05 (2-5%).
    """
    return 0.02


def maximum_detection_range(inherent_contrast, beta, threshold=0.02):
    """
    Maximum range at which target can be detected.

    r_max = -ln(C_threshold / C_inherent) / beta
    """
    if inherent_contrast <= threshold:
        return 0
    return -np.log(threshold / inherent_contrast) / beta


def path_radiance_normalized(distance_km, beta):
    """
    Normalized path radiance (airlight).

    L_path / L_horizon = 1 - exp(-beta * r)
    """
    return 1 - np.exp(-beta * distance_km)


def main():
    args = parse_args()

    print("=" * 70)
    print("ATMOSPHERIC VISIBILITY AND CONTRAST")
    print("=" * 70)
    print(f"\nMeteorological visibility: {args.visibility} km")

    beta = extinction_coefficient(args.visibility)
    print(f"Extinction coefficient: {beta:.4f} km^-1")

    # Visibility conditions
    print("\n" + "-" * 70)
    print("VISIBILITY CONDITIONS")
    print("-" * 70)

    conditions = [
        ("Exceptionally clear", 50),
        ("Very clear", 30),
        ("Clear", 23),
        ("Light haze", 10),
        ("Haze", 5),
        ("Thin fog", 2),
        ("Moderate fog", 1),
        ("Thick fog", 0.5),
        ("Dense fog", 0.1),
    ]

    print(f"\n{'Condition':<20} {'Visibility (km)':>15} {'beta (km^-1)':>12}")
    print("-" * 55)

    for name, vis in conditions:
        b = extinction_coefficient(vis)
        marker = " <--" if abs(vis - args.visibility) < 0.5 else ""
        print(f"{name:<20} {vis:>15.1f} {b:>12.4f}{marker}")

    # Contrast reduction with distance
    print("\n" + "-" * 70)
    print("CONTRAST REDUCTION WITH DISTANCE")
    print("-" * 70)
    print("Black target (C0 = 1.0) against horizon sky")

    distances = [0.1, 0.5, 1, 2, 5, 10, 20, 50]

    print(f"\n{'Distance (km)':>14} {'T_contrast':>12} {'C_apparent':>12} {'% Remaining':>12}")
    print("-" * 55)

    for d in distances:
        if d > args.visibility * 2:
            continue
        T = contrast_transmission(d, beta)
        C = apparent_contrast(1.0, d, beta)
        print(f"{d:>14.1f} {T:>12.4f} {C:>12.4f} {C*100:>11.1f}%")

    # Detection range for different targets
    print("\n" + "-" * 70)
    print("TARGET DETECTION RANGE")
    print("-" * 70)
    print(f"Detection threshold: {minimum_resolvable_contrast()*100:.0f}% contrast")

    targets = [
        ("Black object on white", 1.0),
        ("Dark gray on light gray", 0.5),
        ("Low contrast camouflage", 0.2),
        ("Minimal contrast", 0.05),
    ]

    print(f"\n{'Target Description':<25} {'Inherent C':>12} {'Max Range (km)':>15}")
    print("-" * 55)

    for name, C0 in targets:
        r_max = maximum_detection_range(C0, beta)
        print(f"{name:<25} {C0:>12.2f} {r_max:>15.1f}")

    # Wavelength dependence
    print("\n" + "-" * 70)
    print("WAVELENGTH DEPENDENCE OF VISIBILITY")
    print("-" * 70)
    print("Rayleigh scattering causes blue light to scatter more")

    wavelengths = [400, 450, 500, 550, 600, 650, 700]

    print(f"\n{'Wavelength (nm)':>15} {'Relative tau':>12} {'Visibility (km)':>15}")
    print("-" * 50)

    # Reference at 550 nm
    tau_ref = 0.097  # Rayleigh optical depth at 550 nm

    for wl in wavelengths:
        # Rayleigh scales as lambda^-4
        tau = tau_ref * (550 / wl)**4
        # Effective visibility if only Rayleigh
        vis_rayleigh = 3.912 / (tau / 8.5)  # Assuming 8.5 km scale height

        print(f"{wl:>15} {tau/tau_ref:>12.2f} {vis_rayleigh:>15.1f}")

    # Airlight (path radiance) buildup
    print("\n" + "-" * 70)
    print("AIRLIGHT (PATH RADIANCE) BUILDUP")
    print("-" * 70)
    print("Scattered light adds to apparent brightness")

    print(f"\n{'Distance (km)':>14} {'Target Signal':>14} {'Airlight':>12} {'Total':>12}")
    print("-" * 60)

    for d in [0.1, 1, 5, 10, 20]:
        if d > args.visibility * 2:
            continue
        signal = np.exp(-beta * d)  # Direct transmission
        airlight = 1 - signal       # Scattered light
        total = signal + airlight   # Total (normalized)

        print(f"{d:>14.1f} {signal:>14.4f} {airlight:>12.4f} {total:>12.4f}")

    # Color shift with distance
    print("\n" + "-" * 70)
    print("COLOR SHIFT WITH DISTANCE (Blue Haze)")
    print("-" * 70)

    # Calculate RGB transmission at different distances
    wavelengths_rgb = {"Red (650nm)": 650, "Green (550nm)": 550, "Blue (450nm)": 450}

    print(f"\n{'Distance':>10}", end="")
    for name in wavelengths_rgb:
        print(f" {name:>14}", end="")
    print("")
    print("-" * 60)

    for d in [1, 5, 10, 20]:
        if d > args.visibility * 2:
            continue
        print(f"{d:>10.0f} km", end="")
        for name, wl in wavelengths_rgb.items():
            # Wavelength-dependent beta (Rayleigh component)
            beta_wl = beta * (550 / wl)**2  # Approximate scaling
            T = np.exp(-beta_wl * d)
            print(f" {T:>14.3f}", end="")
        print("")

    print("""
Note: Blue light scatters more (lower transmission at distance).
This causes distant objects to appear bluish (atmospheric blue haze).
Mountains at large distances appear blue due to this effect.
""")

    # Summary
    print("\n" + "=" * 70)
    print("VISIBILITY SUMMARY")
    print("=" * 70)
    print(f"""
For visibility = {args.visibility} km:
- Extinction coefficient beta = {beta:.4f} km^-1
- At 1 km: {contrast_transmission(1, beta)*100:.1f}% contrast remaining
- At visibility range: {contrast_transmission(args.visibility, beta)*100:.1f}% contrast
- Black target detection range: {maximum_detection_range(1.0, beta):.1f} km

Key relationships:
- Koschmieder: Visibility = 3.912 / beta
- Contrast: C(r) = C(0) * exp(-beta * r)
- Airlight: L_path = L_horizon * (1 - exp(-beta * r))
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Visibility and Contrast (V = {args.visibility} km)',
                        fontsize=14, fontweight='bold')

            # Plot 1: Contrast vs distance
            ax1 = axes[0, 0]
            d_plot = np.linspace(0, args.visibility * 2, 100)
            C_plot = contrast_transmission(d_plot, beta)

            ax1.plot(d_plot, C_plot, 'b-', linewidth=2)
            ax1.axhline(0.02, color='red', linestyle='--', label='Detection threshold')
            ax1.axvline(args.visibility, color='green', linestyle='--', label='Visibility')
            ax1.set_xlabel('Distance (km)')
            ax1.set_ylabel('Contrast')
            ax1.set_title('Contrast Reduction with Distance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)

            # Plot 2: Signal vs Airlight
            ax2 = axes[0, 1]
            signal = np.exp(-beta * d_plot)
            airlight = 1 - signal

            ax2.fill_between(d_plot, 0, signal, alpha=0.5, color='blue', label='Target signal')
            ax2.fill_between(d_plot, signal, 1, alpha=0.5, color='gray', label='Airlight')
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Normalized Radiance')
            ax2.set_title('Signal vs Airlight')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Visibility for different conditions
            ax3 = axes[1, 0]
            vis_range = np.array([c[1] for c in conditions if c[1] > 0.05])
            beta_range = 3.912 / vis_range

            ax3.loglog(vis_range, beta_range, 'bo-', linewidth=2, markersize=8)
            ax3.axvline(args.visibility, color='red', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Visibility (km)')
            ax3.set_ylabel('Extinction Coefficient (km^-1)')
            ax3.set_title('Beta vs Visibility (Koschmieder)')
            ax3.grid(True, alpha=0.3)

            # Add condition labels
            for name, vis in conditions:
                if vis in vis_range:
                    ax3.annotate(name, (vis, 3.912/vis),
                                textcoords="offset points", xytext=(5, 5),
                                fontsize=7, rotation=45)

            # Plot 4: RGB transmission
            ax4 = axes[1, 1]
            d_color = np.linspace(0, args.visibility * 1.5, 100)

            for name, wl in wavelengths_rgb.items():
                beta_wl = beta * (550 / wl)**2
                T = np.exp(-beta_wl * d_color)
                color = 'red' if 'Red' in name else ('green' if 'Green' in name else 'blue')
                ax4.plot(d_color, T, color=color, linewidth=2, label=name)

            ax4.set_xlabel('Distance (km)')
            ax4.set_ylabel('Transmission')
            ax4.set_title('Wavelength-Dependent Transmission')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
