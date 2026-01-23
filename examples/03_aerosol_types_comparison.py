#!/usr/bin/env python3
"""
Aerosol Types Comparison
========================

This example compares the optical properties of different aerosol types:
- Sulfate (non-absorbing, from pollution)
- Dust (weakly absorbing, mineral particles)
- Black Carbon (strongly absorbing, from combustion)
- Sea Salt (non-absorbing, marine environments)

The key difference is the single scattering albedo (SSA), which determines
whether aerosols cool or warm the atmosphere.

Usage:
    python 03_aerosol_types_comparison.py
    python 03_aerosol_types_comparison.py --radius 0.3
    python 03_aerosol_types_comparison.py --help

Output:
    - Console: Optical properties comparison table
    - Graph: aerosol_types_comparison.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering import MieScattering
    from raf_tran.scattering.mie import mie_efficiencies, lognormal_size_distribution
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


# Aerosol optical properties database
AEROSOL_TYPES = {
    "Sulfate": {
        "refractive_index": 1.43 + 0.0j,
        "description": "Ammonium sulfate - pollution aerosol",
        "r_g": 0.05,  # Geometric mean radius (μm)
        "sigma_g": 2.0,  # Geometric standard deviation
        "color": "blue"
    },
    "Dust": {
        "refractive_index": 1.53 + 0.008j,
        "description": "Saharan dust - mineral particles",
        "r_g": 0.5,
        "sigma_g": 2.2,
        "color": "brown"
    },
    "Black Carbon": {
        "refractive_index": 1.95 + 0.79j,
        "description": "Soot - combustion product",
        "r_g": 0.02,
        "sigma_g": 1.8,
        "color": "black"
    },
    "Sea Salt": {
        "refractive_index": 1.50 + 0.0j,
        "description": "Marine aerosol",
        "r_g": 0.3,
        "sigma_g": 2.0,
        "color": "cyan"
    },
    "Organic Carbon": {
        "refractive_index": 1.53 + 0.02j,
        "description": "Biomass burning",
        "r_g": 0.1,
        "sigma_g": 1.9,
        "color": "green"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare optical properties of different aerosol types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aerosol types available:
  Sulfate       - Pollution aerosol (non-absorbing)
  Dust          - Mineral particles (weakly absorbing)
  Black Carbon  - Soot from combustion (strongly absorbing)
  Sea Salt      - Marine aerosol (non-absorbing)
  Organic Carbon - Biomass burning (moderately absorbing)

Examples:
  %(prog)s                          # Default comparison
  %(prog)s --radius 0.5             # Fixed radius for all types
  %(prog)s --wavelength 0.55        # Specify wavelength
        """
    )
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Fixed particle radius in μm (default: use type-specific)"
    )
    parser.add_argument(
        "--wavelength", type=float, default=0.55,
        help="Wavelength in μm (default: 0.55)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="aerosol_types_comparison.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("AEROSOL TYPES COMPARISON")
    print("=" * 80)
    print(f"\nWavelength: {args.wavelength * 1000:.0f} nm")
    if args.radius:
        print(f"Fixed radius: {args.radius} μm")

    # Results storage
    results = {}

    print("\n" + "-" * 80)
    print(f"{'Aerosol Type':<16} {'n':<6} {'k':<8} {'x':<8} {'Q_ext':<8} {'Q_sca':<8} "
          f"{'SSA':<8} {'g':<8}")
    print("-" * 80)

    for name, props in AEROSOL_TYPES.items():
        m = props["refractive_index"]
        r = args.radius if args.radius else props["r_g"]

        # Size parameter
        x = 2 * np.pi * r / args.wavelength

        # Calculate Mie properties
        Q_ext, Q_sca, Q_abs, g = mie_efficiencies(x, m)

        # Single scattering albedo
        ssa = Q_sca / Q_ext if Q_ext > 0 else 1.0

        results[name] = {
            "m": m,
            "r": r,
            "x": x,
            "Q_ext": Q_ext,
            "Q_sca": Q_sca,
            "Q_abs": Q_abs,
            "ssa": ssa,
            "g": g
        }

        print(f"{name:<16} {m.real:<6.2f} {m.imag:<8.4f} {x:<8.2f} {Q_ext:<8.3f} "
              f"{Q_sca:<8.3f} {ssa:<8.3f} {g:<8.3f}")

    print("-" * 80)

    # Climate implications
    print("\n" + "=" * 80)
    print("CLIMATE IMPLICATIONS")
    print("=" * 80)

    print("""
Single Scattering Albedo (SSA) determines aerosol climate effect:
""")

    for name, res in sorted(results.items(), key=lambda x: -x[1]['ssa']):
        ssa = res['ssa']
        if ssa > 0.95:
            effect = "COOLING (reflects sunlight)"
        elif ssa > 0.85:
            effect = "Weak cooling"
        elif ssa > 0.7:
            effect = "Mixed (depends on surface)"
        else:
            effect = "WARMING (absorbs sunlight)"

        print(f"  {name:<16} SSA = {ssa:.3f}  →  {effect}")

    # Wavelength dependence analysis
    print("\n" + "-" * 80)
    print("WAVELENGTH DEPENDENCE")
    print("-" * 80)

    wavelengths = np.array([0.35, 0.55, 0.87, 1.02])  # UV, green, NIR, SWIR

    print(f"\n{'Aerosol':<16}", end="")
    for wl in wavelengths:
        print(f" τ_{int(wl*1000):>4}nm", end="")
    print("  Ångström")
    print("-" * 70)

    for name, props in AEROSOL_TYPES.items():
        m = props["refractive_index"]
        r = args.radius if args.radius else props["r_g"]

        tau_values = []
        for wl in wavelengths:
            x = 2 * np.pi * r / wl
            Q_ext, _, _, _ = mie_efficiencies(x, m)
            # Optical depth ∝ Q_ext for fixed particle density
            tau_values.append(Q_ext)

        # Calculate Ångström exponent: τ ∝ λ^(-α)
        log_wl = np.log(wavelengths)
        log_tau = np.log(np.array(tau_values))
        alpha = -np.polyfit(log_wl, log_tau, 1)[0]

        print(f"{name:<16}", end="")
        for tau in tau_values:
            print(f" {tau:>7.3f}", end="")
        print(f"  {alpha:>6.2f}")

    print("""
Ångström exponent (α):
  α > 1.5  : Small particles (pollution, smoke)
  α ~ 1.0  : Mixed aerosols
  α < 0.5  : Large particles (dust, sea salt)
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Aerosol Types: Optical Properties Comparison',
                        fontsize=14, fontweight='bold')

            # Plot 1: Q_ext vs size parameter for different aerosol types
            ax1 = axes[0, 0]
            x_range = np.logspace(-1, 2, 200)

            for name, props in AEROSOL_TYPES.items():
                m = props["refractive_index"]
                Q_ext_arr = []
                for x in x_range:
                    Q_ext, _, _, _ = mie_efficiencies(x, m)
                    Q_ext_arr.append(Q_ext)
                ax1.loglog(x_range, Q_ext_arr, label=name, color=props["color"], linewidth=2)

            ax1.set_xlabel('Size Parameter x = 2πr/λ')
            ax1.set_ylabel('Extinction Efficiency Q_ext')
            ax1.set_title('Extinction Efficiency vs Size Parameter')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0.1, 100)
            ax1.axhline(2, color='gray', linestyle='--', alpha=0.5, label='Large particle limit')

            # Plot 2: Single Scattering Albedo vs size parameter
            ax2 = axes[0, 1]

            for name, props in AEROSOL_TYPES.items():
                m = props["refractive_index"]
                ssa_arr = []
                for x in x_range:
                    Q_ext, Q_sca, _, _ = mie_efficiencies(x, m)
                    ssa = Q_sca / Q_ext if Q_ext > 0 else 1.0
                    ssa_arr.append(ssa)
                ax2.semilogx(x_range, ssa_arr, label=name, color=props["color"], linewidth=2)

            ax2.set_xlabel('Size Parameter x = 2πr/λ')
            ax2.set_ylabel('Single Scattering Albedo (SSA)')
            ax2.set_title('SSA vs Size Parameter')
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0.1, 100)
            ax2.set_ylim(0, 1.05)
            ax2.axhline(0.85, color='gray', linestyle='--', alpha=0.5)
            ax2.text(0.15, 0.87, 'Critical SSA (warming/cooling threshold)', fontsize=8)

            # Plot 3: Asymmetry parameter vs size parameter
            ax3 = axes[1, 0]

            for name, props in AEROSOL_TYPES.items():
                m = props["refractive_index"]
                g_arr = []
                for x in x_range:
                    _, _, _, g = mie_efficiencies(x, m)
                    g_arr.append(g)
                ax3.semilogx(x_range, g_arr, label=name, color=props["color"], linewidth=2)

            ax3.set_xlabel('Size Parameter x = 2πr/λ')
            ax3.set_ylabel('Asymmetry Parameter g')
            ax3.set_title('Asymmetry Parameter vs Size Parameter')
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0.1, 100)
            ax3.set_ylim(-0.5, 1)
            ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax3.text(0.15, 0.05, 'g=0: isotropic', fontsize=8)
            ax3.text(0.15, 0.75, 'g→1: forward scattering', fontsize=8)

            # Plot 4: Bar chart comparison at fixed wavelength
            ax4 = axes[1, 1]

            names = list(results.keys())
            x_pos = np.arange(len(names))
            width = 0.25

            ssa_vals = [results[n]['ssa'] for n in names]
            g_vals = [results[n]['g'] for n in names]
            qext_vals = [results[n]['Q_ext'] / max(results[n]['Q_ext'] for n in names) for n in names]

            bars1 = ax4.bar(x_pos - width, ssa_vals, width, label='SSA', color='blue', alpha=0.7)
            bars2 = ax4.bar(x_pos, g_vals, width, label='g', color='green', alpha=0.7)
            bars3 = ax4.bar(x_pos + width, qext_vals, width, label='Q_ext (norm)', color='red', alpha=0.7)

            ax4.set_xlabel('Aerosol Type')
            ax4.set_ylabel('Value')
            ax4.set_title(f'Optical Properties at λ={args.wavelength*1000:.0f}nm')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(names, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.set_ylim(0, 1.1)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
AEROSOL RADIATIVE EFFECTS:

1. SCATTERING vs ABSORPTION:
   - SSA = Q_sca/Q_ext is the fraction of extinction due to scattering
   - High SSA (>0.95): Mostly reflects sunlight → COOLING
   - Low SSA (<0.85): Mostly absorbs sunlight → WARMING

2. PARTICLE SIZE EFFECTS:
   - Size parameter x = 2πr/λ determines scattering regime
   - x << 1: Rayleigh regime (σ ∝ r⁶)
   - x ~ 1: Mie regime (strong size dependence)
   - x >> 1: Geometric optics (Q_ext → 2)

3. REFRACTIVE INDEX:
   - Real part (n): Determines scattering strength
   - Imaginary part (k): Determines absorption
   - Black carbon has large k → strong absorber

4. CLIMATE IMPLICATIONS:
   - Sulfate aerosols: Net cooling (negative forcing)
   - Black carbon: Net warming (positive forcing)
   - Dust: Complex, depends on surface albedo below
""")


if __name__ == "__main__":
    main()
