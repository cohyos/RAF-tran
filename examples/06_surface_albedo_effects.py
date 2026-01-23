#!/usr/bin/env python3
"""
Surface Albedo Effects on Radiation
===================================

This example demonstrates how surface albedo (reflectivity) affects:
- Absorbed solar radiation
- Planetary temperature
- Reflected flux observed from space

Different surface types have very different albedos, from ocean (~0.06)
to fresh snow (~0.9).

Usage:
    python 06_surface_albedo_effects.py
    python 06_surface_albedo_effects.py --sza 45
    python 06_surface_albedo_effects.py --help

Output:
    - Console: Radiation budget for different surfaces
    - Graph: surface_albedo_effects.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import StandardAtmosphere
    from raf_tran.scattering import RayleighScattering
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


# Surface albedo database
SURFACE_TYPES = {
    "Fresh Snow": {"albedo": 0.85, "color": "white", "emissivity": 0.99},
    "Old Snow": {"albedo": 0.60, "color": "lightgray", "emissivity": 0.98},
    "Sea Ice": {"albedo": 0.50, "color": "lightblue", "emissivity": 0.97},
    "Desert Sand": {"albedo": 0.40, "color": "tan", "emissivity": 0.92},
    "Bare Soil": {"albedo": 0.20, "color": "brown", "emissivity": 0.95},
    "Grassland": {"albedo": 0.25, "color": "lightgreen", "emissivity": 0.96},
    "Forest": {"albedo": 0.15, "color": "darkgreen", "emissivity": 0.97},
    "Ocean": {"albedo": 0.06, "color": "navy", "emissivity": 0.96},
    "Asphalt": {"albedo": 0.10, "color": "dimgray", "emissivity": 0.95},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Study the effect of surface albedo on radiation budget",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Surface types and typical albedos:
  Fresh Snow:  0.85    Sea Ice:     0.50    Forest:    0.15
  Old Snow:    0.60    Desert Sand: 0.40    Ocean:     0.06
  Grassland:   0.25    Bare Soil:   0.20    Asphalt:   0.10

Examples:
  %(prog)s                          # Default comparison
  %(prog)s --sza 60                 # Sun at 60 deg zenith
  %(prog)s --wavelength 0.55        # Green light only
        """
    )
    parser.add_argument(
        "--sza", type=float, default=30,
        help="Solar zenith angle in degrees (default: 30)"
    )
    parser.add_argument(
        "--wavelength", type=float, default=0.55,
        help="Wavelength in um for atmospheric calculation (default: 0.55)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="surface_albedo_effects.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SURFACE ALBEDO EFFECTS ON RADIATION")
    print("=" * 80)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"Wavelength: {args.wavelength * 1000:.0f} nm")

    mu0 = np.cos(np.radians(args.sza))
    if mu0 <= 0:
        print("Error: Sun below horizon!")
        sys.exit(1)

    # Setup atmosphere
    atmosphere = StandardAtmosphere()
    n_layers = 50
    z_levels = np.linspace(0, 50000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    # Calculate atmospheric optical depth
    rayleigh = RayleighScattering()
    number_density = atmosphere.number_density(z_mid)
    wavelength = np.array([args.wavelength])
    tau_rayleigh = rayleigh.optical_depth(wavelength, number_density, dz).ravel()

    tau_total = np.sum(tau_rayleigh)
    print(f"Atmospheric optical depth (Rayleigh): {tau_total:.4f}")

    # Two-stream solver
    solver = TwoStreamSolver()
    omega = np.ones(n_layers)  # Pure scattering
    g = np.zeros(n_layers)     # Isotropic

    # TOA solar flux
    solar_flux = SOLAR_CONSTANT * mu0  # Account for geometry

    print("\n" + "-" * 80)
    print("SURFACE RADIATION BUDGET")
    print("-" * 80)
    print(f"\n{'Surface':<14} {'Albedo':>8} {'Direct':>10} {'Diffuse':>10} "
          f"{'Absorbed':>10} {'Reflected':>10} {'TOA Up':>10}")
    print(f"{'':14} {'':>8} {'(W/m^2)':>10} {'(W/m^2)':>10} "
          f"{'(W/m^2)':>10} {'(W/m^2)':>10} {'(W/m^2)':>10}")
    print("-" * 80)

    results = {}

    for name, props in SURFACE_TYPES.items():
        albedo = props["albedo"]

        result = solver.solve_solar(
            tau=tau_rayleigh,
            omega=omega,
            g=g,
            mu0=mu0,
            flux_toa=SOLAR_CONSTANT,
            surface_albedo=albedo,
        )

        direct = result.flux_direct[-1]
        diffuse = result.flux_down[-1]
        total_down = direct + diffuse
        reflected = albedo * total_down
        absorbed = (1 - albedo) * total_down
        toa_up = result.flux_up[0]

        results[name] = {
            "albedo": albedo,
            "direct": direct,
            "diffuse": diffuse,
            "total_down": total_down,
            "reflected": reflected,
            "absorbed": absorbed,
            "toa_up": toa_up,
        }

        print(f"{name:<14} {albedo:>8.2f} {direct:>10.1f} {diffuse:>10.1f} "
              f"{absorbed:>10.1f} {reflected:>10.1f} {toa_up:>10.1f}")

    # Equilibrium temperature estimates
    print("\n" + "-" * 80)
    print("EQUILIBRIUM TEMPERATURE ESTIMATES (simplified, no atmosphere)")
    print("-" * 80)
    print(f"\n{'Surface':<14} {'Albedo':>8} {'Absorbed':>12} {'T_eq (K)':>10} {'T_eq ( degC)':>10}")
    print("-" * 80)

    for name, res in results.items():
        # Simple equilibrium: absorbed = emitted
        # For local equilibrium at this SZA (not global average)
        absorbed = res["absorbed"]
        emissivity = SURFACE_TYPES[name]["emissivity"]

        # T^4 = absorbed / (epsilon * sigma)
        if absorbed > 0:
            T_eq = (absorbed / (emissivity * STEFAN_BOLTZMANN))**0.25
        else:
            T_eq = 0

        print(f"{name:<14} {res['albedo']:>8.2f} {absorbed:>12.1f} {T_eq:>10.1f} {T_eq-273.15:>10.1f}")

    # Ice-albedo feedback explanation
    print("\n" + "=" * 80)
    print("ICE-ALBEDO FEEDBACK")
    print("=" * 80)

    ocean_absorbed = results["Ocean"]["absorbed"]
    snow_absorbed = results["Fresh Snow"]["absorbed"]

    print(f"""
The ice-albedo feedback is a key climate feedback mechanism:

1. COMPARISON:
   - Ocean absorbs: {ocean_absorbed:.1f} W/m^2
   - Fresh snow absorbs: {snow_absorbed:.1f} W/m^2
   - Difference: {ocean_absorbed - snow_absorbed:.1f} W/m^2 ({(ocean_absorbed/snow_absorbed - 1)*100:.0f}% more)

2. POSITIVE FEEDBACK LOOP:
   Warming -> Ice melts -> Lower albedo -> More absorption -> More warming

3. ARCTIC AMPLIFICATION:
   This feedback makes polar regions warm faster than the global average.
   As sea ice decreases, the ocean absorbs more solar radiation.

4. TIPPING POINT CONCERN:
   Beyond a certain warming threshold, ice loss may become self-sustaining.
""")

    # Urban heat island
    print("-" * 80)
    print("URBAN HEAT ISLAND EFFECT")
    print("-" * 80)

    grass_absorbed = results["Grassland"]["absorbed"]
    asphalt_absorbed = results["Asphalt"]["absorbed"]

    print(f"""
Urban surfaces are darker than natural vegetation:

   - Grassland absorbs: {grass_absorbed:.1f} W/m^2
   - Asphalt absorbs: {asphalt_absorbed:.1f} W/m^2
   - Extra absorption: {asphalt_absorbed - grass_absorbed:.1f} W/m^2 ({(asphalt_absorbed/grass_absorbed - 1)*100:.0f}% more)

This contributes to urban areas being warmer than surrounding countryside.
"Cool roofs" and "cool pavement" use high-albedo materials to reduce this effect.
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Surface Albedo Effects (SZA = {args.sza} deg)',
                        fontsize=14, fontweight='bold')

            names = list(results.keys())
            albedos = [results[n]["albedo"] for n in names]
            absorbed = [results[n]["absorbed"] for n in names]
            reflected = [results[n]["reflected"] for n in names]
            toa_up = [results[n]["toa_up"] for n in names]
            colors = [SURFACE_TYPES[n]["color"] for n in names]

            # Sort by albedo
            sorted_idx = np.argsort(albedos)
            names = [names[i] for i in sorted_idx]
            albedos = [albedos[i] for i in sorted_idx]
            absorbed = [absorbed[i] for i in sorted_idx]
            reflected = [reflected[i] for i in sorted_idx]
            toa_up = [toa_up[i] for i in sorted_idx]
            colors = [colors[i] for i in sorted_idx]

            # Plot 1: Absorbed vs Reflected
            ax1 = axes[0, 0]
            x_pos = np.arange(len(names))
            width = 0.35

            bars1 = ax1.bar(x_pos - width/2, absorbed, width, label='Absorbed',
                           color='red', alpha=0.7)
            bars2 = ax1.bar(x_pos + width/2, reflected, width, label='Reflected',
                           color='blue', alpha=0.7)

            ax1.set_xlabel('Surface Type')
            ax1.set_ylabel('Flux (W/m^2)')
            ax1.set_title('Absorbed vs Reflected Solar Radiation')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Plot 2: Absorbed flux vs albedo
            ax2 = axes[0, 1]
            for i, name in enumerate(names):
                ax2.scatter(albedos[i], absorbed[i], c=[colors[i]], s=200,
                           label=name, edgecolors='black', linewidths=1)
            ax2.set_xlabel('Surface Albedo')
            ax2.set_ylabel('Absorbed Flux (W/m^2)')
            ax2.set_title('Absorbed Radiation vs Albedo')

            # Add trend line
            z = np.polyfit(albedos, absorbed, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 1, 100)
            ax2.plot(x_line, p(x_line), 'k--', alpha=0.5, label='Linear trend')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 1)

            # Plot 3: TOA upward flux (what satellites see)
            ax3 = axes[1, 0]
            for i, name in enumerate(names):
                ax3.barh(i, toa_up[i], color=colors[i], edgecolor='black')
            ax3.set_yticks(range(len(names)))
            ax3.set_yticklabels(names)
            ax3.set_xlabel('TOA Upward Flux (W/m^2)')
            ax3.set_title('Reflected Radiation at Top of Atmosphere')
            ax3.grid(True, alpha=0.3, axis='x')

            # Plot 4: Energy balance diagram
            ax4 = axes[1, 1]

            # Show for selected surfaces
            selected = ["Ocean", "Grassland", "Fresh Snow"]
            y_positions = [3, 2, 1]

            for name, y in zip(selected, y_positions):
                res = results[name]
                total = res["total_down"]
                abs_frac = res["absorbed"] / total
                ref_frac = res["reflected"] / total

                # Stacked bar
                ax4.barh(y, abs_frac, color='red', alpha=0.7, label='Absorbed' if y == 3 else '')
                ax4.barh(y, ref_frac, left=abs_frac, color='blue', alpha=0.7,
                        label='Reflected' if y == 3 else '')

                # Labels
                ax4.text(abs_frac/2, y, f'{abs_frac*100:.0f}%', ha='center', va='center',
                        fontweight='bold', color='white')
                ax4.text(abs_frac + ref_frac/2, y, f'{ref_frac*100:.0f}%', ha='center',
                        va='center', fontweight='bold', color='white')

            ax4.set_yticks(y_positions)
            ax4.set_yticklabels(selected)
            ax4.set_xlabel('Fraction of Incident Solar Radiation')
            ax4.set_title('Surface Energy Partition')
            ax4.legend(loc='lower right')
            ax4.set_xlim(0, 1)
            ax4.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
SURFACE ALBEDO AND CLIMATE:

1. DEFINITION:
   Albedo = Reflected radiation / Incident radiation
   Range: 0 (perfect absorber) to 1 (perfect reflector)

2. SPECTRAL DEPENDENCE:
   - Most albedos are measured for broadband solar radiation
   - Snow has very different visible vs. IR albedo
   - Vegetation has high NIR reflectance (red edge)

3. CLIMATE EFFECTS:
   - Higher albedo -> Less absorption -> Cooler surface
   - Lower albedo -> More absorption -> Warmer surface

4. GLOBAL AVERAGE:
   - Earth's planetary albedo ~ 0.30
   - Clouds contribute significantly (~0.15)
   - Surface contributes (~0.15)

5. KEY FEEDBACKS:
   - Ice-albedo: Melting ice exposes dark ocean
   - Vegetation-albedo: Forest vs. snow-covered tundra
   - Cloud-albedo: Low clouds have high albedo
""")


if __name__ == "__main__":
    main()
