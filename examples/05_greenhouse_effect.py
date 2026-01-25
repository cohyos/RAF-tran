#!/usr/bin/env python3
"""
Greenhouse Effect Demonstration
===============================

This example demonstrates the greenhouse effect - how atmospheric
absorption of thermal radiation warms the surface.

We compare:
1. Earth with no atmosphere (equilibrium temperature ~255 K)
2. Earth with a gray absorbing atmosphere
3. Effect of increasing optical depth (analog to increasing CO2)

Usage:
    python 05_greenhouse_effect.py
    python 05_greenhouse_effect.py --tau 2.0
    python 05_greenhouse_effect.py --help

Output:
    - Console: Temperature calculations and explanations
    - Graph: greenhouse_effect.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demonstrate the greenhouse effect with radiative transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default parameters
  %(prog)s --tau 3.0                # Stronger greenhouse effect
  %(prog)s --albedo 0.1             # Lower albedo (ocean)
  %(prog)s --solar 1361             # Change solar constant
        """
    )
    parser.add_argument(
        "--tau", type=float, default=1.8,
        help="Total infrared optical depth (default: 1.8 for Earth-like)"
    )
    parser.add_argument(
        "--albedo", type=float, default=0.30,
        help="Planetary albedo (default: 0.30)"
    )
    parser.add_argument(
        "--solar", type=float, default=1361,
        help="Solar constant in W/m^2 (default: 1361)"
    )
    parser.add_argument(
        "--n-layers", type=int, default=20,
        help="Number of atmospheric layers (default: 20)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="greenhouse_effect.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def effective_temperature(solar, albedo):
    """Calculate effective radiating temperature with no atmosphere."""
    # Energy balance: (1-a)*S/4 = sigmaT^4
    return ((1 - albedo) * solar / (4 * STEFAN_BOLTZMANN))**0.25


def simple_greenhouse_temperature(T_eff, tau):
    """Simple 1-layer greenhouse model: T_surface = T_eff * (1 + tau/2)^0.25"""
    return T_eff * (1 + tau / 2)**0.25


def iterative_equilibrium(tau_per_layer, n_layers, solar, albedo, max_iter=100):
    """
    Find radiative equilibrium temperature profile.

    Iteratively solve for temperature profile where each layer
    is in radiative equilibrium.
    """
    solver = TwoStreamSolver()

    # Initial guess: isothermal atmosphere at effective temperature
    T_eff = effective_temperature(solar, albedo)
    absorbed_solar = (1 - albedo) * solar / 4

    # Initialize temperatures
    T_surface = T_eff * 1.2  # Start warmer
    T_layers = np.ones(n_layers) * T_eff

    # Optical properties for thermal IR (pure absorption, no scattering)
    omega = np.zeros(n_layers)
    g = np.zeros(n_layers)

    for iteration in range(max_iter):
        T_surface_old = T_surface

        # Solve thermal radiative transfer
        result = solver.solve_thermal(
            tau=tau_per_layer,
            omega=omega,
            g=g,
            temperature=T_layers,
            surface_temperature=T_surface,
            surface_emissivity=1.0,
        )

        # Surface energy balance: absorbed solar = emitted - backradiation
        # sigmaT_s^4 = absorbed_solar + F_down(surface)
        # After level ordering fix: index 0 = surface, index -1 = TOA
        backradiation = result.flux_down[0]  # Surface level
        T_surface_new = ((absorbed_solar + backradiation) / STEFAN_BOLTZMANN)**0.25

        # Update layer temperatures based on local radiative equilibrium
        for i in range(n_layers):
            # Layer absorbs and emits (layer i is between levels i and i+1)
            # With surface-to-TOA ordering:
            #   F_up entering from below = flux_up[i]
            #   F_up leaving at top = flux_up[i+1]
            #   F_down entering from above = flux_down[i+1]
            #   F_down leaving at bottom = flux_down[i]
            F_up_in = result.flux_up[i]       # From surface side
            F_up_out = result.flux_up[i + 1]  # To TOA side
            F_down_in = result.flux_down[i + 1]  # From TOA side
            F_down_out = result.flux_down[i]     # To surface side

            # Net heating of layer
            net_absorbed = (F_up_in - F_up_out) + (F_down_in - F_down_out)

            # Adjust temperature based on net heating
            dT = 0.1 * net_absorbed / (STEFAN_BOLTZMANN * T_layers[i]**3)
            T_layers[i] = max(100, T_layers[i] + dT)

        # Relaxation
        T_surface = 0.7 * T_surface + 0.3 * T_surface_new

        if abs(T_surface - T_surface_old) < 0.01:
            break

    return T_surface, T_layers, result


def main():
    args = parse_args()

    print("=" * 80)
    print("GREENHOUSE EFFECT DEMONSTRATION")
    print("=" * 80)
    print(f"\nSolar constant: {args.solar} W/m^2")
    print(f"Planetary albedo: {args.albedo}")
    print(f"IR optical depth: {args.tau}")
    print(f"Number of layers: {args.n_layers}")

    # Basic calculations
    T_eff = effective_temperature(args.solar, args.albedo)
    absorbed_solar = (1 - args.albedo) * args.solar / 4

    print("\n" + "=" * 80)
    print("PART 1: EARTH WITHOUT ATMOSPHERE")
    print("=" * 80)
    print(f"""
Energy Balance:
  Absorbed solar = (1 - albedo) * S_0/4
                 = (1 - {args.albedo}) * {args.solar}/4
                 = {absorbed_solar:.1f} W/m^2

  At equilibrium: Absorbed = Emitted
  sigmaT^4 = {absorbed_solar:.1f} W/m^2
  T = ({absorbed_solar:.1f} / {STEFAN_BOLTZMANN:.4e})^0.25
  T = {T_eff:.1f} K ({T_eff - 273.15:.1f} degC)

This is Earth's EFFECTIVE TEMPERATURE - what we'd measure from space.
The actual mean surface temperature is ~288 K (15 degC).
The difference ({288 - T_eff:.0f} K) is the GREENHOUSE EFFECT.
""")

    # Simple 1-layer model
    T_simple = simple_greenhouse_temperature(T_eff, args.tau)
    print("=" * 80)
    print("PART 2: SIMPLE ONE-LAYER MODEL")
    print("=" * 80)
    print(f"""
With a single absorbing layer at optical depth tau = {args.tau}:

  T_surface = T_eff * (1 + tau/2)^0.25
            = {T_eff:.1f} * (1 + {args.tau}/2)^0.25
            = {T_simple:.1f} K ({T_simple - 273.15:.1f} degC)

Greenhouse warming = {T_simple - T_eff:.1f} K
""")

    # Multi-layer radiative equilibrium
    print("=" * 80)
    print("PART 3: MULTI-LAYER RADIATIVE EQUILIBRIUM")
    print("=" * 80)

    # Distribute optical depth through layers
    tau_per_layer = np.ones(args.n_layers) * args.tau / args.n_layers

    T_surface, T_layers, result = iterative_equilibrium(
        tau_per_layer, args.n_layers, args.solar, args.albedo
    )

    print(f"""
With {args.n_layers} atmospheric layers (tau_total = {args.tau}):

  Surface temperature: {T_surface:.1f} K ({T_surface - 273.15:.1f} degC)
  Top layer temperature: {T_layers[-1]:.1f} K
  Greenhouse warming: {T_surface - T_eff:.1f} K

Flux balance:
  Outgoing LW at TOA: {result.flux_up[-1]:.1f} W/m^2
  Absorbed solar: {absorbed_solar:.1f} W/m^2
  Backradiation to surface: {result.flux_down[0]:.1f} W/m^2
""")

    # Sensitivity study
    print("=" * 80)
    print("PART 4: SENSITIVITY TO OPTICAL DEPTH (CLIMATE SENSITIVITY)")
    print("=" * 80)

    tau_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    print(f"\n{'tau':>6} {'T_surface (K)':>14} {'T_surface ( degC)':>14} {'Warming (K)':>14}")
    print("-" * 60)

    sensitivity_results = []
    for tau in tau_values:
        tau_per = np.ones(args.n_layers) * tau / args.n_layers
        T_s, _, _ = iterative_equilibrium(tau_per, args.n_layers, args.solar, args.albedo)
        warming = T_s - T_eff
        sensitivity_results.append((tau, T_s, warming))
        print(f"{tau:>6.1f} {T_s:>14.1f} {T_s - 273.15:>14.1f} {warming:>14.1f}")

    # CO2 doubling analog
    print("\n" + "-" * 60)
    print("CO_2 DOUBLING ANALOG:")
    print("-" * 60)

    # Find climate sensitivity (warming from tau=1.8 to tau=2.7, ~50% increase)
    tau1, tau2 = 1.8, 2.7  # Approximate CO2 doubling effect
    tau_per1 = np.ones(args.n_layers) * tau1 / args.n_layers
    tau_per2 = np.ones(args.n_layers) * tau2 / args.n_layers
    T_s1, _, _ = iterative_equilibrium(tau_per1, args.n_layers, args.solar, args.albedo)
    T_s2, _, _ = iterative_equilibrium(tau_per2, args.n_layers, args.solar, args.albedo)

    print(f"""
Increasing optical depth from {tau1} to {tau2} (~CO_2 doubling equivalent):
  Temperature change: {T_s2 - T_s1:.1f} K

IMPORTANT NOTE ON CLIMATE SENSITIVITY:
  This gray atmosphere model gives {T_s2 - T_s1:.1f} K per "CO_2 doubling".

  The REAL equilibrium climate sensitivity (ECS) is ~2.5-4 K because:
  1. Gray model ignores spectral details (CO_2 absorbs only specific bands)
  2. Gray model ignores water vapor feedback (amplifies warming by ~2x)
  3. Gray model ignores cloud feedbacks (uncertain, +/- effects)
  4. Gray model ignores ice-albedo feedback (amplifies warming)
  5. The 50% tau increase here doesn't match actual CO_2 radiative forcing

  Actual CO_2 doubling forcing: ~3.7 W/m^2 -> ~3 K warming (best estimate).
  IPCC AR6 likely range: 2.5-4.0 K (very likely: 2.0-5.0 K)
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Greenhouse Effect Demonstration', fontsize=14, fontweight='bold')

            # Plot 1: Temperature profile
            ax1 = axes[0, 0]
            z = np.linspace(0, 20, args.n_layers + 1)  # Arbitrary altitude scale
            z_layers = (z[:-1] + z[1:]) / 2

            ax1.plot([T_surface] + list(T_layers), [0] + list(z_layers), 'ro-', linewidth=2,
                    markersize=8, label='Equilibrium profile')
            ax1.axvline(T_eff, color='blue', linestyle='--', linewidth=2,
                       label=f'Effective T = {T_eff:.0f} K')
            ax1.axhline(0, color='brown', linewidth=3, label='Surface')
            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Atmospheric Layer')
            ax1.set_title(f'Temperature Profile (tau = {args.tau})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Flux profiles
            ax2 = axes[0, 1]
            z_levels = np.linspace(0, 20, args.n_layers + 1)
            ax2.plot(result.flux_up, z_levels, 'r-', linewidth=2, label='Upward LW')
            ax2.plot(result.flux_down, z_levels, 'b-', linewidth=2, label='Downward LW')
            ax2.axvline(absorbed_solar, color='orange', linestyle='--',
                       linewidth=2, label=f'Absorbed solar = {absorbed_solar:.0f} W/m^2')
            ax2.set_xlabel('Flux (W/m^2)')
            ax2.set_ylabel('Atmospheric Level')
            ax2.set_title('Thermal Flux Profiles')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Surface temperature vs optical depth
            ax3 = axes[1, 0]
            taus = [r[0] for r in sensitivity_results]
            temps = [r[1] for r in sensitivity_results]
            warmings = [r[2] for r in sensitivity_results]

            ax3.plot(taus, temps, 'ro-', linewidth=2, markersize=8)
            ax3.axhline(288, color='green', linestyle='--', alpha=0.7,
                       label='Actual Earth T_surface ~ 288 K')
            ax3.axhline(T_eff, color='blue', linestyle='--', alpha=0.7,
                       label=f'Effective T = {T_eff:.0f} K')
            ax3.set_xlabel('Infrared Optical Depth (tau)')
            ax3.set_ylabel('Surface Temperature (K)')
            ax3.set_title('Surface Temperature vs Optical Depth')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Add secondary y-axis for  degC
            ax3b = ax3.twinx()
            ax3b.set_ylim(ax3.get_ylim()[0] - 273.15, ax3.get_ylim()[1] - 273.15)
            ax3b.set_ylabel('Temperature ( degC)')

            # Plot 4: Greenhouse warming
            ax4 = axes[1, 1]
            ax4.bar(taus, warmings, width=0.3, color='red', alpha=0.7)
            ax4.set_xlabel('Infrared Optical Depth (tau)')
            ax4.set_ylabel('Greenhouse Warming (K)')
            ax4.set_title('Greenhouse Warming vs Optical Depth')
            ax4.grid(True, alpha=0.3, axis='y')

            # Add annotation
            ax4.annotate(f'Earth-like\n(tau~{args.tau})',
                        xy=(args.tau, T_surface - T_eff),
                        xytext=(args.tau + 0.5, T_surface - T_eff + 5),
                        arrowprops=dict(arrowstyle='->', color='black'),
                        fontsize=10)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
THE GREENHOUSE EFFECT:

1. ENERGY BALANCE:
   - Earth absorbs solar radiation (mostly visible/UV)
   - Earth emits thermal radiation (infrared)
   - At equilibrium: Absorbed = Emitted

2. ATMOSPHERIC ABSORPTION:
   - Atmosphere is transparent to solar radiation
   - Atmosphere ABSORBS infrared (thermal) radiation
   - Main absorbers: H_2O, CO_2, CH_4, N_2O, O_3

3. THE WARMING MECHANISM:
   - Surface emits IR upward
   - Atmosphere absorbs and re-emits IR (both up AND down)
   - Downward emission ("backradiation") warms the surface
   - Surface must be warmer to balance absorbed solar + backradiation

4. INCREASING GREENHOUSE GASES:
   - More CO_2 -> Higher optical depth tau
   - Higher tau -> More backradiation
   - More backradiation -> Warmer surface

5. CLIMATE SENSITIVITY:
   - Doubling CO_2 increases tau by ~10-15%
   - Direct warming: ~1 K
   - With feedbacks (water vapor, ice-albedo): ~2-4.5 K
""")


if __name__ == "__main__":
    main()
