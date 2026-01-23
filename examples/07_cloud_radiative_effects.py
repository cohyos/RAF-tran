#!/usr/bin/env python3
"""
Cloud Radiative Effects
=======================

This example demonstrates how clouds affect both solar (shortwave)
and thermal (longwave) radiation:

- Clouds reflect solar radiation (cooling effect)
- Clouds trap thermal radiation (warming effect)
- Net effect depends on cloud height, thickness, and optical properties

Usage:
    python 07_cloud_radiative_effects.py
    python 07_cloud_radiative_effects.py --cloud-tau 20 --cloud-height 5000
    python 07_cloud_radiative_effects.py --help

Output:
    - Console: Radiative forcing calculations
    - Graph: cloud_radiative_effects.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.atmosphere import StandardAtmosphere
    from raf_tran.scattering import MieScattering
    from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


CLOUD_TYPES = {
    "Cirrus": {
        "height": 10000,  # m
        "thickness": 1000,
        "tau_vis": 0.5,
        "r_eff": 25.0,  # um (ice crystals)
        "description": "High, thin ice clouds"
    },
    "Cirrostratus": {
        "height": 8000,
        "thickness": 2000,
        "tau_vis": 1.5,
        "r_eff": 30.0,
        "description": "High, sheet-like ice clouds"
    },
    "Altostratus": {
        "height": 4000,
        "thickness": 2000,
        "tau_vis": 10.0,
        "r_eff": 10.0,  # um (water droplets)
        "description": "Middle-level gray clouds"
    },
    "Stratocumulus": {
        "height": 1500,
        "thickness": 500,
        "tau_vis": 8.0,
        "r_eff": 8.0,
        "description": "Low, lumpy clouds"
    },
    "Stratus": {
        "height": 500,
        "thickness": 500,
        "tau_vis": 15.0,
        "r_eff": 7.0,
        "description": "Low, uniform gray layer"
    },
    "Cumulus": {
        "height": 1000,
        "thickness": 2000,
        "tau_vis": 20.0,
        "r_eff": 10.0,
        "description": "Fair weather puffy clouds"
    },
    "Cumulonimbus": {
        "height": 2000,
        "thickness": 10000,
        "tau_vis": 100.0,
        "r_eff": 15.0,
        "description": "Deep convective storm clouds"
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate cloud radiative effects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cloud types available:
  Cirrus, Cirrostratus - High ice clouds (warming)
  Altostratus - Middle level (mixed)
  Stratocumulus, Stratus - Low water clouds (cooling)
  Cumulus, Cumulonimbus - Convective clouds

Examples:
  %(prog)s                          # Compare all cloud types
  %(prog)s --cloud-type Cirrus      # Study cirrus clouds
  %(prog)s --cloud-tau 30           # Custom optical depth
        """
    )
    parser.add_argument(
        "--cloud-type", type=str, default=None,
        help="Specific cloud type to analyze"
    )
    parser.add_argument(
        "--cloud-tau", type=float, default=None,
        help="Cloud visible optical depth (overrides cloud type)"
    )
    parser.add_argument(
        "--cloud-height", type=float, default=None,
        help="Cloud base height in meters (overrides cloud type)"
    )
    parser.add_argument(
        "--sza", type=float, default=45,
        help="Solar zenith angle in degrees (default: 45)"
    )
    parser.add_argument(
        "--surface-albedo", type=float, default=0.15,
        help="Surface albedo (default: 0.15)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="cloud_radiative_effects.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def setup_atmosphere(n_layers=40, z_top=20000):
    """Setup atmospheric grid and properties."""
    atmosphere = StandardAtmosphere()
    z_levels = np.linspace(0, z_top, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    temperature = atmosphere.temperature(z_mid)
    pressure = atmosphere.pressure(z_levels)

    return z_levels, z_mid, dz, temperature, pressure


def add_cloud_layer(z_mid, dz, cloud_base, cloud_top, tau_total):
    """Add cloud optical depth to specified layers."""
    n_layers = len(z_mid)
    tau = np.zeros(n_layers)

    # Find cloud layers
    cloud_layers = []
    for i in range(n_layers):
        z_bot = z_mid[i] - dz[i] / 2
        z_top_layer = z_mid[i] + dz[i] / 2

        # Check for overlap with cloud
        if z_top_layer > cloud_base and z_bot < cloud_top:
            cloud_layers.append(i)

    # Distribute optical depth uniformly through cloud
    if cloud_layers:
        tau_per_layer = tau_total / len(cloud_layers)
        for i in cloud_layers:
            tau[i] = tau_per_layer

    return tau, cloud_layers


def calculate_cre(solver, z_levels, z_mid, dz, temperature, pressure,
                  cloud_tau, cloud_layers, mu0, surface_albedo, T_surface):
    """Calculate Cloud Radiative Effect (CRE)."""
    n_layers = len(z_mid)

    # Background atmosphere (small optical depth for Rayleigh)
    tau_clear = np.ones(n_layers) * 0.01

    # Cloud optical properties (water droplet typical values)
    omega_cloud = 0.999  # Single scattering albedo
    g_cloud = 0.85       # Asymmetry parameter

    # Build optical depth arrays
    tau_cloudy = tau_clear + cloud_tau

    omega_clear = np.ones(n_layers)  # Pure Rayleigh scattering
    g_clear = np.zeros(n_layers)

    omega_cloudy = np.ones(n_layers)
    g_cloudy = np.zeros(n_layers)
    for i in cloud_layers:
        # Mix clear and cloud properties
        if tau_cloudy[i] > 0:
            f_cloud = cloud_tau[i] / tau_cloudy[i]
            omega_cloudy[i] = f_cloud * omega_cloud + (1 - f_cloud)
            g_cloudy[i] = f_cloud * g_cloud

    # SHORTWAVE (Solar) calculation
    result_sw_clear = solver.solve_solar(
        tau=tau_clear, omega=omega_clear, g=g_clear,
        mu0=mu0, flux_toa=SOLAR_CONSTANT, surface_albedo=surface_albedo
    )
    result_sw_cloudy = solver.solve_solar(
        tau=tau_cloudy, omega=omega_cloudy, g=g_cloudy,
        mu0=mu0, flux_toa=SOLAR_CONSTANT, surface_albedo=surface_albedo
    )

    # SW at surface
    sw_down_clear = result_sw_clear.flux_direct[-1] + result_sw_clear.flux_down[-1]
    sw_down_cloudy = result_sw_cloudy.flux_direct[-1] + result_sw_cloudy.flux_down[-1]

    # SW at TOA (upward = reflected)
    sw_up_toa_clear = result_sw_clear.flux_up[0]
    sw_up_toa_cloudy = result_sw_cloudy.flux_up[0]

    # LONGWAVE (Thermal) calculation
    # For LW, use optical depth scaled for IR absorption
    tau_lw_clear = tau_clear * 0.1  # Minimal clear-sky LW opacity
    tau_lw_cloudy = tau_clear * 0.1 + cloud_tau * 0.5  # Clouds absorb in IR

    omega_lw = np.zeros(n_layers)  # No scattering in LW
    g_lw = np.zeros(n_layers)

    result_lw_clear = solver.solve_thermal(
        tau=tau_lw_clear, omega=omega_lw, g=g_lw,
        temperature=temperature, surface_temperature=T_surface
    )
    result_lw_cloudy = solver.solve_thermal(
        tau=tau_lw_cloudy, omega=omega_lw, g=g_lw,
        temperature=temperature, surface_temperature=T_surface
    )

    # LW at TOA (OLR)
    olr_clear = result_lw_clear.flux_up[0]
    olr_cloudy = result_lw_cloudy.flux_up[0]

    # LW at surface (downward = backradiation)
    lw_down_clear = result_lw_clear.flux_down[-1]
    lw_down_cloudy = result_lw_cloudy.flux_down[-1]

    # Calculate CRE
    # CRE_SW = (SW_absorbed)_cloudy - (SW_absorbed)_clear
    #        = (SW_down - SW_up)_cloudy - (SW_down - SW_up)_clear at TOA
    # Simplify: CRE_SW = -DeltaSW_up_TOA (negative because more reflection = cooling)
    cre_sw = -(sw_up_toa_cloudy - sw_up_toa_clear)  # Negative = cooling

    # CRE_LW = OLR_clear - OLR_cloudy (less OLR = warming)
    cre_lw = olr_clear - olr_cloudy  # Positive = warming

    # Net CRE
    cre_net = cre_sw + cre_lw

    return {
        "sw_down_clear": sw_down_clear,
        "sw_down_cloudy": sw_down_cloudy,
        "sw_up_toa_clear": sw_up_toa_clear,
        "sw_up_toa_cloudy": sw_up_toa_cloudy,
        "olr_clear": olr_clear,
        "olr_cloudy": olr_cloudy,
        "lw_down_clear": lw_down_clear,
        "lw_down_cloudy": lw_down_cloudy,
        "cre_sw": cre_sw,
        "cre_lw": cre_lw,
        "cre_net": cre_net,
    }


def main():
    args = parse_args()

    print("=" * 80)
    print("CLOUD RADIATIVE EFFECTS")
    print("=" * 80)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"Surface albedo: {args.surface_albedo}")

    mu0 = np.cos(np.radians(args.sza))

    # Setup atmosphere
    z_levels, z_mid, dz, temperature, pressure = setup_atmosphere()
    T_surface = temperature[0]
    print(f"Surface temperature: {T_surface:.1f} K")

    solver = TwoStreamSolver()

    # Analyze cloud types
    if args.cloud_type and args.cloud_type in CLOUD_TYPES:
        clouds_to_analyze = {args.cloud_type: CLOUD_TYPES[args.cloud_type]}
    else:
        clouds_to_analyze = CLOUD_TYPES

    print("\n" + "=" * 80)
    print("CLOUD RADIATIVE FORCING ANALYSIS")
    print("=" * 80)

    print(f"\n{'Cloud Type':<15} {'Height':>8} {'tau_vis':>8} {'CRE_SW':>10} {'CRE_LW':>10} "
          f"{'CRE_Net':>10} {'Effect':>10}")
    print(f"{'':15} {'(km)':>8} {'':>8} {'(W/m^2)':>10} {'(W/m^2)':>10} "
          f"{'(W/m^2)':>10} {'':>10}")
    print("-" * 85)

    results = {}

    for name, props in clouds_to_analyze.items():
        height = args.cloud_height if args.cloud_height else props["height"]
        thickness = props["thickness"]
        tau_vis = args.cloud_tau if args.cloud_tau else props["tau_vis"]

        cloud_base = height
        cloud_top = height + thickness

        # Add cloud layer
        cloud_tau, cloud_layers = add_cloud_layer(z_mid, dz, cloud_base, cloud_top, tau_vis)

        # Calculate CRE
        cre = calculate_cre(solver, z_levels, z_mid, dz, temperature, pressure,
                           cloud_tau, cloud_layers, mu0, args.surface_albedo, T_surface)

        results[name] = {**props, **cre, "height": height, "tau_vis": tau_vis}

        effect = "COOLING" if cre["cre_net"] < -5 else "WARMING" if cre["cre_net"] > 5 else "NEUTRAL"

        print(f"{name:<15} {height/1000:>8.1f} {tau_vis:>8.1f} {cre['cre_sw']:>10.1f} "
              f"{cre['cre_lw']:>10.1f} {cre['cre_net']:>10.1f} {effect:>10}")

    print("-" * 85)

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Cloud Radiative Effect (CRE):

  CRE_SW (Shortwave):
    - Clouds reflect solar radiation back to space
    - Always NEGATIVE (cooling effect)
    - Larger for optically thick, bright clouds

  CRE_LW (Longwave):
    - Clouds absorb and re-emit thermal radiation
    - Emit at cloud-top temperature (colder than surface)
    - POSITIVE (warming effect) - reduces OLR

  CRE_Net = CRE_SW + CRE_LW:
    - LOW clouds: Strong SW cooling, weak LW warming -> NET COOLING
    - HIGH clouds: Weak SW cooling, strong LW warming -> NET WARMING
""")

    # High vs low cloud comparison
    if "Cirrus" in results and "Stratocumulus" in results:
        print("-" * 80)
        print("HIGH vs LOW CLOUD COMPARISON:")
        print("-" * 80)

        cirrus = results["Cirrus"]
        strat = results["Stratocumulus"]

        print(f"""
  Cirrus (high, thin):
    - Cloud top is COLD -> emits little LW -> strong LW warming
    - Thin -> reflects little SW -> weak SW cooling
    - Net: {cirrus['cre_net']:+.1f} W/m^2 (warming)

  Stratocumulus (low, thick):
    - Cloud top is WARM -> emits more LW -> weak LW warming
    - Thick -> reflects much SW -> strong SW cooling
    - Net: {strat['cre_net']:+.1f} W/m^2 (cooling)
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle('Cloud Radiative Effects', fontsize=14, fontweight='bold')

            names = list(results.keys())
            heights = [results[n]["height"] / 1000 for n in names]
            taus = [results[n]["tau_vis"] for n in names]
            cre_sw = [results[n]["cre_sw"] for n in names]
            cre_lw = [results[n]["cre_lw"] for n in names]
            cre_net = [results[n]["cre_net"] for n in names]

            # Plot 1: CRE components
            ax1 = axes[0, 0]
            x_pos = np.arange(len(names))
            width = 0.25

            bars1 = ax1.bar(x_pos - width, cre_sw, width, label='CRE_SW (cooling)',
                           color='blue', alpha=0.7)
            bars2 = ax1.bar(x_pos, cre_lw, width, label='CRE_LW (warming)',
                           color='red', alpha=0.7)
            bars3 = ax1.bar(x_pos + width, cre_net, width, label='CRE_Net',
                           color='green', alpha=0.7)

            ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_xlabel('Cloud Type')
            ax1.set_ylabel('Radiative Forcing (W/m^2)')
            ax1.set_title('Cloud Radiative Effect Components')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')

            # Plot 2: CRE vs cloud height
            ax2 = axes[0, 1]
            ax2.scatter(heights, cre_net, c=cre_net, cmap='RdBu_r', s=200,
                       edgecolors='black', linewidths=1, vmin=-100, vmax=100)
            for i, name in enumerate(names):
                ax2.annotate(name, (heights[i], cre_net[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
            ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Cloud Base Height (km)')
            ax2.set_ylabel('Net CRE (W/m^2)')
            ax2.set_title('Net Cloud Forcing vs Height')
            ax2.grid(True, alpha=0.3)

            # Add text annotations
            ax2.text(0.95, 0.95, 'WARMING', transform=ax2.transAxes,
                    ha='right', va='top', fontsize=10, color='red')
            ax2.text(0.95, 0.05, 'COOLING', transform=ax2.transAxes,
                    ha='right', va='bottom', fontsize=10, color='blue')

            # Plot 3: CRE_SW vs CRE_LW
            ax3 = axes[1, 0]
            colors_by_height = plt.cm.viridis(np.array(heights) / max(heights))
            for i, name in enumerate(names):
                ax3.scatter(cre_sw[i], cre_lw[i], c=[colors_by_height[i]], s=200,
                           edgecolors='black', linewidths=1, label=name)
            ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
            ax3.axvline(0, color='black', linestyle='--', alpha=0.3)

            # Add diagonal line for CRE_net = 0
            lim = max(abs(min(cre_sw)), abs(max(cre_lw))) * 1.1
            ax3.plot([-lim, 0], [lim, 0], 'g--', alpha=0.5, label='Net CRE = 0')

            ax3.set_xlabel('CRE_SW (W/m^2)')
            ax3.set_ylabel('CRE_LW (W/m^2)')
            ax3.set_title('SW vs LW Cloud Forcing')
            ax3.legend(loc='lower left', fontsize=7)
            ax3.grid(True, alpha=0.3)

            # Plot 4: Schematic of cloud effects
            ax4 = axes[1, 1]
            ax4.set_xlim(0, 10)
            ax4.set_ylim(0, 15)

            # Draw atmosphere
            ax4.axhspan(0, 2, color='green', alpha=0.3, label='Surface')
            ax4.axhspan(2, 5, color='lightblue', alpha=0.2)
            ax4.axhspan(5, 10, color='lightblue', alpha=0.15)
            ax4.axhspan(10, 15, color='lightblue', alpha=0.1)

            # Draw low cloud
            ax4.add_patch(plt.Rectangle((1, 3), 2, 1, color='gray', alpha=0.8))
            ax4.annotate('Low Cloud\n(Cooling)', (2, 3.5), ha='center', fontsize=9)

            # Draw high cloud
            ax4.add_patch(plt.Rectangle((6, 10), 2.5, 0.5, color='lightgray', alpha=0.6))
            ax4.annotate('High Cloud\n(Warming)', (7.25, 10.25), ha='center', fontsize=9)

            # Draw arrows for radiation
            # Solar
            ax4.annotate('', xy=(2, 5), xytext=(2, 14),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2))
            ax4.text(2.2, 12, 'Solar', color='orange', fontsize=8)

            # Reflected from low cloud
            ax4.annotate('', xy=(2.5, 14), xytext=(2.5, 4),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2, ls='--'))

            # Thermal from surface
            ax4.annotate('', xy=(7, 14), xytext=(7, 2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax4.text(7.2, 6, 'Thermal\n(IR)', color='red', fontsize=8)

            # Blocked by high cloud
            ax4.annotate('', xy=(7.5, 9.5), xytext=(7.5, 2),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1, ls=':'))

            ax4.set_title('Schematic: Cloud Radiative Effects')
            ax4.axis('off')

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
