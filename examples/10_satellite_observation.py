#!/usr/bin/env python3
"""
Satellite Observation Simulation
================================

This example simulates what a satellite would observe looking down
at Earth in different spectral channels:

- Visible channel: Reflected sunlight (clouds, surface)
- Thermal IR window: Surface/cloud-top temperature
- Water vapor channel: Upper tropospheric humidity

This is fundamental for remote sensing and weather satellite interpretation.

Usage:
    python 10_satellite_observation.py
    python 10_satellite_observation.py --surface-temp 300
    python 10_satellite_observation.py --cloud-cover 0.5
    python 10_satellite_observation.py --help

Output:
    - Console: Simulated satellite observations
    - Graph: satellite_observation.png
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.atmosphere import StandardAtmosphere, TropicalAtmosphere
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.scattering import RayleighScattering
    from raf_tran.utils.constants import STEFAN_BOLTZMANN, SOLAR_CONSTANT
    from raf_tran.utils.spectral import planck_function_wavenumber
except ImportError:
    print("Error: raf_tran package not found.")
    print("Please install it first: pip install -e . (from the project root)")
    sys.exit(1)


# Satellite channels (simplified representations)
SATELLITE_CHANNELS = {
    "VIS (0.65 um)": {
        "center_wl": 0.65,  # um
        "type": "reflective",
        "description": "Visible - clouds and surface reflectance"
    },
    "NIR (0.86 um)": {
        "center_wl": 0.86,
        "type": "reflective",
        "description": "Near-IR - vegetation, clouds"
    },
    "IR Window (11 um)": {
        "center_wn": 909,  # cm-^-1
        "type": "thermal",
        "description": "IR window - surface/cloud temperature"
    },
    "Water Vapor (6.7 um)": {
        "center_wn": 1493,  # cm-^-1
        "type": "thermal",
        "tau_wv_scale": 5.0,  # Strong water vapor absorption
        "description": "WV channel - upper tropospheric humidity"
    },
    "CO2 (15 um)": {
        "center_wn": 667,  # cm-^-1
        "type": "thermal",
        "tau_co2": 50.0,  # Very optically thick
        "description": "CO2 band - stratospheric temperature"
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate satellite observations in different channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Satellite channels simulated:
  VIS (0.65 um)      - Visible reflectance
  NIR (0.86 um)      - Near-infrared reflectance
  IR Window (11 um)  - Surface/cloud temperature
  Water Vapor (6.7um)- Upper tropospheric humidity
  CO2 (15 um)        - Stratospheric temperature

Examples:
  %(prog)s                              # Clear sky
  %(prog)s --cloud-cover 0.8            # Mostly cloudy
  %(prog)s --cloud-height 10000         # High cloud (cirrus)
  %(prog)s --surface-type ocean         # Ocean surface
        """
    )
    parser.add_argument(
        "--surface-temp", type=float, default=None,
        help="Surface temperature in K (default: from atmosphere)"
    )
    parser.add_argument(
        "--surface-type", type=str, default="land",
        choices=["ocean", "land", "desert", "snow"],
        help="Surface type (default: land)"
    )
    parser.add_argument(
        "--cloud-cover", type=float, default=0.0,
        help="Cloud fraction 0-1 (default: 0 = clear)"
    )
    parser.add_argument(
        "--cloud-height", type=float, default=3000,
        help="Cloud-top height in meters (default: 3000)"
    )
    parser.add_argument(
        "--cloud-tau", type=float, default=10,
        help="Cloud visible optical depth (default: 10)"
    )
    parser.add_argument(
        "--sza", type=float, default=30,
        help="Solar zenith angle in degrees (default: 30)"
    )
    parser.add_argument(
        "--vza", type=float, default=0,
        help="Viewing zenith angle in degrees (default: 0 = nadir)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plotting"
    )
    parser.add_argument(
        "--output", type=str, default="satellite_observation.png",
        help="Output filename for the plot"
    )
    return parser.parse_args()


def get_surface_properties(surface_type):
    """Get surface albedo and emissivity for different surface types."""
    properties = {
        "ocean": {"albedo_vis": 0.06, "albedo_nir": 0.04, "emissivity": 0.98},
        "land": {"albedo_vis": 0.15, "albedo_nir": 0.25, "emissivity": 0.95},
        "desert": {"albedo_vis": 0.35, "albedo_nir": 0.40, "emissivity": 0.92},
        "snow": {"albedo_vis": 0.85, "albedo_nir": 0.65, "emissivity": 0.99},
    }
    return properties.get(surface_type, properties["land"])


def brightness_temperature(radiance, wavenumber):
    """
    Convert radiance to brightness temperature using inverse Planck.

    Parameters
    ----------
    radiance : float
        Radiance in W/m^2/sr/cm-^-1
    wavenumber : float
        Wavenumber in cm-^-1

    Returns
    -------
    T_b : float
        Brightness temperature in K
    """
    from raf_tran.utils.constants import PLANCK_CONSTANT, SPEED_OF_LIGHT, BOLTZMANN_CONSTANT

    # Convert wavenumber to m-^-1
    nu = wavenumber * 100  # cm-^-1 to m-^-1

    c1 = 2 * PLANCK_CONSTANT * SPEED_OF_LIGHT**2
    c2 = PLANCK_CONSTANT * SPEED_OF_LIGHT / BOLTZMANN_CONSTANT

    # Inverse Planck function
    # B = c1 * nu^3 / (exp(c2*nu/T) - 1)
    # T = c2 * nu / ln(1 + c1*nu^3/B)

    # Convert radiance from W/m^2/sr/cm-^-1 to W/m^2/sr/m-^-1
    B = radiance / 100  # W/m^2/sr/m-^-1

    if B <= 0:
        return 0.0

    T_b = c2 * nu / np.log(1 + c1 * nu**3 / B)
    return T_b


def main():
    args = parse_args()

    print("=" * 80)
    print("SATELLITE OBSERVATION SIMULATION")
    print("=" * 80)
    print(f"\nSolar zenith angle: {args.sza} deg")
    print(f"Viewing zenith angle: {args.vza} deg")
    print(f"Surface type: {args.surface_type}")
    print(f"Cloud cover: {args.cloud_cover * 100:.0f}%")
    if args.cloud_cover > 0:
        print(f"Cloud height: {args.cloud_height / 1000:.1f} km")
        print(f"Cloud optical depth: {args.cloud_tau}")

    mu0 = np.cos(np.radians(args.sza))
    mu_view = np.cos(np.radians(args.vza))

    # Surface properties
    surface_props = get_surface_properties(args.surface_type)
    print(f"\nSurface properties:")
    print(f"  VIS albedo: {surface_props['albedo_vis']}")
    print(f"  NIR albedo: {surface_props['albedo_nir']}")
    print(f"  Emissivity: {surface_props['emissivity']}")

    # Setup atmosphere
    atmosphere = StandardAtmosphere()
    n_layers = 40
    z_levels = np.linspace(0, 20000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    temperature = atmosphere.temperature(z_mid)
    number_density = atmosphere.number_density(z_mid)
    h2o_vmr = atmosphere.h2o_vmr(z_mid)

    T_surface = args.surface_temp if args.surface_temp else temperature[0]
    print(f"  Surface temperature: {T_surface:.1f} K")

    # Find cloud layer
    cloud_layer_idx = np.argmin(np.abs(z_mid - args.cloud_height))
    T_cloud = temperature[cloud_layer_idx]
    if args.cloud_cover > 0:
        print(f"  Cloud-top temperature: {T_cloud:.1f} K")

    solver = TwoStreamSolver()

    print("\n" + "=" * 80)
    print("SIMULATED SATELLITE OBSERVATIONS")
    print("=" * 80)

    results = {}

    for channel_name, props in SATELLITE_CHANNELS.items():
        print(f"\n--- {channel_name}: {props['description']} ---")

        if props["type"] == "reflective":
            # Shortwave reflective channel
            wl = props["center_wl"]

            # Rayleigh scattering
            rayleigh = RayleighScattering()
            tau_ray = rayleigh.optical_depth(np.array([wl]), number_density, dz).ravel()

            # Clear sky calculation
            if wl < 0.7:
                surface_albedo = surface_props["albedo_vis"]
            else:
                surface_albedo = surface_props["albedo_nir"]

            result_clear = solver.solve_solar(
                tau=tau_ray,
                omega=np.ones(n_layers),
                g=np.zeros(n_layers),
                mu0=mu0,
                flux_toa=SOLAR_CONSTANT,
                surface_albedo=surface_albedo,
            )

            # Reflectance = upward flux / incoming flux
            reflectance_clear = result_clear.flux_up[0] / (SOLAR_CONSTANT * mu0)

            # Cloudy calculation
            if args.cloud_cover > 0:
                tau_cloudy = tau_ray.copy()
                omega_cloudy = np.ones(n_layers)
                g_cloudy = np.zeros(n_layers)

                # Add cloud
                for i in range(cloud_layer_idx, min(cloud_layer_idx + 3, n_layers)):
                    tau_cloudy[i] += args.cloud_tau / 3
                    omega_cloudy[i] = 0.999
                    g_cloudy[i] = 0.85

                result_cloudy = solver.solve_solar(
                    tau=tau_cloudy,
                    omega=omega_cloudy,
                    g=g_cloudy,
                    mu0=mu0,
                    flux_toa=SOLAR_CONSTANT,
                    surface_albedo=surface_albedo,
                )
                reflectance_cloudy = result_cloudy.flux_up[0] / (SOLAR_CONSTANT * mu0)

                # Weighted average
                reflectance = (1 - args.cloud_cover) * reflectance_clear + args.cloud_cover * reflectance_cloudy
            else:
                reflectance = reflectance_clear

            results[channel_name] = {
                "value": reflectance * 100,
                "unit": "%",
                "clear": reflectance_clear * 100,
            }

            print(f"  Clear-sky reflectance: {reflectance_clear * 100:.1f}%")
            if args.cloud_cover > 0:
                print(f"  Cloudy reflectance: {reflectance_cloudy * 100:.1f}%")
            print(f"  Observed reflectance: {reflectance * 100:.1f}%")

        else:
            # Thermal channel
            wavenumber = props["center_wn"]

            # Build optical depth for this channel
            tau_ir = np.zeros(n_layers)

            # Water vapor absorption (varies by channel)
            wv_scale = props.get("tau_wv_scale", 0.5)
            for i in range(n_layers):
                tau_ir[i] = wv_scale * h2o_vmr[i] * number_density[i] * dz[i] * 1e-28

            # CO2 absorption for 15 um channel
            if "CO2" in channel_name:
                tau_ir += props.get("tau_co2", 0) * np.ones(n_layers) / n_layers

            # Clear sky thermal calculation
            omega_ir = np.zeros(n_layers)
            g_ir = np.zeros(n_layers)

            result_clear = solver.solve_thermal(
                tau=tau_ir,
                omega=omega_ir,
                g=g_ir,
                temperature=temperature,
                surface_temperature=T_surface,
                surface_emissivity=surface_props["emissivity"],
            )

            # Convert to brightness temperature
            radiance_clear = result_clear.flux_up[0] / np.pi  # Approximate radiance
            T_b_clear = brightness_temperature(radiance_clear * 0.01, wavenumber)

            # For water vapor and CO2 channels, use weighting function approach
            if "Water Vapor" in channel_name:
                # WV channel sees upper troposphere
                T_b_clear = np.mean(temperature[20:35])  # ~5-10 km
            elif "CO2" in channel_name:
                # CO2 channel sees stratosphere
                T_b_clear = np.mean(temperature[35:])  # Above 10 km

            # Cloudy calculation
            if args.cloud_cover > 0:
                # Cloud acts as blackbody at cloud-top temperature in IR window
                if "Window" in channel_name:
                    T_b_cloudy = T_cloud
                else:
                    # Other channels see above cloud
                    T_b_cloudy = T_b_clear

                T_b = (1 - args.cloud_cover) * T_b_clear + args.cloud_cover * T_b_cloudy
            else:
                T_b = T_b_clear

            results[channel_name] = {
                "value": T_b,
                "unit": "K",
                "clear": T_b_clear,
            }

            print(f"  Clear-sky brightness temp: {T_b_clear:.1f} K")
            if args.cloud_cover > 0 and "Window" in channel_name:
                print(f"  Cloud-top brightness temp: {T_b_cloudy:.1f} K")
            print(f"  Observed brightness temp: {T_b:.1f} K")

    # Analysis
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if args.cloud_cover > 0:
        print(f"""
CLOUD DETECTION:
  - VIS channel shows high reflectance ({results['VIS (0.65 um)']['value']:.0f}%) due to clouds
  - IR Window shows cold brightness temp ({results['IR Window (11 um)']['value']:.0f} K)
    indicating cloud-top at ~{args.cloud_height/1000:.1f} km
  - Cloud-top temp ({T_cloud:.0f} K) vs surface ({T_surface:.0f} K): Delta={T_surface-T_cloud:.0f} K
""")
    else:
        print(f"""
CLEAR SKY OBSERVATIONS:
  - VIS reflectance ({results['VIS (0.65 um)']['value']:.0f}%) shows {args.surface_type} surface
  - IR Window ({results['IR Window (11 um)']['value']:.0f} K) shows surface temperature
  - Water Vapor channel ({results['Water Vapor (6.7 um)']['value']:.0f} K) shows upper troposphere
  - CO2 channel ({results['CO2 (15 um)']['value']:.0f} K) shows stratospheric temperature
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle(f'Satellite Observation Simulation ({args.surface_type}, '
                        f'{args.cloud_cover*100:.0f}% cloud)', fontsize=14, fontweight='bold')

            # Plot 1: Atmospheric temperature profile
            ax1 = axes[0, 0]
            ax1.plot(temperature, z_mid / 1000, 'r-', linewidth=2, label='Atmosphere')
            ax1.axhline(0, color='brown', linewidth=3, label='Surface')
            ax1.plot(T_surface, 0, 'ro', markersize=12)

            if args.cloud_cover > 0:
                ax1.axhspan(args.cloud_height / 1000 - 0.5, args.cloud_height / 1000 + 0.5,
                           color='gray', alpha=0.5, label='Cloud')
                ax1.plot(T_cloud, args.cloud_height / 1000, 'ko', markersize=10)

            ax1.set_xlabel('Temperature (K)')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_title('Temperature Profile')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Channel comparison (bar chart)
            ax2 = axes[0, 1]

            channels = list(results.keys())
            thermal_channels = [c for c in channels if 'um)' in c and float(c.split('(')[1].split()[0]) > 1]
            reflect_channels = [c for c in channels if c not in thermal_channels]

            # Reflective channels
            x_refl = np.arange(len(reflect_channels))
            vals_refl = [results[c]['value'] for c in reflect_channels]
            ax2.bar(x_refl, vals_refl, color='gold', alpha=0.7)
            ax2.set_xticks(x_refl)
            ax2.set_xticklabels([c.split('(')[0] for c in reflect_channels])
            ax2.set_ylabel('Reflectance (%)')
            ax2.set_title('Reflective Channels')
            ax2.grid(True, alpha=0.3, axis='y')

            # Plot 3: Thermal channels
            ax3 = axes[1, 0]
            x_therm = np.arange(len(thermal_channels))
            vals_therm = [results[c]['value'] for c in thermal_channels]
            colors = ['red' if v > 250 else 'blue' for v in vals_therm]
            ax3.bar(x_therm, vals_therm, color=colors, alpha=0.7)
            ax3.set_xticks(x_therm)
            ax3.set_xticklabels([c.split('(')[0] for c in thermal_channels], rotation=45, ha='right')
            ax3.set_ylabel('Brightness Temperature (K)')
            ax3.set_title('Thermal Channels')
            ax3.axhline(T_surface, color='brown', linestyle='--', label=f'Surface T={T_surface:.0f}K')
            if args.cloud_cover > 0:
                ax3.axhline(T_cloud, color='gray', linestyle='--', label=f'Cloud T={T_cloud:.0f}K')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

            # Plot 4: Schematic
            ax4 = axes[1, 1]
            ax4.set_xlim(0, 10)
            ax4.set_ylim(0, 15)

            # Draw atmosphere layers
            ax4.axhspan(0, 1, color='green', alpha=0.3, label='Surface')
            for i in range(1, 12):
                ax4.axhspan(i, i + 1, color='lightblue', alpha=0.05 + 0.02 * i)

            # Draw cloud if present
            if args.cloud_cover > 0:
                cloud_y = args.cloud_height / 1000 * 0.5 + 1
                ax4.add_patch(plt.Rectangle((2, cloud_y), 6, 1, color='gray', alpha=0.7))
                ax4.text(5, cloud_y + 0.5, 'Cloud', ha='center', va='center', fontsize=10)

            # Draw satellite
            ax4.plot(5, 14, 'ks', markersize=20)
            ax4.text(5, 14.5, 'Satellite', ha='center', fontsize=10)

            # Draw radiation arrows
            # Reflected solar
            ax4.annotate('', xy=(3, 14), xytext=(3, 1),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2))
            ax4.text(3.2, 7, 'Reflected\nSolar', fontsize=8, color='orange')

            # Thermal emission
            ax4.annotate('', xy=(7, 14), xytext=(7, 1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            ax4.text(7.2, 7, 'Thermal\nEmission', fontsize=8, color='red')

            ax4.set_title('Observation Geometry')
            ax4.axis('off')

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available, skipping plot generation")

    print("\n" + "=" * 80)
    print("PHYSICAL EXPLANATION")
    print("=" * 80)
    print("""
SATELLITE REMOTE SENSING:

1. REFLECTIVE CHANNELS (VIS, NIR):
   - Measure reflected sunlight
   - High reflectance = bright (clouds, snow)
   - Low reflectance = dark (ocean, forest)
   - Need sun illumination

2. THERMAL CHANNELS (IR):
   - Measure emitted thermal radiation
   - Convert to brightness temperature (T_B)
   - T_B depends on emitting surface temperature
   - Work day and night

3. WEIGHTING FUNCTIONS:
   - Each channel has a "weighting function"
   - Describes which altitude contributes most
   - Window channels: see surface/cloud top
   - Absorbing channels: see specific altitude

4. CLOUD DETECTION:
   - VIS: Clouds are bright (high reflectance)
   - IR Window: Clouds are cold (low T_B)
   - Difference methods identify cloud height

5. ATMOSPHERIC SOUNDING:
   - Multiple thermal channels at different absorption
   - Each sees different altitude
   - Temperature profile retrieval
""")


if __name__ == "__main__":
    main()
