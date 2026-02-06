#!/usr/bin/env python3
"""
Example 32: Configuration File Usage
=====================================

Demonstrates how to use YAML and JSON configuration files with RAF-tran
for reproducible and shareable simulation setups.

Features:
- Load simulation parameters from YAML/JSON files
- Validate configuration before running
- Create default configuration templates
- Modify configurations programmatically

Usage:
    python 32_config_file_demo.py
    python 32_config_file_demo.py --config configs/cloudy_atmosphere.yaml
    python 32_config_file_demo.py --create-default my_config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from raf_tran.utils.config import (
    SimulationConfig,
    AtmosphereConfig,
    AerosolConfig,
    CloudConfig,
    SurfaceConfig,
    SolarConfig,
    load_config,
    create_default_config,
    validate_config,
)
from raf_tran.atmosphere import StandardAtmosphere
from raf_tran.scattering import RayleighScattering, MieScattering
from raf_tran.rte_solver import TwoStreamSolver, TwoStreamMethod
from raf_tran.utils.spectral import planck_function


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configuration file usage demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML or JSON configuration file"
    )
    parser.add_argument(
        "--create-default", type=str, default=None,
        metavar="PATH",
        help="Create a default configuration file at specified path"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate the configuration, don't run simulation"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation"
    )
    return parser.parse_args()


def run_simulation_from_config(config: SimulationConfig, show_plot: bool = True):
    """Run a simplified radiative transfer simulation based on config."""

    print(f"\n{'='*60}")
    print(f"Running Simulation: {config.name}")
    print(f"{'='*60}")
    print(f"Description: {config.description}")

    # Create atmosphere
    print(f"\n--- Atmosphere Configuration ---")
    print(f"  Model: {config.atmosphere.model}")
    print(f"  Layers: {config.atmosphere.n_layers}")
    print(f"  Top altitude: {config.atmosphere.top_altitude} km")
    print(f"  Surface pressure: {config.atmosphere.surface_pressure} hPa")
    print(f"  Surface temperature: {config.atmosphere.surface_temperature} K")

    atmosphere = StandardAtmosphere()

    # Create altitude grid
    z_top = config.atmosphere.top_altitude * 1000  # km to m
    z_levels = np.linspace(0, z_top, config.atmosphere.n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    # Get atmospheric properties
    number_density = atmosphere.number_density(z_mid)

    # Generate wavelength grid
    print(f"\n--- Spectral Configuration ---")
    print(f"  Wavelength range: {config.spectral.wavelength_min} - {config.spectral.wavelength_max} um")
    print(f"  Number of wavelengths: {config.spectral.n_wavelengths}")

    if config.spectral.wavelengths:
        wavelengths = np.array(config.spectral.wavelengths)
    else:
        wavelengths = np.linspace(
            config.spectral.wavelength_min,
            config.spectral.wavelength_max,
            config.spectral.n_wavelengths
        )

    # Solar configuration
    print(f"\n--- Solar Configuration ---")
    print(f"  Solar zenith angle: {config.solar.solar_zenith_angle} deg")
    print(f"  Day of year: {config.solar.day_of_year}")

    mu0 = np.cos(np.radians(config.solar.solar_zenith_angle))

    # Calculate Rayleigh optical depth
    rayleigh = RayleighScattering()

    # Aerosol configuration
    print(f"\n--- Aerosol Configuration ---")
    print(f"  Enabled: {config.aerosol.enabled}")
    if config.aerosol.enabled:
        print(f"  Model: {config.aerosol.model}")
        print(f"  AOD at 550nm: {config.aerosol.aod_550}")
        print(f"  Angstrom exponent: {config.aerosol.angstrom_exponent}")

    # Cloud configuration
    print(f"\n--- Cloud Configuration ---")
    print(f"  Enabled: {config.cloud.enabled}")
    if config.cloud.enabled:
        print(f"  Type: {config.cloud.cloud_type}")
        print(f"  Base: {config.cloud.cloud_base} km")
        print(f"  Top: {config.cloud.cloud_top} km")
        print(f"  Optical depth: {config.cloud.optical_depth}")

    # Surface configuration
    print(f"\n--- Surface Configuration ---")
    print(f"  Type: {config.surface.surface_type}")
    print(f"  Albedo: {config.surface.albedo}")
    print(f"  Emissivity: {config.surface.emissivity}")

    # Run radiative transfer for selected wavelengths
    print(f"\n--- Running Radiative Transfer ---")

    # Map config method string to TwoStreamMethod enum
    method_map = {
        "eddington": TwoStreamMethod.EDDINGTON,
        "quadrature": TwoStreamMethod.QUADRATURE,
        "hemispheric_mean": TwoStreamMethod.HEMISPHERIC_MEAN,
        "delta_eddington": TwoStreamMethod.DELTA_EDDINGTON,
    }
    solver_method = method_map.get(config.solver.method, TwoStreamMethod.DELTA_EDDINGTON)
    solver = TwoStreamSolver(method=solver_method)

    # Sample wavelengths for calculation
    sample_wavelengths = np.array([0.4, 0.55, 0.7, 1.0, 1.5])  # um
    sample_wavelengths = sample_wavelengths[
        (sample_wavelengths >= config.spectral.wavelength_min) &
        (sample_wavelengths <= config.spectral.wavelength_max)
    ]

    results = []
    for wl in sample_wavelengths:
        # Rayleigh optical depth
        tau_ray = rayleigh.optical_depth(
            np.array([wl]), number_density, dz
        ).ravel()

        # Add aerosol optical depth if enabled
        tau_aer = np.zeros_like(tau_ray)
        omega_aer = np.ones_like(tau_ray) * config.aerosol.single_scatter_albedo
        g_aer = np.ones_like(tau_ray) * config.aerosol.asymmetry_parameter

        if config.aerosol.enabled:
            # Scale AOD with wavelength using Angstrom exponent
            aod = config.aerosol.aod_550 * (wl / 0.55) ** (-config.aerosol.angstrom_exponent)
            # Distribute with scale height
            h_scale = config.aerosol.scale_height * 1000  # km to m
            aer_profile = np.exp(-z_mid / h_scale)
            aer_profile /= aer_profile.sum()
            tau_aer = aod * aer_profile * len(aer_profile)

        # Add cloud optical depth if enabled
        tau_cld = np.zeros_like(tau_ray)
        if config.cloud.enabled:
            cloud_base_m = config.cloud.cloud_base * 1000
            cloud_top_m = config.cloud.cloud_top * 1000
            cloud_mask = (z_mid >= cloud_base_m) & (z_mid <= cloud_top_m)
            n_cloud_layers = cloud_mask.sum()
            if n_cloud_layers > 0:
                tau_cld[cloud_mask] = config.cloud.optical_depth / n_cloud_layers

        # Total optical properties
        tau_total = tau_ray + tau_aer + tau_cld
        omega = np.where(
            tau_total > 0,
            (tau_ray * 1.0 + tau_aer * omega_aer + tau_cld * 0.999999) / tau_total,
            1.0
        )
        g = np.where(
            tau_total > 0,
            (tau_ray * 0.0 + tau_aer * g_aer + tau_cld * 0.85) / tau_total,
            0.0
        )

        # Solve
        result = solver.solve_solar(
            tau=tau_total,
            omega=omega,
            g=g,
            mu0=mu0,
            flux_toa=1.0,  # Normalized
            surface_albedo=config.surface.albedo,
        )

        results.append({
            'wavelength': wl,
            'tau_total': tau_total.sum(),
            'tau_ray': tau_ray.sum(),
            'tau_aer': tau_aer.sum(),
            'tau_cld': tau_cld.sum(),
            'direct_surface': result.flux_direct[-1],
            'diffuse_surface': result.flux_down[-1],
            'upwelling_toa': result.flux_up[0],
        })

    # Print results
    print(f"\n{'Wavelength':<12} {'Tau_total':<10} {'Direct':<10} {'Diffuse':<10} {'Reflected':<10}")
    print("-" * 52)
    for r in results:
        print(f"{r['wavelength']:<12.2f} {r['tau_total']:<10.3f} {r['direct_surface']:<10.4f} "
              f"{r['diffuse_surface']:<10.4f} {r['upwelling_toa']:<10.4f}")

    # Calculate broadband quantities
    total_surface = sum(r['direct_surface'] + r['diffuse_surface'] for r in results)
    total_reflected = sum(r['upwelling_toa'] for r in results)

    print(f"\n--- Summary ---")
    print(f"Total optical depth at 550nm: {results[1]['tau_total']:.3f}" if len(results) > 1 else "")
    print(f"Planetary albedo (approx): {total_reflected / len(results):.3f}")

    # Plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"{config.name}: Configuration File Demo", fontsize=14)

            # Plot 1: Optical depth components
            ax = axes[0, 0]
            wls = [r['wavelength'] for r in results]
            tau_rays = [r['tau_ray'] for r in results]
            tau_aers = [r['tau_aer'] for r in results]
            tau_clds = [r['tau_cld'] for r in results]

            ax.bar(wls, tau_rays, width=0.08, label='Rayleigh', alpha=0.8)
            ax.bar(wls, tau_aers, width=0.08, bottom=tau_rays, label='Aerosol', alpha=0.8)
            bottom = [r + a for r, a in zip(tau_rays, tau_aers)]
            ax.bar(wls, tau_clds, width=0.08, bottom=bottom, label='Cloud', alpha=0.8)
            ax.set_xlabel('Wavelength (um)')
            ax.set_ylabel('Optical Depth')
            ax.set_title('Optical Depth Components')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Surface fluxes
            ax = axes[0, 1]
            direct = [r['direct_surface'] for r in results]
            diffuse = [r['diffuse_surface'] for r in results]
            ax.plot(wls, direct, 'o-', label='Direct', linewidth=2, markersize=8)
            ax.plot(wls, diffuse, 's-', label='Diffuse', linewidth=2, markersize=8)
            ax.set_xlabel('Wavelength (um)')
            ax.set_ylabel('Normalized Flux')
            ax.set_title('Surface Irradiance')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 3: Atmospheric profile
            ax = axes[1, 0]
            T = atmosphere.temperature(z_mid)
            ax.plot(T, z_mid / 1000, 'r-', linewidth=2)
            ax.set_xlabel('Temperature (K)')
            ax.set_ylabel('Altitude (km)')
            ax.set_title('Temperature Profile')
            ax.grid(True, alpha=0.3)

            # Add cloud layer indication if enabled
            if config.cloud.enabled:
                ax.axhspan(config.cloud.cloud_base, config.cloud.cloud_top,
                          alpha=0.3, color='gray', label='Cloud layer')
                ax.legend()

            # Plot 4: Configuration summary
            ax = axes[1, 1]
            ax.axis('off')

            config_text = f"""Configuration Summary

Name: {config.name}
Description: {config.description}

Atmosphere:
  Model: {config.atmosphere.model}
  Layers: {config.atmosphere.n_layers}
  Top: {config.atmosphere.top_altitude} km

Aerosol:
  Enabled: {config.aerosol.enabled}
  {'AOD: ' + str(config.aerosol.aod_550) if config.aerosol.enabled else ''}

Cloud:
  Enabled: {config.cloud.enabled}
  {'Tau: ' + str(config.cloud.optical_depth) if config.cloud.enabled else ''}

Solar:
  SZA: {config.solar.solar_zenith_angle} deg
  DOY: {config.solar.day_of_year}

Surface:
  Type: {config.surface.surface_type}
  Albedo: {config.surface.albedo}
"""
            ax.text(0.1, 0.9, config_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.savefig('config_file_demo.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved: config_file_demo.png")
            plt.close()

        except ImportError:
            print("\nMatplotlib not available for plotting")

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("RAF-TRAN CONFIGURATION FILE DEMO")
    print("Example 32: Using YAML/JSON Configuration Files")
    print("=" * 60)

    # Create default config if requested
    if args.create_default:
        path = Path(args.create_default)
        print(f"\nCreating default configuration at: {path}")
        config = create_default_config(path)
        print(f"Configuration saved successfully!")
        print(f"\nYou can now edit this file and run:")
        print(f"  python {Path(__file__).name} --config {path}")
        return

    # Load configuration from file or use default
    if args.config:
        config_path = Path(args.config)
        print(f"\nLoading configuration from: {config_path}")

        if not config_path.exists():
            print(f"ERROR: Configuration file not found: {config_path}")
            sys.exit(1)

        config = load_config(config_path)
        print(f"Configuration loaded: {config.name}")
    else:
        print("\nNo configuration file specified, using defaults.")
        print("Create a config file with: --create-default my_config.yaml")

        # Create a demonstration config programmatically
        config = SimulationConfig(
            name="demo_clear_sky",
            description="Clear sky demonstration with moderate aerosols",
        )
        # Modify some defaults
        config.solar.solar_zenith_angle = 45.0
        config.aerosol.aod_550 = 0.15
        config.surface.albedo = 0.2
        config.surface.surface_type = "grass"

    # Validate configuration
    print("\n--- Validating Configuration ---")
    issues = validate_config(config)

    if issues:
        print("Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        if args.validate_only:
            sys.exit(1)
    else:
        print("Configuration valid!")

    if args.validate_only:
        print("\nValidation complete. Use --config to run a simulation.")
        return

    # Run simulation
    run_simulation_from_config(config, show_plot=not args.no_plot)

    # Demonstrate programmatic modification
    print("\n" + "=" * 60)
    print("PROGRAMMATIC CONFIGURATION MODIFICATION")
    print("=" * 60)

    print("\nModifying configuration programmatically...")
    config.name = "modified_config"
    config.aerosol.aod_550 = 0.5  # Increase aerosol loading
    config.cloud.enabled = True   # Enable clouds
    config.cloud.optical_depth = 5.0

    print(f"  - Increased AOD to {config.aerosol.aod_550}")
    print(f"  - Enabled cloud with tau={config.cloud.optical_depth}")

    # Save modified config
    output_path = Path("modified_config.yaml")
    try:
        config.to_yaml(output_path)
        print(f"\nModified configuration saved to: {output_path}")
    except ImportError as e:
        print(f"\nNote: {e}")
        # Fall back to JSON
        output_path = Path("modified_config.json")
        config.to_json(output_path)
        print(f"Configuration saved as JSON: {output_path}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
