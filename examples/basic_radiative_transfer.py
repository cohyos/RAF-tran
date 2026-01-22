#!/usr/bin/env python3
"""
Basic Radiative Transfer Example
================================

This example demonstrates how to use RAF-tran to perform a simple
radiative transfer calculation for a clear-sky atmosphere.

The example computes:
1. Atmospheric profiles (temperature, pressure, gas concentrations)
2. Rayleigh scattering optical depth
3. Solar flux transmission through the atmosphere
"""

import numpy as np
import matplotlib.pyplot as plt

from raf_tran.atmosphere import StandardAtmosphere
from raf_tran.scattering import RayleighScattering
from raf_tran.rte_solver import TwoStreamSolver


def main():
    # Create standard atmosphere
    print("Creating US Standard Atmosphere 1976...")
    atmosphere = StandardAtmosphere()

    # Define altitude grid (0 to 50 km)
    z_levels = np.linspace(0, 50000, 51)  # 51 levels = 50 layers
    n_layers = len(z_levels) - 1

    # Get atmospheric properties at layer midpoints
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2
    dz = np.diff(z_levels)

    temperature = atmosphere.temperature(z_mid)
    pressure = atmosphere.pressure(z_mid)
    density = atmosphere.density(z_mid)
    number_density = atmosphere.number_density(z_mid)

    print(f"Surface temperature: {temperature[0]:.1f} K")
    print(f"Surface pressure: {pressure[0]:.0f} Pa")
    print(f"Number of layers: {n_layers}")

    # Calculate Rayleigh scattering optical depth at 550 nm
    print("\nCalculating Rayleigh scattering...")
    rayleigh = RayleighScattering()
    wavelength = np.array([0.55])  # 550 nm in micrometers

    tau_rayleigh = rayleigh.optical_depth(wavelength, number_density, dz)
    tau_total_rayleigh = np.sum(tau_rayleigh)

    print(f"Total Rayleigh optical depth at 550 nm: {tau_total_rayleigh:.4f}")

    # Perform two-stream radiative transfer calculation
    print("\nPerforming two-stream calculation...")
    solver = TwoStreamSolver()

    # Solar zenith angle (60 degrees)
    sza = 60.0
    mu0 = np.cos(np.radians(sza))

    # Incoming solar flux at 550 nm (approximate)
    solar_flux = 1.9  # W/m²/nm at 550 nm

    # For Rayleigh: single scattering albedo = 1, asymmetry parameter = 0
    omega = np.ones(n_layers)  # Pure scattering
    g = np.zeros(n_layers)  # Isotropic (for Rayleigh)

    result = solver.solve_solar(
        tau=tau_rayleigh.ravel(),
        omega=omega,
        g=g,
        mu0=mu0,
        flux_toa=solar_flux,
        surface_albedo=0.1,  # 10% surface albedo
    )

    # Results
    print(f"\nResults for solar zenith angle = {sza}°:")
    print(f"  Direct flux at TOA: {result.flux_direct[0]:.4f} W/m²/nm")
    print(f"  Direct flux at surface: {result.flux_direct[-1]:.4f} W/m²/nm")
    print(f"  Diffuse downward flux at surface: {result.flux_down[-1]:.4f} W/m²/nm")
    print(f"  Upward flux at TOA (reflected): {result.flux_up[0]:.4f} W/m²/nm")

    # Calculate transmittance
    direct_transmittance = result.flux_direct[-1] / result.flux_direct[0]
    print(f"  Direct beam transmittance: {direct_transmittance:.4f}")

    # Plot atmospheric profile
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Temperature profile
    ax1 = axes[0]
    ax1.plot(temperature, z_mid / 1000)
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Temperature Profile")
    ax1.grid(True, alpha=0.3)

    # Pressure profile
    ax2 = axes[1]
    ax2.semilogx(pressure, z_mid / 1000)
    ax2.set_xlabel("Pressure (Pa)")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title("Pressure Profile")
    ax2.grid(True, alpha=0.3)

    # Flux profiles
    ax3 = axes[2]
    ax3.plot(result.flux_direct, z_levels / 1000, label="Direct")
    ax3.plot(result.flux_down, z_levels / 1000, label="Diffuse down")
    ax3.plot(result.flux_up, z_levels / 1000, label="Diffuse up")
    ax3.set_xlabel("Flux (W/m²/nm)")
    ax3.set_ylabel("Altitude (km)")
    ax3.set_title("Flux Profiles at 550 nm")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("radiative_transfer_example.png", dpi=150)
    print("Saved plot to: radiative_transfer_example.png")


if __name__ == "__main__":
    main()
