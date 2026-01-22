#!/usr/bin/env python3
"""
Thermal Radiation Example
=========================

This example demonstrates thermal (longwave) radiative transfer
calculations using RAF-tran.

The example computes:
1. Planck blackbody emission at different temperatures
2. Thermal flux in a multi-layer atmosphere
3. Radiative heating rates
"""

import numpy as np
import matplotlib.pyplot as plt

from raf_tran.atmosphere import StandardAtmosphere, TropicalAtmosphere
from raf_tran.utils.spectral import planck_function_wavenumber, stefan_boltzmann_flux
from raf_tran.utils.constants import STEFAN_BOLTZMANN
from raf_tran.rte_solver import TwoStreamSolver


def main():
    print("Thermal Radiation Example")
    print("=" * 50)

    # 1. Planck function at different temperatures
    print("\n1. Planck blackbody emission")
    print("-" * 40)

    wavenumbers = np.linspace(100, 2500, 500)  # cm⁻¹
    temperatures = [220, 260, 300]  # K

    fig, ax = plt.subplots(figsize=(10, 6))

    for T in temperatures:
        B = planck_function_wavenumber(wavenumbers, T)
        ax.plot(wavenumbers, B * 1e3, label=f"T = {T} K")

        # Find peak wavenumber
        peak_idx = np.argmax(B)
        print(f"T = {T} K: Peak at {wavenumbers[peak_idx]:.0f} cm⁻¹ "
              f"({10000/wavenumbers[peak_idx]:.1f} μm)")

    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(r"Spectral Radiance (mW/m$^2$/sr/cm$^{-1}$)")
    ax.set_title("Planck Blackbody Emission")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 2500)

    plt.tight_layout()
    plt.savefig("planck_emission.png", dpi=150)
    print("\nSaved: planck_emission.png")

    # 2. Stefan-Boltzmann law verification
    print("\n2. Stefan-Boltzmann flux")
    print("-" * 40)

    T_surface = 288.15  # K (15°C)
    flux_sb = stefan_boltzmann_flux(T_surface)
    print(f"Surface temperature: {T_surface:.2f} K ({T_surface - 273.15:.2f}°C)")
    print(f"Stefan-Boltzmann flux: {flux_sb:.1f} W/m²")

    # Earth's effective temperature
    solar_constant = 1361  # W/m²
    albedo = 0.3
    T_effective = ((solar_constant * (1 - albedo) / 4) / STEFAN_BOLTZMANN) ** 0.25
    print(f"\nEarth's effective temperature (no atmosphere): {T_effective:.1f} K")
    print(f"Actual mean surface temperature: ~288 K")
    print(f"Greenhouse effect: ~{288 - T_effective:.0f} K warming")

    # 3. Thermal radiative transfer in atmosphere
    print("\n3. Atmospheric thermal radiation")
    print("-" * 40)

    # Create atmospheric profile
    atmosphere = StandardAtmosphere()

    # Define layers
    n_layers = 20
    z_levels = np.linspace(0, 30000, n_layers + 1)
    z_mid = (z_levels[:-1] + z_levels[1:]) / 2

    temperature = atmosphere.temperature(z_mid)
    pressure_levels = atmosphere.pressure(z_levels)

    print(f"Number of layers: {n_layers}")
    print(f"Top of atmosphere: {z_levels[-1]/1000:.0f} km")
    print(f"Surface temperature: {temperature[0]:.1f} K")
    print(f"TOA temperature: {temperature[-1]:.1f} K")

    # Simplified optical depth model for thermal radiation
    # In reality, this would come from gas absorption (H2O, CO2, O3)
    # Here we use a simple gray atmosphere approximation

    # Optical depth decreasing with altitude (water vapor dominated)
    tau_surface = 2.0  # Total optical depth
    scale_height = 2000  # meters

    tau_per_layer = np.zeros(n_layers)
    for i in range(n_layers):
        z_bot = z_levels[i]
        z_top = z_levels[i + 1]
        # Exponential decrease
        tau_per_layer[i] = (
            tau_surface * scale_height / (z_levels[-1])
            * (np.exp(-z_bot / scale_height) - np.exp(-z_top / scale_height))
        )

    total_tau = np.sum(tau_per_layer)
    print(f"Total thermal optical depth: {total_tau:.2f}")

    # Two-stream calculation for thermal (longwave) radiation
    solver = TwoStreamSolver()

    # Pure absorption (no scattering in thermal IR)
    omega = np.zeros(n_layers)
    g = np.zeros(n_layers)

    result = solver.solve_thermal(
        tau=tau_per_layer,
        omega=omega,
        g=g,
        temperature=temperature,
        surface_temperature=temperature[0],
        surface_emissivity=1.0,
    )

    # Calculate heating rates
    heating_rate = solver.compute_heating_rate(
        result.flux_up,
        result.flux_down,
        result.flux_direct,
        pressure_levels,
    )

    # Results
    print(f"\nUpward flux at surface: {result.flux_up[-1]:.1f} W/m²")
    print(f"Upward flux at TOA (OLR): {result.flux_up[0]:.1f} W/m²")
    print(f"Downward flux at surface: {result.flux_down[-1]:.1f} W/m²")

    # Net radiation
    net_surface = result.flux_down[-1] - result.flux_up[-1]
    print(f"Net flux at surface: {net_surface:.1f} W/m²")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Temperature profile
    ax1 = axes[0]
    ax1.plot(temperature, z_mid / 1000)
    ax1.set_xlabel("Temperature (K)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Temperature Profile")
    ax1.grid(True, alpha=0.3)

    # Flux profiles
    ax2 = axes[1]
    ax2.plot(result.flux_up, z_levels / 1000, "r-", label="Upward")
    ax2.plot(result.flux_down, z_levels / 1000, "b-", label="Downward")
    ax2.plot(result.flux_up - result.flux_down, z_levels / 1000, "k--", label="Net")
    ax2.set_xlabel("Flux (W/m²)")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title("Thermal Flux Profiles")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Heating rates
    ax3 = axes[2]
    ax3.plot(heating_rate, z_mid / 1000)
    ax3.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax3.set_xlabel("Heating Rate (K/day)")
    ax3.set_ylabel("Altitude (km)")
    ax3.set_title("Radiative Heating Rate")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("thermal_radiation.png", dpi=150)
    print("\nSaved: thermal_radiation.png")

    # 4. Comparison of atmospheric profiles
    print("\n4. Comparison of different atmospheres")
    print("-" * 40)

    atmospheres = {
        "US Standard": StandardAtmosphere(),
        "Tropical": TropicalAtmosphere(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for name, atm in atmospheres.items():
        T = atm.temperature(z_mid)
        surface_flux = stefan_boltzmann_flux(T[0])
        print(f"{name}: Surface T = {T[0]:.1f} K, "
              f"Surface emission = {surface_flux:.1f} W/m²")

        axes[0].plot(T, z_mid / 1000, label=name)
        axes[1].plot(atm.h2o_vmr(z_mid) * 1e6, z_mid / 1000, label=name)

    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Temperature Profiles")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("H₂O VMR (ppmv)")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].set_title("Water Vapor Profiles")
    axes[1].legend()
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("atmosphere_comparison.png", dpi=150)
    print("\nSaved: atmosphere_comparison.png")


if __name__ == "__main__":
    main()
