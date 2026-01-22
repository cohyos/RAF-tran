#!/usr/bin/env python3
"""
Mie Scattering Example for Aerosols
===================================

This example demonstrates Mie scattering calculations for
atmospheric aerosols using RAF-tran.

The example computes:
1. Mie efficiencies for a range of size parameters
2. Wavelength-dependent optical properties
3. Aerosol optical depth for a lognormal size distribution
"""

import numpy as np
import matplotlib.pyplot as plt

from raf_tran.scattering import MieScattering
from raf_tran.scattering.mie import lognormal_size_distribution, mie_efficiencies


def main():
    print("Mie Scattering Example for Atmospheric Aerosols")
    print("=" * 50)

    # Common aerosol refractive indices
    # Sulfate: ~1.43 + 0i (non-absorbing)
    # Dust: ~1.55 + 0.003i (weakly absorbing)
    # Black carbon: ~1.95 + 0.79i (strongly absorbing)

    print("\n1. Single particle Mie efficiencies")
    print("-" * 40)

    # Calculate Mie efficiencies for sulfate aerosol
    m_sulfate = 1.43 + 0.0j
    size_params = np.logspace(-1, 2, 100)  # x = 0.1 to 100

    Q_ext = np.zeros_like(size_params)
    Q_sca = np.zeros_like(size_params)
    Q_abs = np.zeros_like(size_params)
    g = np.zeros_like(size_params)

    for i, x in enumerate(size_params):
        Q_ext[i], Q_sca[i], Q_abs[i], g[i] = mie_efficiencies(x, m_sulfate)

    print(f"Sulfate refractive index: {m_sulfate}")
    print(f"At x=1: Q_ext={Q_ext[size_params >= 1][0]:.3f}, g={g[size_params >= 1][0]:.3f}")
    print(f"At x=10: Q_ext={Q_ext[size_params >= 10][0]:.3f}, g={g[size_params >= 10][0]:.3f}")

    # Plot Mie efficiencies
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.loglog(size_params, Q_ext, label=r"$Q_{ext}$")
    ax1.loglog(size_params, Q_sca, "--", label=r"$Q_{sca}$")
    ax1.loglog(size_params, Q_abs + 1e-10, ":", label=r"$Q_{abs}$")
    ax1.set_xlabel("Size parameter x")
    ax1.set_ylabel("Efficiency")
    ax1.set_title("Mie Efficiencies (Sulfate)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-4, 10)

    ax2 = axes[1]
    ax2.semilogx(size_params, g)
    ax2.set_xlabel("Size parameter x")
    ax2.set_ylabel("Asymmetry parameter g")
    ax2.set_title("Asymmetry Parameter (Sulfate)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("mie_efficiencies.png", dpi=150)
    print("\nSaved: mie_efficiencies.png")

    # Compare different aerosol types
    print("\n2. Comparison of aerosol types")
    print("-" * 40)

    aerosol_types = {
        "Sulfate": 1.43 + 0.0j,
        "Dust": 1.55 + 0.003j,
        "Black Carbon": 1.95 + 0.79j,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name, m in aerosol_types.items():
        Q_ext = np.zeros_like(size_params)
        Q_sca = np.zeros_like(size_params)
        ssa = np.zeros_like(size_params)

        for i, x in enumerate(size_params):
            Q_ext[i], Q_sca[i], _, _ = mie_efficiencies(x, m)
            ssa[i] = Q_sca[i] / Q_ext[i] if Q_ext[i] > 0 else 1.0

        axes[0].loglog(size_params, Q_ext, label=name)
        axes[1].semilogx(size_params, ssa, label=name)

        # Print values at x=5 (typical for ~0.5 μm particles at visible wavelengths)
        idx = np.argmin(np.abs(size_params - 5))
        print(f"{name}: Q_ext={Q_ext[idx]:.3f}, SSA={ssa[idx]:.3f}")

    axes[0].set_xlabel("Size parameter x")
    axes[0].set_ylabel(r"$Q_{ext}$")
    axes[0].set_title("Extinction Efficiency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Size parameter x")
    axes[1].set_ylabel("Single Scattering Albedo")
    axes[1].set_title("Single Scattering Albedo")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("aerosol_comparison.png", dpi=150)
    print("\nSaved: aerosol_comparison.png")

    # Wavelength-dependent AOD
    print("\n3. Wavelength-dependent aerosol optical depth")
    print("-" * 40)

    mie = MieScattering(refractive_index=1.45 + 0.001j)

    wavelengths = np.linspace(0.3, 1.0, 50)  # μm

    # Aerosol parameters
    r_g = 0.15  # μm, geometric mean radius
    sigma_g = 1.8  # geometric standard deviation
    N_total = 1e9  # particles/m³
    layer_thickness = 2000  # m (2 km boundary layer)

    # Calculate size distribution
    radii = np.logspace(-2, 1, 100)  # 0.01 to 10 μm
    n_r = lognormal_size_distribution(radii, r_g, sigma_g, N_total)

    # Calculate AOD for each wavelength (integrate over size distribution)
    aod = np.zeros_like(wavelengths)

    for i, wl in enumerate(wavelengths):
        # Weighted average Q_ext over size distribution
        Q_ext_avg = 0.0
        cross_section_avg = 0.0

        for j, r in enumerate(radii):
            x = 2 * np.pi * r / wl
            Q_ext_j, _, _, _ = mie_efficiencies(x, mie.refractive_index)
            geometric_cs = np.pi * r**2

            # Weight by size distribution
            if j > 0:
                dr = radii[j] - radii[j - 1]
                cross_section_avg += Q_ext_j * geometric_cs * n_r[j] * dr

        aod[i] = cross_section_avg * layer_thickness * 1e-12  # convert μm² to m²

    # Fit Angstrom exponent
    # AOD ∝ λ^(-α)
    log_wl = np.log(wavelengths)
    log_aod = np.log(aod)
    angstrom = -np.polyfit(log_wl, log_aod, 1)[0]

    print(f"Geometric mean radius: {r_g} μm")
    print(f"Geometric std dev: {sigma_g}")
    print(f"Number concentration: {N_total:.1e} particles/m³")
    print(f"Layer thickness: {layer_thickness/1000} km")
    print(f"AOD at 550 nm: {np.interp(0.55, wavelengths, aod):.4f}")
    print(f"Angstrom exponent: {angstrom:.2f}")

    # Plot AOD
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(wavelengths * 1000, aod, "b-", linewidth=2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Aerosol Optical Depth")
    ax.set_title(f"Aerosol Optical Depth (Ångström α = {angstrom:.2f})")
    ax.grid(True, alpha=0.3)
    ax.axvline(550, color="g", linestyle="--", alpha=0.5, label="550 nm")
    ax.legend()

    plt.tight_layout()
    plt.savefig("aerosol_aod.png", dpi=150)
    print("\nSaved: aerosol_aod.png")


if __name__ == "__main__":
    main()
