#!/usr/bin/env python3
"""
Volcanic Aerosol Radiative Forcing
==================================

This example models the strong cooling effect of stratospheric sulfate
aerosols after major volcanic eruptions.

Key volcanic eruptions:
- Pinatubo 1991: ~0.5 K global cooling, AOD ~0.15
- El Chichon 1982: ~0.3 K cooling
- Krakatoa 1883: 0.3-0.5 K cooling
- Tambora 1815: "Year without a summer"

Physics:
- SO2 oxidizes to H2SO4 droplets in stratosphere
- Small droplets (r ~ 0.1-0.5 um) scatter solar radiation
- Absorb some IR (weak warming effect)
- Net effect: strong cooling

Applications:
- Climate modeling
- Geoengineering studies (stratospheric aerosol injection)
- Paleoclimate analysis

Usage:
    python 24_volcanic_aerosol_forcing.py
    python 24_volcanic_aerosol_forcing.py --aod 0.15 --radius 0.3
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering import MieScattering
    from raf_tran.rte_solver import TwoStreamSolver
    from raf_tran.utils.constants import SOLAR_CONSTANT
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model volcanic aerosol radiative forcing"
    )
    parser.add_argument("--aod", type=float, default=0.15, help="Aerosol optical depth at 550 nm")
    parser.add_argument("--radius", type=float, default=0.3, help="Effective radius (um)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="volcanic_forcing.png")
    return parser.parse_args()


def sulfate_refractive_index(wavelength_um):
    """
    Refractive index of sulfuric acid droplets (75% H2SO4).

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers

    Returns
    -------
    n : complex
        Complex refractive index
    """
    # Simplified model based on Palmer & Williams (1975)
    wl = wavelength_um

    # Real part (dispersion)
    if wl < 0.5:
        n_real = 1.44
    elif wl < 2:
        n_real = 1.43
    elif wl < 10:
        n_real = 1.40 + 0.05 * (wl - 2) / 8
    else:
        n_real = 1.45

    # Imaginary part (absorption)
    if wl < 2:
        n_imag = 1e-8
    elif wl < 4:
        n_imag = 1e-5
    elif wl < 12:
        n_imag = 0.01 + 0.05 * (wl - 4) / 8
    else:
        n_imag = 0.1

    return complex(n_real, n_imag)


def calculate_volcanic_forcing(aod_550, effective_radius, wavelength_um=0.55):
    """
    Calculate radiative forcing from volcanic aerosols.

    Uses simple approximations based on Lacis et al. (1992).
    """
    # Solar weighted properties
    # Forcing ~ -25 W/m^2 per unit AOD for sulfate aerosols
    forcing_per_aod = -25.0  # W/m^2

    # Adjustment for particle size (optimal ~0.3 um)
    size_factor = np.exp(-0.5 * ((effective_radius - 0.3) / 0.2)**2)

    forcing = forcing_per_aod * aod_550 * size_factor

    return forcing


def aod_decay_model(time_months, initial_aod, e_folding_months=12):
    """
    Model AOD decay after eruption.

    Stratospheric aerosols have typical e-folding time of 12-18 months.
    """
    return initial_aod * np.exp(-time_months / e_folding_months)


def main():
    args = parse_args()

    print("=" * 70)
    print("VOLCANIC AEROSOL RADIATIVE FORCING")
    print("=" * 70)
    print(f"\nAerosol optical depth (550 nm): {args.aod}")
    print(f"Effective radius: {args.radius} um")

    # Historical context
    print("\n" + "-" * 70)
    print("MAJOR VOLCANIC ERUPTIONS AND CLIMATE IMPACT")
    print("-" * 70)
    print("""
Eruption         Year    VEI    Peak AOD    Global Cooling
--------         ----    ---    --------    --------------
Tambora          1815    7      ~0.5        ~0.5 K ("Year without summer")
Krakatoa         1883    6      ~0.15       ~0.3 K
El Chichon       1982    5      ~0.08       ~0.2 K
Pinatubo         1991    6      ~0.15       ~0.5 K
Hunga Tonga      2022    5+     ~0.01       ~0.05 K (unusual - water vapor)

VEI = Volcanic Explosivity Index (logarithmic scale)
""")

    # Physics of volcanic aerosols
    print("\n" + "-" * 70)
    print("VOLCANIC AEROSOL PHYSICS")
    print("-" * 70)
    print("""
Formation process:
1. SO2 injected into stratosphere (>15 km altitude)
2. Oxidation: SO2 + OH -> H2SO4 (weeks to months)
3. Nucleation: H2SO4 + H2O -> sulfate droplets
4. Growth: Coagulation and condensation

Typical properties:
- Composition: 75% H2SO4 / 25% H2O
- Radius: 0.1-0.5 um (effective ~0.3 um)
- Lifetime: 12-24 months (stratospheric)
- Distribution: Global spread in weeks to months
""")

    # Calculate Mie scattering properties
    print("\n" + "-" * 70)
    print("MIE SCATTERING PROPERTIES")
    print("-" * 70)

    wavelengths = [0.35, 0.45, 0.55, 0.70, 1.0, 2.0, 4.0, 10.0]

    print(f"\nFor sulfate aerosol with r = {args.radius} um:")
    print(f"{'Wavelength':>12} {'Size Param':>12} {'Q_ext':>10} {'Q_sca':>10} {'SSA':>8} {'g':>8}")
    print("-" * 70)

    mie_results = []
    for wl in wavelengths:
        n = sulfate_refractive_index(wl)
        x = 2 * np.pi * args.radius / wl

        try:
            mie = MieScattering(refractive_index=n)
            result = mie.efficiencies(np.array([wl]), args.radius)

            Q_ext = result['Q_ext'][0]
            Q_sca = result['Q_sca'][0]
            Q_abs = Q_ext - Q_sca
            ssa = Q_sca / Q_ext if Q_ext > 0 else 1.0
            g = result['g'][0]

            mie_results.append((wl, x, Q_ext, Q_sca, ssa, g))
            print(f"{wl:>10.2f} um {x:>12.2f} {Q_ext:>10.3f} {Q_sca:>10.3f} {ssa:>8.3f} {g:>8.3f}")
        except Exception:
            # Simplified calculation if Mie fails
            mie_results.append((wl, x, 2.0, 1.9, 0.95, 0.7))
            print(f"{wl:>10.2f} um {x:>12.2f}      ~2.0      ~1.9    ~0.95    ~0.7")

    # Radiative forcing calculation
    print("\n" + "-" * 70)
    print("RADIATIVE FORCING")
    print("-" * 70)

    forcing = calculate_volcanic_forcing(args.aod, args.radius)

    print(f"""
Volcanic aerosol radiative forcing calculation:

Shortwave (solar) effect:
  - Scattering increases planetary albedo
  - Direct forcing: {forcing:.1f} W/m^2
  - This is the dominant effect

Longwave (thermal) effect:
  - Some IR absorption by sulfate
  - Weak warming effect: ~+2 W/m^2 per unit AOD
  - Partially offsets solar cooling

Net forcing for AOD = {args.aod}, r_eff = {args.radius} um:
  Solar:    {forcing:.1f} W/m^2
  Thermal:  ~{args.aod * 2:.1f} W/m^2
  NET:      ~{forcing + args.aod * 2:.1f} W/m^2
""")

    # Temperature response
    climate_sensitivity = 0.8  # K per W/m^2 (transient)
    temp_change = forcing * climate_sensitivity

    print(f"Estimated global temperature change: {temp_change:.2f} K")
    print(f"(Using transient climate sensitivity of {climate_sensitivity} K/(W/m^2))")

    # AOD decay timeline
    print("\n" + "-" * 70)
    print("AEROSOL DECAY AND CLIMATE RECOVERY")
    print("-" * 70)

    months = [0, 3, 6, 12, 18, 24, 36, 48]
    e_folding = 12  # months

    print(f"\nAOD decay (e-folding time = {e_folding} months):")
    print(f"{'Months':>8} {'AOD':>10} {'Forcing':>12} {'Cooling':>10}")
    print("-" * 45)

    for m in months:
        aod_t = aod_decay_model(m, args.aod, e_folding)
        force_t = calculate_volcanic_forcing(aod_t, args.radius)
        cool_t = force_t * climate_sensitivity
        print(f"{m:>8} {aod_t:>10.3f} {force_t:>10.1f} W/m^2 {cool_t:>8.2f} K")

    # Comparison with Pinatubo
    print("\n" + "-" * 70)
    print("COMPARISON WITH MT. PINATUBO (1991)")
    print("-" * 70)
    print("""
Mt. Pinatubo injected ~20 Mt SO2 into stratosphere

Observed effects:
- Peak AOD: 0.15-0.20 (global average)
- Peak forcing: -4 to -5 W/m^2
- Maximum cooling: 0.5 K (1992)
- Recovery time: ~3-4 years

Additional effects:
- Enhanced ozone depletion (heterogeneous chemistry)
- Spectacular sunsets worldwide
- Stratospheric warming (+3 K)
- Reduced diurnal temperature range
""")

    # Geoengineering context
    print("\n" + "-" * 70)
    print("STRATOSPHERIC AEROSOL INJECTION (SAI) CONTEXT")
    print("-" * 70)
    print("""
SAI geoengineering proposes intentional injection of aerosols:

To offset 2 K warming would require:
- Continuous injection of ~5-10 Mt SO2/year
- Maintaining AOD ~0.05-0.1
- Forcing: ~-2 to -3 W/m^2

Concerns:
1. Regional precipitation changes
2. Ozone depletion
3. Termination shock if stopped
4. Governance and ethics
5. Does not address ocean acidification
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Volcanic aerosol analysis for AOD = {args.aod}, r_eff = {args.radius} um:

Optical properties (at 550 nm):
- Size parameter: {2 * np.pi * args.radius / 0.55:.2f}
- Single scattering albedo: ~0.99 (highly scattering)
- Asymmetry parameter: ~0.7 (forward scattering)

Radiative effects:
- Net forcing: {forcing + args.aod * 2:.1f} W/m^2
- Global cooling: {temp_change:.2f} K
- E-folding decay time: ~{e_folding} months

This demonstrates how volcanic eruptions cause significant but
temporary climate cooling through stratospheric sulfate aerosols.
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Volcanic Aerosol Radiative Effects', fontsize=14, fontweight='bold')

            # Plot 1: AOD decay over time
            ax1 = axes[0, 0]
            time = np.linspace(0, 48, 100)
            aod_decay = aod_decay_model(time, args.aod, 12)

            ax1.plot(time, aod_decay, 'b-', linewidth=2, label=f'e-fold = 12 mo')
            ax1.plot(time, aod_decay_model(time, args.aod, 18), 'g--', linewidth=2, label='e-fold = 18 mo')
            ax1.axhline(args.aod / np.e, color='gray', linestyle=':', label='e-folding level')
            ax1.set_xlabel('Time after eruption (months)')
            ax1.set_ylabel('Aerosol Optical Depth')
            ax1.set_title('Stratospheric Aerosol Decay')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 48)

            # Plot 2: Forcing and temperature response
            ax2 = axes[0, 1]
            forcing_decay = [calculate_volcanic_forcing(aod, args.radius) for aod in aod_decay]
            temp_decay = [f * climate_sensitivity for f in forcing_decay]

            ax2.plot(time, forcing_decay, 'r-', linewidth=2, label='Radiative Forcing')
            ax2.set_xlabel('Time after eruption (months)')
            ax2.set_ylabel('Forcing (W/m^2)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            ax2b = ax2.twinx()
            ax2b.plot(time, temp_decay, 'b-', linewidth=2, label='Temperature Change')
            ax2b.set_ylabel('Temperature Change (K)', color='blue')
            ax2b.tick_params(axis='y', labelcolor='blue')

            ax2.set_title('Forcing and Temperature Response')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Wavelength dependence of scattering
            ax3 = axes[1, 0]
            wl_plot = np.linspace(0.3, 2.0, 50)
            aod_wl = args.aod * (0.55 / wl_plot)**1.5  # Angstrom exponent ~1.5

            ax3.plot(wl_plot, aod_wl, 'b-', linewidth=2)
            ax3.axvline(0.55, color='gray', linestyle='--', label='Reference (550 nm)')
            ax3.set_xlabel('Wavelength (um)')
            ax3.set_ylabel('Aerosol Optical Depth')
            ax3.set_title('Spectral Dependence of AOD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Historical eruptions comparison
            ax4 = axes[1, 1]
            eruptions = ['Tambora\n1815', 'Krakatoa\n1883', 'Agung\n1963',
                        'El Chichon\n1982', 'Pinatubo\n1991', 'This\nsimulation']
            aods = [0.5, 0.15, 0.08, 0.08, 0.15, args.aod]
            coolings = [0.5, 0.3, 0.2, 0.2, 0.5, abs(temp_change)]

            x_pos = np.arange(len(eruptions))
            bars = ax4.bar(x_pos, coolings, color=['red', 'orange', 'yellow',
                                                    'green', 'blue', 'purple'])
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(eruptions, fontsize=9)
            ax4.set_ylabel('Global Cooling (K)')
            ax4.set_title('Volcanic Cooling Comparison')
            ax4.grid(True, alpha=0.3, axis='y')

            # Add AOD as text on bars
            for i, (bar, aod) in enumerate(zip(bars, aods)):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'AOD={aod}', ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
