#!/usr/bin/env python3
"""
Laser Beam Propagation Through Atmosphere
==========================================

This example combines absorption, scattering, and turbulence effects
for realistic laser propagation simulations.

Applications:
- Free-space optical communications
- LIDAR/laser ranging
- Laser weapons/directed energy
- Laser guide star adaptive optics

Includes:
- Molecular and aerosol extinction
- Atmospheric turbulence (scintillation, beam wander)
- Beam spreading
- Link budget calculations

Usage:
    python 21_laser_propagation.py
    python 21_laser_propagation.py --wavelength 1.55 --path-length 10000
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.turbulence import (
        hufnagel_valley_cn2, slc_day_cn2,
        fried_parameter, scintillation_index, rytov_variance,
        beam_wander_variance, strehl_ratio
    )
    from raf_tran.turbulence.cn2_profiles import integrated_cn2
    from raf_tran.utils.constants import SPEED_OF_LIGHT
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate laser beam propagation through atmosphere"
    )
    parser.add_argument("--wavelength", type=float, default=1.55,
                       help="Wavelength in um (default: 1.55 um)")
    parser.add_argument("--path-length", type=float, default=10000,
                       help="Path length in meters (default: 10000)")
    parser.add_argument("--power", type=float, default=1.0,
                       help="Transmitter power in watts (default: 1.0)")
    parser.add_argument("--beam-diameter", type=float, default=0.1,
                       help="Transmitter aperture diameter in m (default: 0.1)")
    parser.add_argument("--receiver-diameter", type=float, default=0.2,
                       help="Receiver aperture diameter in m (default: 0.2)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="laser_propagation.png")
    return parser.parse_args()


def molecular_extinction(wavelength_um, altitude_m=10):
    """
    Approximate molecular extinction coefficient at given wavelength.
    Returns extinction in km^-1.
    """
    # Rayleigh scattering (scales as lambda^-4)
    beta_rayleigh = 0.012 * (0.55 / wavelength_um)**4

    # Water vapor absorption (simplified, around 1.4 um and 1.9 um bands)
    if 1.3 < wavelength_um < 1.5:
        beta_h2o = 0.05
    elif 1.8 < wavelength_um < 2.1:
        beta_h2o = 0.1
    else:
        beta_h2o = 0.001

    return beta_rayleigh + beta_h2o


def aerosol_extinction(wavelength_um, visibility_km=23):
    """
    Aerosol extinction coefficient based on visibility.
    Koschmieder formula.
    """
    # Scale visibility value at 550 nm
    beta_550 = 3.912 / visibility_km  # km^-1

    # Wavelength scaling (Angstrom exponent ~ 1.3 for typical aerosols)
    alpha = 1.3
    beta = beta_550 * (0.55 / wavelength_um)**alpha

    return beta


def diffraction_beam_size(wavelength_m, aperture_m, distance_m):
    """
    Beam diameter at distance due to diffraction.
    For Gaussian beam: w(z) = w0 * sqrt(1 + (z/z_R)^2)
    """
    w0 = aperture_m / 2  # Beam waist
    z_R = np.pi * w0**2 / wavelength_m  # Rayleigh range

    return 2 * w0 * np.sqrt(1 + (distance_m / z_R)**2)


def turbulence_beam_spread(wavelength_m, r0, distance_m, aperture_m):
    """
    Additional beam spreading due to turbulence.
    Short-term: w_ST ~ w_diffraction (coherent)
    Long-term: w_LT ~ sqrt(w_diffraction^2 + w_wander^2)
    """
    w_diff = diffraction_beam_size(wavelength_m, aperture_m, distance_m)

    # Turbulence spreading
    if r0 > 0:
        w_turb = wavelength_m * distance_m / (np.pi * r0)
        w_long_term = np.sqrt(w_diff**2 + w_turb**2)
    else:
        w_long_term = w_diff

    return w_long_term


def main():
    args = parse_args()

    wavelength_m = args.wavelength * 1e-6
    L = args.path_length
    P_tx = args.power
    D_tx = args.beam_diameter
    D_rx = args.receiver_diameter

    print("=" * 70)
    print("LASER BEAM PROPAGATION THROUGH ATMOSPHERE")
    print("=" * 70)
    print(f"\nWavelength: {args.wavelength} um")
    print(f"Path length: {L/1000:.1f} km")
    print(f"Transmitter power: {P_tx} W")
    print(f"Transmitter aperture: {D_tx*100:.0f} cm")
    print(f"Receiver aperture: {D_rx*100:.0f} cm")

    # Atmospheric extinction
    print("\n" + "-" * 70)
    print("ATMOSPHERIC EXTINCTION")
    print("-" * 70)

    beta_mol = molecular_extinction(args.wavelength)
    beta_aer = aerosol_extinction(args.wavelength, visibility_km=23)
    beta_total = beta_mol + beta_aer

    tau_total = beta_total * L / 1000  # Convert L to km

    T_atm = np.exp(-tau_total)

    print(f"\nMolecular extinction: {beta_mol:.4f} km^-1")
    print(f"Aerosol extinction: {beta_aer:.4f} km^-1")
    print(f"Total extinction: {beta_total:.4f} km^-1")
    print(f"Optical depth: {tau_total:.3f}")
    print(f"Atmospheric transmission: {T_atm:.4f} ({T_atm*100:.2f}%)")

    # Turbulence effects
    print("\n" + "-" * 70)
    print("TURBULENCE EFFECTS")
    print("-" * 70)

    # Cn2 profile (horizontal path at 10m altitude)
    cn2_path = slc_day_cn2(10)  # Strong daytime turbulence
    cn2_integrated = cn2_path * L

    print(f"\nCn2 (path avg): {cn2_path:.2e} m^(-2/3)")

    # Fried parameter
    r0 = fried_parameter(wavelength_m, cn2_integrated)
    print(f"Fried parameter r0: {r0*100:.2f} cm")

    # Rytov variance
    sigma_r2 = rytov_variance(wavelength_m, cn2_path, L)
    print(f"Rytov variance: {sigma_r2:.3f}")

    if sigma_r2 < 0.3:
        regime = "Weak fluctuations"
    elif sigma_r2 < 5:
        regime = "Moderate fluctuations"
    else:
        regime = "Strong fluctuations (saturation)"
    print(f"Turbulence regime: {regime}")

    # Scintillation
    si_point = scintillation_index(wavelength_m, cn2_path, L)
    si_aperture = scintillation_index(wavelength_m, cn2_path, L,
                                       aperture_diameter_m=D_rx)
    print(f"\nScintillation index (point): {si_point:.4f}")
    print(f"Scintillation index ({D_rx*100:.0f}cm aperture): {si_aperture:.4f}")
    print(f"Aperture averaging factor: {si_aperture/si_point:.3f}")

    # Beam wander
    sigma_bw2 = beam_wander_variance(wavelength_m, cn2_path, L, D_tx)
    sigma_bw = np.sqrt(sigma_bw2)
    print(f"\nBeam wander RMS: {sigma_bw*1e3:.2f} mm")

    # Beam spreading
    print("\n" + "-" * 70)
    print("BEAM SPREADING")
    print("-" * 70)

    w_diffraction = diffraction_beam_size(wavelength_m, D_tx, L)
    w_turbulence = turbulence_beam_spread(wavelength_m, r0, L, D_tx)

    print(f"\nDiffraction-only beam diameter: {w_diffraction*100:.1f} cm")
    print(f"With turbulence (long-term): {w_turbulence*100:.1f} cm")

    # Geometric spreading loss
    A_tx = np.pi * (D_tx/2)**2
    A_beam = np.pi * (w_turbulence/2)**2
    A_rx = np.pi * (D_rx/2)**2

    if A_beam > A_rx:
        eta_geo = A_rx / A_beam
    else:
        eta_geo = 1.0

    print(f"\nBeam area at receiver: {A_beam*1e4:.1f} cm^2")
    print(f"Receiver aperture area: {A_rx*1e4:.1f} cm^2")
    print(f"Geometric collection efficiency: {eta_geo:.4f}")

    # Link budget
    print("\n" + "-" * 70)
    print("LINK BUDGET")
    print("-" * 70)

    # Power at receiver
    P_rx = P_tx * T_atm * eta_geo

    # Losses in dB
    L_atm_dB = -10 * np.log10(T_atm)
    L_geo_dB = -10 * np.log10(eta_geo)
    L_total_dB = L_atm_dB + L_geo_dB

    print(f"\nTransmitter power: {P_tx:.3f} W ({10*np.log10(P_tx*1000):.1f} dBm)")
    print(f"Atmospheric loss: {L_atm_dB:.2f} dB")
    print(f"Geometric/spreading loss: {L_geo_dB:.2f} dB")
    print(f"Total path loss: {L_total_dB:.2f} dB")
    print(f"Received power: {P_rx*1e6:.3f} uW ({10*np.log10(P_rx*1000):.1f} dBm)")

    # Fade statistics for FSO
    print("\n" + "-" * 70)
    print("FADE STATISTICS (Free-Space Optical Comm)")
    print("-" * 70)

    # Log-normal fading
    sigma_I = np.sqrt(si_aperture)

    # Outage probability (power drops below threshold)
    fade_margins = [3, 6, 10]  # dB

    print(f"\nLog-normal intensity fluctuation sigma_I = {sigma_I:.3f}")
    print(f"\n{'Fade Margin (dB)':>18} {'Outage Prob':>14} {'Availability':>14}")
    print("-" * 50)

    for margin in fade_margins:
        # Threshold normalized by mean
        threshold = 10**(-margin/10)
        # Log-normal CDF (approximate)
        from math import erf
        if sigma_I > 0:
            z = (np.log(threshold) + sigma_I**2/2) / sigma_I
            outage = 0.5 * (1 + erf(z / np.sqrt(2)))
        else:
            outage = 0 if threshold < 1 else 1

        availability = 1 - outage

        print(f"{margin:>18} {outage:>14.2e} {availability*100:>13.4f}%")

    # Wavelength comparison
    print("\n" + "-" * 70)
    print("WAVELENGTH COMPARISON")
    print("-" * 70)

    wavelengths = [0.532, 0.85, 1.06, 1.55, 3.8, 10.6]
    names = ["Nd:YAG 2H", "Diode", "Nd:YAG", "Telecom", "MWIR", "CO2"]

    print(f"\n{'Wavelength':>12} {'Name':>10} {'T_atm':>8} {'r0 (cm)':>10} {'sigma_I':>8}")
    print("-" * 60)

    for wl, name in zip(wavelengths, names):
        wl_m = wl * 1e-6
        beta = molecular_extinction(wl) + aerosol_extinction(wl)
        T = np.exp(-beta * L / 1000)

        r0_wl = fried_parameter(wl_m, cn2_integrated)
        si_wl = scintillation_index(wl_m, cn2_path, L, D_rx)

        print(f"{wl:>10.3f} um {name:>10} {T:>8.4f} {r0_wl*100:>10.2f} {si_wl:>8.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("LASER PROPAGATION SUMMARY")
    print("=" * 70)
    print(f"""
System: {args.wavelength} um laser, {L/1000:.0f} km horizontal path

Performance metrics:
- Atmospheric transmission: {T_atm*100:.1f}%
- Beam diameter at receiver: {w_turbulence*100:.0f} cm
- Fried parameter: {r0*100:.1f} cm
- Scintillation index: {si_aperture:.3f}
- Received power: {P_rx*1e6:.2f} uW

Recommendations:
- Use aperture averaging (large receiver) to reduce scintillation
- Consider longer wavelength to reduce turbulence effects
- Add fade margin for reliable communication
- Avoid boundary layer for horizontal paths if possible
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Laser Propagation ({args.wavelength} um, {L/1000:.0f} km)',
                        fontsize=14, fontweight='bold')

            # Plot 1: Beam spreading
            ax1 = axes[0, 0]
            z_plot = np.linspace(0.1, L, 100)

            w_diff = [diffraction_beam_size(wavelength_m, D_tx, z) for z in z_plot]
            w_turb = [turbulence_beam_spread(wavelength_m, r0, z, D_tx) for z in z_plot]

            ax1.plot(z_plot/1000, np.array(w_diff)*100, 'b--',
                    linewidth=2, label='Diffraction only')
            ax1.plot(z_plot/1000, np.array(w_turb)*100, 'r-',
                    linewidth=2, label='With turbulence')
            ax1.axhline(D_rx*100, color='green', linestyle=':',
                       label=f'Receiver aperture ({D_rx*100:.0f} cm)')
            ax1.set_xlabel('Distance (km)')
            ax1.set_ylabel('Beam Diameter (cm)')
            ax1.set_title('Beam Spreading')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Power vs distance
            ax2 = axes[0, 1]

            P_rx_arr = []
            for z in z_plot:
                tau = beta_total * z / 1000
                T = np.exp(-tau)
                w = turbulence_beam_spread(wavelength_m, r0, z, D_tx)
                A_beam = np.pi * (w/2)**2
                eta = min(A_rx / A_beam, 1.0)
                P_rx_arr.append(P_tx * T * eta)

            ax2.semilogy(z_plot/1000, np.array(P_rx_arr)*1e6, 'b-', linewidth=2)
            ax2.set_xlabel('Distance (km)')
            ax2.set_ylabel('Received Power (uW)')
            ax2.set_title('Received Power vs Distance')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Scintillation vs distance
            ax3 = axes[1, 0]

            si_arr = []
            for z in z_plot:
                cn2_int_z = cn2_path * z
                si_arr.append(scintillation_index(wavelength_m, cn2_path, z, D_rx))

            ax3.plot(z_plot/1000, si_arr, 'r-', linewidth=2)
            ax3.axhline(0.3, color='orange', linestyle='--', alpha=0.5,
                       label='Weak/moderate boundary')
            ax3.axhline(1.0, color='red', linestyle='--', alpha=0.5,
                       label='Saturation')
            ax3.set_xlabel('Distance (km)')
            ax3.set_ylabel('Scintillation Index')
            ax3.set_title('Scintillation vs Distance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Plot 4: Wavelength comparison
            ax4 = axes[1, 1]

            r0_arr = []
            T_arr = []
            for wl in wavelengths:
                wl_m = wl * 1e-6
                beta = molecular_extinction(wl) + aerosol_extinction(wl)
                T_arr.append(np.exp(-beta * L / 1000))
                r0_arr.append(fried_parameter(wl_m, cn2_integrated) * 100)

            ax4_twin = ax4.twinx()
            ax4.bar(np.arange(len(wavelengths)) - 0.15, T_arr, 0.3,
                   color='blue', alpha=0.7, label='Transmission')
            ax4_twin.bar(np.arange(len(wavelengths)) + 0.15, r0_arr, 0.3,
                        color='red', alpha=0.7, label='r0 (cm)')

            ax4.set_xticks(range(len(wavelengths)))
            ax4.set_xticklabels([f'{w}' for w in wavelengths])
            ax4.set_xlabel('Wavelength (um)')
            ax4.set_ylabel('Transmission', color='blue')
            ax4_twin.set_ylabel('Fried Parameter (cm)', color='red')
            ax4.set_title('Wavelength Comparison')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
