#!/usr/bin/env python3
"""
Example 37: Adaptive Optics System Simulation
=============================================

This example demonstrates the adaptive optics (AO) simulation capabilities
including wavefront sensing, deformable mirror correction, and system
performance analysis.

Features demonstrated:
1. AO system configuration
2. Error budget analysis (fitting, temporal, noise)
3. Optimal actuator selection
4. Strehl ratio prediction
5. Zernike mode analysis

Usage:
    python examples/37_adaptive_optics_simulation.py [--no-plot]
"""

import argparse
import sys
import numpy as np

# Plotting imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# RAF-tran imports
from raf_tran.turbulence import (
    hufnagel_valley_cn2,
    fried_parameter,
    greenwood_frequency,
    isoplanatic_angle,
    zernike_variance,
)
from raf_tran.turbulence.adaptive_optics import (
    AOSystemConfig,
    AOPerformance,
    compute_ao_performance,
    fitting_error,
    temporal_error,
    optimal_actuator_count,
    ShackHartmannWFS,
    strehl_from_variance,
)


def main(args):
    print("=" * 70)
    print("Example 37: Adaptive Optics System Simulation")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------------
    # 1. Define Atmospheric Conditions
    # ---------------------------------------------------------------------
    print("1. Atmospheric Turbulence Conditions")
    print("-" * 40)

    # Generate Cn2 profile
    altitudes = np.linspace(0, 20000, 100)
    cn2_ground = 1.7e-14  # HV 5/7 model ground value

    cn2_profile = np.array([hufnagel_valley_cn2(h, cn2_ground) for h in altitudes])

    # Integrated values
    cn2_integrated = np.trapezoid(cn2_profile, altitudes)

    # Calculate key parameters
    wavelength = 500e-9  # 500 nm (visible)
    k = 2 * np.pi / wavelength

    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)
    seeing = 0.98 * wavelength / r0 * 206265  # arcsec

    # Isoplanatic angle
    h53_integral = np.trapezoid(cn2_profile * altitudes**(5/3), altitudes)
    theta0 = (2.91 * k**2 * h53_integral)**(-3/5)

    # Greenwood frequency (assuming 10 m/s wind)
    wind_speed = 10.0  # m/s
    f_g = 0.102 * k**(6/5) * (cn2_integrated * wind_speed**(5/3))**(3/5)

    print(f"   Fried parameter r0: {r0*100:.2f} cm")
    print(f"   Seeing: {seeing:.2f} arcsec")
    print(f"   Isoplanatic angle: {theta0*206265:.2f} arcsec")
    print(f"   Greenwood frequency: {f_g:.1f} Hz")
    print()

    # ---------------------------------------------------------------------
    # 2. Configure AO System
    # ---------------------------------------------------------------------
    print("2. AO System Configuration")
    print("-" * 40)

    # Create AO system for 4m telescope
    aperture = 4.0  # meters
    target_strehl = 0.8

    # Optimal number of actuators
    n_act_optimal = optimal_actuator_count(aperture, r0, target_strehl)
    print(f"   Aperture diameter: {aperture} m")
    print(f"   Target Strehl: {target_strehl}")
    print(f"   Optimal actuators: {n_act_optimal} across diameter")

    # Configure system
    config = AOSystemConfig(
        aperture_diameter=aperture,
        wavelength=wavelength,
        n_actuators=n_act_optimal,
        wfs_type='shack_hartmann',
        loop_frequency=1000.0,  # Hz
        loop_gain=0.5,
        wfs_noise=0.1,  # rad RMS
        dm_stroke=5.0,  # um
    )

    print(f"   Loop frequency: {config.loop_frequency} Hz")
    print(f"   Loop gain: {config.loop_gain}")
    print(f"   WFS noise: {config.wfs_noise} rad RMS")
    print()

    # ---------------------------------------------------------------------
    # 3. Error Budget Analysis
    # ---------------------------------------------------------------------
    print("3. Error Budget Analysis")
    print("-" * 40)

    # Calculate individual error terms
    d_actuator = aperture / config.n_actuators

    sigma_fit2 = fitting_error(aperture, r0, d_actuator)
    print(f"   Fitting error: {sigma_fit2:.4f} rad^2 ({np.sqrt(sigma_fit2):.3f} rad RMS)")

    f_3db = 0.1 * config.loop_frequency * config.loop_gain
    sigma_temp2 = temporal_error(f_g, f_3db, r0, aperture)
    print(f"   Temporal error: {sigma_temp2:.4f} rad^2 ({np.sqrt(sigma_temp2):.3f} rad RMS)")

    sigma_noise2 = config.wfs_noise**2 * 0.3  # Simplified noise propagation
    print(f"   Noise error: {sigma_noise2:.4f} rad^2 ({np.sqrt(sigma_noise2):.3f} rad RMS)")

    # Total error
    sigma_total2 = sigma_fit2 + sigma_temp2 + sigma_noise2
    strehl = strehl_from_variance(sigma_total2)

    print()
    print(f"   Total residual: {sigma_total2:.4f} rad^2 ({np.sqrt(sigma_total2):.3f} rad RMS)")
    print(f"   Predicted Strehl: {strehl:.3f}")
    print()

    # ---------------------------------------------------------------------
    # 4. Comprehensive Performance Analysis
    # ---------------------------------------------------------------------
    print("4. Full System Performance")
    print("-" * 40)

    performance = compute_ao_performance(
        config=config,
        r0=r0,
        greenwood_freq=f_g,
    )

    print(f"   Strehl ratio: {performance.strehl_ratio:.3f}")
    print(f"   Residual variance: {performance.residual_variance:.4f} rad^2")
    print(f"   Corrected modes: ~{performance.n_corrected_modes}")
    print()

    # ---------------------------------------------------------------------
    # 5. Zernike Mode Analysis
    # ---------------------------------------------------------------------
    print("5. Zernike Mode Variance (Uncorrected)")
    print("-" * 40)

    mode_names = ['Piston', 'Tip', 'Tilt', 'Focus', 'Astig1', 'Astig2',
                  'Coma1', 'Coma2', 'Trefoil1', 'Trefoil2', 'Spherical']

    print("   Mode        Variance (rad^2)  RMS (rad)")
    print("   " + "-" * 45)
    for j in range(1, 12):
        var_j = zernike_variance(j, wavelength, cn2_integrated, aperture)
        print(f"   {mode_names[j-1]:10s}    {var_j:10.4f}        {np.sqrt(var_j):.4f}")

    print()

    # ---------------------------------------------------------------------
    # 6. Wavefront Sensor Analysis
    # ---------------------------------------------------------------------
    print("6. Shack-Hartmann WFS Analysis")
    print("-" * 40)

    wfs = ShackHartmannWFS(
        n_subapertures=config.n_actuators,
        pixel_size=0.5,  # arcsec
        wavelength=wavelength,
        read_noise=3.0,  # electrons
        throughput=0.5,
    )

    spot_fwhm = wfs.spot_size_fwhm(r0, aperture)
    print(f"   Subapertures: {wfs.n_subapertures} x {wfs.n_subapertures}")
    print(f"   Spot FWHM: {spot_fwhm:.2f} arcsec")

    # Centroid error vs guide star magnitude
    print()
    print("   Guide Star  Photons/sub   Centroid Error")
    for photons in [100, 500, 1000, 5000, 10000]:
        error = wfs.measurement_error(r0, aperture, photons)
        print(f"   {photons:8d}      {photons:6d}     {error:.4f} arcsec")

    print()

    # ---------------------------------------------------------------------
    # 7. Visualization
    # ---------------------------------------------------------------------
    if not args.no_plot:
        print("7. Creating Visualizations")
        print("-" * 40)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Panel 1: Cn2 profile
        ax1 = axes[0, 0]
        ax1.semilogy(cn2_profile, altitudes/1000, 'b-', linewidth=2)
        ax1.set_xlabel('Cn2 (m^-2/3)')
        ax1.set_ylabel('Altitude (km)')
        ax1.set_title('Turbulence Profile (HV 5/7)')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Error budget pie chart
        ax2 = axes[0, 1]
        errors = [sigma_fit2, sigma_temp2, sigma_noise2]
        labels = ['Fitting', 'Temporal', 'Noise']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        ax2.pie(errors, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=(0.05, 0.05, 0.05))
        ax2.set_title('Error Budget')

        # Panel 3: Zernike variances
        ax3 = axes[0, 2]
        j_values = np.arange(1, 21)
        variances = [zernike_variance(j, wavelength, cn2_integrated, aperture) for j in j_values]
        ax3.bar(j_values, variances, color='steelblue', edgecolor='black')
        ax3.set_xlabel('Zernike Mode (j)')
        ax3.set_ylabel('Variance (rad^2)')
        ax3.set_title('Zernike Mode Variances')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')

        # Panel 4: Strehl vs number of actuators
        ax4 = axes[1, 0]
        n_acts = np.arange(5, 51, 5)
        strehls = []
        for n in n_acts:
            d = aperture / n
            s_fit = fitting_error(aperture, r0, d)
            s_total = s_fit + sigma_temp2 + sigma_noise2
            strehls.append(strehl_from_variance(s_total))
        ax4.plot(n_acts, strehls, 'b-o', linewidth=2, markersize=8)
        ax4.axhline(0.8, color='r', linestyle='--', label='Target (0.8)')
        ax4.axvline(n_act_optimal, color='g', linestyle=':', label=f'Optimal ({n_act_optimal})')
        ax4.set_xlabel('Actuators Across Diameter')
        ax4.set_ylabel('Strehl Ratio')
        ax4.set_title('Strehl vs DM Order')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        # Panel 5: Strehl vs wavelength
        ax5 = axes[1, 1]
        wavelengths = np.linspace(400, 2200, 50) * 1e-9  # nm to m
        strehls_wl = []
        for wl in wavelengths:
            k_wl = 2 * np.pi / wl
            r0_wl = r0 * (wl / 500e-9)**(6/5)
            d = aperture / config.n_actuators
            s_fit = fitting_error(aperture, r0_wl, d)
            s_total = s_fit + sigma_temp2 * (500e-9/wl)**2 + sigma_noise2
            strehls_wl.append(strehl_from_variance(s_total))
        ax5.plot(wavelengths*1e9, strehls_wl, 'b-', linewidth=2)
        ax5.set_xlabel('Wavelength (nm)')
        ax5.set_ylabel('Strehl Ratio')
        ax5.set_title('Strehl vs Wavelength')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, 1)

        # Panel 6: Performance summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        summary_text = f"""
        AO System Performance Summary
        {'='*35}

        Telescope: {aperture:.1f} m aperture
        Wavelength: {wavelength*1e9:.0f} nm

        Atmospheric Conditions:
          r0 = {r0*100:.1f} cm
          Seeing = {seeing:.2f}"
          theta0 = {theta0*206265:.1f}"
          f_G = {f_g:.1f} Hz

        AO System:
          Actuators: {config.n_actuators} x {config.n_actuators}
          Loop rate: {config.loop_frequency:.0f} Hz
          Corrected modes: ~{performance.n_corrected_modes}

        Performance:
          Strehl = {performance.strehl_ratio:.3f}
          Residual = {np.sqrt(performance.residual_variance):.3f} rad RMS
        """
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top')

        plt.tight_layout()
        plt.savefig('outputs/37_adaptive_optics_simulation.png', dpi=150, bbox_inches='tight')
        print("   Saved: outputs/37_adaptive_optics_simulation.png")
        plt.close()

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Adaptive Optics Simulation for {aperture}m Telescope:

    Atmospheric Conditions (HV 5/7):
      - Fried parameter: {r0*100:.1f} cm at 500nm
      - Seeing: {seeing:.2f} arcsec
      - Isoplanatic angle: {theta0*206265:.1f} arcsec

    AO System Design:
      - Optimal actuators: {n_act_optimal} across diameter
      - Loop frequency: {config.loop_frequency:.0f} Hz

    Error Budget:
      - Fitting error: {np.sqrt(sigma_fit2):.3f} rad RMS
      - Temporal error: {np.sqrt(sigma_temp2):.3f} rad RMS
      - Noise error: {np.sqrt(sigma_noise2):.3f} rad RMS

    Performance:
      - Predicted Strehl: {performance.strehl_ratio:.3f}
      - Meets target ({target_strehl}): {'Yes' if performance.strehl_ratio >= target_strehl else 'No'}
    """)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive Optics Simulation')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    import os
    os.makedirs('outputs', exist_ok=True)

    sys.exit(main(args))
