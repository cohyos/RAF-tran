#!/usr/bin/env python3
"""
Water Vapor Feedback and Radiative Forcing
==========================================

This example demonstrates how increased water vapor changes
downwelling longwave radiation and amplifies climate sensitivity.

Key concepts:
- Water vapor is the most important greenhouse gas
- Clausius-Clapeyron: ~7% increase in H2O per degree warming
- Water vapor feedback approximately doubles climate sensitivity
- Runaway greenhouse effect (Venus)

Applications:
- Climate modeling
- Understanding climate sensitivity
- Paleoclimate analysis
- Exoplanet habitability

References:
- Held & Soden (2000). Water vapor feedback and global warming.
- IPCC AR6 Chapter 7: The Earth's Energy Budget.

Usage:
    python 25_water_vapor_feedback.py
    python 25_water_vapor_feedback.py --temp-increase 2.0
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.utils.constants import STEFAN_BOLTZMANN
    from raf_tran.utils.spectral import planck_function
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze water vapor feedback in climate system"
    )
    parser.add_argument("--temp-increase", type=float, default=2.0,
                       help="Temperature increase to analyze (K)")
    parser.add_argument("--base-temp", type=float, default=288,
                       help="Base surface temperature (K)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="water_vapor_feedback.png")
    return parser.parse_args()


def clausius_clapeyron(temperature_k):
    """
    Calculate saturation vapor pressure using Clausius-Clapeyron equation.

    Parameters
    ----------
    temperature_k : float or array_like
        Temperature in Kelvin

    Returns
    -------
    e_sat : float or ndarray
        Saturation vapor pressure in Pa
    """
    T = np.asarray(temperature_k)
    T_c = T - 273.15  # Convert to Celsius

    # Tetens formula (valid for -40 to 50 C)
    e_sat = 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))

    return e_sat


def precipitable_water(temperature_k, relative_humidity=0.8):
    """
    Estimate precipitable water column from surface conditions.

    Parameters
    ----------
    temperature_k : float
        Surface temperature in K
    relative_humidity : float
        Average relative humidity (0-1)

    Returns
    -------
    pw : float
        Precipitable water in kg/m^2 (or mm)
    """
    e_sat = clausius_clapeyron(temperature_k)
    # Approximate total column water from surface conditions
    # Using scale height approach
    scale_height = 2000  # m, for water vapor
    rho_surface = e_sat * relative_humidity / (461.5 * temperature_k)  # kg/m^3

    return rho_surface * scale_height * 1000  # mm


def h2o_longwave_forcing(pw_mm, baseline_pw_mm=25):
    """
    Calculate longwave forcing change from water vapor.

    Based on simplified radiative transfer (logarithmic dependence).

    Parameters
    ----------
    pw_mm : float
        Precipitable water in mm
    baseline_pw_mm : float
        Baseline precipitable water

    Returns
    -------
    forcing : float
        Radiative forcing in W/m^2
    """
    # Empirical relationship from radiative transfer calculations
    # ~2 W/m^2 per doubling of water vapor
    forcing = 2.0 * np.log2(pw_mm / baseline_pw_mm)
    return forcing


def olr_with_h2o(surface_temp, pw_mm):
    """
    Calculate outgoing longwave radiation with water vapor absorption.

    Simplified model accounting for H2O greenhouse effect.
    """
    # Blackbody emission at surface
    F_surface = STEFAN_BOLTZMANN * surface_temp**4

    # Effective emissivity reduction due to H2O
    # More water vapor = lower effective emission temperature
    tau_h2o = 0.05 * np.sqrt(pw_mm)  # Simplified optical depth
    epsilon_eff = np.exp(-tau_h2o)

    # OLR is reduced by atmospheric absorption
    T_eff = (epsilon_eff * F_surface / STEFAN_BOLTZMANN)**0.25
    OLR = STEFAN_BOLTZMANN * T_eff**4 + (1 - epsilon_eff) * 0.8 * F_surface

    return OLR


def no_feedback_sensitivity():
    """Calculate climate sensitivity without feedbacks (Planck response only)."""
    # dT/dF = 1 / (4 * sigma * T^3) at T = 255K (effective emission temp)
    T_eff = 255  # K
    return 1 / (4 * STEFAN_BOLTZMANN * T_eff**3)


def main():
    args = parse_args()

    print("=" * 70)
    print("WATER VAPOR FEEDBACK AND RADIATIVE FORCING")
    print("=" * 70)
    print(f"\nBase surface temperature: {args.base_temp} K ({args.base_temp - 273:.1f} C)")
    print(f"Temperature increase to analyze: {args.temp_increase} K")

    # Clausius-Clapeyron relationship
    print("\n" + "-" * 70)
    print("CLAUSIUS-CLAPEYRON RELATIONSHIP")
    print("-" * 70)
    print("""
The Clausius-Clapeyron equation governs water vapor saturation:

    de_sat/dT = L_v * e_sat / (R_v * T^2)

This gives approximately 7% increase in saturation vapor pressure
per degree Kelvin of warming (at Earth surface temperatures).
""")

    # Calculate saturation vapor pressure vs temperature
    temps = np.array([273, 278, 283, 288, 293, 298, 303])
    e_sat = clausius_clapeyron(temps)

    print(f"{'Temperature':>12} {'e_sat (Pa)':>12} {'e_sat (mbar)':>14} {'% change/K':>12}")
    print("-" * 55)

    for i, (T, e) in enumerate(zip(temps, e_sat)):
        if i > 0:
            pct_change = ((e - e_sat[i - 1]) / e_sat[i - 1]) * 100 / (temps[i] - temps[i - 1])
        else:
            pct_change = 0
        print(f"{T:>10} K {e:>12.1f} {e / 100:>14.2f} {pct_change:>11.1f}%")

    print(f"\nAverage: ~7%/K (as predicted by Clausius-Clapeyron)")

    # Precipitable water analysis
    print("\n" + "-" * 70)
    print("PRECIPITABLE WATER COLUMN")
    print("-" * 70)

    T_base = args.base_temp
    T_warm = args.base_temp + args.temp_increase

    pw_base = precipitable_water(T_base)
    pw_warm = precipitable_water(T_warm)
    pw_change = (pw_warm - pw_base) / pw_base * 100

    print(f"""
Precipitable water (assuming 80% average relative humidity):

  At {T_base} K:     {pw_base:.1f} mm
  At {T_warm} K:     {pw_warm:.1f} mm
  Change:          {pw_change:+.1f}%
  Per degree:      {pw_change / args.temp_increase:+.1f}%/K

This matches the ~7%/K Clausius-Clapeyron prediction.
""")

    # Longwave forcing
    print("\n" + "-" * 70)
    print("WATER VAPOR LONGWAVE FORCING")
    print("-" * 70)

    forcing = h2o_longwave_forcing(pw_warm, pw_base)

    print(f"""
Additional downwelling longwave radiation from water vapor increase:

  Forcing = {forcing:.2f} W/m^2

Water vapor increases the greenhouse effect through:
1. Rotational band (far-IR, >20 um)
2. 6.3 um bending vibration
3. Continuum absorption
4. Near-IR overtones (minor)
""")

    # Climate sensitivity
    print("\n" + "-" * 70)
    print("CLIMATE SENSITIVITY AND FEEDBACKS")
    print("-" * 70)

    lambda_0 = no_feedback_sensitivity()
    print(f"""
No-feedback (Planck) sensitivity:
  lambda_0 = {lambda_0:.3f} K/(W/m^2)
  Doubling CO2 (3.7 W/m^2) would give: {3.7 * lambda_0:.2f} K warming

With water vapor feedback:
  The water vapor feedback factor f_wv ~ 0.4-0.5
  This roughly doubles the sensitivity

  lambda = lambda_0 / (1 - f_wv) ~ {lambda_0 / 0.5:.3f} K/(W/m^2)
  Doubling CO2 would give: {3.7 * lambda_0 / 0.5:.2f} K warming

This is why equilibrium climate sensitivity is ~3 K per CO2 doubling,
not the ~1.1 K from Planck response alone.
""")

    # Feedback calculation
    print("\n" + "-" * 70)
    print("FEEDBACK STRENGTH CALCULATION")
    print("-" * 70)

    # Calculate feedback strength
    dT = args.temp_increase
    dF_wv = forcing
    f_wv = dF_wv / (4 * STEFAN_BOLTZMANN * T_base**3 * dT)

    print(f"""
For {dT} K warming:
  Additional H2O forcing: {dF_wv:.2f} W/m^2
  Planck response: {4 * STEFAN_BOLTZMANN * T_base**3 * dT:.2f} W/m^2
  Feedback factor: f_wv = {f_wv:.3f}

Feedback contributions (IPCC AR6 estimates):
  Water vapor:     +1.8 W/m^2/K  (strong positive)
  Lapse rate:      -0.5 W/m^2/K  (weak negative)
  WV + Lapse rate: +1.3 W/m^2/K  (combined positive)
  Clouds:          +0.4 W/m^2/K  (positive, uncertain)
  Surface albedo:  +0.3 W/m^2/K  (positive)
""")

    # OLR analysis
    print("\n" + "-" * 70)
    print("OUTGOING LONGWAVE RADIATION (OLR) ANALYSIS")
    print("-" * 70)

    olr_base = olr_with_h2o(T_base, pw_base)
    olr_warm = olr_with_h2o(T_warm, pw_warm)

    print(f"""
OLR at different conditions:

  Condition              Surface T    PW (mm)    OLR (W/m^2)
  ---------              ---------    -------    -----------
  Current climate        {T_base:>6.0f} K     {pw_base:>5.0f}      {olr_base:>6.1f}
  Warmed (+{args.temp_increase}K)          {T_warm:>6.0f} K     {pw_warm:>5.0f}      {olr_warm:>6.1f}

OLR increase: {olr_warm - olr_base:.1f} W/m^2
(Less than Planck ~{4 * STEFAN_BOLTZMANN * T_base**3 * args.temp_increase:.1f} W/m^2 due to increased H2O opacity)
""")

    # Runaway greenhouse
    print("\n" + "-" * 70)
    print("RUNAWAY GREENHOUSE EFFECT")
    print("-" * 70)
    print("""
At very high temperatures, water vapor can cause runaway warming:

1. As T increases, more H2O evaporates (Clausius-Clapeyron)
2. More H2O increases greenhouse effect
3. This causes more warming -> more evaporation
4. Process runs away if OLR can't keep up with solar input

Simpson-Nakajima limit:
- Maximum OLR for moist atmosphere: ~280-300 W/m^2
- If absorbed solar exceeds this, oceans boil
- This happened on Venus (solar constant 2600 W/m^2)

Earth is safe because:
- Absorbed solar ~240 W/m^2 < Simpson limit
- But this sets limit on possible warming
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Water vapor feedback analysis for {args.temp_increase} K warming:

1. Clausius-Clapeyron:
   - Saturation vapor pressure increases ~7%/K
   - This is a robust physical relationship

2. Precipitable water:
   - Increases from {pw_base:.0f} to {pw_warm:.0f} mm ({pw_change:+.0f}%)
   - Assuming constant relative humidity

3. Radiative forcing:
   - Additional forcing: {forcing:.1f} W/m^2
   - Feedback factor: {f_wv:.2f}

4. Climate sensitivity amplification:
   - Without feedbacks: ~1.1 K per CO2 doubling
   - With H2O feedback: ~2.2 K per CO2 doubling
   - With all feedbacks: ~3 K per CO2 doubling (IPCC best estimate)

Water vapor feedback is the most important positive feedback in the
climate system, approximately doubling the warming from CO2 alone.
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Water Vapor Feedback and Radiative Effects',
                        fontsize=14, fontweight='bold')

            # Plot 1: Clausius-Clapeyron
            ax1 = axes[0, 0]
            T_range = np.linspace(260, 320, 100)
            e_sat_range = clausius_clapeyron(T_range)

            ax1.semilogy(T_range - 273.15, e_sat_range / 100, 'b-', linewidth=2)
            ax1.axvline(T_base - 273.15, color='red', linestyle='--', label=f'Current ({T_base} K)')
            ax1.set_xlabel('Temperature (C)')
            ax1.set_ylabel('Saturation Vapor Pressure (mbar)')
            ax1.set_title('Clausius-Clapeyron Relationship')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Precipitable water vs temperature
            ax2 = axes[0, 1]
            pw_range = [precipitable_water(T) for T in T_range]

            ax2.plot(T_range - 273.15, pw_range, 'b-', linewidth=2)
            ax2.axvline(T_base - 273.15, color='red', linestyle='--')
            ax2.axhline(pw_base, color='red', linestyle=':', alpha=0.5)
            ax2.fill_between([T_base - 273.15, T_warm - 273.15],
                            [pw_base, pw_base], [pw_base, pw_warm],
                            alpha=0.3, color='orange', label=f'+{args.temp_increase}K warming')
            ax2.set_xlabel('Temperature (C)')
            ax2.set_ylabel('Precipitable Water (mm)')
            ax2.set_title('Precipitable Water Column')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Feedback diagram
            ax3 = axes[1, 0]
            feedbacks = ['Water Vapor', 'Lapse Rate', 'WV+LR', 'Cloud', 'Albedo', 'Total']
            values = [1.8, -0.5, 1.3, 0.4, 0.3, 2.0]
            colors = ['blue' if v > 0 else 'red' for v in values]

            bars = ax3.barh(feedbacks, values, color=colors, alpha=0.7)
            ax3.axvline(0, color='black', linewidth=0.5)
            ax3.set_xlabel('Feedback (W/m^2/K)')
            ax3.set_title('Climate Feedback Strengths (IPCC AR6)')
            ax3.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for bar, val in zip(bars, values):
                x_pos = val + 0.1 if val > 0 else val - 0.1
                ha = 'left' if val > 0 else 'right'
                ax3.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f'{val:+.1f}', ha=ha, va='center')

            # Plot 4: OLR vs surface temperature
            ax4 = axes[1, 1]
            T_olr = np.linspace(280, 340, 100)
            olr_dry = STEFAN_BOLTZMANN * (0.6 * T_olr)**4  # Simplified dry atmosphere
            olr_moist = [olr_with_h2o(T, precipitable_water(T)) for T in T_olr]

            ax4.plot(T_olr, olr_dry, 'b--', linewidth=2, label='Dry atmosphere')
            ax4.plot(T_olr, olr_moist, 'r-', linewidth=2, label='Moist atmosphere')
            ax4.axhline(290, color='gray', linestyle=':', label='Simpson limit (~290 W/m^2)')
            ax4.axhline(240, color='orange', linestyle=':', label='Absorbed solar (~240 W/m^2)')
            ax4.axvline(T_base, color='green', linestyle='--', alpha=0.5, label=f'Current T')
            ax4.set_xlabel('Surface Temperature (K)')
            ax4.set_ylabel('OLR (W/m^2)')
            ax4.set_title('OLR vs Surface Temperature')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
