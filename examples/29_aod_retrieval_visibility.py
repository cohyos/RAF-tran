#!/usr/bin/env python3
"""
AOD Retrieval and Visibility
============================

This example demonstrates Aerosol Optical Depth (AOD) retrieval methods
and visibility calculations for atmospheric applications.

Key concepts:
- Langley plot extrapolation for AOD
- Multi-wavelength Angstrom exponent
- Koschmieder visibility equation
- Sunphotometer measurements (AERONET-style)

Applications:
- Air quality monitoring
- Aviation meteorology
- Climate research
- Remote sensing validation

Usage:
    python 29_aod_retrieval_visibility.py
    python 29_aod_retrieval_visibility.py --aod 0.3
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering.rayleigh import rayleigh_cross_section
    from raf_tran.utils.air_mass import kasten_young_air_mass
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AOD retrieval and visibility calculations"
    )
    parser.add_argument("--aod", type=float, default=0.2, help="AOD at 500 nm")
    parser.add_argument("--angstrom", type=float, default=1.4, help="Angstrom exponent")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="aod_visibility.png")
    return parser.parse_args()


# =============================================================================
# AOD Calculations
# =============================================================================

def rayleigh_optical_depth(wavelength_um, pressure_mb=1013.25):
    """
    Calculate Rayleigh optical depth for the atmosphere.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    pressure_mb : float
        Surface pressure in mbar

    Returns
    -------
    tau_ray : float
        Rayleigh optical depth
    """
    sigma = rayleigh_cross_section(wavelength_um)
    N_column = 2.55e25 * (pressure_mb / 1013.25)  # molecules/m^2
    return sigma * N_column


def angstrom_aod(wavelength_um, aod_500, alpha):
    """
    Calculate AOD at any wavelength using Angstrom law.

    AOD(lambda) = AOD(500nm) * (lambda/500nm)^(-alpha)

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    aod_500 : float
        AOD at 500 nm
    alpha : float
        Angstrom exponent

    Returns
    -------
    aod : float
        AOD at specified wavelength
    """
    return aod_500 * (wavelength_um / 0.5)**(-alpha)


def retrieve_angstrom_exponent(wavelengths, aod_values):
    """
    Retrieve Angstrom exponent from multi-wavelength AOD.

    Uses linear regression on log-log data.

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths in micrometers
    aod_values : array_like
        AOD at each wavelength

    Returns
    -------
    alpha : float
        Angstrom exponent
    aod_500 : float
        AOD at 500 nm (interpolated)
    """
    wl = np.array(wavelengths)
    aod = np.array(aod_values)

    # Log-log linear fit
    log_wl = np.log(wl)
    log_aod = np.log(aod)

    # Linear regression
    n = len(wl)
    sum_x = np.sum(log_wl)
    sum_y = np.sum(log_aod)
    sum_xx = np.sum(log_wl**2)
    sum_xy = np.sum(log_wl * log_aod)

    alpha = -(n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
    intercept = (sum_y + alpha * sum_x) / n

    # AOD at 500 nm
    aod_500 = np.exp(intercept + (-alpha) * np.log(0.5))

    return alpha, aod_500


# =============================================================================
# Langley Calibration
# =============================================================================

def langley_plot_data(sza_values, aod_true, wavelength_um=0.5, v0=1.0):
    """
    Generate synthetic Langley plot data.

    The Langley method uses:
    ln(V) = ln(V0) - tau * m

    where V is measured voltage, V0 is extraterrestrial signal,
    tau is optical depth, and m is air mass.

    Parameters
    ----------
    sza_values : array_like
        Solar zenith angles in degrees
    aod_true : float
        True AOD at wavelength
    wavelength_um : float
        Wavelength in micrometers
    v0 : float
        Extraterrestrial signal (normalized)

    Returns
    -------
    air_masses : array
        Air mass values
    signals : array
        Measured signals (with noise)
    tau_total : float
        Total optical depth
    """
    # Calculate optical depths
    tau_ray = rayleigh_optical_depth(wavelength_um)
    tau_total = tau_ray + aod_true

    air_masses = []
    signals = []

    for sza in sza_values:
        m = kasten_young_air_mass(sza)
        # Add small noise
        noise = 1 + np.random.normal(0, 0.005)
        v = v0 * np.exp(-tau_total * m) * noise
        air_masses.append(m)
        signals.append(v)

    return np.array(air_masses), np.array(signals), tau_total


def langley_aod_retrieval(air_masses, signals, wavelength_um):
    """
    Retrieve AOD using Langley plot method.

    Parameters
    ----------
    air_masses : array_like
        Air mass values
    signals : array_like
        Measured signals
    wavelength_um : float
        Wavelength in micrometers

    Returns
    -------
    aod : float
        Retrieved AOD
    v0 : float
        Retrieved extraterrestrial constant
    r_squared : float
        Correlation coefficient
    """
    m = np.array(air_masses)
    v = np.array(signals)

    # Linear regression on ln(V) vs m
    ln_v = np.log(v)

    n = len(m)
    sum_x = np.sum(m)
    sum_y = np.sum(ln_v)
    sum_xx = np.sum(m**2)
    sum_xy = np.sum(m * ln_v)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
    intercept = (sum_y - slope * sum_x) / n

    # Total optical depth from slope
    tau_total = -slope

    # Remove Rayleigh contribution
    tau_ray = rayleigh_optical_depth(wavelength_um)
    aod = tau_total - tau_ray

    # V0 from intercept
    v0 = np.exp(intercept)

    # R-squared
    mean_y = np.mean(ln_v)
    ss_tot = np.sum((ln_v - mean_y)**2)
    ss_res = np.sum((ln_v - (intercept + slope * m))**2)
    r_squared = 1 - ss_res / ss_tot

    return aod, v0, r_squared


# =============================================================================
# Visibility Calculations
# =============================================================================

def koschmieder_visibility(extinction_coeff, contrast_threshold=0.02):
    """
    Calculate visibility using Koschmieder equation.

    V = -ln(epsilon) / beta

    where epsilon is contrast threshold and beta is extinction.

    Parameters
    ----------
    extinction_coeff : float
        Extinction coefficient (per meter)
    contrast_threshold : float
        Contrast threshold (typically 0.02 or 2%)

    Returns
    -------
    visibility : float
        Visibility in meters
    """
    return -np.log(contrast_threshold) / extinction_coeff


def visibility_from_aod(aod, scale_height=1500, contrast_threshold=0.02):
    """
    Calculate visibility from AOD.

    Assumes aerosol extinction is concentrated in boundary layer.

    Parameters
    ----------
    aod : float
        Aerosol optical depth
    scale_height : float
        Aerosol scale height in meters
    contrast_threshold : float
        Contrast threshold

    Returns
    -------
    visibility : float
        Visibility in meters
    """
    # Extinction coefficient at surface
    beta = aod / scale_height

    return koschmieder_visibility(beta, contrast_threshold)


def aod_from_visibility(visibility, scale_height=1500, contrast_threshold=0.02):
    """
    Estimate AOD from visibility.

    Parameters
    ----------
    visibility : float
        Visibility in meters
    scale_height : float
        Aerosol scale height in meters
    contrast_threshold : float
        Contrast threshold

    Returns
    -------
    aod : float
        Estimated AOD
    """
    beta = -np.log(contrast_threshold) / visibility
    return beta * scale_height


def visibility_category(vis_km):
    """
    Return visibility category (aviation/meteorological).

    Parameters
    ----------
    vis_km : float
        Visibility in kilometers

    Returns
    -------
    category : str
        Visibility category
    """
    if vis_km >= 50:
        return "Exceptionally clear"
    elif vis_km >= 10:
        return "Clear"
    elif vis_km >= 5:
        return "Moderate"
    elif vis_km >= 2:
        return "Light haze"
    elif vis_km >= 1:
        return "Haze"
    elif vis_km >= 0.5:
        return "Moderate fog/mist"
    else:
        return "Dense fog"


# =============================================================================
# Aerosol Size Information
# =============================================================================

def angstrom_to_size(alpha):
    """
    Interpret Angstrom exponent in terms of aerosol size.

    Parameters
    ----------
    alpha : float
        Angstrom exponent

    Returns
    -------
    description : str
        Size interpretation
    """
    if alpha > 1.5:
        return "Fine mode dominant (pollution, smoke)"
    elif alpha > 1.0:
        return "Mixed fine and coarse mode"
    elif alpha > 0.5:
        return "Coarse mode significant (dust, sea salt)"
    else:
        return "Coarse mode dominant (desert dust)"


def aerosol_type_from_spectral(aod_440, aod_870, alpha):
    """
    Estimate aerosol type from spectral properties.

    Based on AERONET classification schemes.

    Parameters
    ----------
    aod_440 : float
        AOD at 440 nm
    aod_870 : float
        AOD at 870 nm
    alpha : float
        Angstrom exponent (440-870 nm)

    Returns
    -------
    aerosol_type : str
        Estimated aerosol type
    """
    if aod_440 < 0.15:
        return "Background/clean continental"
    elif alpha > 1.5 and aod_440 > 0.4:
        return "Urban/industrial pollution"
    elif alpha > 1.2 and aod_440 > 0.3:
        return "Biomass burning smoke"
    elif alpha < 0.5 and aod_440 > 0.3:
        return "Desert dust"
    elif alpha < 0.8 and 0.15 < aod_440 < 0.4:
        return "Marine/sea salt"
    else:
        return "Mixed/indeterminate"


# =============================================================================
# Standard Visibility Conditions
# =============================================================================

def standard_atmospheres():
    """
    Return standard visibility/AOD for different conditions.

    Returns
    -------
    conditions : dict
        Standard atmospheric conditions
    """
    return {
        'Very clear': {'visibility_km': 50, 'aod_500': 0.05},
        'Clear': {'visibility_km': 23, 'aod_500': 0.1},
        'Moderate': {'visibility_km': 10, 'aod_500': 0.2},
        'Hazy': {'visibility_km': 5, 'aod_500': 0.4},
        'Very hazy': {'visibility_km': 2, 'aod_500': 0.8},
        'Polluted': {'visibility_km': 1, 'aod_500': 1.5},
    }


def main():
    args = parse_args()

    print("=" * 70)
    print("AOD RETRIEVAL AND VISIBILITY")
    print("=" * 70)
    print(f"\nInput parameters:")
    print(f"  AOD at 500 nm: {args.aod}")
    print(f"  Angstrom exponent: {args.angstrom}")

    # AOD basics
    print("\n" + "-" * 70)
    print("AEROSOL OPTICAL DEPTH (AOD)")
    print("-" * 70)
    print("""
AOD is a measure of aerosol loading in the atmospheric column:

  tau_aer = integral(beta_ext * dz)

where beta_ext is the aerosol extinction coefficient.

Typical values:
  - Clean background: AOD < 0.1
  - Moderate pollution: AOD ~ 0.2-0.5
  - Heavy pollution/dust: AOD > 1.0
""")

    # Spectral AOD
    print("\n" + "-" * 70)
    print("SPECTRAL AOD (ANGSTROM LAW)")
    print("-" * 70)

    wavelengths = [0.340, 0.380, 0.440, 0.500, 0.675, 0.870, 1.020]
    colors = ['UV', 'UV', 'Blue', 'Green', 'Red', 'NIR', 'NIR']

    print(f"\nWavelength dependence (Angstrom exponent = {args.angstrom}):")
    print(f"{'Wavelength':>12} {'AOD':>10} {'Rayleigh OD':>12} {'Total OD':>12}")
    print("-" * 55)

    aod_values = []
    for wl, color in zip(wavelengths, colors):
        aod = angstrom_aod(wl, args.aod, args.angstrom)
        tau_ray = rayleigh_optical_depth(wl)
        tau_total = aod + tau_ray
        aod_values.append(aod)
        print(f"{wl*1000:>10.0f} nm {aod:>10.4f} {tau_ray:>12.4f} {tau_total:>12.4f}")

    # Retrieve Angstrom exponent
    alpha_retrieved, aod500_retrieved = retrieve_angstrom_exponent(wavelengths, aod_values)
    print(f"\nAngstrom exponent retrieval:")
    print(f"  True alpha: {args.angstrom:.2f}")
    print(f"  Retrieved: {alpha_retrieved:.2f}")
    print(f"  Size interpretation: {angstrom_to_size(alpha_retrieved)}")

    # Langley plot method
    print("\n" + "-" * 70)
    print("LANGLEY PLOT AOD RETRIEVAL")
    print("-" * 70)
    print("""
The Langley method retrieves AOD from multi-airmass measurements:

  ln(V) = ln(V0) - tau * m

where:
  V = measured signal
  V0 = extraterrestrial signal
  tau = total optical depth
  m = air mass

By plotting ln(V) vs m, we get:
  - Slope = -tau
  - Intercept = ln(V0)
""")

    # Generate Langley plot data
    sza_range = np.arange(60, 82, 2)  # Morning measurements
    air_masses, signals, tau_true = langley_plot_data(sza_range, args.aod)

    print(f"\nSimulated Langley measurements at 500 nm:")
    print(f"{'SZA (deg)':>10} {'Air Mass':>12} {'Signal':>12}")
    print("-" * 40)

    for sza, m, v in zip(sza_range[::2], air_masses[::2], signals[::2]):
        print(f"{sza:>10.0f} {m:>12.2f} {v:>12.4f}")

    # Retrieve AOD
    aod_retrieved, v0_retrieved, r_squared = langley_aod_retrieval(air_masses, signals, 0.5)

    tau_ray = rayleigh_optical_depth(0.5)
    print(f"\nLangley retrieval results:")
    print(f"  True AOD: {args.aod:.4f}")
    print(f"  Retrieved AOD: {aod_retrieved:.4f}")
    print(f"  Error: {(aod_retrieved - args.aod)/args.aod * 100:.1f}%")
    print(f"  R-squared: {r_squared:.6f}")
    print(f"  Rayleigh OD: {tau_ray:.4f}")
    print(f"  Total OD (true): {tau_true:.4f}")

    # Visibility
    print("\n" + "-" * 70)
    print("VISIBILITY FROM AOD")
    print("-" * 70)
    print("""
Visibility is related to AOD via the Koschmieder equation:

  V = -ln(epsilon) / beta_ext

where:
  epsilon = contrast threshold (~0.02)
  beta_ext = extinction coefficient

For boundary layer aerosols:
  beta_ext ~ AOD / H_scale

where H_scale is the aerosol scale height (~1.5 km).
""")

    vis = visibility_from_aod(args.aod)
    vis_km = vis / 1000

    print(f"\nVisibility calculation:")
    print(f"  AOD at 500 nm: {args.aod}")
    print(f"  Assumed scale height: 1500 m")
    print(f"  Visibility: {vis_km:.1f} km")
    print(f"  Category: {visibility_category(vis_km)}")

    # Visibility table
    print("\n\nStandard atmospheric conditions:")
    print(f"{'Condition':>15} {'Visibility':>15} {'AOD (500nm)':>15}")
    print("-" * 50)

    for name, vals in standard_atmospheres().items():
        print(f"{name:>15} {vals['visibility_km']:>13.0f} km {vals['aod_500']:>15.2f}")

    # Aerosol type classification
    print("\n" + "-" * 70)
    print("AEROSOL TYPE CLASSIFICATION")
    print("-" * 70)

    aod_440 = angstrom_aod(0.44, args.aod, args.angstrom)
    aod_870 = angstrom_aod(0.87, args.aod, args.angstrom)

    aerosol_type = aerosol_type_from_spectral(aod_440, aod_870, args.angstrom)

    print(f"""
Spectral properties:
  AOD at 440 nm: {aod_440:.3f}
  AOD at 870 nm: {aod_870:.3f}
  Angstrom exponent: {args.angstrom:.2f}

Estimated aerosol type: {aerosol_type}

Classification scheme (AERONET-style):
  - alpha > 1.5, high AOD: Urban/industrial
  - alpha > 1.2, high AOD: Biomass burning
  - alpha < 0.5, high AOD: Desert dust
  - alpha < 0.8, low AOD: Marine
  - Low AOD: Background continental
""")

    # AERONET wavelengths
    print("\n" + "-" * 70)
    print("AERONET SUNPHOTOMETER WAVELENGTHS")
    print("-" * 70)
    print("""
Standard AERONET wavelengths for AOD retrieval:

  340 nm - UV, high Rayleigh, sensitive to fine particles
  380 nm - UV, ozone absorption correction needed
  440 nm - Blue, standard channel
  500 nm - Green, primary reference
  675 nm - Red, low Rayleigh
  870 nm - NIR, primarily aerosol
  1020 nm - NIR, aerosol only, water vapor nearby

Key derived products:
  - AOD at all wavelengths
  - Angstrom exponent (440-870 nm)
  - Fine mode fraction
  - Water vapor column (940 nm)
  - Ozone column (300s nm)
""")

    # Multi-wavelength analysis
    print("\n" + "-" * 70)
    print("MULTI-WAVELENGTH AOD ANALYSIS")
    print("-" * 70)

    print(f"\n{'Wavelength':>12} {'AOD':>10} {'Rayleigh':>10} {'Aer/Tot':>10}")
    print("-" * 50)

    for wl in wavelengths:
        aod = angstrom_aod(wl, args.aod, args.angstrom)
        tau_ray = rayleigh_optical_depth(wl)
        aer_frac = aod / (aod + tau_ray) * 100
        print(f"{wl*1000:>10.0f} nm {aod:>10.4f} {tau_ray:>10.4f} {aer_frac:>9.1f}%")

    print("""
Notes:
- UV: Rayleigh dominates, aerosol contribution smaller
- VIS: Aerosol and Rayleigh comparable
- NIR: Aerosol dominates, Rayleigh negligible
""")

    # Aviation applications
    print("\n" + "-" * 70)
    print("AVIATION VISIBILITY CATEGORIES")
    print("-" * 70)
    print("""
ICAO visibility categories:

  VFR (Visual Flight Rules):     > 8 km
  Marginal VFR:                  5-8 km
  IFR (Instrument Flight Rules): 1.5-5 km
  Low IFR:                       < 1.5 km

METAR visibility codes:
  9999 = Visibility >= 10 km
  0000 = Visibility < 50 m

Current conditions:
""")
    print(f"  Visibility: {vis_km:.1f} km")

    if vis_km >= 8:
        flight_cat = "VFR"
    elif vis_km >= 5:
        flight_cat = "Marginal VFR"
    elif vis_km >= 1.5:
        flight_cat = "IFR"
    else:
        flight_cat = "Low IFR"

    print(f"  Flight category: {flight_cat}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
AOD and Visibility Analysis:

Input parameters:
  AOD at 500 nm: {args.aod}
  Angstrom exponent: {args.angstrom}

Derived quantities:
  AOD at 440 nm: {aod_440:.3f}
  AOD at 870 nm: {aod_870:.3f}
  Visibility: {vis_km:.1f} km
  Visibility category: {visibility_category(vis_km)}

Aerosol characterization:
  Size mode: {angstrom_to_size(args.angstrom)}
  Likely type: {aerosol_type}

Key relationships:
  - Higher Angstrom exponent = finer particles
  - Higher AOD = lower visibility
  - Visibility ~ 23 km / AOD (rule of thumb)
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('AOD Retrieval and Visibility', fontsize=14, fontweight='bold')

            # Plot 1: Spectral AOD
            ax1 = axes[0, 0]
            wl_range = np.linspace(0.3, 1.1, 100)
            aod_spec = [angstrom_aod(wl, args.aod, args.angstrom) for wl in wl_range]
            tau_ray_spec = [rayleigh_optical_depth(wl) for wl in wl_range]

            ax1.semilogy(wl_range * 1000, aod_spec, 'b-', linewidth=2, label='Aerosol')
            ax1.semilogy(wl_range * 1000, tau_ray_spec, 'g--', linewidth=2, label='Rayleigh')
            ax1.semilogy(wl_range * 1000, np.array(aod_spec) + np.array(tau_ray_spec),
                        'r-', linewidth=2, label='Total')
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Optical Depth')
            ax1.set_title(f'Spectral AOD (alpha={args.angstrom})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(300, 1100)

            # Plot 2: Langley plot
            ax2 = axes[0, 1]
            ax2.scatter(air_masses, np.log(signals), c='blue', s=30, alpha=0.7)

            # Fit line
            m_fit = np.linspace(min(air_masses), max(air_masses), 50)
            ln_v_fit = np.log(v0_retrieved) - (tau_ray + aod_retrieved) * m_fit
            ax2.plot(m_fit, ln_v_fit, 'r-', linewidth=2,
                    label=f'Fit: AOD={aod_retrieved:.3f}')

            ax2.set_xlabel('Air Mass')
            ax2.set_ylabel('ln(Signal)')
            ax2.set_title('Langley Plot (500 nm)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Visibility vs AOD
            ax3 = axes[1, 0]
            aod_range = np.linspace(0.02, 2, 100)
            vis_arr = [visibility_from_aod(a) / 1000 for a in aod_range]

            ax3.semilogy(aod_range, vis_arr, 'b-', linewidth=2)
            ax3.axhline(10, color='green', linestyle='--', alpha=0.5, label='Clear (10 km)')
            ax3.axhline(5, color='orange', linestyle='--', alpha=0.5, label='Moderate (5 km)')
            ax3.axhline(1, color='red', linestyle='--', alpha=0.5, label='Poor (1 km)')
            ax3.axvline(args.aod, color='purple', linestyle=':', linewidth=2,
                       label=f'Current AOD={args.aod}')
            ax3.set_xlabel('AOD at 500 nm')
            ax3.set_ylabel('Visibility (km)')
            ax3.set_title('Visibility vs AOD')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 2)
            ax3.set_ylim(0.5, 100)

            # Plot 4: Angstrom exponent classification
            ax4 = axes[1, 1]

            # Create classification regions
            alpha_values = [0.2, 0.5, 1.0, 1.5, 2.0]
            aod_levels = np.linspace(0.05, 1.5, 100)

            # Background colors for regions
            ax4.axhspan(1.5, 2.2, alpha=0.3, color='gray', label='Fine (urban/smoke)')
            ax4.axhspan(1.0, 1.5, alpha=0.3, color='yellow', label='Mixed')
            ax4.axhspan(0.5, 1.0, alpha=0.3, color='orange', label='Coarse present')
            ax4.axhspan(0, 0.5, alpha=0.3, color='brown', label='Coarse (dust)')

            ax4.scatter([args.aod], [args.angstrom], s=200, c='red', marker='*',
                       zorder=5, label='Current')

            ax4.set_xlabel('AOD at 500 nm')
            ax4.set_ylabel('Angstrom Exponent')
            ax4.set_title('Aerosol Type Classification')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 1.5)
            ax4.set_ylim(0, 2.2)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
