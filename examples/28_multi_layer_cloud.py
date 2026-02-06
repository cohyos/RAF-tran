#!/usr/bin/env python3
"""
Multi-Layer Cloud Overlapping Effects
======================================

This example demonstrates cloud overlap assumptions and their impact
on radiative transfer through multi-layer cloud systems.

Cloud Overlap Assumptions:
- Maximum overlap: Clouds vertically aligned (minimum total cover)
- Random overlap: Clouds randomly distributed (moderate cover)
- Maximum-random: Maximum within groups, random between

Key concepts:
- Cloud fraction profiles
- Effective total cloud cover
- Layer-by-layer transmission
- Shortwave and longwave effects

Applications:
- Global climate models (GCMs)
- Weather prediction
- Satellite retrievals
- Radiative budget studies

Usage:
    python 28_multi_layer_cloud.py
    python 28_multi_layer_cloud.py --layers 5
"""

import argparse
import numpy as np
import sys

sys.path.insert(0, '..')

try:
    from raf_tran.scattering.rayleigh import rayleigh_cross_section
except ImportError:
    print("Error: raf_tran package not found.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze multi-layer cloud overlap effects"
    )
    parser.add_argument("--layers", type=int, default=3, help="Number of cloud layers")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="multi_layer_cloud.png")
    return parser.parse_args()


# =============================================================================
# Cloud Overlap Functions
# =============================================================================

def maximum_overlap_cover(cloud_fractions):
    """
    Calculate total cloud cover assuming maximum overlap.

    In maximum overlap, clouds are vertically aligned so that
    total cover equals the maximum of individual layer fractions.

    Parameters
    ----------
    cloud_fractions : array_like
        Cloud fraction for each layer (0 to 1)

    Returns
    -------
    total_cover : float
        Total cloud cover (0 to 1)
    """
    return np.max(cloud_fractions)


def random_overlap_cover(cloud_fractions):
    """
    Calculate total cloud cover assuming random overlap.

    In random overlap, clouds are randomly distributed between
    layers, so clear sky fractions multiply.

    Parameters
    ----------
    cloud_fractions : array_like
        Cloud fraction for each layer (0 to 1)

    Returns
    -------
    total_cover : float
        Total cloud cover (0 to 1)
    """
    clear_fractions = 1.0 - np.array(cloud_fractions)
    total_clear = np.prod(clear_fractions)
    return 1.0 - total_clear


def maximum_random_overlap_cover(cloud_fractions, layer_heights):
    """
    Calculate total cloud cover using maximum-random overlap.

    Maximum overlap within contiguous cloud groups, random
    overlap between separated groups.

    Parameters
    ----------
    cloud_fractions : array_like
        Cloud fraction for each layer (0 to 1)
    layer_heights : array_like
        Height of each layer (km)

    Returns
    -------
    total_cover : float
        Total cloud cover (0 to 1)
    """
    cf = np.array(cloud_fractions)
    h = np.array(layer_heights)

    if len(cf) == 0:
        return 0.0

    # Find contiguous cloud groups (separated by clear layers)
    groups = []
    current_group = []

    for i, c in enumerate(cf):
        if c > 0:
            current_group.append(c)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    if current_group:
        groups.append(current_group)

    if not groups:
        return 0.0

    # Maximum overlap within groups, random between groups
    group_covers = [max(g) for g in groups]
    total_clear = np.prod([1 - gc for gc in group_covers])

    return 1.0 - total_clear


# =============================================================================
# Cloud Optical Properties
# =============================================================================

def cloud_optical_depth(lwp, r_eff):
    """
    Calculate cloud optical depth from liquid water path.

    Uses the relationship: tau = 3 * LWP / (2 * rho_w * r_eff)

    Parameters
    ----------
    lwp : float
        Liquid water path (g/m^2)
    r_eff : float
        Effective radius (micrometers)

    Returns
    -------
    tau : float
        Cloud optical depth
    """
    rho_w = 1e6  # g/m^3 (water density)
    r_eff_m = r_eff * 1e-6  # convert to meters

    # Optical depth: tau = 3 * LWP / (2 * rho_w * r_eff)
    # With LWP in g/m^2, rho_w in g/m^3, r_eff in m
    tau = 1.5 * lwp / (rho_w * r_eff_m)

    return tau


def cloud_single_scatter_albedo(wavelength_um, phase='liquid'):
    """
    Get cloud single-scatter albedo.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    phase : str
        'liquid' or 'ice'

    Returns
    -------
    ssa : float
        Single-scatter albedo (0 to 1)
    """
    # Clouds are highly scattering in visible/NIR
    if wavelength_um < 2.5:
        ssa = 0.999 if phase == 'liquid' else 0.998
    elif wavelength_um < 4.0:
        # Absorbing bands
        ssa = 0.98 if phase == 'liquid' else 0.95
    else:
        # Thermal infrared - more absorbing
        ssa = 0.5  # Approximate
    return ssa


def cloud_asymmetry_parameter(wavelength_um, r_eff, phase='liquid'):
    """
    Get cloud asymmetry parameter.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    r_eff : float
        Effective radius (micrometers)
    phase : str
        'liquid' or 'ice'

    Returns
    -------
    g : float
        Asymmetry parameter (-1 to 1)
    """
    # Typical values for cloud droplets
    if phase == 'liquid':
        # Mie theory gives g ~ 0.85 for water clouds
        g = 0.85
    else:
        # Ice crystals are more forward scattering
        g = 0.75  # Lower due to non-spherical shapes

    # Size dependence (larger droplets more forward scattering)
    if r_eff > 20:
        g = min(0.90, g + 0.02)
    elif r_eff < 8:
        g = max(0.75, g - 0.05)

    return g


# =============================================================================
# Layer-by-Layer Radiative Transfer
# =============================================================================

def layer_transmission_shortwave(tau, ssa, g, mu0=1.0):
    """
    Calculate layer transmission for shortwave radiation.

    Uses delta-Eddington approximation.

    Parameters
    ----------
    tau : float
        Optical depth
    ssa : float
        Single-scatter albedo
    g : float
        Asymmetry parameter
    mu0 : float
        Cosine of solar zenith angle

    Returns
    -------
    T_dir : float
        Direct beam transmission
    T_dif : float
        Diffuse transmission
    """
    # Direct beam transmission
    T_dir = np.exp(-tau / mu0)

    # Delta scaling
    f = g**2
    tau_prime = (1 - ssa * f) * tau
    ssa_prime = ssa * (1 - f) / (1 - ssa * f)
    g_prime = (g - f) / (1 - f)

    # Two-stream diffuse transmission (simplified)
    gamma = np.sqrt(3 * (1 - ssa_prime) * (1 - ssa_prime * g_prime))
    T_dif = np.exp(-gamma * tau_prime)

    return T_dir, T_dif


def layer_transmission_longwave(tau_lw, temperature_K):
    """
    Calculate layer transmission and emission for longwave.

    Parameters
    ----------
    tau_lw : float
        Longwave optical depth
    temperature_K : float
        Layer temperature (K)

    Returns
    -------
    T_lw : float
        Longwave transmission
    emissivity : float
        Layer emissivity
    """
    T_lw = np.exp(-tau_lw)
    emissivity = 1 - T_lw

    return T_lw, emissivity


def multilayer_transmission(cloud_fractions, optical_depths, ssa_values, g_values,
                           overlap='random', mu0=1.0):
    """
    Calculate transmission through multi-layer cloud system.

    Parameters
    ----------
    cloud_fractions : array_like
        Cloud fraction for each layer
    optical_depths : array_like
        Optical depth for each layer
    ssa_values : array_like
        Single-scatter albedo for each layer
    g_values : array_like
        Asymmetry parameter for each layer
    overlap : str
        Overlap assumption: 'maximum', 'random', or 'max-random'
    mu0 : float
        Cosine of solar zenith angle

    Returns
    -------
    T_total : float
        Total transmission (direct + diffuse weighted by clear/cloudy)
    """
    n_layers = len(cloud_fractions)

    if overlap == 'maximum':
        # Maximum overlap - all cloudy or all clear
        total_cf = maximum_overlap_cover(cloud_fractions)
        # Combined optical depth where clouds exist
        total_tau = sum(optical_depths)
        avg_ssa = np.mean(ssa_values)
        avg_g = np.mean(g_values)

        T_dir, T_dif = layer_transmission_shortwave(total_tau, avg_ssa, avg_g, mu0)
        T_cloudy = 0.5 * (T_dir + T_dif)  # Average of direct and diffuse

        T_total = total_cf * T_cloudy + (1 - total_cf) * 1.0

    elif overlap == 'random':
        # Random overlap - process layer by layer
        T_clear = 1.0  # Transmission through clear sky path
        T_cloudy_cumulative = 1.0  # Transmission through cloudy path

        for i in range(n_layers):
            cf = cloud_fractions[i]
            tau = optical_depths[i]
            ssa = ssa_values[i]
            g = g_values[i]

            T_dir, T_dif = layer_transmission_shortwave(tau, ssa, g, mu0)
            T_layer = 0.5 * (T_dir + T_dif)

            # Partial cloud coverage
            T_cloudy_cumulative *= (cf * T_layer + (1 - cf) * 1.0)

        T_total = T_cloudy_cumulative

    else:  # max-random
        # Simplified max-random (treat as random for this example)
        T_total = multilayer_transmission(
            cloud_fractions, optical_depths, ssa_values, g_values,
            overlap='random', mu0=mu0
        )

    return T_total


# =============================================================================
# Cloud Radiative Effect
# =============================================================================

def cloud_radiative_effect_sw(cloud_fraction, optical_depth, ssa, g,
                              solar_flux=1361, mu0=0.5, albedo_sfc=0.1):
    """
    Calculate shortwave cloud radiative effect (CRE).

    Parameters
    ----------
    cloud_fraction : float
        Total cloud fraction
    optical_depth : float
        Cloud optical depth
    ssa : float
        Single-scatter albedo
    g : float
        Asymmetry parameter
    solar_flux : float
        Incoming solar flux (W/m^2)
    mu0 : float
        Cosine of solar zenith angle
    albedo_sfc : float
        Surface albedo

    Returns
    -------
    cre_sw : float
        Shortwave CRE (W/m^2, negative = cooling)
    """
    # Clear sky
    flux_clear = solar_flux * mu0 * (1 - albedo_sfc)

    # Cloudy sky
    T_dir, T_dif = layer_transmission_shortwave(optical_depth, ssa, g, mu0)
    T_cloud = 0.5 * (T_dir + T_dif)

    # Cloud reflectance (simplified)
    R_cloud = 1 - T_cloud - 0.05  # Small absorption

    flux_cloudy = solar_flux * mu0 * T_cloud * (1 - albedo_sfc)

    # Weighted average
    flux_all = cloud_fraction * flux_cloudy + (1 - cloud_fraction) * flux_clear

    # CRE = cloudy - clear (at TOA, reflected flux difference)
    reflected_clear = solar_flux * mu0 * albedo_sfc
    reflected_cloudy = solar_flux * mu0 * R_cloud

    cre_sw = cloud_fraction * (reflected_cloudy - reflected_clear)

    return -cre_sw  # Negative = cooling


def cloud_radiative_effect_lw(cloud_fraction, cloud_top_temp, surface_temp=288):
    """
    Calculate longwave cloud radiative effect.

    Parameters
    ----------
    cloud_fraction : float
        Total cloud fraction
    cloud_top_temp : float
        Cloud top temperature (K)
    surface_temp : float
        Surface temperature (K)

    Returns
    -------
    cre_lw : float
        Longwave CRE (W/m^2, positive = warming)
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant

    # Clear sky OLR (simplified - assumes some atmosphere)
    olr_clear = sigma * surface_temp**4 * 0.6  # ~0.6 for atmosphere

    # Cloudy sky OLR
    olr_cloudy = sigma * cloud_top_temp**4

    # CRE = clear - cloudy (lower OLR = warming)
    cre_lw = cloud_fraction * (olr_clear - olr_cloudy)

    return cre_lw


# =============================================================================
# Standard Atmosphere Cloud Profiles
# =============================================================================

def typical_cloud_profile(cloud_type='stratocumulus'):
    """
    Get typical cloud layer properties.

    Parameters
    ----------
    cloud_type : str
        Cloud type: 'stratocumulus', 'cumulus', 'cirrus', 'multilayer'

    Returns
    -------
    layers : dict
        Dictionary with layer properties
    """
    if cloud_type == 'stratocumulus':
        return {
            'heights': [1.0],  # km
            'fractions': [0.8],
            'lwp': [100],  # g/m^2
            'r_eff': [10],  # um
            'phase': ['liquid'],
            'temps': [283],  # K
        }

    elif cloud_type == 'cumulus':
        return {
            'heights': [2.0],
            'fractions': [0.3],
            'lwp': [200],
            'r_eff': [12],
            'phase': ['liquid'],
            'temps': [275],
        }

    elif cloud_type == 'cirrus':
        return {
            'heights': [10.0],
            'fractions': [0.4],
            'lwp': [20],  # Actually IWP for ice
            'r_eff': [30],
            'phase': ['ice'],
            'temps': [220],
        }

    elif cloud_type == 'multilayer':
        return {
            'heights': [1.5, 5.0, 10.0],
            'fractions': [0.6, 0.3, 0.4],
            'lwp': [80, 100, 20],
            'r_eff': [10, 12, 30],
            'phase': ['liquid', 'liquid', 'ice'],
            'temps': [283, 260, 220],
        }

    else:
        raise ValueError(f"Unknown cloud type: {cloud_type}")


def main():
    args = parse_args()

    print("=" * 70)
    print("MULTI-LAYER CLOUD OVERLAPPING EFFECTS")
    print("=" * 70)

    # Cloud overlap assumptions
    print("\n" + "-" * 70)
    print("CLOUD OVERLAP ASSUMPTIONS")
    print("-" * 70)
    print("""
Cloud overlap determines how clouds in different layers combine:

1. MAXIMUM OVERLAP
   - Clouds are vertically aligned
   - Total cover = max(individual fractions)
   - Minimum total cloud cover
   - Valid for deep convective systems

2. RANDOM OVERLAP
   - Clouds randomly distributed
   - Total clear = product of clear fractions
   - Moderate total cloud cover
   - Valid for well-mixed atmospheres

3. MAXIMUM-RANDOM OVERLAP
   - Maximum within contiguous groups
   - Random between separated groups
   - Most realistic for GCMs
   - Standard in many climate models
""")

    # Example with 3 layers
    print("\n" + "-" * 70)
    print("EXAMPLE: THREE-LAYER CLOUD SYSTEM")
    print("-" * 70)

    # Define layers
    heights = [2.0, 5.0, 10.0]  # km
    fractions = [0.5, 0.4, 0.3]  # cloud fractions
    lwp = [100, 80, 20]  # g/m^2
    r_eff = [10, 12, 30]  # um
    phases = ['liquid', 'liquid', 'ice']
    temps = [280, 260, 220]  # K

    print("\nLayer properties:")
    print(f"{'Layer':>6} {'Height':>10} {'Fraction':>10} {'LWP':>10} {'r_eff':>10} {'Phase':>10} {'Temp':>10}")
    print("-" * 75)

    for i in range(len(heights)):
        print(f"{i+1:>6} {heights[i]:>10.1f} km {fractions[i]:>10.1%} "
              f"{lwp[i]:>8} g/m2 {r_eff[i]:>8} um {phases[i]:>10} {temps[i]:>8} K")

    # Calculate total cover under different assumptions
    print("\n\nTotal cloud cover by overlap assumption:")
    print("-" * 40)

    cover_max = maximum_overlap_cover(fractions)
    cover_random = random_overlap_cover(fractions)
    cover_maxrand = maximum_random_overlap_cover(fractions, heights)

    print(f"  Maximum overlap:     {cover_max:.1%}")
    print(f"  Random overlap:      {cover_random:.1%}")
    print(f"  Maximum-random:      {cover_maxrand:.1%}")

    print(f"\nDifference: {(cover_random - cover_max)/cover_max * 100:.1f}% more cover with random vs maximum")

    # Calculate optical properties
    print("\n" + "-" * 70)
    print("OPTICAL PROPERTIES")
    print("-" * 70)

    optical_depths = []
    ssa_values = []
    g_values = []

    print(f"\n{'Layer':>6} {'tau':>10} {'SSA':>10} {'g':>10}")
    print("-" * 45)

    for i in range(len(heights)):
        tau = cloud_optical_depth(lwp[i], r_eff[i])
        ssa = cloud_single_scatter_albedo(0.55, phases[i])
        g = cloud_asymmetry_parameter(0.55, r_eff[i], phases[i])

        optical_depths.append(tau)
        ssa_values.append(ssa)
        g_values.append(g)

        print(f"{i+1:>6} {tau:>10.1f} {ssa:>10.4f} {g:>10.3f}")

    # Shortwave transmission
    print("\n" + "-" * 70)
    print("SHORTWAVE TRANSMISSION")
    print("-" * 70)

    sza_values = [0, 30, 60, 75]
    print(f"\n{'SZA':>6} {'mu0':>8} {'T(max)':>12} {'T(random)':>12}")
    print("-" * 45)

    for sza in sza_values:
        mu0 = np.cos(np.radians(sza))

        T_max = multilayer_transmission(
            fractions, optical_depths, ssa_values, g_values,
            overlap='maximum', mu0=mu0
        )

        T_rand = multilayer_transmission(
            fractions, optical_depths, ssa_values, g_values,
            overlap='random', mu0=mu0
        )

        print(f"{sza:>6} deg {mu0:>8.3f} {T_max:>12.3f} {T_rand:>12.3f}")

    # Cloud radiative effect
    print("\n" + "-" * 70)
    print("CLOUD RADIATIVE EFFECT (CRE)")
    print("-" * 70)

    total_tau = sum(optical_depths)
    avg_ssa = np.mean(ssa_values)
    avg_g = np.mean(g_values)

    print(f"\nSystem-averaged properties:")
    print(f"  Total optical depth: {total_tau:.1f}")
    print(f"  Average SSA: {avg_ssa:.4f}")
    print(f"  Average g: {avg_g:.3f}")

    # Calculate CRE for different cloud fractions
    print("\n\nCRE as function of cloud fraction (SZA=60 deg):")
    print(f"{'Cloud Frac':>12} {'CRE_SW':>12} {'CRE_LW':>12} {'CRE_net':>12}")
    print("-" * 52)

    cf_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    cloud_top_temp = temps[-1]  # Use highest cloud for LW

    for cf in cf_values:
        cre_sw = cloud_radiative_effect_sw(cf, total_tau, avg_ssa, avg_g, mu0=0.5)
        cre_lw = cloud_radiative_effect_lw(cf, cloud_top_temp)
        cre_net = cre_sw + cre_lw

        print(f"{cf:>12.0%} {cre_sw:>10.1f} W/m2 {cre_lw:>10.1f} W/m2 {cre_net:>10.1f} W/m2")

    print("""
Notes:
- Negative SW CRE = cooling (clouds reflect sunlight)
- Positive LW CRE = warming (clouds trap outgoing radiation)
- Net effect depends on cloud height (high cold clouds trap more)
- Low thick clouds: strong SW cooling dominates
- High thin clouds: LW warming can dominate
""")

    # Cloud type comparison
    print("\n" + "-" * 70)
    print("COMPARISON OF CLOUD TYPES")
    print("-" * 70)

    cloud_types = ['stratocumulus', 'cumulus', 'cirrus', 'multilayer']

    print(f"\n{'Type':>15} {'Cover':>10} {'tau':>10} {'CRE_SW':>10} {'CRE_LW':>10} {'Net':>10}")
    print("-" * 70)

    for ctype in cloud_types:
        profile = typical_cloud_profile(ctype)

        cf_total = random_overlap_cover(profile['fractions'])

        total_tau = 0
        for i, lwp_val in enumerate(profile['lwp']):
            total_tau += cloud_optical_depth(lwp_val, profile['r_eff'][i])

        cre_sw = cloud_radiative_effect_sw(cf_total, total_tau, 0.999, 0.85, mu0=0.5)
        cre_lw = cloud_radiative_effect_lw(cf_total, profile['temps'][-1])
        cre_net = cre_sw + cre_lw

        print(f"{ctype:>15} {cf_total:>10.0%} {total_tau:>10.1f} "
              f"{cre_sw:>8.1f} W/m2 {cre_lw:>8.1f} W/m2 {cre_net:>8.1f} W/m2")

    # Impact on climate
    print("\n" + "-" * 70)
    print("IMPORTANCE OF CLOUD OVERLAP IN CLIMATE MODELS")
    print("-" * 70)
    print("""
Cloud overlap assumptions significantly impact:

1. TOTAL CLOUD RADIATIVE EFFECT
   - Random overlap gives ~20-40% more cloud cover than maximum
   - This translates to larger cooling effect in SW
   - Critical for global energy budget

2. PRECIPITATION
   - Overlap affects when clouds overlap -> when rain falls
   - Maximum overlap concentrates precipitation
   - Random overlap spreads it more uniformly

3. CLIMATE SENSITIVITY
   - Cloud feedbacks are largest uncertainty
   - Overlap assumptions affect feedback strength
   - Low clouds especially sensitive to assumptions

4. SATELLITE RETRIEVALS
   - Passive sensors see only total column
   - Overlap must be assumed to retrieve layer properties
   - Affects cloud property climatologies

Modern GCMs typically use:
- Maximum-random overlap (Geleyn & Hollingsworth, 1979)
- Exponential-random overlap (Hogan & Illingworth, 2000)
- Generalized overlap (Pincus et al., 2003)
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Multi-layer cloud analysis:

Number of layers: {len(heights)}
Layer heights: {heights} km
Layer fractions: {[f'{f:.0%}' for f in fractions]}

Total cloud cover:
  Maximum overlap: {cover_max:.1%}
  Random overlap: {cover_random:.1%}
  Maximum-random: {cover_maxrand:.1%}

Radiative effects (SZA=60 deg, random overlap):
  Shortwave CRE: {cloud_radiative_effect_sw(cover_random, total_tau, avg_ssa, avg_g, mu0=0.5):.1f} W/m^2
  Longwave CRE: {cloud_radiative_effect_lw(cover_random, temps[-1]):.1f} W/m^2

Key findings:
1. Overlap assumption can change cloud cover by 20-40%
2. This significantly affects radiative budget
3. Low thick clouds cool, high thin clouds can warm
4. GCMs use sophisticated overlap schemes for accuracy
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Multi-Layer Cloud Overlap Effects', fontsize=14, fontweight='bold')

            # Plot 1: Overlap comparison
            ax1 = axes[0, 0]

            # Vary middle layer fraction
            cf_middle = np.linspace(0, 1, 50)
            cover_max_arr = []
            cover_rand_arr = []

            for cf in cf_middle:
                test_fractions = [0.5, cf, 0.3]
                cover_max_arr.append(maximum_overlap_cover(test_fractions))
                cover_rand_arr.append(random_overlap_cover(test_fractions))

            ax1.plot(cf_middle * 100, np.array(cover_max_arr) * 100, 'b-',
                     linewidth=2, label='Maximum overlap')
            ax1.plot(cf_middle * 100, np.array(cover_rand_arr) * 100, 'r-',
                     linewidth=2, label='Random overlap')
            ax1.fill_between(cf_middle * 100, np.array(cover_max_arr) * 100,
                            np.array(cover_rand_arr) * 100, alpha=0.3)
            ax1.set_xlabel('Middle Layer Cloud Fraction (%)')
            ax1.set_ylabel('Total Cloud Cover (%)')
            ax1.set_title('Total Cover vs Overlap Assumption')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(0, 100)

            # Plot 2: Transmission vs SZA
            ax2 = axes[0, 1]

            sza_range = np.linspace(0, 80, 50)
            T_max_arr = []
            T_rand_arr = []

            for sza in sza_range:
                mu0 = np.cos(np.radians(sza))
                T_max_arr.append(multilayer_transmission(
                    fractions, optical_depths, ssa_values, g_values,
                    overlap='maximum', mu0=mu0
                ))
                T_rand_arr.append(multilayer_transmission(
                    fractions, optical_depths, ssa_values, g_values,
                    overlap='random', mu0=mu0
                ))

            ax2.plot(sza_range, T_max_arr, 'b-', linewidth=2, label='Maximum')
            ax2.plot(sza_range, T_rand_arr, 'r-', linewidth=2, label='Random')
            ax2.set_xlabel('Solar Zenith Angle (deg)')
            ax2.set_ylabel('Transmission')
            ax2.set_title('SW Transmission vs SZA')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 80)

            # Plot 3: CRE components
            ax3 = axes[1, 0]

            cf_range = np.linspace(0.1, 1.0, 50)
            cre_sw_arr = []
            cre_lw_arr = []
            cre_net_arr = []

            for cf in cf_range:
                sw = cloud_radiative_effect_sw(cf, total_tau, avg_ssa, avg_g, mu0=0.5)
                lw = cloud_radiative_effect_lw(cf, temps[-1])
                cre_sw_arr.append(sw)
                cre_lw_arr.append(lw)
                cre_net_arr.append(sw + lw)

            ax3.plot(cf_range * 100, cre_sw_arr, 'b-', linewidth=2, label='SW (cooling)')
            ax3.plot(cf_range * 100, cre_lw_arr, 'r-', linewidth=2, label='LW (warming)')
            ax3.plot(cf_range * 100, cre_net_arr, 'k--', linewidth=2, label='Net')
            ax3.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax3.set_xlabel('Cloud Fraction (%)')
            ax3.set_ylabel('CRE (W/m^2)')
            ax3.set_title('Cloud Radiative Effect vs Coverage')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(10, 100)

            # Plot 4: Layer structure
            ax4 = axes[1, 1]

            # Draw cloud layers
            for i, (h, cf) in enumerate(zip(heights, fractions)):
                ax4.barh(h, cf * 100, height=0.8, color=f'C{i}', alpha=0.7,
                        label=f'Layer {i+1}: {cf:.0%}')
                ax4.text(cf * 100 + 2, h, f'{phases[i]}, T={temps[i]}K',
                        va='center', fontsize=9)

            ax4.set_xlabel('Cloud Fraction (%)')
            ax4.set_ylabel('Height (km)')
            ax4.set_title('Cloud Layer Structure')
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 100)
            ax4.set_ylim(0, 12)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
