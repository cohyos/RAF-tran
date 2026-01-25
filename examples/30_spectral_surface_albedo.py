#!/usr/bin/env python3
"""
Spectral Surface Albedo Effects
===============================

This example demonstrates how different surface types affect
radiative transfer through their spectral albedo properties.

Surface types covered:
- Snow and ice (high albedo, spectral features)
- Vegetation (red edge, NDVI)
- Ocean (low albedo, sun glint)
- Desert/soil (moderate albedo)
- Urban surfaces

Applications:
- Climate modeling
- Remote sensing (surface correction)
- Energy balance studies
- Agricultural monitoring

Usage:
    python 30_spectral_surface_albedo.py
    python 30_spectral_surface_albedo.py --surface snow
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
        description="Analyze spectral surface albedo effects"
    )
    parser.add_argument("--surface", type=str, default="all",
                       choices=['snow', 'vegetation', 'ocean', 'desert', 'urban', 'all'],
                       help="Surface type to analyze")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="spectral_albedo.png")
    return parser.parse_args()


# =============================================================================
# Spectral Albedo Models
# =============================================================================

def snow_albedo(wavelength_um, grain_size_um=100, contamination=0.0):
    """
    Calculate snow albedo as function of wavelength.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    grain_size_um : float
        Snow grain effective radius in micrometers
    contamination : float
        Black carbon contamination (fraction, 0-1)

    Returns
    -------
    albedo : float
        Spectral albedo (0-1)
    """
    wl = wavelength_um

    # Base albedo (clean snow)
    # Very high in visible, decreases in NIR due to ice absorption
    if wl < 0.4:
        base = 0.95
    elif wl < 0.8:
        # High reflectance in visible
        base = 0.98 - 0.02 * (wl - 0.4) / 0.4
    elif wl < 1.4:
        # Ice absorption bands start
        # Strong absorption at 1.03 and 1.27 um
        absorption = 0.15 * np.exp(-((wl - 1.03) / 0.1)**2)
        absorption += 0.20 * np.exp(-((wl - 1.27) / 0.1)**2)
        base = 0.85 - 0.30 * (wl - 0.8) / 0.6 - absorption
    elif wl < 2.0:
        # Strong absorption at 1.5 and 2.0 um
        base = 0.4 - 0.35 * (wl - 1.4) / 0.6
    else:
        base = 0.05

    # Grain size effect (larger grains = more absorption in NIR)
    size_factor = 1.0 - 0.001 * (grain_size_um - 100) * max(0, wl - 0.8)
    size_factor = max(0.3, min(1.0, size_factor))

    # Contamination effect (reduces visible albedo most)
    if wl < 0.7:
        contam_factor = 1.0 - contamination * 0.5
    else:
        contam_factor = 1.0 - contamination * 0.2

    albedo = base * size_factor * contam_factor
    return max(0.01, min(0.99, albedo))


def vegetation_albedo(wavelength_um, lai=3.0, vegetation_type='grass'):
    """
    Calculate vegetation albedo with red edge feature.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    lai : float
        Leaf Area Index
    vegetation_type : str
        'grass', 'forest', 'crop'

    Returns
    -------
    albedo : float
        Spectral albedo (0-1)
    """
    wl = wavelength_um

    # Chlorophyll absorption bands
    # Strong absorption at 450 nm (blue) and 680 nm (red)
    # High reflectance in NIR (plateau at 0.8-1.3 um)

    if wl < 0.5:
        # Blue - moderate absorption by chlorophyll
        chloro_abs = 0.4 * np.exp(-((wl - 0.45) / 0.03)**2)
        base = 0.08 - chloro_abs * 0.04
    elif wl < 0.55:
        # Green peak - less absorption
        base = 0.12
    elif wl < 0.68:
        # Yellow-red - increasing absorption
        base = 0.12 - 0.07 * (wl - 0.55) / 0.13
    elif wl < 0.70:
        # Red edge start - strong chlorophyll absorption
        base = 0.05
    elif wl < 0.75:
        # Red edge transition
        base = 0.05 + 0.40 * (wl - 0.70) / 0.05
    elif wl < 1.3:
        # NIR plateau
        base = 0.45
    elif wl < 1.45:
        # Water absorption
        base = 0.40 - 0.20 * (wl - 1.3) / 0.15
    elif wl < 1.8:
        base = 0.20
    elif wl < 2.0:
        # Water absorption
        base = 0.15 - 0.10 * (wl - 1.8) / 0.2
    else:
        base = 0.05

    # LAI effect (more leaves = more reflectance in NIR, more absorption in VIS)
    lai_factor = 1 - np.exp(-0.5 * lai)
    if wl > 0.7:
        albedo = base * (0.7 + 0.3 * lai_factor)
    else:
        albedo = base * (1.0 - 0.2 * lai_factor)

    # Vegetation type
    if vegetation_type == 'forest':
        albedo *= 0.85  # Forests slightly darker
    elif vegetation_type == 'crop':
        albedo *= 1.05  # Crops slightly brighter

    return max(0.01, min(0.60, albedo))


def ocean_albedo(wavelength_um, sza_deg=30, wind_speed_ms=5):
    """
    Calculate ocean albedo including sun glint.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    sza_deg : float
        Solar zenith angle in degrees
    wind_speed_ms : float
        Wind speed in m/s (affects roughness)

    Returns
    -------
    albedo : float
        Spectral albedo (0-1)
    """
    wl = wavelength_um
    sza_rad = np.radians(sza_deg)
    mu0 = np.cos(sza_rad)

    # Base Fresnel reflectance (normal incidence ~ 2%)
    # Increases at grazing angles
    if sza_deg < 70:
        fresnel = 0.02 + 0.02 * (1 - mu0)
    else:
        # Rapid increase near horizon
        fresnel = 0.04 + 0.30 * (sza_deg - 70) / 20

    # Wavelength dependence (minimal for open ocean)
    # Slightly higher in blue due to scattering
    if wl < 0.5:
        wl_factor = 1.2 - 0.4 * wl
    else:
        wl_factor = 1.0

    # Wind speed effect (more roughness = less glint, more diffuse)
    if wind_speed_ms < 2:
        wind_factor = 1.2  # Calm = more specular
    elif wind_speed_ms < 10:
        wind_factor = 1.0 - 0.02 * (wind_speed_ms - 2)
    else:
        wind_factor = 0.85

    albedo = fresnel * wl_factor * wind_factor

    # Add whitecap contribution at high wind speeds
    if wind_speed_ms > 7:
        whitecap_frac = 0.0003 * (wind_speed_ms - 7)**2
        if wl < 1.0:
            albedo += whitecap_frac * 0.5  # Whitecaps are bright

    return max(0.01, min(0.30, albedo))


def desert_albedo(wavelength_um, soil_type='sand'):
    """
    Calculate desert/soil albedo.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    soil_type : str
        'sand', 'clay', 'loam'

    Returns
    -------
    albedo : float
        Spectral albedo (0-1)
    """
    wl = wavelength_um

    if soil_type == 'sand':
        # Bright sand - increases with wavelength in visible
        if wl < 0.5:
            base = 0.15 + 0.10 * (wl - 0.35) / 0.15
        elif wl < 1.0:
            base = 0.25 + 0.15 * (wl - 0.5) / 0.5
        elif wl < 2.0:
            base = 0.40 - 0.10 * (wl - 1.0)
        else:
            base = 0.30

    elif soil_type == 'clay':
        # Dark clay - flatter spectrum
        if wl < 0.6:
            base = 0.10 + 0.05 * (wl - 0.35) / 0.25
        else:
            base = 0.15 + 0.05 * (wl - 0.6) / 0.4

    else:  # loam
        if wl < 0.6:
            base = 0.12 + 0.08 * (wl - 0.35) / 0.25
        else:
            base = 0.20 + 0.08 * (wl - 0.6) / 0.4

    return max(0.05, min(0.50, base))


def urban_albedo(wavelength_um, surface_type='mixed'):
    """
    Calculate urban surface albedo.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in micrometers
    surface_type : str
        'asphalt', 'concrete', 'rooftop', 'mixed'

    Returns
    -------
    albedo : float
        Spectral albedo (0-1)
    """
    wl = wavelength_um

    if surface_type == 'asphalt':
        # Dark, relatively flat spectrum
        base = 0.08 + 0.04 * (wl - 0.35) / 0.65
    elif surface_type == 'concrete':
        # Moderate, slightly increasing with wavelength
        base = 0.20 + 0.10 * (wl - 0.35) / 0.65
    elif surface_type == 'rooftop':
        # Depends heavily on material
        if wl < 0.7:
            base = 0.25
        else:
            base = 0.35
    else:  # mixed
        # Weighted average
        base = 0.12 + 0.08 * (wl - 0.35) / 0.65

    return max(0.05, min(0.40, base))


# =============================================================================
# Vegetation Indices
# =============================================================================

def calculate_ndvi(red_reflectance, nir_reflectance):
    """
    Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    red_reflectance : float
        Reflectance at red wavelength (~650-680 nm)
    nir_reflectance : float
        Reflectance at NIR wavelength (~800-900 nm)

    Returns
    -------
    ndvi : float
        NDVI value (-1 to 1)
    """
    if (nir_reflectance + red_reflectance) == 0:
        return 0.0
    return (nir_reflectance - red_reflectance) / (nir_reflectance + red_reflectance)


def interpret_ndvi(ndvi):
    """
    Interpret NDVI value.

    Parameters
    ----------
    ndvi : float
        NDVI value

    Returns
    -------
    interpretation : str
        Land cover interpretation
    """
    if ndvi < -0.1:
        return "Water or snow"
    elif ndvi < 0.1:
        return "Bare soil, rock, sand"
    elif ndvi < 0.2:
        return "Sparse vegetation, urban"
    elif ndvi < 0.4:
        return "Moderate vegetation, shrubland"
    elif ndvi < 0.6:
        return "Dense vegetation, grassland"
    else:
        return "Very dense vegetation, forest"


# =============================================================================
# Albedo Effects on Radiative Transfer
# =============================================================================

def surface_forcing(albedo_old, albedo_new, solar_flux=1361, sza_deg=30):
    """
    Calculate radiative forcing from albedo change.

    Parameters
    ----------
    albedo_old : float
        Original albedo
    albedo_new : float
        New albedo
    solar_flux : float
        Solar constant (W/m^2)
    sza_deg : float
        Solar zenith angle (degrees)

    Returns
    -------
    forcing : float
        Radiative forcing (W/m^2)
        Negative = cooling, Positive = warming
    """
    mu0 = np.cos(np.radians(sza_deg))

    # Change in absorbed flux
    absorbed_old = solar_flux * mu0 * (1 - albedo_old)
    absorbed_new = solar_flux * mu0 * (1 - albedo_new)

    # Forcing = change in absorbed (lower albedo = more absorbed = warming)
    forcing = absorbed_new - absorbed_old

    return forcing


def effective_broadband_albedo(surface_func, sza_deg=30, **kwargs):
    """
    Calculate effective broadband albedo weighted by solar spectrum.

    Parameters
    ----------
    surface_func : callable
        Function returning spectral albedo
    sza_deg : float
        Solar zenith angle
    **kwargs : dict
        Additional arguments for surface function

    Returns
    -------
    albedo_bb : float
        Broadband albedo
    """
    # Solar spectral weighting (simplified, vectorized)
    wavelengths = np.linspace(0.35, 2.5, 100)

    # Vectorized solar weight calculation
    solar_weights = np.where(
        wavelengths < 0.5,
        1.0,
        np.where(
            wavelengths < 1.0,
            1.2 - 0.4 * (wavelengths - 0.5),
            0.8 - 0.3 * (wavelengths - 1.0)
        )
    )
    solar_weights = np.maximum(0.1, solar_weights)

    # Calculate weighted average (vectorized)
    albedos = np.array([surface_func(wl, **kwargs) for wl in wavelengths])
    albedo_bb = np.sum(albedos * solar_weights) / np.sum(solar_weights)

    return albedo_bb


# =============================================================================
# Ice-Albedo Feedback
# =============================================================================

def ice_albedo_feedback_sensitivity(temp_change, ice_cover_init=0.1, land_fraction=0.3):
    """
    Estimate ice-albedo feedback effect.

    Parameters
    ----------
    temp_change : float
        Temperature change (K)
    ice_cover_init : float
        Initial ice-covered fraction of planet
    land_fraction : float
        Fraction of surface that is land

    Returns
    -------
    feedback_factor : float
        Additional warming fraction from feedback
    """
    # Simplified model
    # Ice retreats ~5% of coverage per degree warming
    ice_retreat_rate = 0.05

    # Change in ice coverage
    delta_ice = -ice_retreat_rate * ice_cover_init * temp_change
    delta_ice = max(-ice_cover_init, delta_ice)  # Can't go negative

    # Albedo change (ice ~0.7, land ~0.2, ocean ~0.06)
    albedo_ice = 0.70
    albedo_land = 0.20
    albedo_ocean = 0.06

    # New surface exposed is proportional to land/ocean ratio
    albedo_exposed = land_fraction * albedo_land + (1 - land_fraction) * albedo_ocean

    # Change in planetary albedo
    delta_albedo = delta_ice * (albedo_exposed - albedo_ice)

    # Radiative forcing (global average solar = 340 W/m^2)
    forcing = -340 * delta_albedo

    # Feedback factor (additional warming per initial warming)
    # Using lambda = 3.2 W/m^2/K for climate sensitivity
    lambda_param = 3.2
    feedback_warming = forcing / lambda_param

    feedback_factor = feedback_warming / temp_change if temp_change != 0 else 0

    return feedback_factor


def main():
    args = parse_args()

    print("=" * 70)
    print("SPECTRAL SURFACE ALBEDO EFFECTS")
    print("=" * 70)

    # Surface albedo overview
    print("\n" + "-" * 70)
    print("SURFACE ALBEDO OVERVIEW")
    print("-" * 70)
    print("""
Albedo (reflectance) varies significantly by:
  - Surface type (snow, vegetation, ocean, urban)
  - Wavelength (UV, visible, NIR)
  - Illumination angle (especially water)
  - Surface condition (wet/dry, grain size, contamination)

Typical broadband albedos:
  Fresh snow:     0.80-0.95
  Old snow:       0.50-0.70
  Ice:            0.30-0.60
  Forest:         0.10-0.20
  Grassland:      0.15-0.25
  Desert sand:    0.30-0.45
  Ocean:          0.03-0.10
  Urban:          0.10-0.20
""")

    # Spectral comparison
    print("\n" + "-" * 70)
    print("SPECTRAL ALBEDO COMPARISON")
    print("-" * 70)

    wavelengths = [0.40, 0.50, 0.55, 0.65, 0.70, 0.75, 0.80, 0.90, 1.0, 1.5, 2.0]

    print(f"\n{'Wavelength':>10} {'Snow':>8} {'Veg':>8} {'Ocean':>8} {'Desert':>8} {'Urban':>8}")
    print("-" * 60)

    for wl in wavelengths:
        a_snow = snow_albedo(wl)
        a_veg = vegetation_albedo(wl)
        a_ocean = ocean_albedo(wl)
        a_desert = desert_albedo(wl)
        a_urban = urban_albedo(wl)

        print(f"{wl*1000:>8.0f} nm {a_snow:>8.2f} {a_veg:>8.2f} {a_ocean:>8.2f} "
              f"{a_desert:>8.2f} {a_urban:>8.2f}")

    # Snow properties
    print("\n" + "-" * 70)
    print("SNOW ALBEDO PROPERTIES")
    print("-" * 70)
    print("""
Snow albedo depends on:
  - Grain size: Larger grains = more absorption (especially NIR)
  - Contamination: Black carbon reduces visible albedo
  - Ice absorption bands: 1.03, 1.27, 1.5, 2.0 um

This is why:
  - Fresh powder is brightest (small grains, clean)
  - Melting snow darkens (larger grains, contaminants)
  - Spring snowmelt accelerates (positive feedback)
""")

    # Compare grain sizes
    print(f"\nEffect of grain size at key wavelengths:")
    print(f"{'Wavelength':>12} {'50 um':>10} {'100 um':>10} {'500 um':>10}")
    print("-" * 50)

    for wl in [0.5, 0.8, 1.0, 1.5]:
        a_50 = snow_albedo(wl, grain_size_um=50)
        a_100 = snow_albedo(wl, grain_size_um=100)
        a_500 = snow_albedo(wl, grain_size_um=500)
        print(f"{wl*1000:>10.0f} nm {a_50:>10.3f} {a_100:>10.3f} {a_500:>10.3f}")

    # Vegetation red edge
    print("\n" + "-" * 70)
    print("VEGETATION RED EDGE")
    print("-" * 70)
    print("""
The vegetation 'red edge' is the sharp increase in reflectance
from ~680 to 750 nm:

  < 680 nm: Strong chlorophyll absorption (red)
  680-750 nm: Transition zone (red edge)
  > 750 nm: High reflectance (NIR plateau)

This creates high contrast in NIR for:
  - Vegetation mapping
  - Health assessment
  - Agricultural monitoring
""")

    # Red edge transition
    print(f"\nRed edge transition:")
    print(f"{'Wavelength':>12} {'Albedo':>10}")
    print("-" * 25)

    for wl in np.arange(0.65, 0.82, 0.02):
        a = vegetation_albedo(wl)
        print(f"{wl*1000:>10.0f} nm {a:>10.3f}")

    # NDVI calculation
    print("\n" + "-" * 70)
    print("NDVI (NORMALIZED DIFFERENCE VEGETATION INDEX)")
    print("-" * 70)

    surfaces = {
        'Dense forest': (vegetation_albedo, {'lai': 6.0}),
        'Grassland': (vegetation_albedo, {'lai': 2.0}),
        'Sparse veg': (vegetation_albedo, {'lai': 0.5}),
        'Bare soil': (desert_albedo, {'soil_type': 'loam'}),
        'Ocean': (ocean_albedo, {}),
        'Snow': (snow_albedo, {}),
    }

    print(f"\n{'Surface':>15} {'Red (650)':>12} {'NIR (850)':>12} {'NDVI':>10} {'Interpretation':>25}")
    print("-" * 80)

    for name, (func, kwargs) in surfaces.items():
        red = func(0.65, **kwargs)
        nir = func(0.85, **kwargs)
        ndvi = calculate_ndvi(red, nir)
        interp = interpret_ndvi(ndvi)
        print(f"{name:>15} {red:>12.3f} {nir:>12.3f} {ndvi:>10.2f} {interp:>25}")

    # Ocean sun glint
    print("\n" + "-" * 70)
    print("OCEAN ALBEDO AND SUN GLINT")
    print("-" * 70)
    print("""
Ocean albedo depends on:
  - Solar zenith angle (Fresnel reflectance)
  - Wind speed (surface roughness)
  - Wavelength (weakly)
  - Whitecap coverage (high winds)
""")

    print(f"\nOcean albedo vs solar zenith angle:")
    print(f"{'SZA (deg)':>12} {'Albedo':>10}")
    print("-" * 25)

    for sza in [0, 20, 40, 60, 70, 80, 85]:
        a = ocean_albedo(0.5, sza_deg=sza)
        print(f"{sza:>12} {a:>10.3f}")

    # Broadband albedos
    print("\n" + "-" * 70)
    print("BROADBAND EFFECTIVE ALBEDOS")
    print("-" * 70)

    bb_albedos = {
        'Fresh snow (50 um)': effective_broadband_albedo(snow_albedo, grain_size_um=50),
        'Old snow (500 um)': effective_broadband_albedo(snow_albedo, grain_size_um=500),
        'Dense forest': effective_broadband_albedo(vegetation_albedo, lai=6.0),
        'Grassland': effective_broadband_albedo(vegetation_albedo, lai=2.0),
        'Ocean (calm)': effective_broadband_albedo(ocean_albedo, wind_speed_ms=2),
        'Ocean (rough)': effective_broadband_albedo(ocean_albedo, wind_speed_ms=15),
        'Desert sand': effective_broadband_albedo(desert_albedo, soil_type='sand'),
        'Urban mixed': effective_broadband_albedo(urban_albedo),
    }

    print(f"\n{'Surface':>25} {'Broadband Albedo':>20}")
    print("-" * 50)

    for name, albedo in bb_albedos.items():
        print(f"{name:>25} {albedo:>20.3f}")

    # Albedo change forcing
    print("\n" + "-" * 70)
    print("RADIATIVE FORCING FROM ALBEDO CHANGE")
    print("-" * 70)

    print(f"\nScenarios (SZA=30 deg, Solar=1361 W/m^2):")
    print(f"{'Scenario':>30} {'Old':>8} {'New':>8} {'Forcing':>12}")
    print("-" * 65)

    scenarios = [
        ('Snow melt (fresh to old)', 0.85, 0.55),
        ('Deforestation', 0.15, 0.25),
        ('Urban expansion', 0.20, 0.12),
        ('Desert irrigation', 0.35, 0.20),
        ('Arctic ice loss', 0.70, 0.06),
    ]

    for name, old, new in scenarios:
        forcing = surface_forcing(old, new)
        sign = '+' if forcing > 0 else ''
        print(f"{name:>30} {old:>8.2f} {new:>8.2f} {sign}{forcing:>10.1f} W/m^2")

    # Ice-albedo feedback
    print("\n" + "-" * 70)
    print("ICE-ALBEDO FEEDBACK")
    print("-" * 70)
    print("""
The ice-albedo feedback is a positive climate feedback:

  Warming -> Ice melts -> Darker surface exposed
         -> More solar absorbed -> More warming

This amplifies initial warming by ~10-20%.
""")

    print(f"\nFeedback strength vs initial warming:")
    print(f"{'Warming (K)':>15} {'Feedback Factor':>20} {'Additional warming':>20}")
    print("-" * 60)

    for dT in [0.5, 1.0, 2.0, 3.0, 5.0]:
        ff = ice_albedo_feedback_sensitivity(dT)
        add_warm = dT * ff
        print(f"{dT:>15.1f} {ff:>20.2f} {add_warm:>18.2f} K")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:

1. SNOW/ICE
   - Highest albedo surfaces (0.5-0.95)
   - Strong wavelength dependence (ice absorption in NIR)
   - Sensitive to grain size and contamination

2. VEGETATION
   - Characteristic red edge at 700 nm
   - Low visible albedo (chlorophyll absorption)
   - High NIR albedo (leaf structure scattering)
   - NDVI captures this contrast

3. OCEAN
   - Lowest albedo (~0.03-0.10)
   - Strong SZA dependence (Fresnel reflectance)
   - Wind affects roughness and whitecaps

4. ALBEDO CHANGES
   - Ice loss: Major warming effect (strong feedback)
   - Deforestation: Small cooling (higher albedo)
   - Urbanization: Net warming (lower albedo + heat island)

Implications:
- Climate models must resolve spectral albedo
- Remote sensing relies on spectral differences
- Land use changes affect regional and global climate
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Spectral Surface Albedo Effects', fontsize=14, fontweight='bold')

            # Plot 1: Spectral albedo comparison
            ax1 = axes[0, 0]
            wl_range = np.linspace(0.35, 2.2, 200)

            a_snow = [snow_albedo(wl) for wl in wl_range]
            a_veg = [vegetation_albedo(wl) for wl in wl_range]
            a_ocean = [ocean_albedo(wl) for wl in wl_range]
            a_desert = [desert_albedo(wl) for wl in wl_range]
            a_urban = [urban_albedo(wl) for wl in wl_range]

            ax1.plot(wl_range * 1000, a_snow, 'c-', linewidth=2, label='Snow')
            ax1.plot(wl_range * 1000, a_veg, 'g-', linewidth=2, label='Vegetation')
            ax1.plot(wl_range * 1000, a_ocean, 'b-', linewidth=2, label='Ocean')
            ax1.plot(wl_range * 1000, a_desert, 'orange', linewidth=2, label='Desert')
            ax1.plot(wl_range * 1000, a_urban, 'gray', linewidth=2, label='Urban')

            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Albedo')
            ax1.set_title('Spectral Albedo by Surface Type')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(350, 2200)
            ax1.set_ylim(0, 1)

            # Plot 2: Red edge detail
            ax2 = axes[0, 1]
            wl_edge = np.linspace(0.55, 0.95, 100)

            for lai in [0.5, 2.0, 4.0, 6.0]:
                a = [vegetation_albedo(wl, lai=lai) for wl in wl_edge]
                ax2.plot(wl_edge * 1000, a, linewidth=2, label=f'LAI={lai}')

            ax2.axvline(680, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(750, color='red', linestyle='--', alpha=0.5)
            ax2.text(715, 0.55, 'Red Edge', ha='center', fontsize=10)

            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Albedo')
            ax2.set_title('Vegetation Red Edge vs LAI')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(550, 950)

            # Plot 3: Snow grain size effect
            ax3 = axes[1, 0]

            for grain in [50, 100, 200, 500, 1000]:
                a = [snow_albedo(wl, grain_size_um=grain) for wl in wl_range]
                ax3.plot(wl_range * 1000, a, linewidth=2, label=f'{grain} um')

            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Albedo')
            ax3.set_title('Snow Albedo vs Grain Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(350, 2200)
            ax3.set_ylim(0, 1)

            # Plot 4: Ocean albedo vs SZA
            ax4 = axes[1, 1]
            sza_range = np.linspace(0, 85, 50)

            for ws in [2, 5, 10, 15]:
                a = [ocean_albedo(0.5, sza_deg=sza, wind_speed_ms=ws) for sza in sza_range]
                ax4.plot(sza_range, a, linewidth=2, label=f'Wind {ws} m/s')

            ax4.set_xlabel('Solar Zenith Angle (deg)')
            ax4.set_ylabel('Albedo')
            ax4.set_title('Ocean Albedo vs SZA and Wind Speed')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, 85)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
