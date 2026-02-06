#!/usr/bin/env python3
"""
Limb Viewing Geometry and Retrievals
=====================================

This example demonstrates limb viewing geometry for satellite
atmospheric remote sensing.

Key concepts:
- Tangent altitude and limb path
- Extended path length through atmosphere
- Vertical resolution advantages
- Limb scattering and emission
- Onion-peeling retrieval technique

Applications:
- Stratospheric ozone monitoring
- Aerosol layer detection
- Temperature profiling
- Trace gas measurements
- Airglow observations

Usage:
    python 31_limb_viewing_geometry.py
    python 31_limb_viewing_geometry.py --tangent-height 30
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
        description="Analyze limb viewing geometry"
    )
    parser.add_argument("--tangent-height", type=float, default=25,
                       help="Tangent height in km")
    parser.add_argument("--satellite-alt", type=float, default=600,
                       help="Satellite altitude in km")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--output", type=str, default="limb_viewing.png")
    return parser.parse_args()


# Earth parameters
R_EARTH = 6371.0  # km


# =============================================================================
# Limb Geometry Functions
# =============================================================================

def limb_path_geometry(tangent_height_km, satellite_alt_km=600):
    """
    Calculate limb viewing geometry.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude in km
    satellite_alt_km : float
        Satellite altitude in km

    Returns
    -------
    geometry : dict
        Dictionary with geometric parameters
    """
    R = R_EARTH
    h_t = tangent_height_km
    h_s = satellite_alt_km

    # Distance from Earth center to tangent point
    r_t = R + h_t

    # Distance from Earth center to satellite
    r_s = R + h_s

    # Limb angle at satellite
    # sin(theta) = r_t / r_s
    sin_theta = r_t / r_s
    theta_rad = np.arcsin(sin_theta)
    theta_deg = np.degrees(theta_rad)

    # Path length from satellite to tangent point
    # d = sqrt(r_s^2 - r_t^2)
    d_to_tangent = np.sqrt(r_s**2 - r_t**2)

    # Total path through atmosphere (both sides)
    total_path = 2 * d_to_tangent

    return {
        'tangent_height_km': h_t,
        'satellite_alt_km': h_s,
        'limb_angle_deg': theta_deg,
        'path_to_tangent_km': d_to_tangent,
        'total_path_km': total_path,
    }


def path_length_through_layer(tangent_height_km, layer_bottom_km, layer_top_km):
    """
    Calculate path length through a single atmospheric layer.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude
    layer_bottom_km : float
        Layer bottom altitude
    layer_top_km : float
        Layer top altitude

    Returns
    -------
    path_km : float
        Path length through layer (km)
    """
    R = R_EARTH
    h_t = tangent_height_km

    # Tangent radius
    r_t = R + h_t

    # Layer radii
    r_bot = R + layer_bottom_km
    r_top = R + layer_top_km

    # If tangent point is above layer, path is zero
    if h_t >= layer_top_km:
        return 0.0

    # If tangent point is within layer
    if h_t >= layer_bottom_km:
        # Path from tangent to top of layer (both sides)
        d_top = np.sqrt(r_top**2 - r_t**2)
        return 2 * d_top

    # Tangent point is below layer
    # Path from bottom to top (both sides)
    d_top = np.sqrt(r_top**2 - r_t**2)
    d_bot = np.sqrt(r_bot**2 - r_t**2)
    return 2 * (d_top - d_bot)


def limb_weighting_function(tangent_height_km, altitude_km, scale_height_km=7.0):
    """
    Calculate limb weighting function for a given tangent height.

    The weighting function describes the contribution of each
    altitude to the limb measurement.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude
    altitude_km : float or array
        Altitude(s) to evaluate
    scale_height_km : float
        Atmospheric scale height

    Returns
    -------
    weight : float or array
        Weighting function value(s)
    """
    h = np.asarray(altitude_km)
    h_t = tangent_height_km
    H = scale_height_km
    R = R_EARTH

    # Only layers above tangent point contribute
    weight = np.zeros_like(h, dtype=float)
    mask = h >= h_t

    if np.any(mask):
        # Approximate weighting function
        # Peak at tangent height, decreasing above
        delta_h = h[mask] - h_t

        # Path length factor (longer path near tangent)
        r_t = R + h_t
        r_h = R + h[mask]
        path_factor = np.sqrt(2 * r_t * (r_h - r_t)) / (r_h - r_t + 0.1)

        # Density factor (exponential atmosphere)
        density_factor = np.exp(-delta_h / H)

        weight[mask] = path_factor * density_factor

    # Normalize
    if np.max(weight) > 0:
        weight = weight / np.max(weight)

    return weight


# =============================================================================
# Limb Radiance Calculations
# =============================================================================

def rayleigh_optical_depth_limb(tangent_height_km, wavelength_um, scale_height_km=8.5):
    """
    Calculate Rayleigh optical depth for limb path.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude
    wavelength_um : float
        Wavelength in micrometers
    scale_height_km : float
        Scale height for Rayleigh scattering

    Returns
    -------
    tau : float
        Optical depth along limb path
    """
    # Rayleigh cross section
    sigma = rayleigh_cross_section(wavelength_um)

    # Column density at sea level
    N_0 = 2.55e25  # molecules/m^2 per km vertical

    # Effective path length at tangent height
    R = R_EARTH * 1e3  # meters
    H = scale_height_km * 1e3  # meters
    h_t = tangent_height_km * 1e3  # meters

    # Chapman function approximation for limb path
    # L_eff = sqrt(2 * pi * (R + h_t) * H)
    L_eff = np.sqrt(2 * np.pi * (R + h_t) * H)

    # Column density along path
    N_column = N_0 * np.exp(-h_t / (H / 1e3)) * (L_eff / 1e3) / H * 1e3

    tau = sigma * N_column

    return tau


def limb_transmission(tangent_height_km, wavelength_um):
    """
    Calculate limb transmission at given tangent height.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude
    wavelength_um : float
        Wavelength in micrometers

    Returns
    -------
    T : float
        Transmission (0 to 1)
    """
    tau = rayleigh_optical_depth_limb(tangent_height_km, wavelength_um)
    return np.exp(-tau)


def limb_scattering_radiance(tangent_height_km, wavelength_um, sza_deg=60):
    """
    Calculate scattered solar radiance in limb view.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude
    wavelength_um : float
        Wavelength in micrometers
    sza_deg : float
        Solar zenith angle at tangent point

    Returns
    -------
    radiance : float
        Relative scattered radiance
    """
    # Simple single-scattering model
    mu0 = np.cos(np.radians(sza_deg))

    # Optical depth
    tau = rayleigh_optical_depth_limb(tangent_height_km, wavelength_um)

    # Single-scatter approximation
    # Radiance proportional to tau for optically thin
    # Saturates for optically thick
    if tau < 0.1:
        radiance = tau * mu0
    else:
        radiance = (1 - np.exp(-tau)) * mu0

    return radiance


def limb_emission_radiance(tangent_height_km, wavelength_um, temp_profile):
    """
    Calculate thermal emission in limb view.

    Parameters
    ----------
    tangent_height_km : float
        Tangent point altitude
    wavelength_um : float
        Wavelength in micrometers
    temp_profile : callable
        Temperature as function of altitude (K)

    Returns
    -------
    radiance : float
        Relative emission radiance
    """
    # Temperature at tangent height
    T = temp_profile(tangent_height_km)

    # Planck function (relative)
    c1 = 1.191e8  # W um^4 / (m^2 sr)
    c2 = 1.439e4  # um K

    wl = wavelength_um
    B = c1 / (wl**5 * (np.exp(c2 / (wl * T)) - 1))

    # Weight by atmospheric density (optically thin approximation)
    scale_height = 7.0  # km
    density_factor = np.exp(-tangent_height_km / scale_height)

    return B * density_factor


# =============================================================================
# Onion-Peeling Retrieval
# =============================================================================

def onion_peel_retrieval(limb_radiances, tangent_heights):
    """
    Demonstrate onion-peeling retrieval concept.

    Parameters
    ----------
    limb_radiances : array_like
        Measured limb radiances at each tangent height
    tangent_heights : array_like
        Tangent heights (km)

    Returns
    -------
    retrieved_values : array
        Retrieved atmospheric quantity at each altitude
    """
    n = len(tangent_heights)
    radiances = np.array(limb_radiances)
    heights = np.array(tangent_heights)

    # Start from top (highest tangent height)
    # Each layer only contributes to measurements at that tangent height or lower

    retrieved = np.zeros(n)

    # Simple onion-peeling
    for i in range(n - 1, -1, -1):
        # Contribution from layers above (already retrieved)
        contribution_above = 0
        for j in range(i + 1, n):
            # Path length through layer j for tangent height i
            if j > i:
                path = path_length_through_layer(heights[i], heights[j-1], heights[j])
                contribution_above += retrieved[j] * path

        # Solve for this layer
        path_this = path_length_through_layer(heights[i], heights[i], heights[i] + 1)
        if path_this > 0:
            retrieved[i] = (radiances[i] - contribution_above) / path_this
        else:
            retrieved[i] = radiances[i]

    return retrieved


# =============================================================================
# Standard Atmosphere Profiles
# =============================================================================

def standard_temperature(altitude_km):
    """
    Standard atmosphere temperature profile.

    Parameters
    ----------
    altitude_km : float
        Altitude in km

    Returns
    -------
    T : float
        Temperature in K
    """
    h = altitude_km

    if h < 11:
        # Troposphere
        T = 288.15 - 6.5 * h
    elif h < 20:
        # Lower stratosphere (isothermal)
        T = 216.65
    elif h < 47:
        # Upper stratosphere
        T = 216.65 + 1.0 * (h - 20)
    elif h < 51:
        # Stratopause
        T = 270.65
    elif h < 71:
        # Mesosphere
        T = 270.65 - 2.8 * (h - 51)
    else:
        # Upper mesosphere
        T = 214.65 - 2.0 * (h - 71)

    return max(180, T)


def ozone_profile(altitude_km):
    """
    Simplified ozone number density profile.

    Parameters
    ----------
    altitude_km : float
        Altitude in km

    Returns
    -------
    n_o3 : float
        Relative ozone density (peak = 1)
    """
    h = altitude_km

    # Ozone layer peaks around 20-25 km
    peak_alt = 22
    width = 6

    n_o3 = np.exp(-((h - peak_alt) / width)**2)

    return n_o3


def main():
    args = parse_args()

    print("=" * 70)
    print("LIMB VIEWING GEOMETRY AND RETRIEVALS")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Tangent height: {args.tangent_height} km")
    print(f"  Satellite altitude: {args.satellite_alt} km")

    # Limb viewing overview
    print("\n" + "-" * 70)
    print("LIMB VIEWING OVERVIEW")
    print("-" * 70)
    print("""
Limb viewing observes the atmosphere tangentially:

  Satellite -----> Tangent Point -----> Space
                        |
                    Earth Surface

Advantages over nadir viewing:
  - Very long path through thin layers (~100-1000x longer)
  - Excellent vertical resolution (~1-3 km)
  - Enhanced sensitivity to trace species
  - Self-calibrating (view of space available)

Disadvantages:
  - Horizontal averaging over long path
  - Cloud contamination
  - Complex radiative transfer
  - Limited coverage (no surface)
""")

    # Geometry calculations
    print("\n" + "-" * 70)
    print("LIMB GEOMETRY")
    print("-" * 70)

    geom = limb_path_geometry(args.tangent_height, args.satellite_alt)

    print(f"""
Geometry for tangent height = {geom['tangent_height_km']} km:

  Satellite altitude: {geom['satellite_alt_km']} km
  Limb angle (from nadir): {geom['limb_angle_deg']:.2f} deg
  Distance to tangent point: {geom['path_to_tangent_km']:.0f} km
  Total atmospheric path: {geom['total_path_km']:.0f} km
""")

    # Geometry vs tangent height
    print("\nGeometry for different tangent heights:")
    print(f"{'Tangent (km)':>15} {'Limb Angle':>15} {'Path to Tang':>15} {'Total Path':>15}")
    print("-" * 65)

    for h_t in [0, 10, 20, 30, 50, 80, 100]:
        g = limb_path_geometry(h_t, args.satellite_alt)
        print(f"{h_t:>15} {g['limb_angle_deg']:>13.2f} deg {g['path_to_tangent_km']:>13.0f} km "
              f"{g['total_path_km']:>13.0f} km")

    # Path through layers
    print("\n" + "-" * 70)
    print("PATH LENGTH THROUGH ATMOSPHERIC LAYERS")
    print("-" * 70)

    layer_bottoms = [0, 10, 20, 30, 40, 50]
    layer_tops = [10, 20, 30, 40, 50, 60]

    print(f"\nPath lengths for tangent height = {args.tangent_height} km:")
    print(f"{'Layer':>15} {'Path Length':>15}")
    print("-" * 35)

    for bot, top in zip(layer_bottoms, layer_tops):
        path = path_length_through_layer(args.tangent_height, bot, top)
        print(f"{bot:>6}-{top:<6} km {path:>13.1f} km")

    # Compare with nadir
    print(f"""
Comparison with nadir viewing:
  - Nadir through 10 km layer: 10 km
  - Limb at 25 km through same layer: ~{path_length_through_layer(25, 20, 30):.0f} km
  - Enhancement factor: ~{path_length_through_layer(25, 20, 30)/10:.0f}x
""")

    # Weighting functions
    print("\n" + "-" * 70)
    print("LIMB WEIGHTING FUNCTIONS")
    print("-" * 70)
    print("""
The weighting function describes the contribution of each altitude
to the limb measurement. It peaks at the tangent height and decreases
above due to decreasing density.
""")

    altitudes = np.arange(0, 80, 2)

    print(f"\nWeighting function for tangent heights:")
    print(f"{'Altitude':>10}", end="")
    for h_t in [10, 20, 30, 50]:
        print(f"{h_t:>10} km", end="")
    print()
    print("-" * 55)

    for alt in [0, 10, 15, 20, 25, 30, 35, 40, 50, 60]:
        print(f"{alt:>10} km", end="")
        for h_t in [10, 20, 30, 50]:
            w = limb_weighting_function(h_t, alt)
            print(f"{w:>12.2f}", end="")
        print()

    # Limb transmission
    print("\n" + "-" * 70)
    print("LIMB TRANSMISSION (RAYLEIGH)")
    print("-" * 70)

    wavelengths = [0.30, 0.40, 0.50, 0.65, 0.80]
    tangent_heights = [10, 20, 30, 40, 50]

    print(f"\nTransmission at different wavelengths and tangent heights:")
    print(f"{'Tangent (km)':>15}", end="")
    for wl in wavelengths:
        print(f"{wl*1000:>10.0f} nm", end="")
    print()
    print("-" * 70)

    for h_t in tangent_heights:
        print(f"{h_t:>15}", end="")
        for wl in wavelengths:
            T = limb_transmission(h_t, wl)
            print(f"{T:>12.3f}", end="")
        print()

    print("""
Notes:
  - Short wavelengths (UV/blue) scatter more - lower transmission
  - Low tangent heights have longer paths - lower transmission
  - Above ~30 km, atmosphere becomes optically thin at all wavelengths
""")

    # Limb radiance profile
    print("\n" + "-" * 70)
    print("LIMB RADIANCE PROFILES")
    print("-" * 70)

    print(f"\nScattered solar radiance vs tangent height (500 nm, SZA=60 deg):")
    print(f"{'Tangent (km)':>15} {'Radiance':>15} {'Description':>30}")
    print("-" * 65)

    for h_t in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
        rad = limb_scattering_radiance(h_t, 0.5, sza_deg=60)
        if h_t < 12:
            desc = "Dense troposphere"
        elif h_t < 25:
            desc = "Ozone layer region"
        elif h_t < 50:
            desc = "Upper stratosphere"
        else:
            desc = "Mesosphere"
        print(f"{h_t:>15} {rad:>15.4f} {desc:>30}")

    # Onion-peeling concept
    print("\n" + "-" * 70)
    print("ONION-PEELING RETRIEVAL")
    print("-" * 70)
    print("""
Onion-peeling is a simple limb retrieval technique:

1. Start from the highest tangent height (topmost layer)
   - Only the top layer contributes
   - Directly retrieve top layer properties

2. Move to next lower tangent height
   - Contribution from above is known
   - Subtract it and retrieve this layer

3. Continue downward layer by layer

Advantages:
  - Simple and intuitive
  - Explicit layer-by-layer solution
  - Good for optically thin atmosphere

Limitations:
  - Error propagation downward
  - Sensitive to noise at high altitudes
  - Assumes horizontal homogeneity
""")

    # Example retrieval
    heights = np.array([15, 20, 25, 30, 35, 40, 45, 50])
    true_ozone = np.array([ozone_profile(h) for h in heights])

    # Simulate measurements (simplified)
    measurements = []
    for h_t in heights:
        # Sum contributions from all layers above
        meas = 0
        for i, h in enumerate(heights):
            if h >= h_t:
                path = path_length_through_layer(h_t, h, h + 5)
                meas += true_ozone[i] * path * 0.01
        measurements.append(meas)

    measurements = np.array(measurements)
    retrieved = onion_peel_retrieval(measurements, heights)

    # Normalize for display
    retrieved = retrieved / np.max(retrieved) if np.max(retrieved) > 0 else retrieved

    print(f"\nExample ozone retrieval:")
    print(f"{'Altitude':>12} {'True O3':>12} {'Retrieved':>12} {'Error':>12}")
    print("-" * 55)

    for i, h in enumerate(heights):
        err = (retrieved[i] - true_ozone[i]) / true_ozone[i] * 100 if true_ozone[i] > 0.01 else 0
        print(f"{h:>12} km {true_ozone[i]:>12.3f} {retrieved[i]:>12.3f} {err:>10.1f}%")

    # Limb instruments
    print("\n" + "-" * 70)
    print("LIMB-VIEWING SATELLITE INSTRUMENTS")
    print("-" * 70)
    print("""
Notable limb-viewing instruments:

SCATTERING (UV-VIS):
  - SCIAMACHY (2002-2012): O3, NO2, BrO, aerosols
  - OSIRIS (2001-present): O3, NO2, aerosols
  - OMPS Limb Profiler (2011-present): O3 profiles

EMISSION (IR):
  - MIPAS (2002-2012): T, O3, H2O, N2O, CH4, CFCs
  - MLS (2004-present): O3, H2O, HNO3, ClO
  - HIRDLS (2004-2008): T, O3, aerosols

SOLAR/STELLAR OCCULTATION:
  - SAGE series (1984-2005): O3, NO2, aerosols
  - ACE-FTS (2003-present): O3, H2O, many trace gases
  - GOMOS (2002-2012): O3, NO2, NO3

Key products:
  - Stratospheric ozone profiles (climate, UV)
  - Temperature profiles
  - Trace gas distributions
  - Aerosol extinction profiles
  - Polar stratospheric clouds
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    g = limb_path_geometry(args.tangent_height, args.satellite_alt)

    print(f"""
Limb viewing analysis for tangent height = {args.tangent_height} km:

Geometry:
  - Total path length: {g['total_path_km']:.0f} km
  - Limb angle: {g['limb_angle_deg']:.1f} deg from nadir
  - Enhancement over nadir: ~{g['total_path_km']/10:.0f}x for 10 km layer

Optical properties at 500 nm:
  - Rayleigh optical depth: {rayleigh_optical_depth_limb(args.tangent_height, 0.5):.2f}
  - Transmission: {limb_transmission(args.tangent_height, 0.5):.3f}

Key characteristics:
  - Excellent vertical resolution (~1-3 km)
  - High sensitivity to stratospheric composition
  - Self-calibrating (space reference)
  - Horizontal averaging over ~400 km
  - Limited by clouds and aerosols at low altitudes

Primary applications:
  - Stratospheric ozone monitoring
  - Climate research (temperature, trace gases)
  - Atmospheric chemistry studies
  - Aerosol profiling
""")

    # Plotting
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Limb Viewing Geometry', fontsize=14, fontweight='bold')

            # Plot 1: Geometry schematic
            ax1 = axes[0, 0]

            # Draw Earth
            theta = np.linspace(0, 2 * np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)

            # Atmosphere edge
            r_atm = 1 + 100 / R_EARTH
            ax1.plot(r_atm * np.cos(theta), r_atm * np.sin(theta), 'c--', alpha=0.5)

            # Satellite position
            r_sat = 1 + args.satellite_alt / R_EARTH
            sat_angle = np.radians(30)  # Position on orbit
            sat_x = r_sat * np.cos(sat_angle)
            sat_y = r_sat * np.sin(sat_angle)
            ax1.plot(sat_x, sat_y, 'ro', markersize=10)
            ax1.annotate('Satellite', (sat_x, sat_y), xytext=(10, 10),
                        textcoords='offset points')

            # Tangent point
            r_tang = 1 + args.tangent_height / R_EARTH
            # Calculate tangent point angle
            geom = limb_path_geometry(args.tangent_height, args.satellite_alt)

            # Limb ray (simplified)
            tang_angle = sat_angle - np.radians(geom['limb_angle_deg'] - 90)
            tang_x = r_tang * np.cos(tang_angle)
            tang_y = r_tang * np.sin(tang_angle)
            ax1.plot([sat_x, tang_x], [sat_y, tang_y], 'r-', linewidth=1.5)
            ax1.plot(tang_x, tang_y, 'g*', markersize=15)
            ax1.annotate('Tangent', (tang_x, tang_y), xytext=(-30, -20),
                        textcoords='offset points')

            ax1.set_xlim(-1.5, 1.5)
            ax1.set_ylim(-1.5, 1.5)
            ax1.set_aspect('equal')
            ax1.set_title(f'Limb Geometry (h_tang = {args.tangent_height} km)')
            ax1.axis('off')

            # Plot 2: Path length vs tangent height
            ax2 = axes[0, 1]
            h_range = np.linspace(5, 80, 50)
            paths = [limb_path_geometry(h, args.satellite_alt)['total_path_km'] for h in h_range]

            ax2.semilogy(h_range, paths, 'b-', linewidth=2)
            ax2.axhline(args.satellite_alt, color='gray', linestyle='--', alpha=0.5,
                       label='Satellite alt')
            ax2.axvline(args.tangent_height, color='red', linestyle='--',
                       label=f'h_t = {args.tangent_height} km')
            ax2.set_xlabel('Tangent Height (km)')
            ax2.set_ylabel('Total Path Length (km)')
            ax2.set_title('Atmospheric Path Length')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Weighting functions
            ax3 = axes[1, 0]
            alt = np.linspace(0, 80, 100)

            for h_t in [15, 25, 35, 50]:
                w = [limb_weighting_function(h_t, a) for a in alt]
                ax3.plot(w, alt, linewidth=2, label=f'h_t = {h_t} km')

            ax3.set_xlabel('Weighting Function')
            ax3.set_ylabel('Altitude (km)')
            ax3.set_title('Limb Weighting Functions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 1.1)

            # Plot 4: Limb radiance profiles
            ax4 = axes[1, 1]
            h_range = np.linspace(5, 70, 50)

            for wl in [0.3, 0.5, 0.8]:
                rad = [limb_scattering_radiance(h, wl) for h in h_range]
                ax4.plot(rad, h_range, linewidth=2, label=f'{wl*1000:.0f} nm')

            ax4.set_xlabel('Scattered Radiance (relative)')
            ax4.set_ylabel('Tangent Height (km)')
            ax4.set_title('Limb Scattering Profiles')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {args.output}")

        except ImportError:
            print("\nNote: matplotlib not available")


if __name__ == "__main__":
    main()
