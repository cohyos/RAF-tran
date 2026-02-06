"""
Cn2 (refractive index structure constant) profile models.

This module provides various models for the vertical profile of Cn2,
the refractive index structure constant that characterizes optical
turbulence strength in the atmosphere.

Typical Cn2 values:
- Strong turbulence (near surface, midday): 10^-13 m^(-2/3)
- Moderate turbulence: 10^-15 m^(-2/3)
- Weak turbulence (high altitude): 10^-17 m^(-2/3)

References
----------
- Hufnagel, R.E. (1974). Variations of atmospheric turbulence.
- Valley, G.C. (1980). Isoplanatic degradation of tilt correction.
- Miller, M.G. & Zieske, P.L. (1979). Turbulence environment
  characterization. RADC-TR-79-131.
"""

import numpy as np


def hufnagel_valley_cn2(altitude_m, wind_speed_rms=21.0, cn2_ground=1.7e-14):
    """
    Hufnagel-Valley 5/7 model for Cn2 vertical profile.

    The HV 5/7 model is widely used for astronomical site characterization.
    The "5/7" refers to achieving 5 cm Fried parameter and 7 urad isoplanatic
    angle at 0.5 um wavelength for vertical path.

    Parameters
    ----------
    altitude_m : float or array_like
        Altitude above ground in meters
    wind_speed_rms : float, optional
        RMS high-altitude wind speed in m/s (default: 21 m/s)
        Controls tropopause turbulence peak
    cn2_ground : float, optional
        Cn2 at ground level in m^(-2/3) (default: 1.7e-14)
        Controls boundary layer turbulence

    Returns
    -------
    cn2 : float or ndarray
        Refractive index structure constant in m^(-2/3)

    Notes
    -----
    The HV model has three components:
    1. Boundary layer: Exponential decay from surface
    2. Tropopause: Peak at ~10-15 km altitude
    3. Upper atmosphere: Gradual decrease

    The formula is:
        Cn2(h) = 0.00594 * (W/27)^2 * (10^-5 * h)^10 * exp(-h/1000)
               + 2.7e-16 * exp(-h/1500)
               + A * exp(-h/100)

    where h is altitude in meters, W is wind speed, A is ground term.

    Examples
    --------
    >>> cn2 = hufnagel_valley_cn2(1000)  # At 1 km altitude
    >>> print(f"Cn2 = {cn2:.2e} m^(-2/3)")
    """
    h = np.asarray(altitude_m, dtype=float)
    h_km = h / 1000.0

    # Tropopause/jet stream term (high altitude turbulence)
    term1 = 0.00594 * (wind_speed_rms / 27.0)**2 * \
            (1e-5 * h)**10 * np.exp(-h / 1000.0)

    # Free atmosphere term
    term2 = 2.7e-16 * np.exp(-h / 1500.0)

    # Boundary layer term
    term3 = cn2_ground * np.exp(-h / 100.0)

    return term1 + term2 + term3


def slc_day_cn2(altitude_m):
    """
    SLC (Submarine Laser Communication) daytime Cn2 model.

    Empirical model for daytime conditions with convective boundary layer.

    Parameters
    ----------
    altitude_m : float or array_like
        Altitude above ground in meters

    Returns
    -------
    cn2 : float or ndarray
        Refractive index structure constant in m^(-2/3)

    Notes
    -----
    Daytime model reflects enhanced turbulence in the convective
    boundary layer (typically 0-2 km during day).
    """
    h = np.asarray(altitude_m, dtype=float)

    # Piecewise model based on altitude
    cn2 = np.zeros_like(h)

    # Surface layer (0-19 m) - avoid h=0 singularity
    mask1 = (h > 0) & (h <= 19)
    cn2[mask1] = 4.008e-13 * h[mask1]**(-1.054)
    # At h=0, use value at h=1m
    mask0 = h == 0
    if np.any(mask0):
        cn2[mask0] = 4.008e-13 * 1.0**(-1.054)

    # Convective boundary layer (19 m - 230 m)
    mask2 = (h > 19) & (h <= 230)
    cn2[mask2] = 1.300e-15

    # Mixed layer (230 m - 850 m)
    mask3 = (h > 230) & (h <= 850)
    cn2[mask3] = 6.352e-7 * h[mask3]**(-2.966)

    # Free atmosphere (850 m - 7000 m)
    mask4 = (h > 850) & (h <= 7000)
    cn2[mask4] = 6.209e-16 * h[mask4]**(-0.6229)

    # Upper atmosphere (> 7000 m)
    mask5 = h > 7000
    cn2[mask5] = 1.0e-17 * np.exp(-(h[mask5] - 7000) / 5000)

    # Handle scalar input
    if cn2.ndim == 0:
        return float(cn2)
    return cn2


def slc_night_cn2(altitude_m):
    """
    SLC (Submarine Laser Communication) nighttime Cn2 model.

    Empirical model for nighttime conditions with stable boundary layer.

    Parameters
    ----------
    altitude_m : float or array_like
        Altitude above ground in meters

    Returns
    -------
    cn2 : float or ndarray
        Refractive index structure constant in m^(-2/3)

    Notes
    -----
    Nighttime model reflects reduced turbulence due to stable
    stratification in the nocturnal boundary layer.
    """
    h = np.asarray(altitude_m, dtype=float)

    # Piecewise model based on altitude
    cn2 = np.zeros_like(h)

    # Surface layer (0-19 m) - weaker than daytime, avoid h=0 singularity
    mask1 = (h > 0) & (h <= 19)
    cn2[mask1] = 8.40e-14 * h[mask1]**(-1.054)
    # At h=0, use value at h=1m
    mask0 = h == 0
    if np.any(mask0):
        cn2[mask0] = 8.40e-14 * 1.0**(-1.054)

    # Stable boundary layer (19 m - 230 m)
    mask2 = (h > 19) & (h <= 230)
    cn2[mask2] = 2.87e-16

    # Residual layer (230 m - 850 m)
    mask3 = (h > 230) & (h <= 850)
    cn2[mask3] = 2.5e-16 * h[mask3]**(-0.5)

    # Free atmosphere (850 m - 7000 m)
    mask4 = (h > 850) & (h <= 7000)
    cn2[mask4] = 3.0e-16 * h[mask4]**(-0.5)

    # Upper atmosphere (> 7000 m)
    mask5 = h > 7000
    cn2[mask5] = 5.0e-18 * np.exp(-(h[mask5] - 7000) / 5000)

    # Handle scalar input
    if cn2.ndim == 0:
        return float(cn2)
    return cn2


def cn2_from_weather(altitude_m, temperature_k, wind_speed_ms,
                     humidity_percent=50.0, solar_elevation_deg=45.0):
    """
    Estimate Cn2 from meteorological parameters.

    Simple parametric model relating Cn2 to local weather conditions.
    Useful when standard profiles don't match local conditions.

    Parameters
    ----------
    altitude_m : float or array_like
        Altitude above ground in meters
    temperature_k : float or array_like
        Air temperature in Kelvin
    wind_speed_ms : float or array_like
        Wind speed in m/s
    humidity_percent : float, optional
        Relative humidity in percent (default: 50%)
    solar_elevation_deg : float, optional
        Solar elevation angle in degrees (default: 45 deg)
        Affects convective turbulence strength

    Returns
    -------
    cn2 : float or ndarray
        Estimated Cn2 in m^(-2/3)

    Notes
    -----
    This is a simplified empirical model. For accurate predictions,
    use direct measurements or more sophisticated turbulence models.

    The model accounts for:
    - Temperature affects refractive index fluctuations
    - Wind shear generates mechanical turbulence
    - Solar heating drives convective turbulence
    - Humidity affects thermal stratification
    """
    h = np.asarray(altitude_m, dtype=float)
    T = np.asarray(temperature_k, dtype=float)
    v = np.asarray(wind_speed_ms, dtype=float)

    # Pressure from standard atmosphere (simplified)
    P = 101325 * np.exp(-h / 8500)  # Pa

    # Base Cn2 from Tatarskii formula: Cn2 ~ (P/T^2)^2 * CT^2
    # where CT^2 is temperature structure constant
    base_cn2 = 7.9e-13 * (P / T**2)**2

    # Scale by altitude (decrease with height)
    altitude_factor = np.exp(-h / 2000)

    # Solar heating factor (more turbulence during day)
    solar_factor = 1.0 + 2.0 * np.sin(np.radians(max(0, solar_elevation_deg)))

    # Wind shear contribution (mechanical turbulence)
    wind_factor = 1.0 + 0.1 * v

    # Humidity damping (high humidity = more stable)
    humidity_factor = 1.0 - 0.005 * humidity_percent

    cn2 = base_cn2 * altitude_factor * solar_factor * wind_factor * humidity_factor

    # Apply reasonable bounds
    cn2 = np.clip(cn2, 1e-18, 1e-12)

    return cn2


def integrated_cn2(altitude_m, cn2_profile, zenith_angle_deg=0):
    """
    Calculate path-integrated Cn2 along slant path.

    Parameters
    ----------
    altitude_m : array_like
        Altitude levels in meters (must be sorted ascending)
    cn2_profile : array_like
        Cn2 values at each altitude level
    zenith_angle_deg : float, optional
        Zenith angle of propagation path in degrees (default: 0 for vertical)

    Returns
    -------
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)

    Notes
    -----
    For vertical path: integral of Cn2(h) dh
    For slant path: integral of Cn2(h) sec(z) dh

    This integrated value is used to compute Fried parameter and other
    atmospheric parameters.
    """
    h = np.asarray(altitude_m)
    cn2 = np.asarray(cn2_profile)

    # Slant path factor
    sec_z = 1.0 / np.cos(np.radians(zenith_angle_deg))

    # Trapezoidal integration
    dh = np.diff(h)
    cn2_mid = 0.5 * (cn2[:-1] + cn2[1:])

    return sec_z * np.sum(cn2_mid * dh)
