"""
Optical beam propagation through atmospheric turbulence.

This module provides functions for calculating key parameters that describe
the effects of atmospheric turbulence on optical wave propagation:

- Fried parameter (r0): Atmospheric coherence length
- Scintillation index: Intensity fluctuation variance
- Rytov variance: Measure of turbulence strength for propagation
- Isoplanatic angle: Angular coherence
- Beam wander: Random beam displacement

Theory
------
Most calculations are based on:
1. Kolmogorov turbulence theory (5/3 power spectrum)
2. Rytov approximation (weak fluctuation regime)
3. Andrews-Phillips extended theory (moderate-strong fluctuations)

References
----------
- Andrews, L.C. & Phillips, R.L. (2005). Laser Beam Propagation through
  Random Media. SPIE Press.
- Fried, D.L. (1966). Optical resolution through a randomly inhomogeneous
  medium for very long and very short exposures. JOSA, 56(10), 1372-1379.
- Tatarskii, V.I. (1971). The Effects of the Turbulent Atmosphere on
  Wave Propagation.
"""

import numpy as np


def fried_parameter(wavelength_m, cn2_integrated, zenith_angle_deg=0):
    """
    Calculate Fried parameter (atmospheric coherence length).

    The Fried parameter r0 is the diameter of a circular aperture that
    collects the same amount of coherent flux as an infinitely large
    aperture in the presence of turbulence.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3) from integrated_cn2()
    zenith_angle_deg : float, optional
        Zenith angle in degrees (default: 0)

    Returns
    -------
    r0 : float
        Fried parameter in meters

    Notes
    -----
    The Fried parameter is defined as:

        r0 = (0.423 * k^2 * sec(z) * integral(Cn2(h) dh))^(-3/5)

    where k = 2*pi/wavelength.

    Typical values:
    - Good seeing (astronomical): r0 ~ 20-30 cm at 500 nm
    - Average seeing: r0 ~ 10-15 cm at 500 nm
    - Poor seeing: r0 ~ 5 cm at 500 nm

    Wavelength scaling: r0 ~ wavelength^(6/5)

    Examples
    --------
    >>> # For cn2_integrated = 1e-13 m^(1/3), wavelength 500 nm
    >>> r0 = fried_parameter(0.5e-6, 1e-13)
    >>> print(f"Fried parameter: {r0*100:.1f} cm")
    """
    k = 2 * np.pi / wavelength_m
    sec_z = 1.0 / np.cos(np.radians(zenith_angle_deg))

    # Fried's formula
    r0 = (0.423 * k**2 * sec_z * cn2_integrated)**(-3/5)

    return r0


def isoplanatic_angle(wavelength_m, cn2_profile, altitude_m, zenith_angle_deg=0):
    """
    Calculate isoplanatic angle.

    The isoplanatic angle theta0 is the angular separation within which
    the wavefront distortion is correlated. Beyond this angle, adaptive
    optics correction degrades.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_profile : array_like
        Cn2 values at each altitude level
    altitude_m : array_like
        Altitude levels in meters
    zenith_angle_deg : float, optional
        Zenith angle in degrees (default: 0)

    Returns
    -------
    theta0 : float
        Isoplanatic angle in radians

    Notes
    -----
    The isoplanatic angle is:

        theta0 = (2.91 * k^2 * sec(z)^(8/3) * integral(Cn2(h) * h^(5/3) dh))^(-3/5)

    Typical values at 500 nm:
    - Good conditions: theta0 ~ 5-10 arcsec
    - Average: theta0 ~ 2-3 arcsec
    """
    k = 2 * np.pi / wavelength_m
    sec_z = 1.0 / np.cos(np.radians(zenith_angle_deg))

    h = np.asarray(altitude_m)
    cn2 = np.asarray(cn2_profile)

    # Weighted integral
    dh = np.diff(h)
    cn2_mid = 0.5 * (cn2[:-1] + cn2[1:])
    h_mid = 0.5 * (h[:-1] + h[1:])

    integral = np.sum(cn2_mid * h_mid**(5/3) * dh)

    theta0 = (2.91 * k**2 * sec_z**(8/3) * integral)**(-3/5)

    return theta0


def rytov_variance(wavelength_m, cn2_integrated, path_length_m):
    """
    Calculate Rytov variance for plane wave.

    The Rytov variance sigma_R^2 is a measure of turbulence strength
    that determines the fluctuation regime (weak, moderate, strong).

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-averaged Cn2 in m^(-2/3)
    path_length_m : float
        Propagation path length in meters

    Returns
    -------
    sigma_r2 : float
        Rytov variance (dimensionless)

    Notes
    -----
    Rytov variance for plane wave:

        sigma_R^2 = 1.23 * Cn2 * k^(7/6) * L^(11/6)

    Fluctuation regimes:
    - Weak: sigma_R^2 < 0.3
    - Moderate: 0.3 < sigma_R^2 < 5
    - Strong (saturation): sigma_R^2 > 5
    """
    k = 2 * np.pi / wavelength_m
    L = path_length_m

    sigma_r2 = 1.23 * cn2_integrated * k**(7/6) * L**(11/6)

    return sigma_r2


def scintillation_index(wavelength_m, cn2_integrated, path_length_m,
                        aperture_diameter_m=None):
    """
    Calculate scintillation index (normalized intensity variance).

    The scintillation index sigma_I^2 is the normalized variance of
    intensity fluctuations: sigma_I^2 = <I^2>/<I>^2 - 1

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-averaged Cn2 in m^(-2/3)
    path_length_m : float
        Propagation path length in meters
    aperture_diameter_m : float, optional
        Receiving aperture diameter in meters
        If specified, aperture averaging is applied

    Returns
    -------
    sigma_i2 : float
        Scintillation index (dimensionless)

    Notes
    -----
    For a point receiver in weak fluctuation regime:

        sigma_I^2 = exp(4 * sigma_chi^2) - 1 ~ 4 * sigma_chi^2

    where sigma_chi^2 is the log-amplitude variance.

    In strong fluctuation (saturation), sigma_I^2 -> 1.

    Aperture averaging reduces scintillation for apertures larger
    than the Fresnel zone sqrt(wavelength * L).

    Examples
    --------
    >>> # 1.55 um laser over 10 km horizontal path
    >>> si = scintillation_index(1.55e-6, 1e-15, 10000)
    >>> print(f"Scintillation index: {si:.3f}")
    """
    # Rytov variance (plane wave)
    sigma_r2 = rytov_variance(wavelength_m, cn2_integrated, path_length_m)

    # Weak fluctuation (Rytov) approximation
    if sigma_r2 < 0.3:
        sigma_i2 = np.exp(4 * 0.307 * sigma_r2) - 1
    else:
        # Strong fluctuation regime - use Andrews-Phillips model
        # Approaches saturation value of 1
        sigma_i2 = 1.0 - np.exp(-0.49 * sigma_r2 / (1 + 0.65 * sigma_r2 +
                                                     1.11 * sigma_r2**(6/5)))

    # Aperture averaging
    if aperture_diameter_m is not None and aperture_diameter_m > 0:
        # Fresnel zone size
        fresnel_zone = np.sqrt(wavelength_m * path_length_m)

        # Aperture averaging factor
        if aperture_diameter_m > fresnel_zone:
            A_factor = (fresnel_zone / aperture_diameter_m)**2
        else:
            A_factor = 1.0

        sigma_i2 *= A_factor

    return sigma_i2


def greenwood_frequency(wavelength_m, cn2_profile, altitude_m, wind_profile_ms,
                        zenith_angle_deg=0):
    """
    Calculate Greenwood frequency for adaptive optics.

    The Greenwood frequency f_G determines the required bandwidth for
    adaptive optics correction of atmospheric turbulence.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_profile : array_like
        Cn2 values at each altitude level
    altitude_m : array_like
        Altitude levels in meters
    wind_profile_ms : array_like
        Wind speed at each altitude in m/s
    zenith_angle_deg : float, optional
        Zenith angle in degrees (default: 0)

    Returns
    -------
    f_g : float
        Greenwood frequency in Hz

    Notes
    -----
    The Greenwood frequency is:

        f_G = (0.102 * k^2 * sec(z) * integral(Cn2(h) * V(h)^(5/3) dh))^(3/5)

    Typical values: 10-100 Hz for ground-based astronomy
    """
    k = 2 * np.pi / wavelength_m
    sec_z = 1.0 / np.cos(np.radians(zenith_angle_deg))

    h = np.asarray(altitude_m)
    cn2 = np.asarray(cn2_profile)
    v = np.asarray(wind_profile_ms)

    # Weighted integral
    dh = np.diff(h)
    cn2_mid = 0.5 * (cn2[:-1] + cn2[1:])
    v_mid = 0.5 * (v[:-1] + v[1:])

    integral = np.sum(cn2_mid * v_mid**(5/3) * dh)

    f_g = (0.102 * k**2 * sec_z * integral)**(3/5)

    return f_g


def beam_wander_variance(wavelength_m, cn2_integrated, path_length_m,
                         beam_diameter_m):
    """
    Calculate beam wander variance for Gaussian beam.

    Beam wander is the random displacement of the beam centroid caused
    by large-scale turbulence eddies.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-averaged Cn2 in m^(-2/3)
    path_length_m : float
        Propagation path length in meters
    beam_diameter_m : float
        Initial beam diameter (1/e^2 intensity) in meters

    Returns
    -------
    sigma_bw2 : float
        Beam wander variance in m^2 (RMS displacement = sqrt(sigma_bw2))

    Notes
    -----
    For a collimated Gaussian beam:

        <r_c^2> = 0.54 * (L/k*W0)^2 * (2*W0/r0)^(5/3) * (W0/L)^(1/3)

    where W0 is beam radius at waist, k is wavenumber, L is path length.
    """
    k = 2 * np.pi / wavelength_m
    L = path_length_m
    W0 = beam_diameter_m / 2  # Radius

    # Fried parameter from integrated Cn2
    r0 = (0.423 * k**2 * cn2_integrated * L)**(-3/5)

    # Beam wander variance (collimated beam)
    if r0 > 0 and W0 > 0:
        sigma_bw2 = 0.54 * (L / (k * W0))**2 * (2 * W0 / r0)**(5/3) * (W0 / L)**(1/3)
    else:
        sigma_bw2 = 0.0

    return sigma_bw2


def log_amplitude_variance(wavelength_m, cn2_integrated, path_length_m):
    """
    Calculate log-amplitude variance (chi variance).

    The log-amplitude variance sigma_chi^2 describes fluctuations in
    the log of the field amplitude, related to scintillation.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-averaged Cn2 in m^(-2/3)
    path_length_m : float
        Propagation path length in meters

    Returns
    -------
    sigma_chi2 : float
        Log-amplitude variance (dimensionless)

    Notes
    -----
    For plane wave in weak fluctuation:

        sigma_chi^2 = 0.307 * sigma_R^2

    where sigma_R^2 is the Rytov variance.

    The intensity scintillation index relates to chi:

        sigma_I^2 ~ 4 * sigma_chi^2 (weak fluctuations)
    """
    sigma_r2 = rytov_variance(wavelength_m, cn2_integrated, path_length_m)

    # Weak fluctuation regime
    sigma_chi2 = 0.307 * sigma_r2

    return sigma_chi2


def coherence_time(wavelength_m, cn2_integrated, wind_speed_ms):
    """
    Calculate atmospheric coherence time (Greenwood time).

    The coherence time tau0 is the timescale over which the wavefront
    remains correlated - determines required AO update rate.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    wind_speed_ms : float
        Effective wind speed in m/s (often ~10-20 m/s)

    Returns
    -------
    tau0 : float
        Coherence time in seconds

    Notes
    -----
    tau0 ~ r0 / V_wind

    Typical values: 1-10 ms for visible wavelengths
    """
    k = 2 * np.pi / wavelength_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Coherence time
    tau0 = 0.314 * r0 / wind_speed_ms

    return tau0


def strehl_ratio(wavelength_m, rms_wavefront_error_m=None, r0=None,
                 aperture_diameter_m=None):
    """
    Calculate Strehl ratio from wavefront error or turbulence.

    The Strehl ratio is the ratio of peak intensity of an aberrated
    image to that of a diffraction-limited image.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    rms_wavefront_error_m : float, optional
        RMS wavefront error in meters
    r0 : float, optional
        Fried parameter in meters (if no wavefront error specified)
    aperture_diameter_m : float, optional
        Aperture diameter in meters (required if using r0)

    Returns
    -------
    S : float
        Strehl ratio (0 to 1, where 1 is diffraction-limited)

    Notes
    -----
    Marechal approximation (small aberrations):
        S ~ exp(-(2*pi*sigma/lambda)^2)

    For turbulence-limited:
        S ~ (r0/D)^2  for D >> r0
        S ~ 1 - (D/r0)^(5/3)  for D << r0

    Examples
    --------
    >>> # From wavefront error
    >>> S = strehl_ratio(0.5e-6, rms_wavefront_error_m=50e-9)
    >>> print(f"Strehl ratio: {S:.3f}")
    """
    if rms_wavefront_error_m is not None:
        # Marechal approximation
        phase_variance = (2 * np.pi * rms_wavefront_error_m / wavelength_m)**2
        S = np.exp(-phase_variance)
    elif r0 is not None and aperture_diameter_m is not None:
        D = aperture_diameter_m
        if D <= r0:
            # Small aperture
            S = 1.0 - (D / r0)**(5/3)
        else:
            # Large aperture (turbulence-limited)
            S = (r0 / D)**2
    else:
        raise ValueError("Must specify either rms_wavefront_error_m or both r0 and aperture_diameter_m")

    return np.clip(S, 0.0, 1.0)
