"""
Adaptive Optics Simulation Module
=================================

This module provides comprehensive adaptive optics (AO) simulation capabilities
for modeling wavefront sensing, correction, and system performance in turbulent
atmospheres.

Applications
------------
- Ground-based telescope AO systems
- Laser communication terminals
- Free-space optical links
- Directed energy systems
- Remote sensing with AO correction

Key Components
--------------
- Wavefront sensor models (Shack-Hartmann, pyramid, curvature)
- Deformable mirror models (continuous, segmented)
- Control loop dynamics and servo lag
- Fitting error and temporal error analysis
- Multi-conjugate AO (MCAO) basics

References
----------
- Hardy, J.W. (1998). Adaptive Optics for Astronomical Telescopes. Oxford.
- Roddier, F. (1999). Adaptive Optics in Astronomy. Cambridge.
- Tyson, R.K. (2015). Principles of Adaptive Optics. CRC Press.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class AOSystemConfig:
    """
    Configuration for an adaptive optics system.

    Attributes
    ----------
    aperture_diameter : float
        Telescope/system aperture diameter in meters
    wavelength : float
        Operating wavelength in meters
    n_actuators : int
        Number of deformable mirror actuators across diameter
    wfs_type : str
        Wavefront sensor type: 'shack_hartmann', 'pyramid', 'curvature'
    loop_frequency : float
        Control loop update frequency in Hz
    loop_gain : float
        Feedback loop gain (typically 0.3-0.7)
    wfs_noise : float
        WFS measurement noise in rad RMS
    dm_stroke : float
        Maximum DM stroke in micrometers
    """
    aperture_diameter: float
    wavelength: float
    n_actuators: int = 20
    wfs_type: str = 'shack_hartmann'
    loop_frequency: float = 1000.0
    loop_gain: float = 0.5
    wfs_noise: float = 0.1
    dm_stroke: float = 5.0


@dataclass
class AOPerformance:
    """
    Results from AO performance analysis.

    Attributes
    ----------
    strehl_ratio : float
        Expected Strehl ratio (0-1)
    residual_variance : float
        Total residual phase variance in rad^2
    fitting_error : float
        Fitting error variance in rad^2
    temporal_error : float
        Temporal (servo lag) error variance in rad^2
    noise_error : float
        WFS noise propagated error in rad^2
    anisoplanatism_error : float
        Angular anisoplanatism error in rad^2
    n_corrected_modes : int
        Effective number of corrected modes
    """
    strehl_ratio: float
    residual_variance: float
    fitting_error: float
    temporal_error: float
    noise_error: float
    anisoplanatism_error: float = 0.0
    n_corrected_modes: int = 0


def fitting_error(aperture_diameter: float, r0: float, d_actuator: float) -> float:
    """
    Calculate deformable mirror fitting error.

    The fitting error arises from the finite spatial sampling of the
    deformable mirror, which cannot correct high-order aberrations.

    Parameters
    ----------
    aperture_diameter : float
        Aperture diameter in meters
    r0 : float
        Fried parameter in meters
    d_actuator : float
        Inter-actuator spacing in meters

    Returns
    -------
    sigma_fit2 : float
        Fitting error variance in rad^2

    Notes
    -----
    For continuous-facesheet DM:

        sigma_fit^2 = 0.28 * (d/r0)^(5/3)

    where d is the inter-actuator spacing.

    For segmented DM (piston-only):

        sigma_fit^2 = 0.134 * (d/r0)^(5/3)
    """
    # Continuous facesheet DM fitting coefficient
    kappa_fit = 0.28

    sigma_fit2 = kappa_fit * (d_actuator / r0)**(5/3)

    return sigma_fit2


def temporal_error(greenwood_freq: float, loop_bandwidth: float,
                   r0: float, aperture_diameter: float) -> float:
    """
    Calculate servo lag (temporal) error.

    Temporal error arises from the delay between wavefront sensing
    and correction due to finite loop bandwidth.

    Parameters
    ----------
    greenwood_freq : float
        Greenwood frequency in Hz
    loop_bandwidth : float
        Closed-loop control bandwidth in Hz
    r0 : float
        Fried parameter in meters
    aperture_diameter : float
        Aperture diameter in meters

    Returns
    -------
    sigma_temp2 : float
        Temporal error variance in rad^2

    Notes
    -----
    For a simple integrator servo:

        sigma_temp^2 = (f_G / f_3dB)^(5/3) * (D/r0)^(5/3)

    where f_G is Greenwood frequency and f_3dB is 3dB bandwidth.

    Typical ratio: f_3dB ~ 0.1 * f_loop (loop frequency)
    """
    if loop_bandwidth <= 0:
        return np.inf

    # Temporal error coefficient
    sigma_temp2 = (greenwood_freq / loop_bandwidth)**(5/3) * (aperture_diameter / r0)**(5/3)

    return sigma_temp2


def wfs_noise_propagation(wfs_noise_rad: float, n_actuators: int,
                          loop_gain: float) -> float:
    """
    Calculate WFS noise propagation to residual error.

    Wavefront sensor noise propagates through the control loop,
    contributing to residual phase error.

    Parameters
    ----------
    wfs_noise_rad : float
        WFS measurement noise in rad RMS (per subaperture)
    n_actuators : int
        Number of actuators across diameter
    loop_gain : float
        Control loop gain

    Returns
    -------
    sigma_noise2 : float
        Noise propagation error variance in rad^2

    Notes
    -----
    For a Shack-Hartmann WFS with modal reconstruction:

        sigma_noise^2 ~ n_act * sigma_wfs^2 * noise_propagation_coeff

    The noise propagation coefficient depends on the reconstruction
    algorithm (typically 0.2-0.5 for least-squares).
    """
    # Noise propagation coefficient for least-squares reconstruction
    # Approximately 0.3 for well-conditioned system
    np_coeff = 0.3

    # Noise variance scales with number of modes fitted
    n_modes = n_actuators**2

    # Closed-loop noise rejection: ~gain/(2-gain) for type-1 servo
    cl_factor = loop_gain / (2 - loop_gain) if loop_gain < 2 else 1.0

    sigma_noise2 = np_coeff * wfs_noise_rad**2 * cl_factor

    return sigma_noise2


def angular_anisoplanatism(theta_rad: float, wavelength: float,
                           cn2_profile: np.ndarray,
                           altitude_profile: np.ndarray) -> float:
    """
    Calculate angular anisoplanatism error.

    Angular anisoplanatism occurs when the guide star and science
    target are separated in angle, so their wavefronts sample different
    turbulence volumes.

    Parameters
    ----------
    theta_rad : float
        Angular separation between guide star and target in radians
    wavelength : float
        Wavelength in meters
    cn2_profile : ndarray
        Cn2 values at each altitude in m^(-2/3)
    altitude_profile : ndarray
        Altitudes in meters

    Returns
    -------
    sigma_aniso2 : float
        Anisoplanatism error variance in rad^2

    Notes
    -----
    The anisoplanatism error depends on the isoplanatic angle theta_0:

        sigma_aniso^2 = (theta / theta_0)^(5/3)

    where theta_0 = (2.91 * k^2 * integral(Cn2 * h^(5/3) dh))^(-3/5)

    This assumes a single conjugate AO system (ground-layer correction).
    """
    k = 2 * np.pi / wavelength

    # Compute isoplanatic angle integral
    # theta_0 = (2.91 * k^2 * integral(Cn2 * h^(5/3) dh))^(-3/5)

    cn2 = np.asarray(cn2_profile)
    h = np.asarray(altitude_profile)

    # Use trapezoidal integration
    integrand = cn2 * h**(5/3)
    integral = np.trapezoid(integrand, h)

    theta_0 = (2.91 * k**2 * integral)**(-3/5)

    # Anisoplanatism error
    sigma_aniso2 = (theta_rad / theta_0)**(5/3)

    return sigma_aniso2


def focus_anisoplanatism(altitude_focus: float, altitude_turbulence: float,
                         aperture_diameter: float, r0: float) -> float:
    """
    Calculate focus anisoplanatism (cone effect) for laser guide stars.

    When using a laser guide star at finite altitude, the cone of light
    does not sample the same turbulence volume as a parallel beam from
    a celestial source.

    Parameters
    ----------
    altitude_focus : float
        Laser guide star altitude in meters
    altitude_turbulence : float
        Effective turbulence altitude in meters
    aperture_diameter : float
        Aperture diameter in meters
    r0 : float
        Fried parameter in meters

    Returns
    -------
    sigma_cone2 : float
        Focus anisoplanatism error variance in rad^2

    Notes
    -----
    The cone effect error increases with:
    - Higher turbulence altitude
    - Lower LGS altitude
    - Larger aperture

    For h_turb << h_LGS:
        sigma_cone^2 ~ (D * h_turb / (h_LGS * r0))^(5/3)
    """
    if altitude_focus <= altitude_turbulence:
        return np.inf

    # Geometric factor
    d_cone = aperture_diameter * altitude_turbulence / altitude_focus

    # Focus anisoplanatism
    sigma_cone2 = (d_cone / r0)**(5/3)

    return sigma_cone2


def compute_ao_performance(
    config: AOSystemConfig,
    r0: float,
    greenwood_freq: float,
    theta_target: float = 0.0,
    cn2_profile: Optional[np.ndarray] = None,
    altitude_profile: Optional[np.ndarray] = None,
) -> AOPerformance:
    """
    Compute comprehensive AO system performance.

    Parameters
    ----------
    config : AOSystemConfig
        AO system configuration
    r0 : float
        Fried parameter at operating wavelength in meters
    greenwood_freq : float
        Greenwood frequency in Hz
    theta_target : float, optional
        Off-axis angle to target in radians (default: 0, on-axis)
    cn2_profile : ndarray, optional
        Cn2 profile for anisoplanatism calculation
    altitude_profile : ndarray, optional
        Altitude profile in meters

    Returns
    -------
    performance : AOPerformance
        AO system performance metrics
    """
    D = config.aperture_diameter

    # Inter-actuator spacing
    d_act = D / config.n_actuators

    # Number of corrected modes (approximately)
    n_modes = int(np.pi * (config.n_actuators / 2)**2)

    # 1. Fitting error
    sigma_fit2 = fitting_error(D, r0, d_act)

    # 2. Temporal error
    # 3dB bandwidth approximately 0.1 * loop frequency for integrator
    f_3db = 0.1 * config.loop_frequency * config.loop_gain
    sigma_temp2 = temporal_error(greenwood_freq, f_3db, r0, D)

    # 3. WFS noise propagation
    sigma_noise2 = wfs_noise_propagation(
        config.wfs_noise, config.n_actuators, config.loop_gain
    )

    # 4. Anisoplanatism (if off-axis)
    sigma_aniso2 = 0.0
    if theta_target > 0 and cn2_profile is not None and altitude_profile is not None:
        sigma_aniso2 = angular_anisoplanatism(
            theta_target, config.wavelength, cn2_profile, altitude_profile
        )

    # Total residual variance
    sigma_total2 = sigma_fit2 + sigma_temp2 + sigma_noise2 + sigma_aniso2

    # Strehl ratio (extended Marechal approximation)
    # Valid for sigma^2 < ~2 rad^2
    if sigma_total2 < 4:
        strehl = np.exp(-sigma_total2)
    else:
        # Large aberration regime
        strehl = (r0 / D)**2

    return AOPerformance(
        strehl_ratio=np.clip(strehl, 0, 1),
        residual_variance=sigma_total2,
        fitting_error=sigma_fit2,
        temporal_error=sigma_temp2,
        noise_error=sigma_noise2,
        anisoplanatism_error=sigma_aniso2,
        n_corrected_modes=n_modes,
    )


def optimal_actuator_count(aperture_diameter: float, r0: float,
                           target_strehl: float = 0.8) -> int:
    """
    Estimate optimal number of actuators for target Strehl ratio.

    Parameters
    ----------
    aperture_diameter : float
        Aperture diameter in meters
    r0 : float
        Fried parameter in meters
    target_strehl : float
        Target Strehl ratio (default: 0.8)

    Returns
    -------
    n_actuators : int
        Recommended number of actuators across diameter
    """
    D = aperture_diameter

    # Target residual variance from Strehl
    sigma_target2 = -np.log(target_strehl)

    # Assume fitting error dominates, solve for d_act
    # sigma^2 = 0.28 * (d/r0)^(5/3) = sigma_target^2
    d_act = r0 * (sigma_target2 / 0.28)**(3/5)

    # Number of actuators
    n_act = max(int(np.ceil(D / d_act)), 4)

    return n_act


def zernike_temporal_psd(j: int, wavelength: float, wind_speed: float,
                         r0: float, frequencies: np.ndarray) -> np.ndarray:
    """
    Temporal power spectral density of Zernike coefficients.

    The PSD describes the temporal behavior of each Zernike mode
    due to wind-blown turbulence (Taylor frozen flow hypothesis).

    Parameters
    ----------
    j : int
        Zernike mode index (Noll ordering)
    wavelength : float
        Wavelength in meters
    wind_speed : float
        Effective wind speed in m/s
    r0 : float
        Fried parameter in meters
    frequencies : ndarray
        Temporal frequencies in Hz

    Returns
    -------
    psd : ndarray
        Power spectral density in rad^2/Hz

    Notes
    -----
    Under Taylor hypothesis, turbulence advects across the aperture
    at wind speed V. The temporal PSD follows:

        PSD(f) ~ f^(-2n/3 - 17/3) for f >> f_knee
        PSD(f) ~ constant for f << f_knee

    where f_knee ~ V / D and n is the radial order of the Zernike mode.
    """
    f = np.asarray(frequencies)

    # Get radial order n from Zernike index j
    # j = n(n+1)/2 + m + 1 approximately
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))

    # Knee frequency (where temporal spectrum rolls off)
    f_knee = wind_speed / (2 * r0)

    # Low and high frequency behavior
    # High-f: f^(-8/3) for tip/tilt, steeper for higher orders
    alpha = -(2 * n + 17) / 3

    # Variance of this mode (from Noll coefficients, simplified)
    var_j = 0.2944 * j**(-11/6)  # Asymptotic formula

    # Construct PSD (simplified model)
    with np.errstate(divide='ignore', invalid='ignore'):
        psd = var_j * f_knee / (1 + (f / f_knee)**(-alpha))
        psd = np.where(f > 0, psd, var_j)

    return psd


@dataclass
class ShackHartmannWFS:
    """
    Shack-Hartmann wavefront sensor model.

    Attributes
    ----------
    n_subapertures : int
        Number of subapertures across diameter
    pixel_size : float
        Detector pixel size in arcsec
    wavelength : float
        Sensing wavelength in meters
    read_noise : float
        Detector read noise in electrons
    throughput : float
        Optical throughput (0-1)
    """
    n_subapertures: int
    pixel_size: float
    wavelength: float
    read_noise: float = 3.0
    throughput: float = 0.5

    def spot_size_fwhm(self, r0: float, aperture_diameter: float) -> float:
        """
        Calculate spot FWHM in each subaperture.

        Parameters
        ----------
        r0 : float
            Fried parameter in meters
        aperture_diameter : float
            Full aperture diameter in meters

        Returns
        -------
        fwhm : float
            Spot FWHM in arcsec
        """
        d_sub = aperture_diameter / self.n_subapertures

        # Diffraction limit of subaperture
        fwhm_diff = 1.22 * self.wavelength / d_sub * 206265  # arcsec

        # Seeing-limited size (if d_sub >> r0)
        if d_sub > r0:
            fwhm_seeing = 0.98 * self.wavelength / r0 * 206265
            fwhm = np.sqrt(fwhm_diff**2 + fwhm_seeing**2)
        else:
            fwhm = fwhm_diff

        return fwhm

    def measurement_error(self, r0: float, aperture_diameter: float,
                          photons_per_subaperture: float) -> float:
        """
        Calculate centroid measurement error.

        Parameters
        ----------
        r0 : float
            Fried parameter in meters
        aperture_diameter : float
            Full aperture diameter in meters
        photons_per_subaperture : float
            Number of detected photons per subaperture per frame

        Returns
        -------
        sigma_cent : float
            Centroid error in arcsec
        """
        fwhm = self.spot_size_fwhm(r0, aperture_diameter)

        # Photon noise contribution
        sigma_photon = fwhm / np.sqrt(photons_per_subaperture)

        # Read noise contribution
        # Assuming 4x4 pixels per spot
        n_pixels = 16
        sigma_read = self.pixel_size * np.sqrt(n_pixels) * self.read_noise / \
                     photons_per_subaperture

        sigma_cent = np.sqrt(sigma_photon**2 + sigma_read**2)

        return sigma_cent


def strehl_from_variance(phase_variance_rad2: float) -> float:
    """
    Compute Strehl ratio from phase variance.

    Parameters
    ----------
    phase_variance_rad2 : float
        Phase variance in rad^2

    Returns
    -------
    strehl : float
        Strehl ratio (0-1)

    Notes
    -----
    Uses extended Marechal approximation:
        S = exp(-sigma^2) for sigma^2 < ~2
        S = 1 - sigma^2 for small sigma^2 (linear regime)
    """
    if phase_variance_rad2 < 0.5:
        # Linear regime
        strehl = 1 - phase_variance_rad2
    elif phase_variance_rad2 < 4:
        # Extended Marechal
        strehl = np.exp(-phase_variance_rad2)
    else:
        # Deep turbulence - Strehl very low
        strehl = np.exp(-4) * (4 / phase_variance_rad2)

    return np.clip(strehl, 0, 1)


def variance_from_strehl(strehl: float) -> float:
    """
    Compute phase variance from Strehl ratio.

    Parameters
    ----------
    strehl : float
        Strehl ratio (0-1)

    Returns
    -------
    phase_variance : float
        Phase variance in rad^2
    """
    if strehl <= 0 or strehl > 1:
        raise ValueError("Strehl must be in (0, 1]")

    return -np.log(strehl)


def multi_conjugate_fitting_error(
    aperture_diameter: float,
    r0: float,
    dm_altitudes: List[float],
    cn2_profile: np.ndarray,
    altitude_profile: np.ndarray,
    n_actuators: int = 20,
) -> float:
    """
    Fitting error for multi-conjugate adaptive optics (MCAO).

    MCAO uses multiple deformable mirrors conjugated to different
    altitudes to correct 3D turbulence volume.

    Parameters
    ----------
    aperture_diameter : float
        Aperture diameter in meters
    r0 : float
        Fried parameter at ground in meters
    dm_altitudes : list of float
        DM conjugate altitudes in meters
    cn2_profile : ndarray
        Cn2 profile values in m^(-2/3)
    altitude_profile : ndarray
        Altitudes in meters
    n_actuators : int
        Actuators per DM

    Returns
    -------
    sigma_fit2 : float
        MCAO fitting error variance in rad^2

    Notes
    -----
    Each DM corrects turbulence within its conjugate range.
    The total fitting error is reduced compared to single-conjugate AO.
    """
    d_act = aperture_diameter / n_actuators

    cn2 = np.asarray(cn2_profile)
    h = np.asarray(altitude_profile)
    dm_h = np.sort(dm_altitudes)

    # Add ground and very high altitude boundaries
    boundaries = [0] + list(dm_h) + [np.max(h) * 2]

    total_fit_error = 0.0

    # For each layer, find closest DM and compute fitting error
    for i in range(len(altitude_profile)):
        # Find which DM range this altitude falls into
        closest_dm_dist = np.min(np.abs(dm_h - h[i]))

        # Projected actuator spacing at this altitude
        # Increases with altitude due to geometric projection
        d_eff = d_act * (1 + h[i] / 10000)  # Simple model

        # Layer r0
        if cn2[i] > 0:
            r0_layer = (0.423 * (2*np.pi/0.5e-6)**2 * cn2[i])**(-3/5)
            layer_fit = 0.28 * (d_eff / r0_layer)**(5/3) * (cn2[i] / cn2.sum())
            total_fit_error += layer_fit

    return total_fit_error
