"""
Kolmogorov turbulence spectrum and structure functions.

This module provides functions for the Kolmogorov (and modified)
power spectra that describe atmospheric optical turbulence.

Theory
------
Kolmogorov turbulence assumes:
1. Locally isotropic turbulence
2. Energy cascade from large to small scales
3. Inertial subrange with -5/3 power law

The refractive index fluctuations follow:
    Phi_n(kappa) = 0.033 * Cn2 * kappa^(-11/3)

References
----------
- Kolmogorov, A.N. (1941). The local structure of turbulence in
  incompressible viscous fluid for very large Reynolds numbers.
- Tatarskii, V.I. (1961). Wave Propagation in a Turbulent Medium.
- von Karman, T. (1948). Progress in the statistical theory of turbulence.
"""

import numpy as np


def kolmogorov_spectrum(kappa, cn2):
    """
    Kolmogorov power spectrum of refractive index fluctuations.

    Parameters
    ----------
    kappa : float or array_like
        Spatial frequency in rad/m
    cn2 : float
        Refractive index structure constant in m^(-2/3)

    Returns
    -------
    phi_n : float or ndarray
        Power spectral density in m^3

    Notes
    -----
    The Kolmogorov spectrum is:

        Phi_n(kappa) = 0.033 * Cn2 * kappa^(-11/3)

    Valid for: l0 << 1/kappa << L0
    where l0 is inner scale (~mm) and L0 is outer scale (~m to 100 m).

    Outside this range, use von_karman_spectrum for proper behavior.
    """
    kappa = np.asarray(kappa)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        phi_n = 0.033 * cn2 * kappa**(-11/3)
        phi_n = np.where(kappa > 0, phi_n, 0)

    return phi_n


def von_karman_spectrum(kappa, cn2, L0=100.0, l0=0.001):
    """
    Modified von Karman power spectrum.

    Extends Kolmogorov spectrum with inner and outer scale cutoffs,
    providing more realistic behavior at extreme spatial frequencies.

    Parameters
    ----------
    kappa : float or array_like
        Spatial frequency in rad/m
    cn2 : float
        Refractive index structure constant in m^(-2/3)
    L0 : float, optional
        Outer scale in meters (default: 100 m)
        Defines low-frequency cutoff
    l0 : float, optional
        Inner scale in meters (default: 1 mm)
        Defines high-frequency cutoff (viscous dissipation)

    Returns
    -------
    phi_n : float or ndarray
        Power spectral density in m^3

    Notes
    -----
    The modified von Karman spectrum is:

        Phi_n(kappa) = 0.033 * Cn2 * exp(-kappa^2/kappa_m^2) /
                       (kappa^2 + kappa_0^2)^(11/6)

    where:
        kappa_0 = 2*pi/L0 (outer scale wavenumber)
        kappa_m = 5.92/l0 (inner scale wavenumber)

    Typical scales:
        Outer scale L0: 1-100 m (larger near surface)
        Inner scale l0: 1-10 mm (smaller at higher altitude)
    """
    kappa = np.asarray(kappa, dtype=float)

    # Scale wavenumbers
    kappa_0 = 2 * np.pi / L0  # Outer scale
    kappa_m = 5.92 / l0       # Inner scale (Hill bump)

    # von Karman spectrum with inner scale exponential cutoff
    phi_n = 0.033 * cn2 * np.exp(-kappa**2 / kappa_m**2) / \
            (kappa**2 + kappa_0**2)**(11/6)

    return phi_n


def structure_function(separation_m, cn2, order=2):
    """
    Kolmogorov structure function for refractive index.

    The structure function describes the mean-square difference of
    a field between two points separated by distance r.

    Parameters
    ----------
    separation_m : float or array_like
        Separation distance in meters
    cn2 : float
        Refractive index structure constant in m^(-2/3)
    order : int, optional
        Order of structure function (default: 2)

    Returns
    -------
    D_n : float or ndarray
        Structure function value

    Notes
    -----
    Second-order structure function (order=2):

        D_n(r) = <[n(x+r) - n(x)]^2> = Cn2 * r^(2/3)

    This is the defining relation for Cn2.

    Third-order: D_n(r) ~ r^1 (skewness)
    """
    r = np.asarray(separation_m)

    if order == 2:
        # Second-order (variance)
        D_n = cn2 * r**(2/3)
    elif order == 3:
        # Third-order (approximately)
        D_n = cn2 * r
    else:
        raise ValueError(f"Order {order} not implemented. Use 2 or 3.")

    return D_n


def phase_structure_function(separation_m, wavelength_m, cn2_integrated,
                             path_length_m):
    """
    Atmospheric phase structure function.

    Describes the mean-square phase difference between two points
    in the aperture plane after propagation through turbulence.

    Parameters
    ----------
    separation_m : float or array_like
        Separation in aperture plane in meters
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    path_length_m : float
        Propagation path length in meters

    Returns
    -------
    D_phi : float or ndarray
        Phase structure function in rad^2

    Notes
    -----
    For Kolmogorov turbulence:

        D_phi(r) = 6.88 * (r/r0)^(5/3)

    where r0 is the Fried parameter.
    """
    r = np.asarray(separation_m)
    k = 2 * np.pi / wavelength_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Phase structure function
    D_phi = 6.88 * (r / r0)**(5/3)

    return D_phi


def coherence_function(separation_m, wavelength_m, cn2_integrated):
    """
    Mutual coherence function for atmospheric propagation.

    The coherence function describes the correlation of the field
    between two points in the observation plane.

    Parameters
    ----------
    separation_m : float or array_like
        Separation distance in meters
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)

    Returns
    -------
    gamma : float or ndarray
        Coherence function (0 to 1)

    Notes
    -----
    For atmospheric turbulence:

        gamma(r) = exp(-D_phi(r)/2)
                 = exp(-3.44 * (r/r0)^(5/3))

    where r0 is the Fried parameter.
    """
    r = np.asarray(separation_m)
    k = 2 * np.pi / wavelength_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Coherence function
    gamma = np.exp(-3.44 * (r / r0)**(5/3))

    return gamma


def spatial_filter_function(kappa, aperture_diameter_m, mode='circular'):
    """
    Spatial filter function for aperture averaging.

    Describes how an aperture filters turbulence spatial frequencies.

    Parameters
    ----------
    kappa : float or array_like
        Spatial frequency in rad/m
    aperture_diameter_m : float
        Aperture diameter in meters
    mode : str, optional
        Aperture shape: 'circular' or 'square' (default: 'circular')

    Returns
    -------
    F : float or ndarray
        Filter function (0 to 1)

    Notes
    -----
    For circular aperture:
        F(kappa) = [2*J1(kappa*D/2) / (kappa*D/2)]^2

    where J1 is the first-order Bessel function.
    """
    kappa = np.asarray(kappa)
    D = aperture_diameter_m

    if mode == 'circular':
        from scipy.special import j1

        # Argument for Bessel function
        x = kappa * D / 2

        # Avoid division by zero at kappa=0
        with np.errstate(divide='ignore', invalid='ignore'):
            F = np.where(x > 0, (2 * j1(x) / x)**2, 1.0)

    elif mode == 'square':
        x = kappa * D / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            F = np.where(x > 0, (np.sin(x) / x)**4, 1.0)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'circular' or 'square'.")

    return F


def turbulence_mtf(spatial_freq, wavelength_m, cn2_integrated, path_length_m):
    """
    Atmospheric turbulence modulation transfer function (MTF).

    The MTF describes the contrast reduction of spatial frequencies
    in an image due to atmospheric turbulence.

    Parameters
    ----------
    spatial_freq : float or array_like
        Spatial frequency in cycles/m (in image plane)
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    path_length_m : float
        Propagation path length in meters

    Returns
    -------
    mtf : float or ndarray
        MTF value (0 to 1)

    Notes
    -----
    Long-exposure MTF for Kolmogorov turbulence:

        MTF(f) = exp(-3.44 * (lambda*f*L/r0)^(5/3))

    where L is path length and r0 is Fried parameter.
    """
    f = np.asarray(spatial_freq)
    k = 2 * np.pi / wavelength_m
    L = path_length_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Effective separation in pupil
    r_eff = wavelength_m * f * L

    # MTF
    mtf = np.exp(-3.44 * (r_eff / r0)**(5/3))

    return mtf


# =============================================================================
# Phase-Aware Functions
# =============================================================================

def phase_variance(wavelength_m, cn2_integrated, aperture_diameter_m):
    """
    Calculate total phase variance over a circular aperture.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    aperture_diameter_m : float
        Aperture diameter in meters

    Returns
    -------
    sigma_phi2 : float
        Total phase variance in rad^2

    Notes
    -----
    For Kolmogorov turbulence, the phase variance is:

        sigma_phi^2 = 1.03 * (D/r0)^(5/3)

    where D is aperture diameter and r0 is Fried parameter.

    This represents the total variance including piston, tip, tilt,
    and all higher-order aberrations.
    """
    k = 2 * np.pi / wavelength_m
    D = aperture_diameter_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Total phase variance (Noll 1976)
    sigma_phi2 = 1.03 * (D / r0)**(5/3)

    return sigma_phi2


def tilt_removed_phase_variance(wavelength_m, cn2_integrated, aperture_diameter_m):
    """
    Calculate phase variance with piston and tilt removed.

    This is the residual phase variance after perfect tip-tilt correction,
    relevant for adaptive optics systems.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    aperture_diameter_m : float
        Aperture diameter in meters

    Returns
    -------
    sigma_phi2 : float
        Tilt-removed phase variance in rad^2

    Notes
    -----
    Removing piston (j=1), tip (j=2), and tilt (j=3):

        sigma_phi^2 (tilt-removed) = 0.134 * (D/r0)^(5/3)

    This is ~13% of the total phase variance (Noll 1976).

    References
    ----------
    Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence.
    JOSA, 66(3), 207-211.
    """
    k = 2 * np.pi / wavelength_m
    D = aperture_diameter_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Tilt-removed phase variance (from Noll 1976)
    # sigma^2 - sigma_1^2 - sigma_2^2 - sigma_3^2 = 0.134 * (D/r0)^(5/3)
    sigma_phi2 = 0.134 * (D / r0)**(5/3)

    return sigma_phi2


def angle_of_arrival_variance(wavelength_m, cn2_integrated, aperture_diameter_m):
    """
    Calculate angle-of-arrival (wavefront tilt) variance.

    The angle of arrival is the local wavefront gradient, causing
    image motion in imaging systems.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    aperture_diameter_m : float
        Aperture diameter in meters

    Returns
    -------
    sigma_alpha2 : float
        Angle-of-arrival variance in rad^2

    Notes
    -----
    For a circular aperture:

        sigma_alpha^2 = 0.364 * (D/r0)^(-1/3) * (wavelength/D)^2

    Or equivalently:

        sigma_alpha^2 = 0.364 * wavelength^2 / (D^(5/3) * r0^(1/3))

    RMS image motion = sqrt(sigma_alpha^2)
    """
    k = 2 * np.pi / wavelength_m
    D = aperture_diameter_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Angle of arrival variance (Fried 1965)
    sigma_alpha2 = 0.364 * (wavelength_m / D)**2 * (D / r0)**(-1/3)

    return sigma_alpha2


def zernike_variance(j, wavelength_m, cn2_integrated, aperture_diameter_m):
    """
    Calculate variance of Zernike coefficient for atmospheric turbulence.

    Zernike polynomials decompose the wavefront into orthogonal modes.
    This function gives the variance of each coefficient.

    Parameters
    ----------
    j : int
        Zernike mode index (Noll ordering, j >= 1)
        j=1: piston, j=2: tip, j=3: tilt, j=4: defocus, etc.
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    aperture_diameter_m : float
        Aperture diameter in meters

    Returns
    -------
    sigma_j2 : float
        Variance of Zernike coefficient j in rad^2

    Notes
    -----
    For Kolmogorov turbulence, the variance follows:

        sigma_j^2 ~ (D/r0)^(5/3) * Delta_j

    where Delta_j depends on mode index j. For large j:

        Delta_j ~ 0.2944 * j^(-11/6) * (-1)^(n(j))

    References
    ----------
    Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence.
    JOSA, 66(3), 207-211.
    """
    if j < 1:
        raise ValueError("Zernike index j must be >= 1")

    k = 2 * np.pi / wavelength_m
    D = aperture_diameter_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Noll (1976) Table 1 coefficients for first few modes
    # Delta_j values (divided by (D/r0)^(5/3) to get sigma_j^2)
    noll_coefficients = {
        1: 1.030,    # Piston
        2: 0.582,    # Tip (x-tilt)
        3: 0.582,    # Tilt (y-tilt)
        4: 0.134,    # Defocus
        5: 0.111,    # Astigmatism
        6: 0.111,    # Astigmatism
        7: 0.0880,   # Coma
        8: 0.0880,   # Coma
        9: 0.0648,   # Trefoil
        10: 0.0648,  # Trefoil
        11: 0.0587,  # Spherical
    }

    if j in noll_coefficients:
        delta_j = noll_coefficients[j]
    else:
        # Asymptotic formula for large j (Noll 1976)
        # Convert j to radial order n
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))
        delta_j = 0.2944 * j**(-11/6)

    sigma_j2 = delta_j * (D / r0)**(5/3)

    return sigma_j2


def phase_power_spectrum(spatial_freq, wavelength_m, cn2_integrated):
    """
    Power spectrum of atmospheric phase fluctuations.

    Parameters
    ----------
    spatial_freq : float or array_like
        Spatial frequency in cycles/m
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)

    Returns
    -------
    W_phi : float or ndarray
        Phase power spectrum in rad^2 * m^2

    Notes
    -----
    For Kolmogorov turbulence:

        W_phi(f) = 0.023 * r0^(-5/3) * f^(-11/3)

    where f is spatial frequency in cycles/m.

    This is the Fourier transform of the phase covariance function.
    """
    f = np.asarray(spatial_freq)
    k = 2 * np.pi / wavelength_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Phase power spectrum
    with np.errstate(divide='ignore', invalid='ignore'):
        W_phi = 0.023 * r0**(-5/3) * f**(-11/3)
        W_phi = np.where(f > 0, W_phi, 0)

    return W_phi


def residual_phase_variance_ao(n_modes, wavelength_m, cn2_integrated, aperture_diameter_m):
    """
    Residual phase variance after adaptive optics correction.

    Calculates the remaining phase variance after correcting
    the first N Zernike modes.

    Parameters
    ----------
    n_modes : int
        Number of Zernike modes corrected (including piston)
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    aperture_diameter_m : float
        Aperture diameter in meters

    Returns
    -------
    sigma_residual2 : float
        Residual phase variance in rad^2

    Notes
    -----
    The residual variance after correcting J modes is:

        sigma_residual^2 = sum_{j>J} sigma_j^2

    For large J, this approaches:

        sigma_residual^2 ~ 0.2944 * (D/r0)^(5/3) * J^(-5/6)

    From this, the Strehl ratio can be estimated as:

        S ~ exp(-sigma_residual^2)

    References
    ----------
    Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence.
    """
    k = 2 * np.pi / wavelength_m
    D = aperture_diameter_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    # Cumulative residual variance (Noll 1976, Table 3)
    # These are Delta_J values = sum_{j>J} Delta_j
    cumulative_residuals = {
        1: 1.030,    # No correction
        2: 0.582,    # Piston removed
        3: 0.134,    # Tip removed (piston + tip)
        4: 0.111,    # Tilt removed (piston + tip + tilt)
        5: 0.0880,   # Focus removed
        6: 0.0648,   # Astig removed
        7: 0.0587,   # Astig removed
        10: 0.0401,  # First 10 modes
        20: 0.0208,  # First 20 modes
    }

    if n_modes in cumulative_residuals:
        delta_residual = cumulative_residuals[n_modes]
    else:
        # Asymptotic formula for large n_modes
        delta_residual = 0.2944 * n_modes**(-5/6)

    sigma_residual2 = delta_residual * (D / r0)**(5/3)

    return sigma_residual2


def long_exposure_strehl(wavelength_m, cn2_integrated, aperture_diameter_m):
    """
    Long-exposure Strehl ratio due to atmospheric turbulence.

    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
    cn2_integrated : float
        Path-integrated Cn2 in m^(1/3)
    aperture_diameter_m : float
        Aperture diameter in meters

    Returns
    -------
    S : float
        Long-exposure Strehl ratio (0 to 1)

    Notes
    -----
    For D >> r0 (turbulence-limited):

        S = (r0/D)^2

    For D << r0 (diffraction-limited):

        S ~ 1 - (D/r0)^(5/3)

    This differs from the extended Marechal approximation which uses
    phase variance.
    """
    k = 2 * np.pi / wavelength_m
    D = aperture_diameter_m

    # Fried parameter
    r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    if D <= r0:
        # Near diffraction-limited
        S = 1.0 - (D / r0)**(5/3)
    else:
        # Turbulence-limited
        S = (r0 / D)**2

    return np.clip(S, 0.0, 1.0)
