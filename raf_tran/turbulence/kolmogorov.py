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
