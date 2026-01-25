"""
Air mass functions for atmospheric path length calculations.

This module provides functions for calculating optical air mass,
including the Chapman function for curved-Earth geometry at high
solar zenith angles.

References
----------
- Chapman, S. (1931). The absorption and dissociative or ionizing
  effect of monochromatic radiation in an atmosphere on a rotating
  earth. Proc. Phys. Soc., 43, 26-45.
- Kasten, F. & Young, A.T. (1989). Revised optical air mass tables
  and approximation formula. Applied Optics, 28(22), 4735-4738.
"""

import numpy as np
from raf_tran.utils.constants import EARTH_RADIUS


def plane_parallel_air_mass(sza_deg):
    """
    Calculate air mass using plane-parallel approximation.

    Simple 1/cos(SZA) formula valid for SZA < 70 deg.

    Parameters
    ----------
    sza_deg : float or array_like
        Solar zenith angle in degrees

    Returns
    -------
    air_mass : float or ndarray
        Optical air mass (dimensionless)

    Notes
    -----
    This approximation breaks down for SZA > 70 deg due to
    Earth curvature effects. Use chapman_air_mass() instead.
    """
    sza_rad = np.radians(sza_deg)
    mu0 = np.cos(sza_rad)

    if np.any(mu0 <= 0):
        raise ValueError("Solar zenith angle must be < 90 deg (sun above horizon)")

    return 1.0 / mu0


def kasten_young_air_mass(sza_deg):
    """
    Calculate air mass using Kasten & Young (1989) formula.

    Empirical formula accurate to 0.1% for SZA up to 90 deg.

    Parameters
    ----------
    sza_deg : float or array_like
        Solar zenith angle in degrees

    Returns
    -------
    air_mass : float or ndarray
        Optical air mass (dimensionless)

    References
    ----------
    Kasten, F. & Young, A.T. (1989). Revised optical air mass tables
    and approximation formula. Applied Optics, 28(22), 4735-4738.
    """
    sza = np.asarray(sza_deg)

    if np.any(sza >= 90):
        raise ValueError("Solar zenith angle must be < 90 deg")

    # Kasten-Young formula
    # m = 1 / [cos(z) + 0.50572 * (96.07995 - z)^(-1.6364)]
    a = 0.50572
    b = 96.07995
    c = 1.6364

    return 1.0 / (np.cos(np.radians(sza)) + a * (b - sza)**(-c))


def chapman_function(chi, x):
    """
    Chapman grazing incidence function for spherical atmosphere.

    Calculates the ratio of slant column to vertical column for
    a spherical atmosphere with exponential density profile.

    Parameters
    ----------
    chi : float or array_like
        Solar zenith angle in radians
    x : float
        Ratio of observer altitude to atmospheric scale height (h/H)
        For ground level with H=8.5 km: x = R_earth / H ~ 750

    Returns
    -------
    ch : float or ndarray
        Chapman function value (air mass factor)

    Notes
    -----
    The Chapman function is defined as:

        Ch(chi, x) = sqrt(pi*x/2) * exp(x*(1-sin(chi))) * erfc(sqrt(x/2)*(1-sin(chi))^0.5)

    For chi < 90 deg (sun above horizon).

    For large x (surface observations), simpler approximations exist.

    References
    ----------
    Chapman, S. (1931). The absorption and dissociative or ionizing
    effect of monochromatic radiation in an atmosphere on a rotating
    earth. Proc. Phys. Soc., 43, 26-45.
    """
    from scipy.special import erfc

    chi = np.asarray(chi)
    sin_chi = np.sin(chi)
    cos_chi = np.cos(chi)

    # For chi < 90 degrees
    if np.all(chi < np.pi / 2):
        # Simple approximation valid for surface observations
        # Ch ~ sec(chi) * (1 - (H/R) * tan^2(chi) / 2)
        # where H is scale height, R is Earth radius
        tan_chi = np.tan(chi)
        return (1.0 / cos_chi) * (1.0 - (1.0 / (2 * x)) * tan_chi**2)

    # Full Chapman function for chi >= 90 deg (grazing incidence)
    result = np.zeros_like(chi, dtype=float)

    # Below horizon case would need different treatment
    mask_above = chi < np.pi / 2

    if np.any(mask_above):
        tan_chi_above = np.tan(chi[mask_above])
        result[mask_above] = (1.0 / np.cos(chi[mask_above])) * \
                             (1.0 - (1.0 / (2 * x)) * tan_chi_above**2)

    # Near-horizon case (chi ~ 90 deg): use full formula
    mask_horizon = ~mask_above
    if np.any(mask_horizon):
        # Approximate formula for grazing incidence
        y = np.sqrt(x / 2) * np.abs(cos_chi[mask_horizon])
        result[mask_horizon] = np.sqrt(np.pi * x / 2) * np.exp(y**2) * erfc(y)

    return result


def optical_air_mass(sza_deg, altitude_m=0, scale_height_m=8500, method='auto'):
    """
    Calculate optical air mass with automatic method selection.

    Parameters
    ----------
    sza_deg : float or array_like
        Solar zenith angle in degrees
    altitude_m : float, optional
        Observer altitude in meters (default: 0 for sea level)
    scale_height_m : float, optional
        Atmospheric scale height in meters (default: 8500 m)
    method : str, optional
        Calculation method:
        - 'auto': Automatically select based on SZA (default)
        - 'plane_parallel': Simple 1/cos(SZA)
        - 'kasten_young': Empirical formula (Kasten & Young 1989)
        - 'chapman': Chapman function for spherical atmosphere

    Returns
    -------
    air_mass : float or ndarray
        Optical air mass (dimensionless)

    Raises
    ------
    ValueError
        If SZA >= 90 deg (sun below horizon)

    Examples
    --------
    >>> optical_air_mass(0)  # Overhead sun
    1.0
    >>> optical_air_mass(60)  # 60 deg SZA
    2.0
    >>> optical_air_mass(85)  # High SZA - uses Kasten-Young
    10.4
    """
    sza = np.asarray(sza_deg)

    if np.any(sza >= 90):
        raise ValueError(
            f"Solar zenith angle must be < 90 deg (sun above horizon). "
            f"Got SZA = {np.max(sza):.1f} deg"
        )

    if method == 'auto':
        # Use plane-parallel for SZA < 70, Kasten-Young for higher
        if np.all(sza < 70):
            method = 'plane_parallel'
        else:
            method = 'kasten_young'

    if method == 'plane_parallel':
        return plane_parallel_air_mass(sza_deg)

    elif method == 'kasten_young':
        return kasten_young_air_mass(sza_deg)

    elif method == 'chapman':
        chi = np.radians(sza)
        x = (EARTH_RADIUS + altitude_m) / scale_height_m
        return chapman_function(chi, x)

    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Use 'auto', 'plane_parallel', 'kasten_young', or 'chapman'")


def validate_solar_geometry(sza_deg, allow_horizon=False):
    """
    Validate solar zenith angle and return cosine.

    Parameters
    ----------
    sza_deg : float or array_like
        Solar zenith angle in degrees
    allow_horizon : bool, optional
        If True, allow SZA up to 90 deg (default: False)

    Returns
    -------
    mu0 : float or ndarray
        Cosine of solar zenith angle

    Raises
    ------
    ValueError
        If SZA is invalid (negative or sun below horizon)
    """
    sza = np.asarray(sza_deg)

    if np.any(sza < 0):
        raise ValueError(f"Solar zenith angle cannot be negative. Got {np.min(sza):.1f} deg")

    max_sza = 90 if allow_horizon else 89.99

    if np.any(sza >= max_sza):
        raise ValueError(
            f"Solar zenith angle must be < {max_sza:.0f} deg (sun above horizon). "
            f"Got SZA = {np.max(sza):.1f} deg"
        )

    return np.cos(np.radians(sza))
