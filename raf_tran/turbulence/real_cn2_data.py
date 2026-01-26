"""
Real Cn2 Data Integration Module
================================

This module provides interfaces for importing and using real atmospheric
turbulence (Cn2) measurements from various sources for high-fidelity
simulation.

Data Sources (all optional, offline fallback available)
-------------------------------------------------------
1. SCIDAR/SLODAR measurements
2. Balloon radiosonde profiles
3. NOAA/NCEP atmospheric models (online)
4. Site-specific measurement databases
5. User-provided profiles

OFFLINE OPERATION
-----------------
This module works fully offline using built-in climatological models.
External data sources are optional enhancements for site-specific accuracy.

References
----------
- Tokovinin, A. (2002). From differential image motion to seeing.
  PASP 114:1156-1166.
- Vernin, J. & Munoz-Tunon, C. (1992). Optical seeing at La Palma.
  A&A 257:811-816.
"""

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union, Callable
from pathlib import Path
import json

import numpy as np

# Check for optional dependencies
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class Cn2Profile:
    """
    Container for Cn2 profile data.

    Attributes
    ----------
    altitudes : ndarray
        Altitude levels in meters above ground
    cn2 : ndarray
        Cn2 values at each altitude in m^(-2/3)
    source : str
        Data source description
    timestamp : str, optional
        Measurement timestamp (ISO format)
    location : tuple, optional
        (latitude, longitude) of measurement site
    integrated_cn2 : float
        Path-integrated Cn2 in m^(1/3)
    r0_500nm : float
        Fried parameter at 500nm in meters
    seeing_arcsec : float
        Seeing FWHM at 500nm in arcsec
    """
    altitudes: np.ndarray
    cn2: np.ndarray
    source: str = "unknown"
    timestamp: Optional[str] = None
    location: Optional[Tuple[float, float]] = None

    @property
    def integrated_cn2(self) -> float:
        """Path-integrated Cn2."""
        return np.trapezoid(self.cn2, self.altitudes)

    @property
    def r0_500nm(self) -> float:
        """Fried parameter at 500nm."""
        k = 2 * np.pi / 500e-9
        return (0.423 * k**2 * self.integrated_cn2)**(-3/5)

    @property
    def seeing_arcsec(self) -> float:
        """Seeing FWHM at 500nm in arcsec."""
        return 0.98 * 500e-9 / self.r0_500nm * 206265

    def scale_to_wavelength(self, wavelength_m: float) -> float:
        """
        Get r0 scaled to a different wavelength.

        Parameters
        ----------
        wavelength_m : float
            Target wavelength in meters

        Returns
        -------
        r0 : float
            Fried parameter at target wavelength
        """
        # r0 scales as lambda^(6/5)
        return self.r0_500nm * (wavelength_m / 500e-9)**(6/5)


# ============================================================================
# Built-in Climatological Profiles (Offline)
# ============================================================================

def get_climatological_profile(
    site: str = "generic_good",
    season: str = "annual",
) -> Cn2Profile:
    """
    Get a built-in climatological Cn2 profile.

    These profiles work offline and provide reasonable defaults
    for various site types.

    Parameters
    ----------
    site : str
        Site type: 'generic_good', 'generic_median', 'generic_poor',
                   'mountaintop', 'coastal', 'desert', 'continental'
    season : str
        Season: 'annual', 'winter', 'summer'

    Returns
    -------
    profile : Cn2Profile
        Climatological Cn2 profile
    """
    # Standard altitude grid
    altitudes = np.array([
        0, 100, 200, 500, 1000, 2000, 3000, 4000, 5000,
        6000, 8000, 10000, 12000, 15000, 20000
    ], dtype=float)

    # Ground-level Cn2 values for different sites (m^-2/3)
    ground_cn2 = {
        'generic_good': 1e-15,
        'generic_median': 3e-15,
        'generic_poor': 1e-14,
        'mountaintop': 5e-16,
        'coastal': 2e-15,
        'desert': 8e-16,
        'continental': 4e-15,
    }

    if site not in ground_cn2:
        warnings.warn(f"Unknown site '{site}', using 'generic_median'")
        site = 'generic_median'

    # Construct profile using modified Hufnagel-Valley model
    cn2_0 = ground_cn2[site]

    # Seasonal adjustment
    seasonal_factor = {
        'annual': 1.0,
        'winter': 0.7,  # Generally calmer
        'summer': 1.5,  # More convection
    }
    factor = seasonal_factor.get(season, 1.0)
    cn2_0 *= factor

    # Build profile
    cn2 = np.zeros_like(altitudes)
    for i, h in enumerate(altitudes):
        if h < 1000:
            # Boundary layer (exponential decay)
            cn2[i] = cn2_0 * np.exp(-h / 100)
        elif h < 10000:
            # Free atmosphere (HV-like)
            cn2[i] = 2.7e-16 * np.exp(-h / 1500) + \
                     1e-17 * np.exp(-((h - 8000) / 1500)**2)
        else:
            # Tropopause and above
            cn2[i] = 1e-17 * np.exp(-(h - 10000) / 3000)

    return Cn2Profile(
        altitudes=altitudes,
        cn2=cn2,
        source=f"climatological_{site}_{season}",
    )


# ============================================================================
# Standard Profile Models
# ============================================================================

def hufnagel_valley_57_profile(
    ground_cn2: float = 1.7e-14,
    high_altitude_wind: float = 21.0,
    altitudes: Optional[np.ndarray] = None,
) -> Cn2Profile:
    """
    Generate Hufnagel-Valley 5/7 model profile.

    This is the standard HV model that produces 5 arcsec seeing
    with 7 cm isoplanatic angle at 500nm.

    Parameters
    ----------
    ground_cn2 : float
        Ground-level Cn2 in m^(-2/3) (default: 1.7e-14)
    high_altitude_wind : float
        High-altitude wind speed in m/s (default: 21)
    altitudes : ndarray, optional
        Custom altitude grid in meters

    Returns
    -------
    profile : Cn2Profile
        HV 5/7 Cn2 profile
    """
    if altitudes is None:
        altitudes = np.linspace(0, 20000, 100)

    h = np.asarray(altitudes)
    A = ground_cn2
    v = high_altitude_wind

    # HV 5/7 model
    cn2 = 0.00594 * (v / 27)**2 * (1e-5 * h)**10 * np.exp(-h / 1000) + \
          2.7e-16 * np.exp(-h / 1500) + \
          A * np.exp(-h / 100)

    return Cn2Profile(
        altitudes=h,
        cn2=cn2,
        source="hufnagel_valley_57",
    )


def bufton_wind_profile(altitude: np.ndarray) -> np.ndarray:
    """
    Bufton wind speed model for atmospheric wind profile.

    Parameters
    ----------
    altitude : ndarray
        Altitudes in meters

    Returns
    -------
    wind_speed : ndarray
        Wind speed in m/s at each altitude
    """
    h = np.asarray(altitude)

    # Bufton model (1973)
    v = 5 + 30 * np.exp(-((h - 9400) / 4800)**2)

    return v


# ============================================================================
# External Data Sources (Optional, Online)
# ============================================================================

def fetch_noaa_profile(
    latitude: float,
    longitude: float,
    date: Optional[str] = None,
    timeout: float = 10.0,
) -> Optional[Cn2Profile]:
    """
    Fetch atmospheric profile from NOAA GFS model (requires internet).

    This function is OPTIONAL - simulation works offline without it.

    Parameters
    ----------
    latitude : float
        Latitude in degrees (-90 to 90)
    longitude : float
        Longitude in degrees (-180 to 180)
    date : str, optional
        Date in YYYY-MM-DD format (default: latest)
    timeout : float
        Request timeout in seconds

    Returns
    -------
    profile : Cn2Profile or None
        Profile if successful, None if offline/unavailable

    Notes
    -----
    NOAA GFS provides temperature and wind profiles which are
    converted to Cn2 using the Tatarskii formula.
    """
    if not REQUESTS_AVAILABLE:
        warnings.warn(
            "requests library not available. Install with: pip install requests\n"
            "Using offline climatological profile instead."
        )
        return None

    # NOAA NOMADS GFS endpoint (example - actual API may vary)
    # This is a simplified example; real implementation would use proper GRIB/NetCDF
    base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

    try:
        # Note: This is a placeholder - actual NOAA API requires GRIB processing
        warnings.warn(
            "NOAA profile fetch is a placeholder. "
            "For production use, implement proper GRIB/NetCDF handling."
        )
        return None

    except Exception as e:
        warnings.warn(f"Could not fetch NOAA data: {e}. Using offline profile.")
        return None


def load_profile_from_file(
    filepath: Union[str, Path],
    format: str = "auto",
) -> Cn2Profile:
    """
    Load Cn2 profile from a file.

    Supported formats:
    - JSON: {"altitudes": [...], "cn2": [...], "source": "..."}
    - CSV: altitude,cn2 columns
    - NPZ: numpy archive with 'altitudes' and 'cn2' arrays

    Parameters
    ----------
    filepath : str or Path
        Path to profile file
    format : str
        File format: 'auto', 'json', 'csv', 'npz'

    Returns
    -------
    profile : Cn2Profile
        Loaded profile
    """
    filepath = Path(filepath)

    if format == "auto":
        suffix = filepath.suffix.lower()
        format = {'json': 'json', '.csv': 'csv', '.npz': 'npz'}.get(suffix, 'json')

    if format == 'json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return Cn2Profile(
            altitudes=np.array(data['altitudes']),
            cn2=np.array(data['cn2']),
            source=data.get('source', str(filepath)),
            timestamp=data.get('timestamp'),
            location=tuple(data['location']) if 'location' in data else None,
        )

    elif format == 'csv':
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        return Cn2Profile(
            altitudes=data[:, 0],
            cn2=data[:, 1],
            source=str(filepath),
        )

    elif format == 'npz':
        data = np.load(filepath)
        return Cn2Profile(
            altitudes=data['altitudes'],
            cn2=data['cn2'],
            source=str(filepath),
        )

    else:
        raise ValueError(f"Unknown format: {format}")


def save_profile_to_file(
    profile: Cn2Profile,
    filepath: Union[str, Path],
    format: str = "json",
) -> None:
    """
    Save Cn2 profile to a file.

    Parameters
    ----------
    profile : Cn2Profile
        Profile to save
    filepath : str or Path
        Output file path
    format : str
        File format: 'json', 'csv', 'npz'
    """
    filepath = Path(filepath)

    if format == 'json':
        data = {
            'altitudes': profile.altitudes.tolist(),
            'cn2': profile.cn2.tolist(),
            'source': profile.source,
            'timestamp': profile.timestamp,
            'location': list(profile.location) if profile.location else None,
            'integrated_cn2': float(profile.integrated_cn2),
            'r0_500nm': float(profile.r0_500nm),
            'seeing_arcsec': float(profile.seeing_arcsec),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    elif format == 'csv':
        header = "altitude_m,cn2_m23"
        np.savetxt(
            filepath,
            np.column_stack([profile.altitudes, profile.cn2]),
            delimiter=',',
            header=header,
            comments='',
        )

    elif format == 'npz':
        np.savez(
            filepath,
            altitudes=profile.altitudes,
            cn2=profile.cn2,
            source=profile.source,
        )

    else:
        raise ValueError(f"Unknown format: {format}")


# ============================================================================
# Profile Utilities
# ============================================================================

def interpolate_profile(
    profile: Cn2Profile,
    new_altitudes: np.ndarray,
    method: str = "log_linear",
) -> Cn2Profile:
    """
    Interpolate Cn2 profile to new altitude grid.

    Parameters
    ----------
    profile : Cn2Profile
        Input profile
    new_altitudes : ndarray
        New altitude grid in meters
    method : str
        Interpolation method: 'linear', 'log_linear' (default)

    Returns
    -------
    new_profile : Cn2Profile
        Interpolated profile
    """
    new_h = np.asarray(new_altitudes)

    if method == "linear":
        new_cn2 = np.interp(new_h, profile.altitudes, profile.cn2)

    elif method == "log_linear":
        # Interpolate in log space (better for exponential decay)
        log_cn2 = np.log(np.maximum(profile.cn2, 1e-20))
        new_log_cn2 = np.interp(new_h, profile.altitudes, log_cn2)
        new_cn2 = np.exp(new_log_cn2)

    else:
        raise ValueError(f"Unknown method: {method}")

    return Cn2Profile(
        altitudes=new_h,
        cn2=new_cn2,
        source=f"{profile.source}_interpolated",
        timestamp=profile.timestamp,
        location=profile.location,
    )


def combine_profiles(
    profiles: List[Cn2Profile],
    weights: Optional[List[float]] = None,
    altitude_grid: Optional[np.ndarray] = None,
) -> Cn2Profile:
    """
    Combine multiple Cn2 profiles with weighted averaging.

    Useful for ensemble forecasting or averaging measurements.

    Parameters
    ----------
    profiles : list of Cn2Profile
        Profiles to combine
    weights : list of float, optional
        Weights for each profile (default: equal weights)
    altitude_grid : ndarray, optional
        Common altitude grid (default: first profile's grid)

    Returns
    -------
    combined : Cn2Profile
        Combined profile
    """
    if not profiles:
        raise ValueError("Need at least one profile")

    if weights is None:
        weights = [1.0 / len(profiles)] * len(profiles)

    if len(weights) != len(profiles):
        raise ValueError("Number of weights must match number of profiles")

    weights = np.array(weights)
    weights /= weights.sum()

    if altitude_grid is None:
        altitude_grid = profiles[0].altitudes

    # Interpolate all profiles to common grid and average
    combined_cn2 = np.zeros_like(altitude_grid)

    for prof, w in zip(profiles, weights):
        interp_prof = interpolate_profile(prof, altitude_grid)
        combined_cn2 += w * interp_prof.cn2

    return Cn2Profile(
        altitudes=altitude_grid,
        cn2=combined_cn2,
        source="combined_" + "_".join(p.source for p in profiles[:3]),
    )


def add_turbulent_layer(
    profile: Cn2Profile,
    altitude: float,
    strength: float,
    width: float = 500.0,
) -> Cn2Profile:
    """
    Add a localized turbulent layer to a profile.

    Useful for modeling jet stream or tropopause turbulence.

    Parameters
    ----------
    profile : Cn2Profile
        Base profile
    altitude : float
        Layer center altitude in meters
    strength : float
        Layer peak Cn2 in m^(-2/3)
    width : float
        Layer width (Gaussian sigma) in meters

    Returns
    -------
    new_profile : Cn2Profile
        Profile with added layer
    """
    # Gaussian layer
    layer = strength * np.exp(-((profile.altitudes - altitude) / width)**2)

    new_cn2 = profile.cn2 + layer

    return Cn2Profile(
        altitudes=profile.altitudes,
        cn2=new_cn2,
        source=f"{profile.source}_+layer_{altitude}m",
        timestamp=profile.timestamp,
        location=profile.location,
    )


def estimate_cn2_from_weather(
    temperature_profile: np.ndarray,
    altitude_profile: np.ndarray,
    wind_speed_profile: np.ndarray,
    humidity_profile: Optional[np.ndarray] = None,
) -> Cn2Profile:
    """
    Estimate Cn2 from meteorological data using Tatarskii formula.

    Parameters
    ----------
    temperature_profile : ndarray
        Temperature in Kelvin at each altitude
    altitude_profile : ndarray
        Altitudes in meters
    wind_speed_profile : ndarray
        Wind speed in m/s at each altitude
    humidity_profile : ndarray, optional
        Relative humidity (0-1) at each altitude

    Returns
    -------
    profile : Cn2Profile
        Estimated Cn2 profile

    Notes
    -----
    Uses simplified Tatarskii formula:
        Cn2 ~ (dT/dz)^2 * outer_scale^(4/3)

    The outer scale is estimated from wind shear.
    """
    T = np.asarray(temperature_profile)
    h = np.asarray(altitude_profile)
    v = np.asarray(wind_speed_profile)

    # Temperature gradient (potential temperature)
    # dtheta/dz = dT/dz + gamma_d, where gamma_d = 9.8 K/km
    gamma_d = 0.0098  # K/m
    dT_dz = np.gradient(T, h) + gamma_d

    # Wind shear
    dv_dz = np.abs(np.gradient(v, h))

    # Outer scale estimate from wind shear (simplified)
    L0 = np.clip(20 / (1 + 100 * dv_dz), 1, 100)  # 1-100 m range

    # Pressure (standard atmosphere approximation)
    P = 101325 * np.exp(-h / 8500)

    # Tatarskii formula (simplified)
    # Cn2 ~ 2.8 * (79e-6 * P/T^2)^2 * L0^(4/3) * (dT/dz)^2
    Ct2 = 2.8 * L0**(4/3) * dT_dz**2
    cn2 = (79e-6 * P / T**2)**2 * Ct2

    # Apply bounds
    cn2 = np.clip(cn2, 1e-19, 1e-12)

    return Cn2Profile(
        altitudes=h,
        cn2=cn2,
        source="weather_estimated",
    )


def get_profile(
    site: str = "generic_median",
    use_online: bool = False,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    filepath: Optional[Union[str, Path]] = None,
) -> Cn2Profile:
    """
    Get a Cn2 profile with automatic fallback to offline data.

    This is the recommended entry point for getting Cn2 profiles.
    It tries online sources if requested but always falls back to
    offline climatological data.

    Parameters
    ----------
    site : str
        Site type for climatological profile (fallback)
    use_online : bool
        Try to fetch online data first (default: False for offline)
    latitude : float, optional
        Site latitude for online fetch
    longitude : float, optional
        Site longitude for online fetch
    filepath : str or Path, optional
        Path to local profile file

    Returns
    -------
    profile : Cn2Profile
        Cn2 profile (guaranteed to return a valid profile)
    """
    # Option 1: Load from file if provided
    if filepath is not None:
        try:
            return load_profile_from_file(filepath)
        except Exception as e:
            warnings.warn(f"Could not load profile from {filepath}: {e}")

    # Option 2: Try online if requested
    if use_online and latitude is not None and longitude is not None:
        profile = fetch_noaa_profile(latitude, longitude)
        if profile is not None:
            return profile

    # Option 3: Climatological fallback (always works)
    return get_climatological_profile(site)


# Alias for backwards compatibility
def can_run_offline() -> bool:
    """
    Check if turbulence simulation can run offline.

    Returns
    -------
    bool
        Always True - offline operation is always supported
    """
    return True
