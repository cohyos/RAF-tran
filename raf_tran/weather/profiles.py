"""
Atmospheric Profile Data (Offline)
==================================

This module provides built-in atmospheric profiles that work
completely offline. These include standard atmospheres and
climatological models.

All functions in this module work without internet connection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


@dataclass
class AtmosphericProfile:
    """
    Container for atmospheric profile data.

    Attributes
    ----------
    altitudes : ndarray
        Altitude levels in meters
    temperature : ndarray
        Temperature in Kelvin at each altitude
    pressure : ndarray
        Pressure in Pa at each altitude
    density : ndarray
        Air density in kg/m^3 at each altitude
    humidity : ndarray, optional
        Relative humidity (0-1) or specific humidity (kg/kg)
    ozone : ndarray, optional
        Ozone mixing ratio (ppmv or kg/kg)
    wind_speed : ndarray, optional
        Wind speed in m/s
    wind_direction : ndarray, optional
        Wind direction in degrees (from north)
    source : str
        Description of data source
    """
    altitudes: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    humidity: Optional[np.ndarray] = None
    ozone: Optional[np.ndarray] = None
    wind_speed: Optional[np.ndarray] = None
    wind_direction: Optional[np.ndarray] = None
    source: str = "unknown"

    @property
    def n_levels(self) -> int:
        """Number of altitude levels."""
        return len(self.altitudes)

    @property
    def surface_temperature(self) -> float:
        """Surface temperature in K."""
        return self.temperature[0]

    @property
    def surface_pressure(self) -> float:
        """Surface pressure in Pa."""
        return self.pressure[0]

    @property
    def scale_height(self) -> float:
        """Approximate atmospheric scale height in meters."""
        # From ideal gas: H = RT/(Mg)
        R = 8.314  # J/(mol*K)
        M = 0.029  # kg/mol (air)
        g = 9.81  # m/s^2
        return R * self.surface_temperature / (M * g)

    def interpolate_to(self, new_altitudes: np.ndarray) -> 'AtmosphericProfile':
        """
        Interpolate profile to new altitude grid.

        Parameters
        ----------
        new_altitudes : ndarray
            New altitude levels in meters

        Returns
        -------
        new_profile : AtmosphericProfile
            Interpolated profile
        """
        new_temp = np.interp(new_altitudes, self.altitudes, self.temperature)

        # Interpolate pressure in log space
        log_p = np.log(self.pressure)
        new_log_p = np.interp(new_altitudes, self.altitudes, log_p)
        new_pressure = np.exp(new_log_p)

        # Interpolate density in log space
        log_rho = np.log(self.density)
        new_log_rho = np.interp(new_altitudes, self.altitudes, log_rho)
        new_density = np.exp(new_log_rho)

        new_humidity = None
        if self.humidity is not None:
            new_humidity = np.interp(new_altitudes, self.altitudes, self.humidity)

        new_ozone = None
        if self.ozone is not None:
            new_ozone = np.interp(new_altitudes, self.altitudes, self.ozone)

        return AtmosphericProfile(
            altitudes=new_altitudes,
            temperature=new_temp,
            pressure=new_pressure,
            density=new_density,
            humidity=new_humidity,
            ozone=new_ozone,
            source=f"{self.source}_interpolated",
        )


# =============================================================================
# US Standard Atmosphere 1976
# =============================================================================

def us_standard_atmosphere(
    altitudes: Optional[np.ndarray] = None,
    max_altitude: float = 86000.0,
    n_levels: int = 87,
) -> AtmosphericProfile:
    """
    US Standard Atmosphere 1976.

    Parameters
    ----------
    altitudes : ndarray, optional
        Custom altitude grid in meters
    max_altitude : float
        Maximum altitude in meters (default: 86 km)
    n_levels : int
        Number of altitude levels if using default grid

    Returns
    -------
    profile : AtmosphericProfile
        US Standard Atmosphere profile

    Notes
    -----
    Valid from sea level to 86 km. Based on:
    - Surface: T=288.15 K, P=101325 Pa
    - Tropopause: 11 km, T=216.65 K
    - Stratopause: 47 km, T=270.65 K
    - Mesopause: 86 km, T=186.87 K
    """
    if altitudes is None:
        altitudes = np.linspace(0, max_altitude, n_levels)

    # Layer boundaries and lapse rates (K/m)
    layers = [
        (0, 11000, -0.0065),       # Troposphere
        (11000, 20000, 0.0),       # Tropopause
        (20000, 32000, 0.001),     # Stratosphere lower
        (32000, 47000, 0.0028),    # Stratosphere upper
        (47000, 51000, 0.0),       # Stratopause
        (51000, 71000, -0.0028),   # Mesosphere lower
        (71000, 86000, -0.002),    # Mesosphere upper
    ]

    # Base values at sea level
    T_0 = 288.15  # K
    P_0 = 101325  # Pa
    g = 9.80665   # m/s^2
    R = 287.05    # J/(kg*K) specific gas constant for air
    M = 0.0289644 # kg/mol

    temperature = np.zeros_like(altitudes)
    pressure = np.zeros_like(altitudes)

    for i, h in enumerate(altitudes):
        T_base = T_0
        P_base = P_0
        h_base = 0

        for h_start, h_end, lapse in layers:
            if h <= h_start:
                break

            h_layer = min(h, h_end) - h_base

            if lapse == 0:
                # Isothermal layer
                T_top = T_base
                P_top = P_base * np.exp(-g * h_layer / (R * T_base))
            else:
                # Linear temperature gradient
                T_top = T_base + lapse * h_layer
                P_top = P_base * (T_top / T_base) ** (-g / (lapse * R))

            if h <= h_end:
                # Within this layer
                if lapse == 0:
                    temperature[i] = T_base
                    pressure[i] = P_base * np.exp(-g * (h - h_base) / (R * T_base))
                else:
                    temperature[i] = T_base + lapse * (h - h_base)
                    pressure[i] = P_base * (temperature[i] / T_base) ** (-g / (lapse * R))
                break

            T_base = T_top
            P_base = P_top
            h_base = h_end

        else:
            # Above all defined layers - extrapolate
            temperature[i] = T_base + layers[-1][2] * (h - h_base)
            pressure[i] = P_base * np.exp(-g * (h - h_base) / (R * T_base))

    # Calculate density from ideal gas law
    density = pressure / (R * temperature)

    # Standard ozone profile (Dobson units equivalent)
    ozone = 8e-6 * np.exp(-altitudes / 5000) * \
            np.exp(-((altitudes - 25000) / 8000)**2)

    return AtmosphericProfile(
        altitudes=altitudes,
        temperature=temperature,
        pressure=pressure,
        density=density,
        ozone=ozone,
        source="US_Standard_Atmosphere_1976",
    )


# =============================================================================
# CIRA-86 Climatology
# =============================================================================

def cira86_atmosphere(
    latitude: float = 45.0,
    month: int = 7,
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """
    CIRA-86 (COSPAR International Reference Atmosphere).

    Provides latitude and season-dependent atmospheric profiles.

    Parameters
    ----------
    latitude : float
        Latitude in degrees (-90 to 90)
    month : int
        Month (1-12)
    altitudes : ndarray, optional
        Custom altitude grid in meters

    Returns
    -------
    profile : AtmosphericProfile
        CIRA-86 profile

    References
    ----------
    Rees, D. (ed.) (1988). COSPAR International Reference Atmosphere:
    1986. Pergamon Press.
    """
    if altitudes is None:
        altitudes = np.linspace(0, 120000, 121)

    # Determine season
    # Northern hemisphere: summer = May-Oct (5-10)
    if latitude >= 0:
        is_summer = 5 <= month <= 10
    else:
        is_summer = not (5 <= month <= 10)

    # Latitude band
    abs_lat = abs(latitude)

    # Base surface temperature varies with latitude and season
    if abs_lat < 30:
        T_surface = 300 - 0.1 * abs_lat
    elif abs_lat < 60:
        T_surface = 290 - 0.5 * (abs_lat - 30) + (10 if is_summer else -10)
    else:
        T_surface = 275 - 0.8 * (abs_lat - 60) + (15 if is_summer else -20)

    # Tropopause height varies with latitude
    if abs_lat < 30:
        h_tropo = 17000
    elif abs_lat < 60:
        h_tropo = 17000 - 200 * (abs_lat - 30)
    else:
        h_tropo = 11000 - 100 * (abs_lat - 60)

    # Build temperature profile
    temperature = np.zeros_like(altitudes)
    for i, h in enumerate(altitudes):
        if h < h_tropo:
            # Troposphere with latitude-dependent lapse rate
            lapse = 0.0065 + 0.0005 * (abs_lat / 90)
            temperature[i] = T_surface - lapse * h
        elif h < 20000:
            # Tropopause
            temperature[i] = T_surface - 0.0065 * h_tropo
        elif h < 50000:
            # Stratosphere
            T_tropo = T_surface - 0.0065 * h_tropo
            temperature[i] = T_tropo + 0.001 * (h - 20000)
        elif h < 85000:
            # Mesosphere
            T_strato = T_surface - 0.0065 * h_tropo + 0.001 * 30000
            temperature[i] = T_strato - 0.003 * (h - 50000)
        else:
            # Thermosphere
            T_meso = T_surface - 0.0065 * h_tropo + 0.001 * 30000 - 0.003 * 35000
            temperature[i] = T_meso + 0.01 * (h - 85000)

    # Ensure minimum temperature
    temperature = np.maximum(temperature, 150)

    # Pressure from hydrostatic equation
    g = 9.81
    R = 287.05
    P_0 = 101325

    pressure = np.zeros_like(altitudes)
    pressure[0] = P_0

    for i in range(1, len(altitudes)):
        dh = altitudes[i] - altitudes[i-1]
        T_avg = 0.5 * (temperature[i] + temperature[i-1])
        pressure[i] = pressure[i-1] * np.exp(-g * dh / (R * T_avg))

    density = pressure / (R * temperature)

    return AtmosphericProfile(
        altitudes=altitudes,
        temperature=temperature,
        pressure=pressure,
        density=density,
        source=f"CIRA-86_lat{latitude}_month{month}",
    )


# =============================================================================
# Standard Atmosphere Profiles (AFGL)
# =============================================================================

def _create_standard_profile(
    name: str,
    T_surface: float,
    h_tropopause: float,
    T_tropopause: float,
    humidity_surface: float,
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """Create a standard atmosphere profile."""
    if altitudes is None:
        altitudes = np.linspace(0, 100000, 101)

    g = 9.81
    R = 287.05
    P_0 = 101325

    temperature = np.zeros_like(altitudes)
    for i, h in enumerate(altitudes):
        if h < h_tropopause:
            # Troposphere
            lapse = (T_surface - T_tropopause) / h_tropopause
            temperature[i] = T_surface - lapse * h
        elif h < 20000:
            temperature[i] = T_tropopause
        elif h < 32000:
            temperature[i] = T_tropopause + 0.001 * (h - 20000)
        elif h < 47000:
            temperature[i] = T_tropopause + 12 + 0.0028 * (h - 32000)
        elif h < 51000:
            temperature[i] = T_tropopause + 12 + 42
        elif h < 71000:
            temperature[i] = T_tropopause + 54 - 0.0028 * (h - 51000)
        else:
            temperature[i] = max(T_tropopause + 54 - 56 - 0.002 * (h - 71000), 180)

    pressure = np.zeros_like(altitudes)
    pressure[0] = P_0
    for i in range(1, len(altitudes)):
        dh = altitudes[i] - altitudes[i-1]
        T_avg = 0.5 * (temperature[i] + temperature[i-1])
        pressure[i] = pressure[i-1] * np.exp(-g * dh / (R * T_avg))

    density = pressure / (R * temperature)

    # Humidity profile (exponential decay)
    humidity = humidity_surface * np.exp(-altitudes / 2000)
    humidity = np.minimum(humidity, 1.0)

    return AtmosphericProfile(
        altitudes=altitudes,
        temperature=temperature,
        pressure=pressure,
        density=density,
        humidity=humidity,
        source=name,
    )


def tropical_atmosphere(
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """
    Tropical atmosphere (AFGL).

    Surface temperature: 300 K
    Tropopause: 17 km, 195 K
    High humidity
    """
    return _create_standard_profile(
        name="Tropical",
        T_surface=300.0,
        h_tropopause=17000.0,
        T_tropopause=195.0,
        humidity_surface=0.85,
        altitudes=altitudes,
    )


def midlatitude_summer(
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """
    Midlatitude Summer atmosphere (AFGL).

    Surface temperature: 294 K
    Tropopause: 13 km, 218 K
    """
    return _create_standard_profile(
        name="Midlatitude_Summer",
        T_surface=294.0,
        h_tropopause=13000.0,
        T_tropopause=218.0,
        humidity_surface=0.70,
        altitudes=altitudes,
    )


def midlatitude_winter(
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """
    Midlatitude Winter atmosphere (AFGL).

    Surface temperature: 272 K
    Tropopause: 10 km, 218 K
    """
    return _create_standard_profile(
        name="Midlatitude_Winter",
        T_surface=272.0,
        h_tropopause=10000.0,
        T_tropopause=218.0,
        humidity_surface=0.50,
        altitudes=altitudes,
    )


def subarctic_summer(
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """
    Subarctic Summer atmosphere (AFGL).

    Surface temperature: 287 K
    Tropopause: 10 km, 220 K
    """
    return _create_standard_profile(
        name="Subarctic_Summer",
        T_surface=287.0,
        h_tropopause=10000.0,
        T_tropopause=220.0,
        humidity_surface=0.60,
        altitudes=altitudes,
    )


def subarctic_winter(
    altitudes: Optional[np.ndarray] = None,
) -> AtmosphericProfile:
    """
    Subarctic Winter atmosphere (AFGL).

    Surface temperature: 257 K
    Tropopause: 9 km, 217 K
    Low humidity
    """
    return _create_standard_profile(
        name="Subarctic_Winter",
        T_surface=257.0,
        h_tropopause=9000.0,
        T_tropopause=217.0,
        humidity_surface=0.30,
        altitudes=altitudes,
    )


# =============================================================================
# Profile Selection
# =============================================================================

def get_atmospheric_profile(
    latitude: float = 45.0,
    longitude: Optional[float] = None,
    month: Optional[int] = None,
    datetime_str: Optional[str] = None,
    use_online: bool = False,
    altitudes: Optional[np.ndarray] = None,
    profile_type: str = "auto",
) -> AtmosphericProfile:
    """
    Get an atmospheric profile for given location/time.

    This function always returns a valid profile, using online data
    if requested and available, otherwise falling back to offline
    climatological data.

    Parameters
    ----------
    latitude : float
        Latitude in degrees (-90 to 90)
    longitude : float, optional
        Longitude in degrees (for online data)
    month : int, optional
        Month (1-12) for climatology selection
    datetime_str : str, optional
        ISO format datetime for online data
    use_online : bool
        Try to fetch online data (default: False)
    altitudes : ndarray, optional
        Custom altitude grid in meters
    profile_type : str
        Profile type: 'auto', 'us_standard', 'cira86', 'tropical',
                     'midlat_summer', 'midlat_winter', 'subarctic_summer',
                     'subarctic_winter'

    Returns
    -------
    profile : AtmosphericProfile
        Atmospheric profile (always returns valid data)

    Examples
    --------
    >>> # Offline usage (always works)
    >>> profile = get_atmospheric_profile(latitude=40, month=7)
    >>>
    >>> # Try online (falls back to offline if unavailable)
    >>> profile = get_atmospheric_profile(
    ...     latitude=40, longitude=-75,
    ...     datetime_str='2024-06-15T12:00:00',
    ...     use_online=True
    ... )
    """
    import warnings

    # Try online data if requested
    if use_online and longitude is not None and datetime_str is not None:
        try:
            from raf_tran.weather.online import fetch_gfs_profile

            online_profile = fetch_gfs_profile(
                latitude, longitude, datetime_str
            )
            if online_profile is not None:
                if altitudes is not None:
                    online_profile = online_profile.interpolate_to(altitudes)
                return online_profile
        except Exception as e:
            warnings.warn(f"Online fetch failed: {e}. Using offline data.")

    # Select offline profile based on type or location
    if profile_type == "auto":
        abs_lat = abs(latitude)

        if month is None:
            # Default to July for northern hemisphere, January for southern
            month = 7 if latitude >= 0 else 1

        # Determine if summer or winter
        is_summer = (5 <= month <= 10) if latitude >= 0 else not (5 <= month <= 10)

        if abs_lat < 25:
            profile = tropical_atmosphere(altitudes)
        elif abs_lat < 55:
            profile = midlatitude_summer(altitudes) if is_summer else midlatitude_winter(altitudes)
        else:
            profile = subarctic_summer(altitudes) if is_summer else subarctic_winter(altitudes)

    elif profile_type == "us_standard":
        profile = us_standard_atmosphere(altitudes)
    elif profile_type == "cira86":
        profile = cira86_atmosphere(latitude, month or 7, altitudes)
    elif profile_type == "tropical":
        profile = tropical_atmosphere(altitudes)
    elif profile_type == "midlat_summer":
        profile = midlatitude_summer(altitudes)
    elif profile_type == "midlat_winter":
        profile = midlatitude_winter(altitudes)
    elif profile_type == "subarctic_summer":
        profile = subarctic_summer(altitudes)
    elif profile_type == "subarctic_winter":
        profile = subarctic_winter(altitudes)
    else:
        warnings.warn(f"Unknown profile type '{profile_type}', using US Standard")
        profile = us_standard_atmosphere(altitudes)

    return profile
