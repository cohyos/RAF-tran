"""
Spherical Earth Geometry Calculations
=====================================

This module provides geometry calculations for a spherical Earth,
essential for accurate radiative transfer at high solar zenith angles,
limb viewing, and satellite remote sensing.

All calculations work fully offline.
"""

import numpy as np
from math import erf
from typing import Tuple, Optional
from datetime import datetime, timezone


# =============================================================================
# Earth Constants
# =============================================================================

# Mean Earth radius in meters
EARTH_RADIUS = 6371000.0

# WGS84 ellipsoid parameters
EARTH_RADIUS_EQUATORIAL = 6378137.0  # meters
EARTH_RADIUS_POLAR = 6356752.3  # meters
EARTH_FLATTENING = 1 / 298.257223563


def earth_radius_at_latitude(latitude_deg: float) -> float:
    """
    Calculate Earth radius at a given latitude (WGS84 ellipsoid).

    Parameters
    ----------
    latitude_deg : float
        Geodetic latitude in degrees

    Returns
    -------
    radius : float
        Earth radius at latitude in meters
    """
    lat = np.radians(latitude_deg)

    a = EARTH_RADIUS_EQUATORIAL
    b = EARTH_RADIUS_POLAR

    # Radius of curvature in the meridian plane
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)

    numerator = (a**2 * cos_lat)**2 + (b**2 * sin_lat)**2
    denominator = (a * cos_lat)**2 + (b * sin_lat)**2

    return np.sqrt(numerator / denominator)


# =============================================================================
# Path Geometry
# =============================================================================

def slant_path_length(
    altitude_start: float,
    altitude_end: float,
    zenith_angle_deg: float,
    earth_radius: float = EARTH_RADIUS,
) -> float:
    """
    Calculate slant path length through atmosphere.

    Parameters
    ----------
    altitude_start : float
        Starting altitude in meters (e.g., observer altitude)
    altitude_end : float
        Ending altitude in meters (e.g., TOA)
    zenith_angle_deg : float
        Zenith angle in degrees (0 = vertical, 90 = horizontal)
    earth_radius : float
        Local Earth radius in meters

    Returns
    -------
    path_length : float
        Slant path length in meters

    Notes
    -----
    Uses spherical geometry for accurate results at high zenith angles.
    For plane-parallel (small angles): L = (h2 - h1) / cos(theta)
    """
    theta = np.radians(zenith_angle_deg)
    R = earth_radius
    h1 = altitude_start
    h2 = altitude_end

    if zenith_angle_deg < 75:
        # Plane-parallel approximation (accurate for small angles)
        return (h2 - h1) / np.cos(theta)

    # Spherical geometry
    r1 = R + h1
    r2 = R + h2

    # Using law of cosines in spherical triangle
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Path length from geometry
    # L = sqrt(r2^2 - r1^2 * sin^2(theta)) - r1 * cos(theta)
    discriminant = r2**2 - (r1 * sin_theta)**2

    if discriminant < 0:
        # Ray doesn't reach altitude_end (below tangent height)
        return np.inf

    L = np.sqrt(discriminant) - r1 * cos_theta

    return L


def tangent_height(
    observer_altitude: float,
    zenith_angle_deg: float,
    earth_radius: float = EARTH_RADIUS,
) -> float:
    """
    Calculate tangent height for a ray at given zenith angle.

    The tangent height is the minimum altitude reached by a ray.

    Parameters
    ----------
    observer_altitude : float
        Observer altitude in meters
    zenith_angle_deg : float
        Zenith angle in degrees (must be > 90 for limb viewing)
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    h_tan : float
        Tangent height in meters (negative if ray hits surface)
    """
    theta = np.radians(zenith_angle_deg)
    R = earth_radius
    h_obs = observer_altitude

    r_obs = R + h_obs

    # Tangent radius: r_tan = r_obs * sin(theta)
    r_tan = r_obs * np.sin(theta)

    h_tan = r_tan - R

    return h_tan


def line_of_sight_altitudes(
    observer_altitude: float,
    zenith_angle_deg: float,
    n_points: int = 100,
    max_altitude: float = 100000.0,
    earth_radius: float = EARTH_RADIUS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate altitudes along a line of sight.

    Parameters
    ----------
    observer_altitude : float
        Observer altitude in meters
    zenith_angle_deg : float
        Zenith angle in degrees
    n_points : int
        Number of points along path
    max_altitude : float
        Maximum altitude to consider in meters
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    distances : ndarray
        Path distances from observer in meters
    altitudes : ndarray
        Altitudes at each point in meters
    """
    theta = np.radians(zenith_angle_deg)
    R = earth_radius
    r_obs = R + observer_altitude

    # Determine path extent
    if zenith_angle_deg <= 90:
        # Upward looking
        max_distance = slant_path_length(
            observer_altitude, max_altitude, zenith_angle_deg, earth_radius
        )
    else:
        # Limb viewing - go to tangent point and back up
        h_tan = tangent_height(observer_altitude, zenith_angle_deg, earth_radius)
        if h_tan < 0:
            h_tan = 0  # Ray hits surface

        # Distance to tangent point
        r_tan = R + max(h_tan, 0)
        d_tan = np.sqrt(r_obs**2 - r_tan**2)

        # Continue to max altitude on far side
        d_far = slant_path_length(h_tan, max_altitude, 0, earth_radius)
        max_distance = d_tan + d_far

    distances = np.linspace(0, min(max_distance, 1e7), n_points)

    # Calculate altitude at each distance
    altitudes = np.zeros_like(distances)

    for i, d in enumerate(distances):
        # Using spherical geometry
        # r^2 = r_obs^2 + d^2 - 2*r_obs*d*cos(theta)
        r_squared = r_obs**2 + d**2 - 2 * r_obs * d * np.cos(theta)
        r = np.sqrt(max(r_squared, R**2))
        altitudes[i] = r - R

    return distances, altitudes


def chapman_function(
    zenith_angle_deg: float,
    altitude: float,
    scale_height: float = 8500.0,
    earth_radius: float = EARTH_RADIUS,
) -> float:
    """
    Chapman grazing incidence function.

    Calculates the enhancement factor for column density at
    high solar zenith angles due to Earth curvature.

    Parameters
    ----------
    zenith_angle_deg : float
        Solar zenith angle in degrees
    altitude : float
        Altitude in meters
    scale_height : float
        Atmospheric scale height in meters
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    ch : float
        Chapman function value

    Notes
    -----
    For small zenith angles: ch ~ sec(theta)
    For grazing incidence: ch accounts for spherical geometry

    References
    ----------
    Chapman, S. (1931). The absorption and dissociative or ionizing
    effect of monochromatic radiation in an atmosphere on a rotating
    earth. Proc. Phys. Soc. 43, 26-45.
    """
    theta = np.radians(zenith_angle_deg)
    R = earth_radius
    H = scale_height
    z = altitude

    # Parameter X = (R + z) / H
    X = (R + z) / H

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    if zenith_angle_deg < 75:
        # Simple secant for small angles
        return 1.0 / cos_theta

    elif zenith_angle_deg < 90:
        # Spherical correction
        y = np.sqrt(X / 2) * np.abs(cos_theta)

        # Chapman function approximation (Swider, 1964)
        if y < 8:
            # erfc approximation
            ch = np.sqrt(np.pi * X / 2) * np.exp(y**2) * (1 - erf(y))
        else:
            # Asymptotic expansion
            ch = 1.0 / (y * np.sqrt(np.pi))

        return ch

    else:
        # Grazing/setting sun (theta > 90)
        y = np.sqrt(X / 2) * np.abs(cos_theta)

        # Double Chapman for grazing incidence
        ch_90 = np.sqrt(np.pi * X / 2)

        if y < 8:
            ch = 2 * ch_90 * np.exp(X * (1 - sin_theta)) - \
                 np.sqrt(np.pi * X / 2) * np.exp(y**2) * (1 - erf(y))
        else:
            ch = 2 * ch_90 * np.exp(X * (1 - sin_theta))

        return max(ch, 1.0)


# =============================================================================
# Solar Geometry
# =============================================================================

def solar_zenith_angle(
    latitude_deg: float,
    longitude_deg: float,
    datetime_utc: datetime,
) -> float:
    """
    Calculate solar zenith angle for a given location and time.

    Parameters
    ----------
    latitude_deg : float
        Latitude in degrees (-90 to 90)
    longitude_deg : float
        Longitude in degrees (-180 to 180)
    datetime_utc : datetime
        UTC datetime

    Returns
    -------
    sza : float
        Solar zenith angle in degrees
    """
    # Ensure UTC
    if datetime_utc.tzinfo is None:
        datetime_utc = datetime_utc.replace(tzinfo=timezone.utc)

    lat = np.radians(latitude_deg)
    lon = np.radians(longitude_deg)

    # Day of year
    day_of_year = datetime_utc.timetuple().tm_yday

    # Hour angle
    hour = datetime_utc.hour + datetime_utc.minute / 60 + datetime_utc.second / 3600
    hour_angle = np.radians(15 * (hour - 12) + longitude_deg)

    # Solar declination (approximate)
    declination = np.radians(23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81))))

    # Solar zenith angle
    cos_sza = (np.sin(lat) * np.sin(declination) +
               np.cos(lat) * np.cos(declination) * np.cos(hour_angle))

    sza = np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))

    return sza


def solar_azimuth_angle(
    latitude_deg: float,
    longitude_deg: float,
    datetime_utc: datetime,
) -> float:
    """
    Calculate solar azimuth angle for a given location and time.

    Parameters
    ----------
    latitude_deg : float
        Latitude in degrees
    longitude_deg : float
        Longitude in degrees
    datetime_utc : datetime
        UTC datetime

    Returns
    -------
    azimuth : float
        Solar azimuth angle in degrees (0 = North, 90 = East)
    """
    if datetime_utc.tzinfo is None:
        datetime_utc = datetime_utc.replace(tzinfo=timezone.utc)

    lat = np.radians(latitude_deg)

    day_of_year = datetime_utc.timetuple().tm_yday
    hour = datetime_utc.hour + datetime_utc.minute / 60 + datetime_utc.second / 3600
    hour_angle = np.radians(15 * (hour - 12) + longitude_deg)

    declination = np.radians(23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81))))

    # Calculate azimuth
    sza = np.radians(solar_zenith_angle(latitude_deg, longitude_deg, datetime_utc))

    sin_azimuth = np.cos(declination) * np.sin(hour_angle) / np.sin(sza)
    cos_azimuth = (np.sin(declination) - np.sin(lat) * np.cos(sza)) / \
                  (np.cos(lat) * np.sin(sza))

    azimuth = np.degrees(np.arctan2(sin_azimuth, cos_azimuth))

    # Convert to 0-360 range
    if azimuth < 0:
        azimuth += 360

    return azimuth


def air_mass_kasten(zenith_angle_deg: float) -> float:
    """
    Calculate air mass using Kasten & Young (1989) formula.

    Accurate for solar zenith angles up to 90 degrees.

    Parameters
    ----------
    zenith_angle_deg : float
        Solar zenith angle in degrees

    Returns
    -------
    air_mass : float
        Relative air mass

    References
    ----------
    Kasten, F. & Young, A.T. (1989). Revised optical air mass tables
    and approximation formula. Applied Optics 28(22), 4735-4738.
    """
    if zenith_angle_deg >= 90:
        return 40.0  # Approximate maximum

    theta = zenith_angle_deg

    # Kasten-Young formula
    air_mass = 1.0 / (np.cos(np.radians(theta)) +
                       0.50572 * (96.07995 - theta)**(-1.6364))

    return air_mass


def refracted_solar_zenith(
    apparent_zenith_deg: float,
    altitude: float = 0,
    temperature: float = 288.15,
    pressure: float = 101325,
) -> float:
    """
    Calculate true solar zenith angle accounting for atmospheric refraction.

    Parameters
    ----------
    apparent_zenith_deg : float
        Apparent (observed) zenith angle in degrees
    altitude : float
        Observer altitude in meters
    temperature : float
        Temperature in Kelvin
    pressure : float
        Pressure in Pa

    Returns
    -------
    true_zenith : float
        True (geometric) zenith angle in degrees

    Notes
    -----
    Refraction makes the sun appear higher than it actually is.
    Effect is largest near horizon (~0.5 degrees at sunset).
    """
    # Standard refraction correction (arcminutes)
    z = apparent_zenith_deg

    if z > 90.5:
        return z  # Below horizon, no correction

    # Bennett's formula for refraction in arcminutes
    h = 90 - z  # Elevation angle

    if h < 0:
        h = 0

    # Refraction in arcminutes
    R = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))

    # Pressure and temperature correction
    R *= (pressure / 101325) * (283 / temperature)

    # True zenith = apparent zenith + refraction
    true_zenith = z + R / 60  # Convert arcmin to degrees

    return true_zenith


def sunrise_sunset_times(
    latitude_deg: float,
    longitude_deg: float,
    date: datetime,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Calculate sunrise and sunset times for a location.

    Parameters
    ----------
    latitude_deg : float
        Latitude in degrees
    longitude_deg : float
        Longitude in degrees
    date : datetime
        Date for calculation

    Returns
    -------
    sunrise : datetime or None
        Sunrise time in UTC (None for polar day)
    sunset : datetime or None
        Sunset time in UTC (None for polar night)
    """
    lat = np.radians(latitude_deg)

    day_of_year = date.timetuple().tm_yday
    declination = np.radians(23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81))))

    # Hour angle at sunrise/sunset (cos(h) when sza = 90)
    cos_h = -np.tan(lat) * np.tan(declination)

    if cos_h < -1:
        # Polar day - sun never sets
        return None, None
    elif cos_h > 1:
        # Polar night - sun never rises
        return None, None

    hour_angle = np.degrees(np.arccos(cos_h))

    # Solar noon in hours (local time)
    solar_noon = 12 - longitude_deg / 15

    # Sunrise and sunset hours (UTC)
    sunrise_hour = solar_noon - hour_angle / 15
    sunset_hour = solar_noon + hour_angle / 15

    # Create datetime objects
    base_date = date.replace(hour=0, minute=0, second=0, microsecond=0)

    def hours_to_datetime(hours):
        total_seconds = int(hours * 3600)
        return base_date + timedelta(seconds=total_seconds)

    from datetime import timedelta

    sunrise = base_date + timedelta(hours=sunrise_hour)
    sunset = base_date + timedelta(hours=sunset_hour)

    return sunrise, sunset


# =============================================================================
# Coordinate Transforms
# =============================================================================

def geodetic_to_ecef(
    latitude_deg: float,
    longitude_deg: float,
    altitude: float,
) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates to Earth-Centered Earth-Fixed (ECEF).

    Parameters
    ----------
    latitude_deg : float
        Geodetic latitude in degrees
    longitude_deg : float
        Longitude in degrees
    altitude : float
        Altitude above ellipsoid in meters

    Returns
    -------
    x, y, z : tuple of float
        ECEF coordinates in meters
    """
    lat = np.radians(latitude_deg)
    lon = np.radians(longitude_deg)

    a = EARTH_RADIUS_EQUATORIAL
    e2 = 1 - (EARTH_RADIUS_POLAR / a)**2

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    x = (N + altitude) * np.cos(lat) * np.cos(lon)
    y = (N + altitude) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + altitude) * np.sin(lat)

    return x, y, z


def ecef_to_geodetic(
    x: float,
    y: float,
    z: float,
) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic.

    Parameters
    ----------
    x, y, z : float
        ECEF coordinates in meters

    Returns
    -------
    latitude : float
        Geodetic latitude in degrees
    longitude : float
        Longitude in degrees
    altitude : float
        Altitude above ellipsoid in meters
    """
    a = EARTH_RADIUS_EQUATORIAL
    b = EARTH_RADIUS_POLAR
    e2 = 1 - (b / a)**2
    ep2 = (a / b)**2 - 1

    lon = np.degrees(np.arctan2(y, x))

    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)

    lat = np.arctan2(
        z + ep2 * b * np.sin(theta)**3,
        p - e2 * a * np.cos(theta)**3
    )

    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), lon, alt


def local_to_ecef(
    east: float,
    north: float,
    up: float,
    ref_latitude_deg: float,
    ref_longitude_deg: float,
    ref_altitude: float,
) -> Tuple[float, float, float]:
    """
    Convert local East-North-Up (ENU) coordinates to ECEF.

    Parameters
    ----------
    east, north, up : float
        Local ENU coordinates in meters
    ref_latitude_deg : float
        Reference point latitude in degrees
    ref_longitude_deg : float
        Reference point longitude in degrees
    ref_altitude : float
        Reference point altitude in meters

    Returns
    -------
    x, y, z : tuple of float
        ECEF coordinates in meters
    """
    lat = np.radians(ref_latitude_deg)
    lon = np.radians(ref_longitude_deg)

    # Rotation matrix from ENU to ECEF
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # Transform
    dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    dy = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    dz = cos_lat * north + sin_lat * up

    # Reference point in ECEF
    x0, y0, z0 = geodetic_to_ecef(ref_latitude_deg, ref_longitude_deg, ref_altitude)

    return x0 + dx, y0 + dy, z0 + dz


def viewing_geometry(
    observer_lat: float,
    observer_lon: float,
    observer_alt: float,
    target_lat: float,
    target_lon: float,
    target_alt: float,
) -> Tuple[float, float, float]:
    """
    Calculate viewing geometry between observer and target.

    Parameters
    ----------
    observer_lat, observer_lon : float
        Observer geodetic coordinates in degrees
    observer_alt : float
        Observer altitude in meters
    target_lat, target_lon : float
        Target geodetic coordinates in degrees
    target_alt : float
        Target altitude in meters

    Returns
    -------
    zenith_angle : float
        Zenith angle to target in degrees
    azimuth : float
        Azimuth to target in degrees (0 = North)
    distance : float
        Slant range distance in meters
    """
    # Convert to ECEF
    x1, y1, z1 = geodetic_to_ecef(observer_lat, observer_lon, observer_alt)
    x2, y2, z2 = geodetic_to_ecef(target_lat, target_lon, target_alt)

    # Vector from observer to target
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    # Local up vector at observer
    lat = np.radians(observer_lat)
    lon = np.radians(observer_lon)

    up_x = np.cos(lat) * np.cos(lon)
    up_y = np.cos(lat) * np.sin(lon)
    up_z = np.sin(lat)

    # Zenith angle
    cos_zenith = (dx * up_x + dy * up_y + dz * up_z) / distance
    zenith_angle = np.degrees(np.arccos(np.clip(cos_zenith, -1, 1)))

    # Local East and North vectors
    east_x = -np.sin(lon)
    east_y = np.cos(lon)
    east_z = 0

    north_x = -np.sin(lat) * np.cos(lon)
    north_y = -np.sin(lat) * np.sin(lon)
    north_z = np.cos(lat)

    # Project to horizontal plane
    east_comp = dx * east_x + dy * east_y + dz * east_z
    north_comp = dx * north_x + dy * north_y + dz * north_z

    azimuth = np.degrees(np.arctan2(east_comp, north_comp))
    if azimuth < 0:
        azimuth += 360

    return zenith_angle, azimuth, distance
