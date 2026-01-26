"""
Path Integration and Ray Tracing
================================

This module provides tools for defining and integrating along
atmospheric paths, including slant paths and limb viewing geometry.

All calculations work fully offline.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union

from raf_tran.geometry.spherical import (
    EARTH_RADIUS,
    slant_path_length,
    tangent_height,
    line_of_sight_altitudes,
)


@dataclass
class PathSegment:
    """
    A segment of an atmospheric path.

    Attributes
    ----------
    altitude_start : float
        Starting altitude in meters
    altitude_end : float
        Ending altitude in meters
    path_length : float
        Geometric path length of segment in meters
    zenith_angle : float
        Local zenith angle in degrees
    pressure_start : float, optional
        Starting pressure in Pa
    pressure_end : float, optional
        Ending pressure in Pa
    temperature : float, optional
        Mean temperature in K
    """
    altitude_start: float
    altitude_end: float
    path_length: float
    zenith_angle: float
    pressure_start: Optional[float] = None
    pressure_end: Optional[float] = None
    temperature: Optional[float] = None

    @property
    def altitude_mid(self) -> float:
        """Middle altitude of segment."""
        return (self.altitude_start + self.altitude_end) / 2

    @property
    def altitude_thickness(self) -> float:
        """Vertical thickness of segment."""
        return abs(self.altitude_end - self.altitude_start)


@dataclass
class SlantPath:
    """
    Complete slant path through the atmosphere.

    Attributes
    ----------
    segments : list of PathSegment
        Path segments from observer to end
    observer_altitude : float
        Observer altitude in meters
    zenith_angle : float
        Initial zenith angle in degrees
    total_path_length : float
        Total geometric path length in meters
    is_limb : bool
        True if this is a limb-viewing path
    tangent_altitude : float, optional
        Tangent altitude for limb paths in meters
    """
    segments: List[PathSegment]
    observer_altitude: float
    zenith_angle: float
    total_path_length: float = 0.0
    is_limb: bool = False
    tangent_altitude: Optional[float] = None

    def __post_init__(self):
        if self.total_path_length == 0:
            self.total_path_length = sum(s.path_length for s in self.segments)

    @property
    def altitudes(self) -> np.ndarray:
        """Array of segment boundary altitudes."""
        alts = [self.segments[0].altitude_start]
        for seg in self.segments:
            alts.append(seg.altitude_end)
        return np.array(alts)

    @property
    def cumulative_path(self) -> np.ndarray:
        """Cumulative path length at segment boundaries."""
        cum = [0.0]
        for seg in self.segments:
            cum.append(cum[-1] + seg.path_length)
        return np.array(cum)


def create_vertical_path(
    surface_altitude: float = 0.0,
    top_altitude: float = 100000.0,
    n_layers: int = 50,
    scale_height: float = 8500.0,
) -> SlantPath:
    """
    Create a vertical (nadir) path through the atmosphere.

    Parameters
    ----------
    surface_altitude : float
        Surface altitude in meters
    top_altitude : float
        Top of atmosphere altitude in meters
    n_layers : int
        Number of atmospheric layers
    scale_height : float
        Scale height for layer spacing in meters

    Returns
    -------
    path : SlantPath
        Vertical atmospheric path
    """
    # Use exponential spacing for better resolution near surface
    z_norm = np.linspace(0, 1, n_layers + 1)
    altitudes = surface_altitude + (top_altitude - surface_altitude) * \
                (1 - np.exp(-3 * z_norm)) / (1 - np.exp(-3))

    segments = []
    for i in range(n_layers):
        seg = PathSegment(
            altitude_start=altitudes[i],
            altitude_end=altitudes[i + 1],
            path_length=altitudes[i + 1] - altitudes[i],
            zenith_angle=0.0,
        )
        segments.append(seg)

    return SlantPath(
        segments=segments,
        observer_altitude=surface_altitude,
        zenith_angle=0.0,
    )


def create_slant_path(
    observer_altitude: float,
    zenith_angle_deg: float,
    top_altitude: float = 100000.0,
    n_layers: int = 50,
    earth_radius: float = EARTH_RADIUS,
) -> SlantPath:
    """
    Create a slant path at a given zenith angle.

    Parameters
    ----------
    observer_altitude : float
        Observer altitude in meters
    zenith_angle_deg : float
        Zenith angle in degrees (0 = vertical, 90 = horizontal)
    top_altitude : float
        Top of atmosphere altitude in meters
    n_layers : int
        Number of atmospheric layers
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    path : SlantPath
        Slant atmospheric path

    Notes
    -----
    For zenith angles < 90, path goes from observer to TOA.
    For zenith angles >= 90, use create_limb_path instead.
    """
    if zenith_angle_deg >= 90:
        raise ValueError(
            "For zenith angles >= 90, use create_limb_path instead"
        )

    theta = np.radians(zenith_angle_deg)

    # Create altitude grid with better resolution near surface
    z_norm = np.linspace(0, 1, n_layers + 1)
    altitudes = observer_altitude + (top_altitude - observer_altitude) * \
                (1 - np.exp(-3 * z_norm)) / (1 - np.exp(-3))

    segments = []
    for i in range(n_layers):
        h1 = altitudes[i]
        h2 = altitudes[i + 1]

        # Slant path length for this layer
        if zenith_angle_deg < 75:
            # Plane-parallel approximation
            pl = (h2 - h1) / np.cos(theta)
            local_zenith = zenith_angle_deg
        else:
            # Spherical geometry
            pl = slant_path_length(h1, h2, zenith_angle_deg, earth_radius)

            # Local zenith angle changes along path due to curvature
            r1 = earth_radius + h1
            r_mid = earth_radius + (h1 + h2) / 2

            # Snell's law for refractive index = 1
            sin_local = r1 * np.sin(theta) / r_mid
            local_zenith = np.degrees(np.arcsin(np.clip(sin_local, -1, 1)))

        seg = PathSegment(
            altitude_start=h1,
            altitude_end=h2,
            path_length=pl,
            zenith_angle=local_zenith,
        )
        segments.append(seg)

    return SlantPath(
        segments=segments,
        observer_altitude=observer_altitude,
        zenith_angle=zenith_angle_deg,
    )


def create_limb_path(
    observer_altitude: float,
    tangent_alt: float,
    top_altitude: float = 100000.0,
    n_layers_per_side: int = 25,
    earth_radius: float = EARTH_RADIUS,
) -> SlantPath:
    """
    Create a limb-viewing path through the atmosphere.

    Parameters
    ----------
    observer_altitude : float
        Observer (e.g., satellite) altitude in meters
    tangent_alt : float
        Tangent point altitude in meters
    top_altitude : float
        Top of atmosphere altitude in meters
    n_layers_per_side : int
        Number of layers on each side of tangent point
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    path : SlantPath
        Limb viewing path (goes down to tangent, then up to TOA)

    Notes
    -----
    The path is symmetric about the tangent point. Segments are
    ordered from observer -> tangent -> far limb (TOA).
    """
    R = earth_radius
    r_obs = R + observer_altitude
    r_tan = R + tangent_alt

    # Calculate zenith angle for this tangent altitude
    sin_zenith = r_tan / r_obs
    if sin_zenith > 1:
        raise ValueError(
            f"Tangent altitude {tangent_alt}m is above observer altitude {observer_altitude}m"
        )
    zenith_angle_deg = np.degrees(np.arcsin(sin_zenith))
    zenith_angle_deg = 180 - zenith_angle_deg  # Limb viewing is > 90 deg

    # Create altitude grid from tangent to TOA
    z_norm = np.linspace(0, 1, n_layers_per_side + 1)
    altitudes_up = tangent_alt + (top_altitude - tangent_alt) * \
                   (1 - np.exp(-3 * z_norm)) / (1 - np.exp(-3))

    # Path from observer down to tangent
    segments_down = []
    altitudes_down = np.linspace(observer_altitude, tangent_alt, n_layers_per_side + 1)

    for i in range(n_layers_per_side):
        h1 = altitudes_down[i]
        h2 = altitudes_down[i + 1]

        r1 = R + h1
        r2 = R + h2

        # Path length using spherical geometry
        # From triangle: d^2 = r1^2 + r2^2 - 2*r1*r2*cos(angle)
        # Where angle is the Earth-central angle
        cos_angle = (r1**2 + r_tan**2 - r2**2) / (2 * r1 * r_tan) if i > 0 else 1

        # Simplified path length calculation
        pl = np.sqrt(r1**2 - r_tan**2) - np.sqrt(r2**2 - r_tan**2)

        seg = PathSegment(
            altitude_start=h1,
            altitude_end=h2,
            path_length=abs(pl),
            zenith_angle=zenith_angle_deg,
        )
        segments_down.append(seg)

    # Path from tangent up to TOA (far side)
    segments_up = []
    for i in range(n_layers_per_side):
        h1 = altitudes_up[i]
        h2 = altitudes_up[i + 1]

        r1 = R + h1
        r2 = R + h2

        pl = np.sqrt(r2**2 - r_tan**2) - np.sqrt(r1**2 - r_tan**2)

        seg = PathSegment(
            altitude_start=h1,
            altitude_end=h2,
            path_length=abs(pl),
            zenith_angle=180 - zenith_angle_deg,  # Upward on far side
        )
        segments_up.append(seg)

    # Combine: observer -> tangent -> TOA
    all_segments = segments_down + segments_up

    return SlantPath(
        segments=all_segments,
        observer_altitude=observer_altitude,
        zenith_angle=zenith_angle_deg,
        is_limb=True,
        tangent_altitude=tangent_alt,
    )


def create_ground_to_space_path(
    surface_altitude: float,
    satellite_altitude: float,
    zenith_angle_deg: float,
    n_layers: int = 50,
    earth_radius: float = EARTH_RADIUS,
) -> SlantPath:
    """
    Create a path from ground to a satellite.

    Parameters
    ----------
    surface_altitude : float
        Ground/surface altitude in meters
    satellite_altitude : float
        Satellite altitude in meters
    zenith_angle_deg : float
        Zenith angle to satellite in degrees
    n_layers : int
        Number of atmospheric layers
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    path : SlantPath
        Ground-to-space atmospheric path
    """
    return create_slant_path(
        observer_altitude=surface_altitude,
        zenith_angle_deg=zenith_angle_deg,
        top_altitude=satellite_altitude,
        n_layers=n_layers,
        earth_radius=earth_radius,
    )


def integrate_along_path(
    path: SlantPath,
    quantity_profile: Callable[[float], float],
    method: str = 'trapezoid',
) -> float:
    """
    Integrate a quantity along an atmospheric path.

    Parameters
    ----------
    path : SlantPath
        Atmospheric path to integrate along
    quantity_profile : callable
        Function that returns quantity value at a given altitude
        Signature: quantity(altitude_m) -> value
    method : str
        Integration method: 'trapezoid', 'midpoint', 'simpson'

    Returns
    -------
    integral : float
        Path-integrated quantity

    Examples
    --------
    >>> # Integrate extinction coefficient
    >>> def extinction(h):
    ...     return 0.1 * np.exp(-h / 8000)
    >>> tau = integrate_along_path(path, extinction)
    """
    integral = 0.0

    for seg in path.segments:
        if method == 'trapezoid':
            q_start = quantity_profile(seg.altitude_start)
            q_end = quantity_profile(seg.altitude_end)
            seg_integral = 0.5 * (q_start + q_end) * seg.path_length

        elif method == 'midpoint':
            q_mid = quantity_profile(seg.altitude_mid)
            seg_integral = q_mid * seg.path_length

        elif method == 'simpson':
            q_start = quantity_profile(seg.altitude_start)
            q_mid = quantity_profile(seg.altitude_mid)
            q_end = quantity_profile(seg.altitude_end)
            seg_integral = (q_start + 4 * q_mid + q_end) / 6 * seg.path_length

        else:
            raise ValueError(f"Unknown method: {method}")

        integral += seg_integral

    return integral


def trace_ray(
    start_position: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    max_distance: float = 1e7,
    n_steps: int = 1000,
    earth_radius: float = EARTH_RADIUS,
    min_altitude: float = 0.0,
    max_altitude: float = 100000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trace a ray through the atmosphere (no refraction).

    Parameters
    ----------
    start_position : tuple
        Starting position (x, y, z) in ECEF coordinates (meters)
    direction : tuple
        Ray direction unit vector (dx, dy, dz)
    max_distance : float
        Maximum ray tracing distance in meters
    n_steps : int
        Number of ray tracing steps
    earth_radius : float
        Earth radius in meters
    min_altitude : float
        Stop if ray drops below this altitude
    max_altitude : float
        Stop if ray rises above this altitude

    Returns
    -------
    positions : ndarray
        Ray positions, shape (n_points, 3)
    altitudes : ndarray
        Altitude at each position
    """
    pos = np.array(start_position, dtype=float)
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    step_size = max_distance / n_steps

    positions = [pos.copy()]
    altitudes = [np.linalg.norm(pos) - earth_radius]

    for _ in range(n_steps):
        pos = pos + direction * step_size

        r = np.linalg.norm(pos)
        alt = r - earth_radius

        positions.append(pos.copy())
        altitudes.append(alt)

        # Check termination conditions
        if alt < min_altitude or alt > max_altitude:
            break

    return np.array(positions), np.array(altitudes)


def refracted_path(
    observer_altitude: float,
    zenith_angle_deg: float,
    wavelength_um: float = 0.5,
    top_altitude: float = 100000.0,
    n_layers: int = 100,
    earth_radius: float = EARTH_RADIUS,
) -> SlantPath:
    """
    Create a path accounting for atmospheric refraction.

    Parameters
    ----------
    observer_altitude : float
        Observer altitude in meters
    zenith_angle_deg : float
        Initial zenith angle in degrees
    wavelength_um : float
        Wavelength in micrometers (affects refractive index)
    top_altitude : float
        Top of atmosphere in meters
    n_layers : int
        Number of layers for integration
    earth_radius : float
        Earth radius in meters

    Returns
    -------
    path : SlantPath
        Refracted atmospheric path

    Notes
    -----
    Uses Snell's law with atmospheric refractive index profile.
    Refraction is most significant near the horizon.
    """
    # Refractive index at sea level (approximate)
    n_0 = 1.000293  # At 589 nm

    # Wavelength-dependent adjustment (Cauchy formula approximation)
    n_0 = 1 + (n_0 - 1) * (0.589 / wavelength_um)**2

    # Scale height for refractivity
    H_n = 8000.0  # meters

    R = earth_radius
    theta = np.radians(zenith_angle_deg)

    # Create altitude grid
    altitudes = np.linspace(observer_altitude, top_altitude, n_layers + 1)

    segments = []

    # Track cumulative bending
    sin_theta_0 = np.sin(theta)
    r_0 = R + observer_altitude
    n_at_observer = 1 + (n_0 - 1) * np.exp(-observer_altitude / H_n)

    for i in range(n_layers):
        h1 = altitudes[i]
        h2 = altitudes[i + 1]

        # Refractive index at layer midpoint
        h_mid = (h1 + h2) / 2
        n_layer = 1 + (n_0 - 1) * np.exp(-h_mid / H_n)

        # Apply Snell's law (in spherical coordinates)
        # n_0 * r_0 * sin(theta_0) = n * r * sin(theta)
        r_mid = R + h_mid

        sin_theta_local = n_at_observer * r_0 * sin_theta_0 / (n_layer * r_mid)

        if abs(sin_theta_local) > 1:
            # Total internal reflection - ray bends back
            break

        theta_local = np.arcsin(sin_theta_local)

        # Path length through layer (approximate)
        if np.cos(theta_local) > 0.01:
            pl = (h2 - h1) / np.cos(theta_local)
        else:
            # Near horizontal - use spherical formula
            r1 = R + h1
            r2 = R + h2
            pl = np.sqrt(r2**2 - (r1 * np.sin(theta_local))**2) - \
                 r1 * np.cos(theta_local)

        seg = PathSegment(
            altitude_start=h1,
            altitude_end=h2,
            path_length=abs(pl),
            zenith_angle=np.degrees(theta_local),
        )
        segments.append(seg)

    # Calculate effective zenith angle at TOA
    if segments:
        final_zenith = segments[-1].zenith_angle
    else:
        final_zenith = zenith_angle_deg

    return SlantPath(
        segments=segments,
        observer_altitude=observer_altitude,
        zenith_angle=zenith_angle_deg,
    )
