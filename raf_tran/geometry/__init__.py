"""
3D Geometry Module for Radiative Transfer
==========================================

This module provides 3D geometry calculations beyond plane-parallel
approximation, including:

- Spherical Earth geometry
- Line-of-sight path calculations
- Limb viewing geometry
- Slant path integration
- Earth curvature corrections

OFFLINE OPERATION
-----------------
All geometry calculations work fully offline using built-in Earth models.

Applications
------------
- Satellite remote sensing
- Limb sounding
- Aircraft-based sensors
- Long horizontal paths
- High solar zenith angle calculations

References
----------
- Liou, K.N. (2002). An Introduction to Atmospheric Radiation. Academic Press.
- Stephens, G.L. (1994). Remote Sensing of the Lower Atmosphere. Oxford.
"""

from raf_tran.geometry.spherical import (
    # Earth constants
    EARTH_RADIUS,
    EARTH_RADIUS_EQUATORIAL,
    EARTH_RADIUS_POLAR,
    earth_radius_at_latitude,
    # Path geometry
    slant_path_length,
    tangent_height,
    line_of_sight_altitudes,
    chapman_function,
    # Solar geometry
    solar_zenith_angle,
    solar_azimuth_angle,
    air_mass_kasten,
    refracted_solar_zenith,
    sunrise_sunset_times,
    # Coordinate transforms
    geodetic_to_ecef,
    ecef_to_geodetic,
    local_to_ecef,
    viewing_geometry,
)

from raf_tran.geometry.paths import (
    # Path integration
    PathSegment,
    SlantPath,
    create_vertical_path,
    create_slant_path,
    create_limb_path,
    create_ground_to_space_path,
    integrate_along_path,
    # Ray tracing
    trace_ray,
    refracted_path,
)

__all__ = [
    # Earth constants
    "EARTH_RADIUS",
    "EARTH_RADIUS_EQUATORIAL",
    "EARTH_RADIUS_POLAR",
    "earth_radius_at_latitude",
    # Path geometry
    "slant_path_length",
    "tangent_height",
    "line_of_sight_altitudes",
    "chapman_function",
    # Solar geometry
    "solar_zenith_angle",
    "solar_azimuth_angle",
    "air_mass_kasten",
    "refracted_solar_zenith",
    "sunrise_sunset_times",
    # Coordinate transforms
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "local_to_ecef",
    "viewing_geometry",
    # Path classes
    "PathSegment",
    "SlantPath",
    "create_vertical_path",
    "create_slant_path",
    "create_limb_path",
    "create_ground_to_space_path",
    "integrate_along_path",
    # Ray tracing
    "trace_ray",
    "refracted_path",
]
