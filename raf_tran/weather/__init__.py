"""
Weather Data Integration Module
===============================

This module provides interfaces for fetching and using meteorological
data from various sources for atmospheric simulations.

OFFLINE OPERATION
-----------------
All weather functions work fully offline using built-in climatological
data (US Standard Atmosphere, CIRA-86, etc.). External data sources
(ECMWF, NOAA, etc.) are OPTIONAL enhancements for real-time or
site-specific accuracy.

Data Sources (all optional, require internet)
----------------------------------------------
1. ECMWF ERA5 reanalysis
2. NOAA GFS forecasts
3. MERRA-2 reanalysis
4. User-provided local data

Built-in Offline Data
---------------------
- US Standard Atmosphere 1976
- CIRA-86 climatology
- Seasonal/latitudinal profiles
- Typical aerosol profiles

Usage
-----
>>> from raf_tran.weather import get_atmospheric_profile
>>>
>>> # Always works (offline)
>>> profile = get_atmospheric_profile(latitude=45, month=7)
>>>
>>> # Try online if available
>>> profile = get_atmospheric_profile(
...     latitude=45, longitude=-75,
...     datetime='2024-06-15T12:00:00',
...     use_online=True  # Falls back to offline if unavailable
... )
"""

from raf_tran.weather.profiles import (
    AtmosphericProfile,
    get_atmospheric_profile,
    us_standard_atmosphere,
    cira86_atmosphere,
    tropical_atmosphere,
    midlatitude_summer,
    midlatitude_winter,
    subarctic_summer,
    subarctic_winter,
)

from raf_tran.weather.online import (
    ONLINE_AVAILABLE,
    fetch_ecmwf_profile,
    fetch_gfs_profile,
    can_fetch_online,
)

__all__ = [
    # Atmospheric profiles
    "AtmosphericProfile",
    "get_atmospheric_profile",
    "us_standard_atmosphere",
    "cira86_atmosphere",
    "tropical_atmosphere",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    # Online data (optional)
    "ONLINE_AVAILABLE",
    "fetch_ecmwf_profile",
    "fetch_gfs_profile",
    "can_fetch_online",
]


def can_run_offline() -> bool:
    """
    Check if weather module can run offline.

    Returns
    -------
    bool
        Always True - offline operation is always supported
    """
    return True
