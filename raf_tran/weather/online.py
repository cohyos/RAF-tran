"""
Online Weather Data Sources (Optional)
======================================

This module provides interfaces for fetching weather data from
online sources. All functions are OPTIONAL and require internet.

The simulation works fully offline without this module.

Supported Sources
-----------------
- ECMWF ERA5 (reanalysis)
- NOAA GFS (forecast)
- MERRA-2 (reanalysis)

Installation
------------
To use online data, install optional dependencies:
    pip install requests cdsapi netCDF4

Or install RAF-tran with the weather extra:
    pip install raf-tran[weather]
"""

import warnings
from typing import Optional
from datetime import datetime

import numpy as np

# Check for optional dependencies
REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass

CDSAPI_AVAILABLE = False
try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    pass

NETCDF4_AVAILABLE = False
try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    pass

# Overall online availability
ONLINE_AVAILABLE = REQUESTS_AVAILABLE


def can_fetch_online() -> bool:
    """
    Check if online data fetching is possible.

    Returns
    -------
    bool
        True if requests library is available
    """
    return ONLINE_AVAILABLE


def fetch_ecmwf_profile(
    latitude: float,
    longitude: float,
    datetime_str: str,
    product: str = "reanalysis-era5-pressure-levels",
    timeout: float = 30.0,
):
    """
    Fetch atmospheric profile from ECMWF ERA5 (requires CDS API key).

    OPTIONAL: Requires internet and CDS API credentials.

    Parameters
    ----------
    latitude : float
        Latitude in degrees
    longitude : float
        Longitude in degrees
    datetime_str : str
        ISO format datetime (e.g., '2024-06-15T12:00:00')
    product : str
        ERA5 product name
    timeout : float
        Request timeout in seconds

    Returns
    -------
    profile : AtmosphericProfile or None
        Profile if successful, None if unavailable

    Notes
    -----
    Requires CDS API credentials. Set up at:
    https://cds.climate.copernicus.eu/api-how-to

    Create ~/.cdsapirc with your API key.
    """
    if not CDSAPI_AVAILABLE:
        warnings.warn(
            "cdsapi not available. Install with: pip install cdsapi\n"
            "Using offline climatological profile instead."
        )
        return None

    try:
        # Parse datetime
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))

        # Initialize CDS API client
        c = cdsapi.Client()

        # Request parameters
        # This is a placeholder - actual implementation would download
        # and process the data
        warnings.warn(
            "ECMWF ERA5 fetch is a placeholder. "
            "Requires proper CDS API setup and GRIB/NetCDF processing."
        )
        return None

    except Exception as e:
        warnings.warn(f"ECMWF fetch failed: {e}. Using offline profile.")
        return None


def fetch_gfs_profile(
    latitude: float,
    longitude: float,
    datetime_str: str,
    timeout: float = 30.0,
):
    """
    Fetch atmospheric profile from NOAA GFS (open access).

    OPTIONAL: Requires internet connection.

    Parameters
    ----------
    latitude : float
        Latitude in degrees
    longitude : float
        Longitude in degrees
    datetime_str : str
        ISO format datetime
    timeout : float
        Request timeout in seconds

    Returns
    -------
    profile : AtmosphericProfile or None
        Profile if successful, None if unavailable

    Notes
    -----
    GFS data is available at:
    https://nomads.ncep.noaa.gov/

    Data is free but requires GRIB processing.
    """
    if not REQUESTS_AVAILABLE:
        warnings.warn(
            "requests library not available. Install with: pip install requests\n"
            "Using offline climatological profile instead."
        )
        return None

    try:
        # Parse datetime
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))

        # NOAA NOMADS endpoint
        # This is a placeholder - actual implementation would need proper
        # GRIB handling and coordinate interpolation
        warnings.warn(
            "GFS fetch is a placeholder. "
            "Requires GRIB file processing (pygrib or cfgrib)."
        )
        return None

    except Exception as e:
        warnings.warn(f"GFS fetch failed: {e}. Using offline profile.")
        return None


def fetch_merra2_profile(
    latitude: float,
    longitude: float,
    datetime_str: str,
    timeout: float = 30.0,
):
    """
    Fetch atmospheric profile from NASA MERRA-2 reanalysis.

    OPTIONAL: Requires internet and NASA Earthdata credentials.

    Parameters
    ----------
    latitude : float
        Latitude in degrees
    longitude : float
        Longitude in degrees
    datetime_str : str
        ISO format datetime
    timeout : float
        Request timeout in seconds

    Returns
    -------
    profile : AtmosphericProfile or None
        Profile if successful, None if unavailable

    Notes
    -----
    Requires NASA Earthdata login. Register at:
    https://urs.earthdata.nasa.gov/
    """
    if not REQUESTS_AVAILABLE or not NETCDF4_AVAILABLE:
        warnings.warn(
            "requests and/or netCDF4 not available. "
            "Install with: pip install requests netCDF4\n"
            "Using offline climatological profile instead."
        )
        return None

    try:
        warnings.warn(
            "MERRA-2 fetch is a placeholder. "
            "Requires NASA Earthdata authentication and OPeNDAP handling."
        )
        return None

    except Exception as e:
        warnings.warn(f"MERRA-2 fetch failed: {e}. Using offline profile.")
        return None


def get_available_sources() -> dict:
    """
    Get information about available online data sources.

    Returns
    -------
    sources : dict
        Dictionary with source name and availability status
    """
    return {
        "ecmwf_era5": {
            "available": CDSAPI_AVAILABLE,
            "description": "ECMWF ERA5 reanalysis",
            "requires": "cdsapi, CDS API credentials",
            "url": "https://cds.climate.copernicus.eu/",
        },
        "noaa_gfs": {
            "available": REQUESTS_AVAILABLE,
            "description": "NOAA GFS forecast",
            "requires": "requests, pygrib or cfgrib",
            "url": "https://nomads.ncep.noaa.gov/",
        },
        "nasa_merra2": {
            "available": REQUESTS_AVAILABLE and NETCDF4_AVAILABLE,
            "description": "NASA MERRA-2 reanalysis",
            "requires": "requests, netCDF4, NASA Earthdata credentials",
            "url": "https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/",
        },
    }


def print_setup_instructions():
    """Print instructions for setting up online data sources."""
    print("""
Online Weather Data Setup Instructions
======================================

RAF-tran works fully offline using built-in climatological data.
Online data sources are OPTIONAL enhancements for site-specific accuracy.

1. ECMWF ERA5 (recommended for historical data)
   - Install: pip install cdsapi
   - Register: https://cds.climate.copernicus.eu/
   - Create ~/.cdsapirc with your API key

2. NOAA GFS (for forecast data)
   - Install: pip install requests pygrib
   - No registration required
   - Data available at: https://nomads.ncep.noaa.gov/

3. NASA MERRA-2 (alternative reanalysis)
   - Install: pip install requests netCDF4
   - Register: https://urs.earthdata.nasa.gov/

Note: All online sources are optional. The simulation runs fully
offline using US Standard Atmosphere, CIRA-86, and AFGL profiles.
""")
