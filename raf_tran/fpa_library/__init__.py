"""
FPA Library - Focal Plane Array Database and Analysis
=====================================================

A modular library for comparing and analyzing infrared Focal Plane Arrays
(FPAs), Readout Integrated Circuits (ROICs), and detector technologies
from major vendors.

This library can be used standalone or as part of the RAF-tran radiative
transfer framework.

Vendors Covered
---------------
- SCD (Semi Conductor Devices)
- Teledyne FLIR
- L3Harris
- Raytheon Vision Systems
- DRS / Leonardo DRS
- Axiom Optics

Spectral Bands
--------------
- SWIR (0.9-1.7 um)
- MWIR (3.0-5.0 um)
- LWIR (8.0-14.0 um)
- Dual-band MW/LW

Key Features
------------
- Comprehensive FPA and ROIC specifications database
- Search and filter by vendor, band, cooling, resolution, etc.
- Johnson criteria DRI range calculations
- SWaP-C analysis and trade studies
- Configuration save/load (JSON)
- Visualization functions for comparison charts

Usage
-----
>>> from raf_tran.fpa_library import get_fpa_database, search_fpas
>>> from raf_tran.fpa_library.models import SpectralBand, Vendor
>>>
>>> # Get all MWIR FPAs
>>> mwir = search_fpas(spectral_band=SpectralBand.MWIR)
>>>
>>> # Get SCD portfolio
>>> scd = search_fpas(vendor=Vendor.SCD)
>>>
>>> # Compare specific FPAs
>>> from raf_tran.fpa_library.analysis import compare_fpas
>>> results = compare_fpas(mwir, focal_length_mm=100)
"""

# Models
from raf_tran.fpa_library.models import (
    FPASpec,
    ROICSpec,
    ArrayFormat,
    SpectralRange,
    SpectralBand,
    DetectorType,
    CoolingType,
    IntegrationMode,
    InterfaceType,
    ApplicationDomain,
    Vendor,
)

# Database
from raf_tran.fpa_library.database import (
    get_fpa_database,
    get_roic_database,
    get_fpa,
    get_roic,
    list_fpas,
    list_roics,
    search_fpas,
    get_vendor_portfolio,
    get_band_options,
)

# Config
from raf_tran.fpa_library.config import (
    save_fpa_config,
    load_fpa_config,
    export_database_json,
    load_custom_fpas,
    merge_databases,
    create_comparison_config,
    save_comparison_config,
    load_comparison_config,
)

# Analysis
from raf_tran.fpa_library.analysis import (
    compare_fpas,
    rank_fpas,
    compute_ifov_urad,
    compute_fov_deg,
    compute_dri_ranges,
    compute_swap_score,
    compute_sensitivity_score,
    pitch_miniaturization_factor,
    hot_reliability_gain,
    spectral_band_comparison,
)

__all__ = [
    # Models
    "FPASpec", "ROICSpec", "ArrayFormat", "SpectralRange",
    "SpectralBand", "DetectorType", "CoolingType", "IntegrationMode",
    "InterfaceType", "ApplicationDomain", "Vendor",
    # Database
    "get_fpa_database", "get_roic_database", "get_fpa", "get_roic",
    "list_fpas", "list_roics", "search_fpas",
    "get_vendor_portfolio", "get_band_options",
    # Config
    "save_fpa_config", "load_fpa_config", "export_database_json",
    "load_custom_fpas", "merge_databases",
    "create_comparison_config", "save_comparison_config", "load_comparison_config",
    # Analysis
    "compare_fpas", "rank_fpas",
    "compute_ifov_urad", "compute_fov_deg", "compute_dri_ranges",
    "compute_swap_score", "compute_sensitivity_score",
    "pitch_miniaturization_factor", "hot_reliability_gain",
    "spectral_band_comparison",
]
