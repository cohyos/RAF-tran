"""
Data management module for RAF-Tran.

This module provides:
- Data Ingestor: ETL tool for converting HITRAN data to local HDF5 format
- Spectral Database: Interface for reading/writing spectral line data
- Offline data management utilities
"""

from raf_tran.data.ingestor import DataIngestor
from raf_tran.data.spectral_db import SpectralDatabase

__all__ = [
    "DataIngestor",
    "SpectralDatabase",
]
