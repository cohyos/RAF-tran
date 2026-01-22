"""
Configuration management for RAF-Tran simulations.

This module provides:
- SimulationConfig: Data class for simulation parameters
- ConfigurationManager: Loading and validation of configurations
- Standard atmosphere profiles
"""

from raf_tran.config.settings import SimulationConfig
from raf_tran.config.manager import ConfigurationManager
from raf_tran.config.atmosphere import AtmosphereProfile, StandardAtmospheres

__all__ = [
    "SimulationConfig",
    "ConfigurationManager",
    "AtmosphereProfile",
    "StandardAtmospheres",
]
