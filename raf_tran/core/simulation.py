"""
Main Simulation class for RAF-Tran radiative transfer calculations.

Provides a high-level interface that orchestrates all components:
- Configuration management
- Atmosphere profile loading
- Gas absorption calculations
- Scattering calculations
- RTE integration
- Output formatting
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from raf_tran.config.settings import SimulationConfig
from raf_tran.config.manager import ConfigurationManager, LoadedConfiguration
from raf_tran.config.atmosphere import AtmosphereProfile, StandardAtmospheres
from raf_tran.core.constants import PATH_TYPES
from raf_tran.data.spectral_db import SpectralDatabase

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Complete simulation results.

    Attributes:
        wavenumber: Spectral grid [cm^-1]
        wavelength_um: Wavelength grid [um]
        transmittance: Path transmittance [0-1]
        radiance: Spectral radiance [W/(cm²·sr·cm^-1)]
        optical_depth: Total optical depth
        thermal_emission: Thermal emission contribution
        config: Configuration used for simulation
        metadata: Additional metadata about the simulation
    """
    wavenumber: np.ndarray
    wavelength_um: np.ndarray
    transmittance: np.ndarray
    radiance: np.ndarray
    optical_depth: np.ndarray
    thermal_emission: np.ndarray
    config: SimulationConfig
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "wavenumber": self.wavenumber.tolist(),
            "wavelength_um": self.wavelength_um.tolist(),
            "transmittance": self.transmittance.tolist(),
            "radiance": self.radiance.tolist(),
            "optical_depth": self.optical_depth.tolist(),
            "thermal_emission": self.thermal_emission.tolist(),
            "metadata": self.metadata,
        }


class Simulation:
    """High-level simulation interface for RAF-Tran.

    Provides a simple API for running radiative transfer simulations
    with automatic setup of all components.

    Example:
        >>> from raf_tran import Simulation
        >>> config = {
        ...     "atmosphere": {"model": "US_STANDARD_1976"},
        ...     "spectral": {"min_wavenumber": 2000, "max_wavenumber": 2500},
        ...     "geometry": {"path_type": "HORIZONTAL", "path_length_km": 1.0}
        ... }
        >>> sim = Simulation(config)
        >>> result = sim.run()
        >>> print(f"Mean transmittance: {result.transmittance.mean():.3f}")
    """

    # Default molecules to include
    DEFAULT_MOLECULES = ["H2O", "CO2", "O3", "CH4", "N2O"]

    def __init__(
        self,
        config: Dict[str, Any] | str | SimulationConfig,
        molecules: Optional[List[str]] = None,
    ):
        """Initialize the simulation.

        Args:
            config: Configuration dictionary, JSON path, or SimulationConfig
            molecules: List of molecules to include (defaults to major gases)
        """
        # Load configuration
        self.config_manager = ConfigurationManager()

        if isinstance(config, SimulationConfig):
            self.config = config
            self.atmosphere = StandardAtmospheres.get_profile(config.atmosphere.model)
            if config.atmosphere.custom_concentrations:
                self.atmosphere.apply_gas_overrides(config.atmosphere.custom_concentrations)
        elif isinstance(config, dict):
            loaded = self.config_manager.load_config(config)
            self.config = loaded.config
            self.atmosphere = loaded.atmosphere
        elif isinstance(config, str):
            loaded = self.config_manager.load_config(config)
            self.config = loaded.config
            self.atmosphere = loaded.atmosphere
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        # Set molecules
        self.molecules = molecules if molecules else self.DEFAULT_MOLECULES

        # Initialize engines lazily
        self._gas_engine = None
        self._scattering_engine = None
        self._rte_solver = None
        self._spectral_db = None

        # Validation
        errors = self.config.validate()
        if errors:
            for error in errors:
                logger.warning(f"Configuration warning: {error}")

    @property
    def gas_engine(self):
        """Get or create gas engine (lazy initialization)."""
        if self._gas_engine is None:
            from raf_tran.core.gas_engine import GasEngine
            self._gas_engine = GasEngine(
                spectral_db=self.spectral_db,
                use_gpu=self.config.system.use_gpu,
            )
        return self._gas_engine

    @property
    def scattering_engine(self):
        """Get or create scattering engine (lazy initialization)."""
        if self._scattering_engine is None:
            from raf_tran.core.scattering_engine import ScatteringEngine
            self._scattering_engine = ScatteringEngine()
        return self._scattering_engine

    @property
    def rte_solver(self):
        """Get or create RTE solver (lazy initialization)."""
        if self._rte_solver is None:
            from raf_tran.core.rte_solver import RTESolver
            self._rte_solver = RTESolver(
                gas_engine=self.gas_engine,
                scattering_engine=self.scattering_engine,
            )
        return self._rte_solver

    @property
    def spectral_db(self):
        """Get or create spectral database (lazy initialization)."""
        if self._spectral_db is None:
            db_path = Path(self.config.system.database_path)

            if not db_path.exists():
                # Try to generate synthetic database for testing
                logger.warning(
                    f"Database not found at {db_path}. "
                    "Generating synthetic database for testing."
                )
                self._create_synthetic_database(db_path)

            self._spectral_db = SpectralDatabase(str(db_path))

        return self._spectral_db

    def _create_synthetic_database(self, db_path: Path) -> None:
        """Create synthetic database if real one doesn't exist."""
        from raf_tran.data.ingestor import DataIngestor

        db_path.parent.mkdir(parents=True, exist_ok=True)
        ingestor = DataIngestor()
        ingestor.generate_synthetic_database(
            output_path=str(db_path),
            wavenumber_range=(
                self.config.spectral.min_wavenumber,
                self.config.spectral.max_wavenumber,
            ),
            molecules=self.molecules,
        )

    def run(
        self,
        include_thermal_emission: bool = True,
        include_intermediate: bool = False,
    ) -> SimulationResult:
        """Run the radiative transfer simulation.

        Args:
            include_thermal_emission: Include thermal emission from atmosphere
            include_intermediate: Save per-layer intermediate results

        Returns:
            SimulationResult with computed quantities
        """
        from raf_tran.core.rte_solver import PathGeometry

        logger.info(
            f"Running simulation: {self.config.atmosphere.model}, "
            f"{self.config.spectral.min_wavenumber}-{self.config.spectral.max_wavenumber} cm^-1"
        )

        # Build path geometry
        geometry = PathGeometry(
            path_type=self.config.geometry.path_type,
            h1_km=self.config.geometry.h1_km,
            h2_km=self.config.geometry.h2_km,
            zenith_angle_deg=self.config.geometry.angle_deg,
            path_length_km=self.config.geometry.path_length_km,
            include_earth_curvature=self.config.geometry.earth_curvature,
        )

        # Run RTE solver
        rte_result = self.rte_solver.solve(
            wavenumber_range=(
                self.config.spectral.min_wavenumber,
                self.config.spectral.max_wavenumber,
            ),
            atmosphere=self.atmosphere,
            geometry=geometry,
            molecules=self.molecules,
            aerosol_type=self.config.atmosphere.aerosols.type,
            visibility_km=self.config.atmosphere.aerosols.visibility_km,
            resolution=self.config.spectral.resolution,
            include_thermal_emission=include_thermal_emission,
            include_intermediate=include_intermediate,
        )

        # Build metadata
        metadata = {
            "atmosphere_model": self.config.atmosphere.model,
            "molecules": self.molecules,
            "aerosol_type": self.config.atmosphere.aerosols.type,
            "visibility_km": self.config.atmosphere.aerosols.visibility_km,
            "path_type": self.config.geometry.path_type,
            "num_spectral_points": len(rte_result.wavenumber),
            "num_layers": len(self.atmosphere.layers),
        }

        return SimulationResult(
            wavenumber=rte_result.wavenumber,
            wavelength_um=rte_result.wavelength_um,
            transmittance=rte_result.transmittance,
            radiance=rte_result.radiance,
            optical_depth=rte_result.optical_depth,
            thermal_emission=rte_result.thermal_emission,
            config=self.config,
            metadata=metadata,
        )

    def compute_transmittance(
        self,
        wavenumber_range: Optional[tuple] = None,
    ) -> tuple:
        """Quick transmittance calculation.

        Args:
            wavenumber_range: Optional override for spectral range

        Returns:
            Tuple of (wavenumber, transmittance) arrays
        """
        from raf_tran.core.rte_solver import PathGeometry

        if wavenumber_range is None:
            wavenumber_range = (
                self.config.spectral.min_wavenumber,
                self.config.spectral.max_wavenumber,
            )

        geometry = PathGeometry(
            path_type=self.config.geometry.path_type,
            h1_km=self.config.geometry.h1_km,
            h2_km=self.config.geometry.h2_km,
            zenith_angle_deg=self.config.geometry.angle_deg,
            path_length_km=self.config.geometry.path_length_km,
            include_earth_curvature=self.config.geometry.earth_curvature,
        )

        return self.rte_solver.compute_transmittance_only(
            wavenumber_range=wavenumber_range,
            atmosphere=self.atmosphere,
            geometry=geometry,
            molecules=self.molecules,
            aerosol_type=self.config.atmosphere.aerosols.type,
            visibility_km=self.config.atmosphere.aerosols.visibility_km,
            resolution=self.config.spectral.resolution,
        )

    def save_result(
        self,
        result: SimulationResult,
        output_path: Optional[str] = None,
        format: Optional[str] = None,
    ) -> str:
        """Save simulation result to file.

        Args:
            result: SimulationResult to save
            output_path: Output file path (defaults to config setting)
            format: Output format (csv, json, netcdf)

        Returns:
            Path to saved file
        """
        from raf_tran.utils.output import OutputFormatter

        if format is None:
            format = self.config.output.format

        if output_path is None:
            output_dir = Path(self.config.output.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"simulation_result.{format}")

        formatter = OutputFormatter()
        return formatter.save(result, output_path, format)

    @classmethod
    def quick_transmittance(
        cls,
        wavenumber_range: tuple,
        atmosphere_model: str = "US_STANDARD_1976",
        path_length_km: float = 1.0,
        molecules: Optional[List[str]] = None,
        resolution: float = 0.1,
    ) -> tuple:
        """Quick one-liner for transmittance calculation.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            atmosphere_model: Standard atmosphere model name
            path_length_km: Horizontal path length [km]
            molecules: Molecules to include
            resolution: Spectral resolution [cm^-1]

        Returns:
            Tuple of (wavenumber, transmittance) arrays
        """
        config = {
            "atmosphere": {"model": atmosphere_model},
            "geometry": {
                "path_type": "HORIZONTAL",
                "h1_km": 0,
                "path_length_km": path_length_km,
            },
            "spectral": {
                "min_wavenumber": wavenumber_range[0],
                "max_wavenumber": wavenumber_range[1],
                "resolution": resolution,
            },
        }

        sim = cls(config, molecules=molecules)
        return sim.compute_transmittance()
