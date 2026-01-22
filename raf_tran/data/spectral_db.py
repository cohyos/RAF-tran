"""
Spectral Database interface for reading HITRAN line data.

Implements FR-12: Local HDF5 database for offline Line-by-Line calculations.

The database structure is optimized for:
- Fast lookup by wavenumber range
- Memory-efficient partial loading
- Caching of frequently accessed data
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


@dataclass
class SpectralLines:
    """Container for spectral line data within a wavenumber range.

    Attributes:
        molecule: Molecule name
        wavenumber: Line center wavenumbers [cm^-1]
        intensity: Line intensities at 296K [cm^-1/(molecule·cm^-2)]
        air_width: Air-broadened half-widths [cm^-1/atm]
        self_width: Self-broadened half-widths [cm^-1/atm]
        lower_energy: Lower state energies [cm^-1]
        temp_exp: Temperature dependence exponent
        pressure_shift: Pressure-induced line shift [cm^-1/atm]
    """
    molecule: str
    wavenumber: np.ndarray
    intensity: np.ndarray
    air_width: np.ndarray
    self_width: np.ndarray
    lower_energy: np.ndarray
    temp_exp: np.ndarray
    pressure_shift: np.ndarray

    @property
    def num_lines(self) -> int:
        """Number of spectral lines."""
        return len(self.wavenumber)

    def filter_by_intensity(self, min_intensity: float = 1e-30) -> "SpectralLines":
        """Filter lines by minimum intensity threshold.

        Args:
            min_intensity: Minimum line intensity

        Returns:
            Filtered SpectralLines object
        """
        mask = self.intensity >= min_intensity
        return SpectralLines(
            molecule=self.molecule,
            wavenumber=self.wavenumber[mask],
            intensity=self.intensity[mask],
            air_width=self.air_width[mask],
            self_width=self.self_width[mask],
            lower_energy=self.lower_energy[mask],
            temp_exp=self.temp_exp[mask],
            pressure_shift=self.pressure_shift[mask],
        )


class SpectralDatabase:
    """Interface for reading spectral line data from HDF5 database.

    Provides efficient access to HITRAN line data stored in the local
    HDF5 database, optimized for Line-by-Line radiative transfer calculations.

    The database is loaded lazily - data is only read when requested.
    Results are cached for repeated access patterns.

    Example:
        >>> db = SpectralDatabase("./data/spectral_db/hitran_lines.h5")
        >>> lines = db.get_lines("H2O", wavenumber_range=(2000, 2500))
        >>> print(f"Found {lines.num_lines} H2O lines")

    Attributes:
        db_path: Path to HDF5 database file
        molecules: List of available molecules
        metadata: Database metadata
    """

    def __init__(
        self,
        db_path: str,
        cache_size_mb: int = 1024,
        preload_molecules: Optional[List[str]] = None,
    ):
        """Initialize the spectral database interface.

        Args:
            db_path: Path to HDF5 database file
            cache_size_mb: Maximum cache size in MB
            preload_molecules: Optional list of molecules to preload

        Raises:
            FileNotFoundError: If database file doesn't exist
            RuntimeError: If h5py is not available
        """
        if not H5PY_AVAILABLE:
            raise RuntimeError(
                "h5py library is required for SpectralDatabase. "
                "Install with: pip install h5py"
            )

        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        self.cache_size_mb = cache_size_mb
        self._cache: Dict[str, SpectralLines] = {}
        self._file: Optional[h5py.File] = None
        self._metadata: Dict = {}
        self._molecules: List[str] = []

        # Load metadata
        self._load_metadata()

        # Preload specified molecules
        if preload_molecules:
            for mol in preload_molecules:
                if mol in self._molecules:
                    self._preload_molecule(mol)

    def _load_metadata(self) -> None:
        """Load database metadata."""
        with h5py.File(self.db_path, 'r') as f:
            self._molecules = list(f["molecules"].keys())

            if "metadata" in f:
                meta = f["metadata"]
                self._metadata = {
                    "total_lines": meta.attrs.get("total_lines", 0),
                    "num_molecules": meta.attrs.get("num_molecules", 0),
                    "wavenumber_min": meta.attrs.get("wavenumber_min", 0),
                    "wavenumber_max": meta.attrs.get("wavenumber_max", 10000),
                    "creation_date": meta.attrs.get("creation_date", "unknown"),
                }

        logger.info(
            f"Loaded spectral database with {len(self._molecules)} molecules, "
            f"{self._metadata.get('total_lines', '?')} total lines"
        )

    @property
    def molecules(self) -> List[str]:
        """List of available molecules in the database."""
        return self._molecules.copy()

    @property
    def metadata(self) -> Dict:
        """Database metadata."""
        return self._metadata.copy()

    def _preload_molecule(self, molecule: str) -> None:
        """Preload all data for a molecule into cache.

        Args:
            molecule: Molecule name
        """
        if molecule in self._cache:
            return

        with h5py.File(self.db_path, 'r') as f:
            if molecule not in f["molecules"]:
                raise KeyError(f"Molecule not in database: {molecule}")

            mol_group = f["molecules"][molecule]

            self._cache[molecule] = SpectralLines(
                molecule=molecule,
                wavenumber=mol_group["wavenumber"][:],
                intensity=mol_group["intensity"][:],
                air_width=mol_group["air_width"][:],
                self_width=mol_group["self_width"][:],
                lower_energy=mol_group["lower_energy"][:],
                temp_exp=mol_group["temp_exp"][:],
                pressure_shift=mol_group["pressure_shift"][:],
            )

        logger.debug(f"Preloaded {self._cache[molecule].num_lines} lines for {molecule}")

    def get_lines(
        self,
        molecule: str,
        wavenumber_range: Optional[Tuple[float, float]] = None,
        min_intensity: float = 1e-30,
    ) -> SpectralLines:
        """Get spectral lines for a molecule within a wavenumber range.

        Args:
            molecule: Molecule name (e.g., 'H2O', 'CO2')
            wavenumber_range: Optional (min, max) wavenumber range [cm^-1]
            min_intensity: Minimum line intensity threshold

        Returns:
            SpectralLines container with line data

        Raises:
            KeyError: If molecule not in database
        """
        # Check cache first
        if molecule in self._cache:
            lines = self._cache[molecule]
        else:
            # Load from file
            self._preload_molecule(molecule)
            lines = self._cache[molecule]

        # Apply wavenumber filter
        if wavenumber_range:
            wn_min, wn_max = wavenumber_range
            mask = (lines.wavenumber >= wn_min) & (lines.wavenumber <= wn_max)

            lines = SpectralLines(
                molecule=molecule,
                wavenumber=lines.wavenumber[mask],
                intensity=lines.intensity[mask],
                air_width=lines.air_width[mask],
                self_width=lines.self_width[mask],
                lower_energy=lines.lower_energy[mask],
                temp_exp=lines.temp_exp[mask],
                pressure_shift=lines.pressure_shift[mask],
            )

        # Apply intensity filter
        if min_intensity > 0:
            lines = lines.filter_by_intensity(min_intensity)

        return lines

    def get_all_lines(
        self,
        molecules: List[str],
        wavenumber_range: Optional[Tuple[float, float]] = None,
        min_intensity: float = 1e-30,
    ) -> Dict[str, SpectralLines]:
        """Get spectral lines for multiple molecules.

        Args:
            molecules: List of molecule names
            wavenumber_range: Optional (min, max) wavenumber range [cm^-1]
            min_intensity: Minimum line intensity threshold

        Returns:
            Dictionary mapping molecule name to SpectralLines
        """
        result = {}
        for mol in molecules:
            if mol in self._molecules:
                result[mol] = self.get_lines(mol, wavenumber_range, min_intensity)
            else:
                logger.warning(f"Molecule {mol} not in database, skipping")
        return result

    def get_molecule_info(self, molecule: str) -> Dict:
        """Get metadata for a specific molecule.

        Args:
            molecule: Molecule name

        Returns:
            Dictionary with molecule statistics
        """
        with h5py.File(self.db_path, 'r') as f:
            if molecule not in f["molecules"]:
                raise KeyError(f"Molecule not in database: {molecule}")

            mol_group = f["molecules"][molecule]
            return {
                "num_lines": mol_group.attrs.get("num_lines", len(mol_group["wavenumber"])),
                "wavenumber_min": mol_group.attrs.get("wavenumber_min"),
                "wavenumber_max": mol_group.attrs.get("wavenumber_max"),
            }

    def clear_cache(self) -> None:
        """Clear the line data cache."""
        self._cache.clear()
        logger.debug("Cleared spectral database cache")

    def get_cache_size_mb(self) -> float:
        """Get current cache memory usage in MB.

        Returns:
            Cache size in megabytes
        """
        total_bytes = 0
        for mol, lines in self._cache.items():
            for arr in [lines.wavenumber, lines.intensity, lines.air_width,
                       lines.self_width, lines.lower_energy, lines.temp_exp,
                       lines.pressure_shift]:
                total_bytes += arr.nbytes
        return total_bytes / (1024 * 1024)

    def estimate_lines_in_range(
        self,
        molecule: str,
        wavenumber_range: Tuple[float, float],
    ) -> int:
        """Estimate number of lines in a wavenumber range without loading data.

        Uses binary search on sorted wavenumber indices for efficiency.

        Args:
            molecule: Molecule name
            wavenumber_range: (min, max) wavenumber range

        Returns:
            Estimated number of lines
        """
        with h5py.File(self.db_path, 'r') as f:
            if molecule not in f["molecules"]:
                return 0

            mol_group = f["molecules"][molecule]
            wn = mol_group["wavenumber"]
            wn_min, wn_max = wavenumber_range

            # Binary search for indices
            idx_min = np.searchsorted(wn[:], wn_min)
            idx_max = np.searchsorted(wn[:], wn_max)

            return idx_max - idx_min

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_cache()
        return False
