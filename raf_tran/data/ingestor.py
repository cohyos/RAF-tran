"""
Data Ingestor (ETL) for HITRAN spectral line data.

Implements FR-14: generate_db.py tool for creating local HDF5 database
from HITRAN online data.

This module should be run ONCE in an online environment to create the
local database file for offline use.

Usage:
    python -m raf_tran.data.ingestor --output ./data/spectral_db/hitran_lines.h5

    Or using the CLI:
    generate-db --output ./data/spectral_db/hitran_lines.h5
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Check for optional HAPI availability
try:
    import hapi
    HAPI_AVAILABLE = True
except ImportError:
    HAPI_AVAILABLE = False
    logger.warning("HAPI not available. Install with: pip install hitran-api")

# Check for h5py
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("h5py not available. Install with: pip install h5py")


@dataclass
class HITRANLineData:
    """Container for HITRAN spectral line data.

    Attributes:
        molecule_id: HITRAN molecule ID
        isotope_id: Isotope number
        wavenumber: Line center wavenumber [cm^-1]
        intensity: Line intensity at 296K [cm^-1/(molecule·cm^-2)]
        einstein_a: Einstein A coefficient [s^-1]
        air_broadened_width: Air-broadened half-width at 296K [cm^-1/atm]
        self_broadened_width: Self-broadened half-width at 296K [cm^-1/atm]
        lower_state_energy: Lower state energy [cm^-1]
        temp_dependence_n: Temperature dependence exponent
        pressure_shift: Air pressure-induced line shift [cm^-1/atm]
    """
    molecule_id: np.ndarray
    isotope_id: np.ndarray
    wavenumber: np.ndarray
    intensity: np.ndarray
    einstein_a: np.ndarray
    air_broadened_width: np.ndarray
    self_broadened_width: np.ndarray
    lower_state_energy: np.ndarray
    temp_dependence_n: np.ndarray
    pressure_shift: np.ndarray


class DataIngestor:
    """ETL tool for converting HITRAN data to local HDF5 format.

    This class implements FR-14, providing functionality to:
    1. Download spectral line data from HITRAN
    2. Filter by wavenumber range and molecule
    3. Convert to optimized binary HDF5 format
    4. Validate the generated database

    The resulting HDF5 file is structured for fast lookup by wavenumber
    and molecule, optimized for Line-by-Line calculations.

    Example:
        >>> ingestor = DataIngestor()
        >>> # Download and process H2O and CO2 in the MWIR band
        >>> ingestor.download_hitran_data(
        ...     molecules=["H2O", "CO2"],
        ...     wavenumber_range=(2000, 3333),
        ... )
        >>> ingestor.save_hdf5("./data/spectral_db/hitran_lines.h5")

    HDF5 Structure:
        /molecules/
            /H2O/
                wavenumber [N]
                intensity [N]
                air_width [N]
                self_width [N]
                lower_energy [N]
                temp_exp [N]
                pressure_shift [N]
            /CO2/
                ...
        /metadata/
            creation_date
            hitran_version
            wavenumber_min
            wavenumber_max
            total_lines
    """

    # HITRAN molecule IDs
    MOLECULE_IDS = {
        "H2O": 1, "CO2": 2, "O3": 3, "N2O": 4, "CO": 5,
        "CH4": 6, "O2": 7, "NO": 8, "SO2": 9, "NO2": 10,
        "NH3": 11, "HNO3": 12, "OH": 13, "HF": 14, "HCl": 15,
        "HBr": 16, "HI": 17, "ClO": 18, "OCS": 19, "H2CO": 20,
    }

    def __init__(self, hapi_data_dir: Optional[str] = None):
        """Initialize the Data Ingestor.

        Args:
            hapi_data_dir: Directory for HAPI data cache.
                          Defaults to ./data/hapi_cache
        """
        self.hapi_data_dir = Path(hapi_data_dir or "./data/hapi_cache")
        self.hapi_data_dir.mkdir(parents=True, exist_ok=True)

        self.line_data: Dict[str, HITRANLineData] = {}
        self.metadata: Dict[str, any] = {}

        if HAPI_AVAILABLE:
            hapi.db_begin(str(self.hapi_data_dir))

    def download_hitran_data(
        self,
        molecules: List[str],
        wavenumber_range: Tuple[float, float],
        isotope_ids: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """Download spectral line data from HITRAN.

        This requires internet access and should be run in an online environment.

        Args:
            molecules: List of molecule names (e.g., ["H2O", "CO2"])
            wavenumber_range: (min, max) wavenumber range [cm^-1]
            isotope_ids: Optional dict mapping molecule to isotope IDs.
                        Defaults to most abundant isotope.

        Raises:
            RuntimeError: If HAPI is not available
            ConnectionError: If unable to connect to HITRAN
        """
        if not HAPI_AVAILABLE:
            raise RuntimeError(
                "HAPI library is required for HITRAN download. "
                "Install with: pip install hitran-api"
            )

        wn_min, wn_max = wavenumber_range
        logger.info(f"Downloading HITRAN data for {molecules} in range {wn_min}-{wn_max} cm^-1")

        for molecule in molecules:
            if molecule not in self.MOLECULE_IDS:
                logger.warning(f"Unknown molecule: {molecule}, skipping")
                continue

            mol_id = self.MOLECULE_IDS[molecule]
            iso_ids = isotope_ids.get(molecule, [1]) if isotope_ids else [1]

            table_name = f"{molecule}_{int(wn_min)}_{int(wn_max)}"

            logger.info(f"Fetching {molecule} (ID={mol_id}, isotopes={iso_ids})")

            try:
                # Use HAPI to fetch data
                hapi.fetch(table_name, mol_id, iso_ids[0], wn_min, wn_max)

                # Extract line parameters
                nu, sw, a, gamma_air, gamma_self, elower, n_air, delta_air = hapi.getColumns(
                    table_name,
                    ['nu', 'sw', 'a', 'gamma_air', 'gamma_self', 'elower', 'n_air', 'delta_air']
                )

                self.line_data[molecule] = HITRANLineData(
                    molecule_id=np.full(len(nu), mol_id, dtype=np.int32),
                    isotope_id=np.full(len(nu), iso_ids[0], dtype=np.int32),
                    wavenumber=np.array(nu, dtype=np.float64),
                    intensity=np.array(sw, dtype=np.float64),
                    einstein_a=np.array(a, dtype=np.float64),
                    air_broadened_width=np.array(gamma_air, dtype=np.float64),
                    self_broadened_width=np.array(gamma_self, dtype=np.float64),
                    lower_state_energy=np.array(elower, dtype=np.float64),
                    temp_dependence_n=np.array(n_air, dtype=np.float64),
                    pressure_shift=np.array(delta_air, dtype=np.float64),
                )

                logger.info(f"Downloaded {len(nu)} lines for {molecule}")

            except Exception as e:
                logger.error(f"Failed to fetch {molecule}: {e}")
                raise

        # Update metadata
        self.metadata.update({
            "wavenumber_min": wn_min,
            "wavenumber_max": wn_max,
            "molecules": molecules,
        })

    def load_par_file(
        self,
        par_path: str,
        wavenumber_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Load HITRAN .par file directly (alternative to online download).

        The .par file format is the standard HITRAN 160-character line format.

        Args:
            par_path: Path to .par file
            wavenumber_range: Optional (min, max) filter range
        """
        logger.info(f"Loading HITRAN .par file: {par_path}")

        path = Path(par_path)
        if not path.exists():
            raise FileNotFoundError(f"HITRAN file not found: {par_path}")

        # Parse .par file (160-character fixed format)
        molecules_data: Dict[str, Dict[str, List]] = {}

        with open(path, 'r') as f:
            for line in f:
                if len(line) < 160:
                    continue

                try:
                    mol_id = int(line[0:2])
                    iso_id = int(line[2:3])
                    nu = float(line[3:15])
                    sw = float(line[15:25])
                    a = float(line[25:35])
                    gamma_air = float(line[35:40])
                    gamma_self = float(line[40:45])
                    elower = float(line[45:55])
                    n_air = float(line[55:59])
                    delta_air = float(line[59:67])

                    # Filter by wavenumber
                    if wavenumber_range:
                        if nu < wavenumber_range[0] or nu > wavenumber_range[1]:
                            continue

                    # Find molecule name
                    mol_name = None
                    for name, mid in self.MOLECULE_IDS.items():
                        if mid == mol_id:
                            mol_name = name
                            break

                    if mol_name is None:
                        continue

                    if mol_name not in molecules_data:
                        molecules_data[mol_name] = {
                            'mol_id': [], 'iso_id': [], 'nu': [], 'sw': [],
                            'a': [], 'gamma_air': [], 'gamma_self': [],
                            'elower': [], 'n_air': [], 'delta_air': []
                        }

                    data = molecules_data[mol_name]
                    data['mol_id'].append(mol_id)
                    data['iso_id'].append(iso_id)
                    data['nu'].append(nu)
                    data['sw'].append(sw)
                    data['a'].append(a)
                    data['gamma_air'].append(gamma_air)
                    data['gamma_self'].append(gamma_self)
                    data['elower'].append(elower)
                    data['n_air'].append(n_air)
                    data['delta_air'].append(delta_air)

                except (ValueError, IndexError) as e:
                    continue

        # Convert to HITRANLineData
        for mol_name, data in molecules_data.items():
            self.line_data[mol_name] = HITRANLineData(
                molecule_id=np.array(data['mol_id'], dtype=np.int32),
                isotope_id=np.array(data['iso_id'], dtype=np.int32),
                wavenumber=np.array(data['nu'], dtype=np.float64),
                intensity=np.array(data['sw'], dtype=np.float64),
                einstein_a=np.array(data['a'], dtype=np.float64),
                air_broadened_width=np.array(data['gamma_air'], dtype=np.float64),
                self_broadened_width=np.array(data['gamma_self'], dtype=np.float64),
                lower_state_energy=np.array(data['elower'], dtype=np.float64),
                temp_dependence_n=np.array(data['n_air'], dtype=np.float64),
                pressure_shift=np.array(data['delta_air'], dtype=np.float64),
            )
            logger.info(f"Loaded {len(data['nu'])} lines for {mol_name}")

    def save_hdf5(self, output_path: str, compression: str = "gzip") -> None:
        """Save spectral data to optimized HDF5 format.

        Creates an HDF5 file optimized for fast Line-by-Line calculations,
        implementing FR-12.

        Args:
            output_path: Path to output HDF5 file
            compression: HDF5 compression type ('gzip', 'lzf', or None)
        """
        if not H5PY_AVAILABLE:
            raise RuntimeError(
                "h5py library is required for HDF5 output. "
                "Install with: pip install h5py"
            )

        if not self.line_data:
            raise ValueError("No line data loaded. Call download_hitran_data or load_par_file first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving HDF5 database to {output_path}")

        compression_opts = {"compression": compression} if compression else {}

        with h5py.File(output_path, 'w') as f:
            # Create molecules group
            molecules_group = f.create_group("molecules")

            total_lines = 0
            for mol_name, data in self.line_data.items():
                mol_group = molecules_group.create_group(mol_name)

                # Save each array as a dataset
                mol_group.create_dataset("wavenumber", data=data.wavenumber, **compression_opts)
                mol_group.create_dataset("intensity", data=data.intensity, **compression_opts)
                mol_group.create_dataset("einstein_a", data=data.einstein_a, **compression_opts)
                mol_group.create_dataset("air_width", data=data.air_broadened_width, **compression_opts)
                mol_group.create_dataset("self_width", data=data.self_broadened_width, **compression_opts)
                mol_group.create_dataset("lower_energy", data=data.lower_state_energy, **compression_opts)
                mol_group.create_dataset("temp_exp", data=data.temp_dependence_n, **compression_opts)
                mol_group.create_dataset("pressure_shift", data=data.pressure_shift, **compression_opts)
                mol_group.create_dataset("molecule_id", data=data.molecule_id, **compression_opts)
                mol_group.create_dataset("isotope_id", data=data.isotope_id, **compression_opts)

                # Add wavenumber index for fast lookup
                sorted_indices = np.argsort(data.wavenumber)
                mol_group.create_dataset("wavenumber_sorted_indices", data=sorted_indices, **compression_opts)

                num_lines = len(data.wavenumber)
                total_lines += num_lines
                mol_group.attrs["num_lines"] = num_lines
                mol_group.attrs["wavenumber_min"] = float(data.wavenumber.min())
                mol_group.attrs["wavenumber_max"] = float(data.wavenumber.max())

                logger.info(f"  {mol_name}: {num_lines} lines")

            # Save metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["total_lines"] = total_lines
            meta_group.attrs["num_molecules"] = len(self.line_data)
            meta_group.attrs["molecules"] = list(self.line_data.keys())
            meta_group.attrs["hitran_format_version"] = "2020"

            if "wavenumber_min" in self.metadata:
                meta_group.attrs["wavenumber_min"] = self.metadata["wavenumber_min"]
                meta_group.attrs["wavenumber_max"] = self.metadata["wavenumber_max"]

            # Add creation timestamp
            from datetime import datetime
            meta_group.attrs["creation_date"] = datetime.now().isoformat()

        logger.info(f"Successfully saved {total_lines} total lines to {output_path}")

    def generate_synthetic_database(
        self,
        output_path: str,
        wavenumber_range: Tuple[float, float] = (2000, 3333),
        molecules: List[str] = None,
    ) -> None:
        """Generate a synthetic database for testing without network access.

        Creates a minimal database with representative spectral features
        for development and testing purposes.

        Args:
            output_path: Path to output HDF5 file
            wavenumber_range: Spectral range [cm^-1]
            molecules: List of molecules to include
        """
        if molecules is None:
            molecules = ["H2O", "CO2", "O3", "CH4", "N2O"]

        wn_min, wn_max = wavenumber_range
        logger.info(f"Generating synthetic database for {molecules}")

        # Generate synthetic line data for each molecule
        rng = np.random.default_rng(42)  # Reproducible random seed

        for mol_name in molecules:
            mol_id = self.MOLECULE_IDS.get(mol_name, 1)

            # Number of lines per 100 cm^-1 (scaled by spectral range)
            # Real HITRAN has ~1000-5000 significant lines per 100 cm^-1
            lines_per_100cm = {
                "H2O": 500,
                "CO2": 300,
                "O3": 200,
                "CH4": 150,
                "N2O": 100,
            }.get(mol_name, 50)

            spectral_range = wn_max - wn_min
            num_lines = int(lines_per_100cm * spectral_range / 100)
            num_lines = max(100, min(num_lines, 10000))  # Reasonable bounds

            # Generate random wavenumber positions
            wavenumbers = rng.uniform(wn_min, wn_max, num_lines)
            wavenumbers.sort()

            # Generate realistic line intensities (log-normal distribution)
            # HITRAN intensities typically range from 1e-30 to 1e-19
            # Most lines are weak, few are strong - use skewed distribution
            log_intensities = rng.normal(-25, 1.5, num_lines)  # log10 scale, narrower spread
            # Clip to realistic HITRAN range
            log_intensities = np.clip(log_intensities, -30, -19)
            intensities = 10 ** log_intensities

            # Generate other parameters
            self.line_data[mol_name] = HITRANLineData(
                molecule_id=np.full(num_lines, mol_id, dtype=np.int32),
                isotope_id=np.ones(num_lines, dtype=np.int32),
                wavenumber=wavenumbers,
                intensity=intensities,
                einstein_a=rng.uniform(0.01, 100, num_lines),
                air_broadened_width=rng.uniform(0.02, 0.1, num_lines),
                self_broadened_width=rng.uniform(0.05, 0.2, num_lines),
                lower_state_energy=rng.uniform(0, 3000, num_lines),
                temp_dependence_n=rng.uniform(0.5, 0.85, num_lines),
                pressure_shift=rng.uniform(-0.01, 0.01, num_lines),
            )

        self.metadata.update({
            "wavenumber_min": wn_min,
            "wavenumber_max": wn_max,
            "molecules": molecules,
            "synthetic": True,
        })

        self.save_hdf5(output_path)
        logger.info(f"Generated synthetic database at {output_path}")


def main():
    """CLI entry point for generate-db command (FR-14)."""
    parser = argparse.ArgumentParser(
        description="Generate local HDF5 database from HITRAN data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download data from HITRAN (requires internet)
    generate-db --output data/hitran_lines.h5 --molecules H2O CO2 O3

    # Generate synthetic data for testing
    generate-db --output data/hitran_lines.h5 --synthetic

    # Load from existing .par file
    generate-db --output data/hitran_lines.h5 --par-file HITRAN2020.par
        """
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/spectral_db/hitran_lines.h5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--molecules", "-m",
        nargs="+",
        default=["H2O", "CO2", "O3", "CH4", "N2O"],
        help="Molecules to include"
    )
    parser.add_argument(
        "--wn-min",
        type=float,
        default=2000.0,
        help="Minimum wavenumber [cm^-1]"
    )
    parser.add_argument(
        "--wn-max",
        type=float,
        default=3333.0,
        help="Maximum wavenumber [cm^-1]"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic database (for testing, no internet required)"
    )
    parser.add_argument(
        "--par-file",
        type=str,
        help="Path to HITRAN .par file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ingestor = DataIngestor()

    if args.synthetic:
        # Generate synthetic database
        ingestor.generate_synthetic_database(
            output_path=args.output,
            wavenumber_range=(args.wn_min, args.wn_max),
            molecules=args.molecules,
        )
    elif args.par_file:
        # Load from .par file
        ingestor.load_par_file(
            par_path=args.par_file,
            wavenumber_range=(args.wn_min, args.wn_max),
        )
        ingestor.save_hdf5(args.output)
    else:
        # Download from HITRAN
        ingestor.download_hitran_data(
            molecules=args.molecules,
            wavenumber_range=(args.wn_min, args.wn_max),
        )
        ingestor.save_hdf5(args.output)

    print(f"Database generated successfully: {args.output}")


if __name__ == "__main__":
    main()
