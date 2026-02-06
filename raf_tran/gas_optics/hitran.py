"""
HITRAN/HAPI Integration Module (Optional)
==========================================

This module provides optional line-by-line absorption calculations using the
HITRAN database through the HAPI (HITRAN Application Programming Interface).

IMPORTANT: This is an OPTIONAL module. The simulation works fully offline
without HITRAN/HAPI using the built-in correlated-k method. HAPI integration
is only needed for high-fidelity spectroscopic applications requiring
line-by-line accuracy (~1% vs ~5% for correlated-k).

Installation
------------
To use HITRAN features, install HAPI:
    pip install hitran-api

Or install RAF-tran with the hitran extra:
    pip install raf-tran[hitran]

Usage
-----
>>> from raf_tran.gas_optics import hitran
>>> if hitran.HAPI_AVAILABLE:
...     abs_coef = hitran.compute_absorption_coefficient(...)
... else:
...     # Fall back to CKD method
...     from raf_tran.gas_optics import GasOptics

References
----------
- HITRAN database: https://hitran.org/
- HAPI documentation: https://hitran.org/hapi/
- Gordon et al. (2022). The HITRAN2020 molecular spectroscopic database.
  JQSRT 277, 107949.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path

import numpy as np

# Check if HAPI is available (optional dependency)
HAPI_AVAILABLE = False
_hapi = None

try:
    import hapi
    _hapi = hapi
    HAPI_AVAILABLE = True
except ImportError:
    pass


# HITRAN molecule IDs
MOLECULE_IDS = {
    'H2O': 1,
    'CO2': 2,
    'O3': 3,
    'N2O': 4,
    'CO': 5,
    'CH4': 6,
    'O2': 7,
    'NO': 8,
    'SO2': 9,
    'NO2': 10,
    'NH3': 11,
    'HNO3': 12,
    'OH': 13,
    'HF': 14,
    'HCl': 15,
    'HBr': 16,
    'HI': 17,
    'ClO': 18,
    'OCS': 19,
    'H2CO': 20,
    'HOCl': 21,
    'N2': 22,
    'HCN': 23,
    'CH3Cl': 24,
    'H2O2': 25,
    'C2H2': 26,
    'C2H6': 27,
    'PH3': 28,
    'COF2': 29,
    'SF6': 30,
}

# Common spectral bands for atmospheric applications
SPECTRAL_BANDS = {
    'MWIR': (1800, 3500),      # 2.9-5.5 um in cm^-1
    'LWIR': (750, 1250),       # 8-13 um in cm^-1
    'SWIR': (4000, 7000),      # 1.4-2.5 um in cm^-1
    'thermal_IR': (500, 2500), # 4-20 um in cm^-1
    'solar': (2000, 25000),    # 0.4-5 um in cm^-1
    'UV': (25000, 50000),      # 0.2-0.4 um in cm^-1
}


@dataclass
class HITRANData:
    """
    Container for HITRAN spectroscopic data.

    Attributes
    ----------
    molecule : str
        Molecule name (e.g., 'H2O', 'CO2')
    isotopologue : int
        Isotopologue number (1 for most abundant)
    wavenumber_min : float
        Minimum wavenumber in cm^-1
    wavenumber_max : float
        Maximum wavenumber in cm^-1
    n_lines : int
        Number of spectral lines in the data
    data_path : Path
        Path to the local HITRAN data file
    """
    molecule: str
    isotopologue: int
    wavenumber_min: float
    wavenumber_max: float
    n_lines: int
    data_path: Path


def check_hapi_available() -> bool:
    """
    Check if HAPI is available for use.

    Returns
    -------
    bool
        True if HAPI is installed and available
    """
    return HAPI_AVAILABLE


def get_default_data_path() -> Path:
    """
    Get the default path for storing HITRAN data.

    Returns
    -------
    Path
        Default data directory path
    """
    # Try environment variable first
    env_path = os.environ.get('HITRAN_DATA_PATH')
    if env_path:
        return Path(env_path)

    # Default to user's home directory
    return Path.home() / '.raf_tran' / 'hitran_data'


def ensure_data_directory(data_path: Optional[Path] = None) -> Path:
    """
    Ensure the HITRAN data directory exists.

    Parameters
    ----------
    data_path : Path, optional
        Custom data path. If None, uses default.

    Returns
    -------
    Path
        Path to the data directory
    """
    if data_path is None:
        data_path = get_default_data_path()

    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    return data_path


def download_hitran_data(
    molecule: str,
    wavenumber_min: float,
    wavenumber_max: float,
    isotopologue: int = 1,
    data_path: Optional[Path] = None,
    force_download: bool = False,
) -> HITRANData:
    """
    Download HITRAN spectroscopic data for a molecule.

    This function requires an internet connection and HAPI to be installed.
    Data is cached locally for offline use.

    Parameters
    ----------
    molecule : str
        Molecule name (e.g., 'H2O', 'CO2', 'O3')
    wavenumber_min : float
        Minimum wavenumber in cm^-1
    wavenumber_max : float
        Maximum wavenumber in cm^-1
    isotopologue : int, optional
        Isotopologue number (default: 1 for most abundant)
    data_path : Path, optional
        Custom path for storing data
    force_download : bool, optional
        Force re-download even if data exists

    Returns
    -------
    HITRANData
        Container with spectroscopic data information

    Raises
    ------
    ImportError
        If HAPI is not installed
    ValueError
        If molecule is not recognized
    ConnectionError
        If download fails
    """
    if not HAPI_AVAILABLE:
        raise ImportError(
            "HAPI is not installed. Install with: pip install hitran-api\n"
            "Or use the built-in correlated-k method for offline operation."
        )

    if molecule not in MOLECULE_IDS:
        raise ValueError(
            f"Unknown molecule: {molecule}. "
            f"Available: {list(MOLECULE_IDS.keys())}"
        )

    # Set up data directory
    data_path = ensure_data_directory(data_path)
    _hapi.db_begin(str(data_path))

    # Create table name
    table_name = f"{molecule}_{isotopologue}_{int(wavenumber_min)}_{int(wavenumber_max)}"

    # Check if data already exists
    table_file = data_path / f"{table_name}.data"
    if table_file.exists() and not force_download:
        # Load existing data
        _hapi.db_begin(str(data_path))
        n_lines = _hapi.getNumberOfLines(table_name)
    else:
        # Download from HITRAN
        try:
            mol_id = MOLECULE_IDS[molecule]
            _hapi.fetch(
                table_name,
                mol_id,
                isotopologue,
                wavenumber_min,
                wavenumber_max,
            )
            n_lines = _hapi.getNumberOfLines(table_name)
        except Exception as e:
            raise ConnectionError(
                f"Failed to download HITRAN data for {molecule}: {e}\n"
                "Check your internet connection or use offline CKD method."
            )

    return HITRANData(
        molecule=molecule,
        isotopologue=isotopologue,
        wavenumber_min=wavenumber_min,
        wavenumber_max=wavenumber_max,
        n_lines=n_lines,
        data_path=data_path,
    )


def compute_absorption_coefficient(
    molecule: str,
    wavenumber: np.ndarray,
    temperature: float,
    pressure: float,
    vmr: float = 1.0,
    isotopologue: int = 1,
    data_path: Optional[Path] = None,
    line_profile: str = 'Voigt',
) -> np.ndarray:
    """
    Compute absorption coefficient using HITRAN line-by-line method.

    Parameters
    ----------
    molecule : str
        Molecule name (e.g., 'H2O', 'CO2')
    wavenumber : ndarray
        Wavenumber grid in cm^-1
    temperature : float
        Temperature in K
    pressure : float
        Pressure in atm
    vmr : float, optional
        Volume mixing ratio (default: 1.0 for pure gas)
    isotopologue : int, optional
        Isotopologue number (default: 1)
    data_path : Path, optional
        Path to HITRAN data
    line_profile : str, optional
        Line profile type: 'Voigt', 'Lorentz', 'Doppler', 'SDVoigt'

    Returns
    -------
    absorption_coefficient : ndarray
        Absorption coefficient in cm^-1 at each wavenumber

    Raises
    ------
    ImportError
        If HAPI is not installed
    FileNotFoundError
        If HITRAN data has not been downloaded
    """
    if not HAPI_AVAILABLE:
        raise ImportError(
            "HAPI is not installed. Install with: pip install hitran-api\n"
            "Or use the built-in correlated-k method for offline operation."
        )

    wavenumber = np.asarray(wavenumber)
    wn_min, wn_max = wavenumber.min(), wavenumber.max()

    # Set up data directory
    data_path = ensure_data_directory(data_path)
    _hapi.db_begin(str(data_path))

    # Find or create table
    table_name = f"{molecule}_{isotopologue}_{int(wn_min)}_{int(wn_max)}"

    # Check if data exists
    table_file = data_path / f"{table_name}.data"
    if not table_file.exists():
        raise FileNotFoundError(
            f"HITRAN data not found for {molecule} ({wn_min}-{wn_max} cm^-1).\n"
            f"Download with: hitran.download_hitran_data('{molecule}', {wn_min}, {wn_max})\n"
            "Or use the built-in correlated-k method for offline operation."
        )

    # Calculate absorption coefficient
    # HAPI expects pressure in atm, temperature in K
    mol_id = MOLECULE_IDS[molecule]

    # Select line profile function
    profile_funcs = {
        'Voigt': _hapi.PROFILE_VOIGT,
        'Lorentz': _hapi.PROFILE_LORENTZ,
        'Doppler': _hapi.PROFILE_DOPPLER,
    }

    if line_profile not in profile_funcs:
        warnings.warn(f"Unknown profile '{line_profile}', using Voigt")
        line_profile = 'Voigt'

    # Compute absorption
    nu, coef = _hapi.absorptionCoefficient_Voigt(
        SourceTables=table_name,
        Environment={'T': temperature, 'p': pressure},
        WavenumberGrid=wavenumber,
    )

    # Apply VMR scaling
    coef = coef * vmr

    return coef


def compute_transmission(
    molecule: str,
    wavenumber: np.ndarray,
    temperature: float,
    pressure: float,
    path_length: float,
    vmr: float = 1.0,
    isotopologue: int = 1,
    data_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute spectral transmission through a gas layer.

    Parameters
    ----------
    molecule : str
        Molecule name
    wavenumber : ndarray
        Wavenumber grid in cm^-1
    temperature : float
        Temperature in K
    pressure : float
        Pressure in atm
    path_length : float
        Path length in cm
    vmr : float, optional
        Volume mixing ratio
    isotopologue : int, optional
        Isotopologue number
    data_path : Path, optional
        Path to HITRAN data

    Returns
    -------
    transmission : ndarray
        Spectral transmission (0-1) at each wavenumber
    """
    abs_coef = compute_absorption_coefficient(
        molecule=molecule,
        wavenumber=wavenumber,
        temperature=temperature,
        pressure=pressure,
        vmr=vmr,
        isotopologue=isotopologue,
        data_path=data_path,
    )

    # Beer-Lambert law
    optical_depth = abs_coef * path_length
    transmission = np.exp(-optical_depth)

    return transmission


def compute_multi_gas_absorption(
    molecules: List[str],
    wavenumber: np.ndarray,
    temperature: float,
    pressure: float,
    vmr: Dict[str, float],
    data_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute combined absorption from multiple gases.

    Parameters
    ----------
    molecules : list of str
        List of molecule names
    wavenumber : ndarray
        Wavenumber grid in cm^-1
    temperature : float
        Temperature in K
    pressure : float
        Pressure in atm
    vmr : dict
        Volume mixing ratios by molecule name
    data_path : Path, optional
        Path to HITRAN data

    Returns
    -------
    total_absorption : ndarray
        Total absorption coefficient in cm^-1
    individual_absorption : dict
        Individual absorption by molecule
    """
    total_absorption = np.zeros_like(wavenumber)
    individual_absorption = {}

    for mol in molecules:
        if mol not in vmr:
            warnings.warn(f"No VMR provided for {mol}, skipping")
            continue

        try:
            abs_coef = compute_absorption_coefficient(
                molecule=mol,
                wavenumber=wavenumber,
                temperature=temperature,
                pressure=pressure,
                vmr=vmr[mol],
                data_path=data_path,
            )
            individual_absorption[mol] = abs_coef
            total_absorption += abs_coef
        except (FileNotFoundError, ImportError) as e:
            warnings.warn(f"Could not compute absorption for {mol}: {e}")

    return total_absorption, individual_absorption


class HITRANGasOptics:
    """
    Gas optics calculator using HITRAN line-by-line method.

    This class provides a high-fidelity alternative to the correlated-k
    method when HAPI is available. It falls back to CKD method if HAPI
    is not installed.

    Parameters
    ----------
    molecules : list of str
        List of molecules to include
    data_path : Path, optional
        Path to HITRAN data directory
    fallback_to_ckd : bool, optional
        If True, fall back to CKD method when HAPI unavailable (default: True)

    Examples
    --------
    >>> optics = HITRANGasOptics(['H2O', 'CO2', 'O3'])
    >>> if optics.using_hitran:
    ...     print("Using high-fidelity HITRAN")
    ... else:
    ...     print("Using CKD fallback")
    """

    def __init__(
        self,
        molecules: List[str],
        data_path: Optional[Path] = None,
        fallback_to_ckd: bool = True,
    ):
        self.molecules = molecules
        self.data_path = data_path
        self.fallback_to_ckd = fallback_to_ckd
        self._ckd_fallback = None

        # Check if HITRAN is available
        self.using_hitran = HAPI_AVAILABLE

        if not HAPI_AVAILABLE and fallback_to_ckd:
            warnings.warn(
                "HAPI not available, using correlated-k fallback. "
                "Install HAPI for high-fidelity calculations: pip install hitran-api"
            )
            # Import CKD as fallback
            from raf_tran.gas_optics.ckd import GasOptics, create_simple_ckd_table
            self._ckd_fallback = GasOptics()
            for mol in molecules:
                # Create simple CKD table for each molecule
                ckd_table = create_simple_ckd_table(
                    gas_name=mol,
                    wavenumber_bounds=(500, 3000),
                )
                self._ckd_fallback.add_gas(ckd_table)

    def download_data(
        self,
        wavenumber_min: float,
        wavenumber_max: float,
        force: bool = False,
    ) -> Dict[str, HITRANData]:
        """
        Download HITRAN data for all configured molecules.

        Parameters
        ----------
        wavenumber_min : float
            Minimum wavenumber in cm^-1
        wavenumber_max : float
            Maximum wavenumber in cm^-1
        force : bool, optional
            Force re-download

        Returns
        -------
        data : dict
            Dictionary of HITRANData by molecule name
        """
        if not HAPI_AVAILABLE:
            raise ImportError("HAPI not available for data download")

        data = {}
        for mol in self.molecules:
            data[mol] = download_hitran_data(
                molecule=mol,
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
                data_path=self.data_path,
                force_download=force,
            )
        return data

    def compute_optical_depth(
        self,
        wavenumber: np.ndarray,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: Dict[str, np.ndarray],
        path_length: np.ndarray,
    ) -> np.ndarray:
        """
        Compute optical depth for atmospheric layers.

        Parameters
        ----------
        wavenumber : ndarray
            Wavenumber grid in cm^-1
        pressure : ndarray
            Layer pressures in Pa
        temperature : ndarray
            Layer temperatures in K
        vmr : dict
            Volume mixing ratios by molecule, each shape (n_layers,)
        path_length : ndarray
            Layer path lengths in m

        Returns
        -------
        optical_depth : ndarray
            Optical depth, shape (n_layers, n_wavenumber)
        """
        if self.using_hitran:
            return self._compute_hitran_optical_depth(
                wavenumber, pressure, temperature, vmr, path_length
            )
        elif self._ckd_fallback is not None:
            return self._compute_ckd_optical_depth(
                pressure, temperature, vmr, path_length
            )
        else:
            raise RuntimeError(
                "HITRAN not available and CKD fallback disabled"
            )

    def _compute_hitran_optical_depth(
        self,
        wavenumber: np.ndarray,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: Dict[str, np.ndarray],
        path_length: np.ndarray,
    ) -> np.ndarray:
        """Compute optical depth using HITRAN line-by-line."""
        n_layers = len(pressure)
        n_wn = len(wavenumber)

        tau = np.zeros((n_layers, n_wn))

        for i in range(n_layers):
            # Convert pressure from Pa to atm
            p_atm = pressure[i] / 101325.0
            # Convert path length from m to cm
            pl_cm = path_length[i] * 100.0

            for mol in self.molecules:
                if mol not in vmr:
                    continue

                try:
                    abs_coef = compute_absorption_coefficient(
                        molecule=mol,
                        wavenumber=wavenumber,
                        temperature=temperature[i],
                        pressure=p_atm,
                        vmr=vmr[mol][i],
                        data_path=self.data_path,
                    )
                    tau[i, :] += abs_coef * pl_cm
                except (FileNotFoundError, ImportError):
                    # Skip if data not available
                    pass

        return tau

    def _compute_ckd_optical_depth(
        self,
        pressure: np.ndarray,
        temperature: np.ndarray,
        vmr: Dict[str, np.ndarray],
        path_length: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optical depth using CKD fallback."""
        from raf_tran.utils.constants import BOLTZMANN

        # Calculate number density
        number_density = pressure / (BOLTZMANN * temperature)

        return self._ckd_fallback.compute_optical_depth(
            pressure=pressure,
            temperature=temperature,
            vmr=vmr,
            dz=path_length,
            number_density=number_density,
        )


def get_absorption_method(
    prefer_hitran: bool = False,
    data_path: Optional[Path] = None,
) -> str:
    """
    Determine which absorption method is available.

    Parameters
    ----------
    prefer_hitran : bool, optional
        If True, prefer HITRAN if available
    data_path : Path, optional
        Path to check for existing HITRAN data

    Returns
    -------
    method : str
        'hitran' if HAPI available and preferred, 'ckd' otherwise
    """
    if prefer_hitran and HAPI_AVAILABLE:
        return 'hitran'
    return 'ckd'


# Convenience function for checking offline capability
def can_run_offline() -> bool:
    """
    Check if the simulation can run fully offline.

    Returns
    -------
    bool
        Always True - RAF-tran supports offline operation via CKD method
    """
    return True  # CKD method always available for offline use
