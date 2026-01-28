"""
FPA Library Configuration
=========================

Save and load FPA configurations and custom sensor definitions.
Supports JSON format for interoperability.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

from raf_tran.fpa_library.models import FPASpec, ROICSpec
from raf_tran.fpa_library.database import get_fpa_database, get_fpa


def save_fpa_config(fpas: Union[FPASpec, List[FPASpec]], filepath: Union[str, Path]) -> None:
    """
    Save FPA configuration(s) to a JSON file.

    Parameters
    ----------
    fpas : FPASpec or list of FPASpec
        FPA(s) to save
    filepath : str or Path
        Output file path
    """
    if isinstance(fpas, FPASpec):
        fpas = [fpas]

    data = {
        'version': '1.0',
        'format': 'fpa_library',
        'fpas': [fpa.to_dict() for fpa in fpas],
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_fpa_config(filepath: Union[str, Path]) -> List[FPASpec]:
    """
    Load FPA configuration(s) from a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Input file path

    Returns
    -------
    fpas : list of FPASpec
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        data = json.load(f)

    fpas = []
    for fpa_dict in data.get('fpas', []):
        fpas.append(FPASpec.from_dict(fpa_dict))

    return fpas


def export_database_json(filepath: Union[str, Path]) -> None:
    """
    Export the entire built-in FPA database to a JSON file.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    """
    db = get_fpa_database()
    all_fpas = list(db.values())
    save_fpa_config(all_fpas, filepath)


def load_custom_fpas(filepath: Union[str, Path]) -> Dict[str, FPASpec]:
    """
    Load custom FPA definitions and return as a name-keyed dictionary.

    Parameters
    ----------
    filepath : str or Path
        JSON file with custom FPA definitions

    Returns
    -------
    custom_fpas : dict
        Name-keyed FPA dictionary
    """
    fpas = load_fpa_config(filepath)
    return {fpa.name: fpa for fpa in fpas}


def merge_databases(custom_filepath: Union[str, Path]) -> Dict[str, FPASpec]:
    """
    Merge the built-in database with custom FPA definitions.

    Custom FPAs override built-in ones if names clash.

    Parameters
    ----------
    custom_filepath : str or Path
        JSON file with custom FPA definitions

    Returns
    -------
    merged : dict
        Merged FPA dictionary
    """
    db = dict(get_fpa_database())
    custom = load_custom_fpas(custom_filepath)

    for name, fpa in custom.items():
        key = name.replace(' ', '_').replace('-', '_')
        db[key] = fpa

    return db


def create_comparison_config(
    fpa_keys: List[str],
    focal_length_mm: float = 100.0,
    target_size_m: float = 2.3,
    atmosphere_model: str = "US_Standard_1976",
    range_km: float = 10.0,
) -> Dict:
    """
    Create a comparison configuration for analysis.

    Parameters
    ----------
    fpa_keys : list of str
        Database keys of FPAs to compare
    focal_length_mm : float
        Reference focal length
    target_size_m : float
        Reference target dimension
    atmosphere_model : str
        Atmosphere model name for propagation analysis
    range_km : float
        Reference observation range

    Returns
    -------
    config : dict
        Configuration dictionary that can be saved/loaded
    """
    fpas = []
    for key in fpa_keys:
        fpa = get_fpa(key)
        if fpa is not None:
            fpas.append(fpa.to_dict())

    return {
        'version': '1.0',
        'format': 'fpa_comparison',
        'fpas': fpas,
        'parameters': {
            'focal_length_mm': focal_length_mm,
            'target_size_m': target_size_m,
            'atmosphere_model': atmosphere_model,
            'range_km': range_km,
        },
    }


def save_comparison_config(config: Dict, filepath: Union[str, Path]) -> None:
    """Save a comparison configuration to JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_comparison_config(filepath: Union[str, Path]) -> Dict:
    """Load a comparison configuration from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
