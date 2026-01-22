"""
Output Formatter for exporting simulation results.

Supports multiple output formats:
- CSV: Simple comma-separated values
- JSON: Full structured output with metadata
- NetCDF: Scientific data format with metadata
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import xarray as xr
    import netCDF4
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False


class OutputFormatter:
    """Formatter for exporting simulation results to various formats.

    Supports:
    - CSV: Tabular data export
    - JSON: Full structured output
    - NetCDF: Scientific data format (CF-compliant)

    Example:
        >>> formatter = OutputFormatter()
        >>> formatter.save(result, "output.json", format="json")
        >>> formatter.save(result, "output.csv", format="csv")
        >>> formatter.save(result, "output.nc", format="netcdf")
    """

    def __init__(self):
        """Initialize the output formatter."""
        pass

    def save(
        self,
        result,  # SimulationResult
        output_path: str,
        format: str = "json",
        **kwargs,
    ) -> str:
        """Save simulation result to file.

        Args:
            result: SimulationResult object
            output_path: Output file path
            format: Output format (csv, json, netcdf)
            **kwargs: Additional format-specific options

        Returns:
            Path to saved file

        Raises:
            ValueError: If format is not supported
        """
        format = format.lower()

        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            return self._save_json(result, output_path, **kwargs)
        elif format == "csv":
            return self._save_csv(result, output_path, **kwargs)
        elif format == "netcdf" or format == "nc":
            return self._save_netcdf(result, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _save_json(
        self,
        result,
        output_path: Path,
        indent: int = 2,
        **kwargs,
    ) -> str:
        """Save result to JSON format.

        Args:
            result: SimulationResult
            output_path: Output file path
            indent: JSON indentation

        Returns:
            Path to saved file
        """
        data = {
            "metadata": {
                "format_version": "1.0",
                "created": datetime.now().isoformat(),
                "software": "RAF-Tran",
                **result.metadata,
            },
            "spectral": {
                "wavenumber_cm1": result.wavenumber.tolist(),
                "wavelength_um": result.wavelength_um.tolist(),
            },
            "results": {
                "transmittance": result.transmittance.tolist(),
                "radiance_W_cm2_sr_cm1": result.radiance.tolist(),
                "optical_depth": result.optical_depth.tolist(),
                "thermal_emission_W_cm2_sr_cm1": result.thermal_emission.tolist(),
            },
            "configuration": result.config.to_dict(),
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)

        logger.info(f"Saved JSON output to {output_path}")
        return str(output_path)

    def _save_csv(
        self,
        result,
        output_path: Path,
        delimiter: str = ",",
        **kwargs,
    ) -> str:
        """Save result to CSV format.

        Args:
            result: SimulationResult
            output_path: Output file path
            delimiter: Column delimiter

        Returns:
            Path to saved file
        """
        if PANDAS_AVAILABLE:
            df = pd.DataFrame({
                "wavenumber_cm1": result.wavenumber,
                "wavelength_um": result.wavelength_um,
                "transmittance": result.transmittance,
                "radiance_W_cm2_sr_cm1": result.radiance,
                "optical_depth": result.optical_depth,
                "thermal_emission_W_cm2_sr_cm1": result.thermal_emission,
            })
            df.to_csv(output_path, index=False, sep=delimiter)
        else:
            # Fallback to numpy
            header = (
                "wavenumber_cm1,wavelength_um,transmittance,"
                "radiance_W_cm2_sr_cm1,optical_depth,thermal_emission_W_cm2_sr_cm1"
            )
            data = np.column_stack([
                result.wavenumber,
                result.wavelength_um,
                result.transmittance,
                result.radiance,
                result.optical_depth,
                result.thermal_emission,
            ])
            np.savetxt(output_path, data, delimiter=delimiter, header=header, comments='')

        logger.info(f"Saved CSV output to {output_path}")
        return str(output_path)

    def _save_netcdf(
        self,
        result,
        output_path: Path,
        **kwargs,
    ) -> str:
        """Save result to NetCDF format.

        Args:
            result: SimulationResult
            output_path: Output file path

        Returns:
            Path to saved file
        """
        if not NETCDF_AVAILABLE:
            raise RuntimeError(
                "NetCDF output requires xarray and netCDF4. "
                "Install with: pip install xarray netCDF4"
            )

        # Create xarray dataset
        ds = xr.Dataset(
            data_vars={
                "transmittance": (["wavenumber"], result.transmittance, {
                    "long_name": "Atmospheric transmittance",
                    "units": "1",
                    "valid_range": [0.0, 1.0],
                }),
                "radiance": (["wavenumber"], result.radiance, {
                    "long_name": "Spectral radiance",
                    "units": "W/(cm^2 sr cm^-1)",
                }),
                "optical_depth": (["wavenumber"], result.optical_depth, {
                    "long_name": "Total optical depth",
                    "units": "1",
                }),
                "thermal_emission": (["wavenumber"], result.thermal_emission, {
                    "long_name": "Thermal emission",
                    "units": "W/(cm^2 sr cm^-1)",
                }),
                "wavelength": (["wavenumber"], result.wavelength_um, {
                    "long_name": "Wavelength",
                    "units": "um",
                }),
            },
            coords={
                "wavenumber": (["wavenumber"], result.wavenumber, {
                    "long_name": "Wavenumber",
                    "units": "cm^-1",
                }),
            },
            attrs={
                "title": "RAF-Tran Radiative Transfer Simulation Results",
                "institution": "RAF-Tran",
                "source": "RAF-Tran atmospheric radiative transfer simulation",
                "history": f"Created {datetime.now().isoformat()}",
                "conventions": "CF-1.8",
                "atmosphere_model": result.metadata.get("atmosphere_model", ""),
                "molecules": str(result.metadata.get("molecules", [])),
                "aerosol_type": result.metadata.get("aerosol_type", ""),
                "path_type": result.metadata.get("path_type", ""),
            },
        )

        ds.to_netcdf(output_path)
        logger.info(f"Saved NetCDF output to {output_path}")
        return str(output_path)


def create_benchmark_json(
    test_case_id: str,
    description: str,
    inputs: Dict[str, Any],
    wavenumber: np.ndarray,
    transmittance: np.ndarray,
    source: str = "RAF-Tran",
) -> Dict[str, Any]:
    """Create a benchmark JSON structure for validation testing.

    Matches the format specified in the SRS validation plan.

    Args:
        test_case_id: Unique test case identifier
        description: Test case description
        inputs: Input parameters dictionary
        wavenumber: Wavenumber array [cm^-1]
        transmittance: Transmittance array
        source: Source of the data

    Returns:
        Benchmark data dictionary

    Example:
        >>> benchmark = create_benchmark_json(
        ...     test_case_id="BENCH_001_HORIZONTAL",
        ...     description="1km horizontal path at sea level",
        ...     inputs={"model_atmosphere": "US_STD_76", "path_length_km": 1.0},
        ...     wavenumber=np.linspace(2000, 3333, 100),
        ...     transmittance=np.ones(100) * 0.85,
        ... )
    """
    # Convert to wavelength
    wavelength_um = 1e4 / wavenumber

    # Build data points
    data_points = []
    for i, (wn, wl, trans) in enumerate(zip(wavenumber, wavelength_um, transmittance)):
        data_points.append({
            "wavenumber": float(wn),
            "wavelength_micron": float(wl),
            "transmittance": float(trans),
        })

    return {
        "test_case_id": test_case_id,
        "description": description,
        "inputs": inputs,
        "ground_truth_data": {
            "source": source,
            "spectral_resolution_cm1": float(wavenumber[1] - wavenumber[0]) if len(wavenumber) > 1 else 0.0,
            "data_points": data_points,
        },
    }


def save_benchmark_json(
    benchmark_data: Dict[str, Any],
    output_path: str,
) -> str:
    """Save benchmark data to JSON file.

    Args:
        benchmark_data: Benchmark dictionary
        output_path: Output file path

    Returns:
        Path to saved file
    """
    with open(output_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    logger.info(f"Saved benchmark data to {output_path}")
    return output_path
