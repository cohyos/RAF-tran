"""
Command-line interface for RAF-Tran.

Provides CLI commands for:
- Running simulations
- Generating spectral databases
- Validation testing
"""

import argparse
import json
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def run_simulation(args: argparse.Namespace) -> int:
    """Run a radiative transfer simulation."""
    from raf_tran import Simulation

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            return 1
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Build config from CLI args
        config = {
            "atmosphere": {"model": args.atmosphere or "US_STANDARD_1976"},
            "geometry": {
                "path_type": args.path_type or "HORIZONTAL",
                "h1_km": args.h1 or 0.0,
                "h2_km": args.h2 or 0.0,
                "angle_deg": args.angle or 0.0,
                "path_length_km": args.path_length or 1.0,
            },
            "spectral": {
                "min_wavenumber": args.wn_min or 2000.0,
                "max_wavenumber": args.wn_max or 3333.0,
                "resolution": args.resolution or 0.1,
            },
        }

    # Set database path if provided
    if args.database:
        config.setdefault("simulation_config", {})["database_path"] = args.database

    # Parse molecules
    molecules = args.molecules.split(",") if args.molecules else None

    # Run simulation
    print(f"Running simulation...")
    sim = Simulation(config, molecules=molecules)
    result = sim.run()

    # Output
    if args.output:
        output_format = args.format or "json"
        output_path = sim.save_result(result, args.output, format=output_format)
        print(f"Results saved to: {output_path}")
    else:
        # Print summary to stdout
        print(f"\nSimulation Results:")
        print(f"  Spectral range: {result.wavenumber[0]:.1f} - {result.wavenumber[-1]:.1f} cm^-1")
        print(f"  Number of points: {len(result.wavenumber)}")
        print(f"  Mean transmittance: {result.transmittance.mean():.4f}")
        print(f"  Mean optical depth: {result.optical_depth.mean():.4f}")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAF-Tran: Open-source atmospheric radiative transfer simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run simulation with config file
    raf-tran --config simulation.json --output results.json

    # Quick simulation with CLI args
    raf-tran --atmosphere MID_LATITUDE_SUMMER --wn-min 2000 --wn-max 2500

    # Generate spectral database
    generate-db --output data/hitran.h5 --molecules H2O CO2 O3
        """,
    )

    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="RAF-Tran 0.1.0",
    )

    # Configuration options
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "-d", "--database",
        type=str,
        help="Path to spectral database (HDF5)",
    )

    # Atmosphere options
    parser.add_argument(
        "-a", "--atmosphere",
        type=str,
        choices=[
            "US_STANDARD_1976", "TROPICAL",
            "MID_LATITUDE_SUMMER", "MID_LATITUDE_WINTER",
            "SUB_ARCTIC_SUMMER", "SUB_ARCTIC_WINTER",
        ],
        help="Standard atmosphere model",
    )

    # Geometry options
    parser.add_argument(
        "--path-type",
        type=str,
        choices=["HORIZONTAL", "SLANT", "VERTICAL"],
        help="Path geometry type",
    )
    parser.add_argument(
        "--h1",
        type=float,
        help="Observer altitude [km]",
    )
    parser.add_argument(
        "--h2",
        type=float,
        help="Target altitude [km]",
    )
    parser.add_argument(
        "--angle",
        type=float,
        help="Zenith angle [degrees]",
    )
    parser.add_argument(
        "--path-length",
        type=float,
        help="Horizontal path length [km]",
    )

    # Spectral options
    parser.add_argument(
        "--wn-min",
        type=float,
        help="Minimum wavenumber [cm^-1]",
    )
    parser.add_argument(
        "--wn-max",
        type=float,
        help="Maximum wavenumber [cm^-1]",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        help="Spectral resolution [cm^-1]",
    )
    parser.add_argument(
        "-m", "--molecules",
        type=str,
        help="Comma-separated list of molecules (e.g., H2O,CO2,O3)",
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path",
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["json", "csv", "netcdf"],
        help="Output format",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    try:
        return run_simulation(args)
    except Exception as e:
        logging.exception(f"Simulation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
