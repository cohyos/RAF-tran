#!/usr/bin/env python
"""
Basic RAF-Tran simulation example.

This script demonstrates a simple transmittance calculation for
a horizontal path through the atmosphere.
"""

from raf_tran import Simulation


def main():
    # Configure simulation
    config = {
        "atmosphere": {
            "model": "US_STANDARD_1976",
            "aerosols": {"type": "RURAL", "visibility_km": 23.0},
        },
        "geometry": {
            "path_type": "HORIZONTAL",
            "h1_km": 0,
            "path_length_km": 1.0,
        },
        "spectral": {
            "min_wavenumber": 2000,  # 5 um
            "max_wavenumber": 2500,  # 4 um
            "resolution": 0.5,  # Lower resolution for faster demo
        },
    }

    # Create and run simulation
    print("Creating simulation...")
    sim = Simulation(config)

    print("Running simulation...")
    result = sim.run()

    # Display results
    print("\n=== Simulation Results ===")
    print(f"Atmosphere: {result.metadata['atmosphere_model']}")
    print(f"Path type: {result.metadata['path_type']}")
    print(f"Molecules: {result.metadata['molecules']}")
    print(f"Aerosol: {result.metadata['aerosol_type']}")
    print(f"\nSpectral range: {result.wavenumber[0]:.1f} - {result.wavenumber[-1]:.1f} cm^-1")
    print(f"               ({result.wavelength_um[-1]:.2f} - {result.wavelength_um[0]:.2f} um)")
    print(f"Number of spectral points: {len(result.wavenumber)}")
    print(f"\nMean transmittance: {result.transmittance.mean():.4f}")
    print(f"Min transmittance: {result.transmittance.min():.4f}")
    print(f"Max transmittance: {result.transmittance.max():.4f}")
    print(f"Mean optical depth: {result.optical_depth.mean():.4f}")

    # Save results
    output_file = "basic_simulation_result.json"
    sim.save_result(result, output_file, format="json")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
