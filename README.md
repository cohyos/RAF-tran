# RAF-tran

Open source atmospheric radiative transfer library based on scientific literature.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

RAF-tran is a Python-native, high-performance atmospheric radiative transfer library designed for:

- **Remote sensing applications**: Simulate satellite and ground-based measurements
- **Climate modeling**: Calculate radiative fluxes and heating rates
- **Atmospheric science research**: Study scattering, absorption, and emission processes

The library implements modern computational techniques including:
- Correlated-k distribution method for efficient gas absorption
- Mie scattering for aerosols and cloud particles
- Two-stream and discrete ordinates RTE solvers
- JAX acceleration for GPU-enabled calculations

## Features

### Atmospheric Profiles
- US Standard Atmosphere 1976
- MODTRAN model atmospheres (tropical, midlatitude, subarctic)
- Custom profile support with temperature, pressure, and gas mixing ratios

### Gas Absorption
- Correlated-k distribution method
- Support for major absorbing gases (H₂O, CO₂, O₃, etc.)
- Pressure and temperature dependent absorption coefficients

### Scattering
- **Rayleigh scattering**: Molecular scattering with wavelength-dependent cross sections
- **Mie scattering**: Full Mie theory implementation for spherical particles
- Lognormal and other particle size distributions

### Radiative Transfer Solvers
- **Two-stream approximation**: Fast calculations for flux computation
  - Eddington, quadrature, hemispheric mean, and delta-Eddington methods
- **Discrete ordinates (DISORT)**: Higher accuracy multi-stream solver
- Solar (shortwave) and thermal (longwave) calculations

## Installation

```bash
# Clone the repository
git clone https://github.com/cohyos/RAF-tran.git
cd RAF-tran

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- JAX ≥ 0.4 (optional, for GPU acceleration)

## Quick Start

### Basic Radiative Transfer Calculation

```python
import numpy as np
from raf_tran.atmosphere import StandardAtmosphere
from raf_tran.scattering import RayleighScattering
from raf_tran.rte_solver import TwoStreamSolver

# Create atmospheric profile
atmosphere = StandardAtmosphere()

# Define altitude grid
z_levels = np.linspace(0, 50000, 51)  # 0-50 km
z_mid = (z_levels[:-1] + z_levels[1:]) / 2
dz = np.diff(z_levels)

# Get atmospheric properties
number_density = atmosphere.number_density(z_mid)

# Calculate Rayleigh optical depth at 550 nm
rayleigh = RayleighScattering()
wavelength = np.array([0.55])  # micrometers
tau = rayleigh.optical_depth(wavelength, number_density, dz)

# Solve radiative transfer
solver = TwoStreamSolver()
result = solver.solve_solar(
    tau=tau.ravel(),
    omega=np.ones(50),  # pure scattering
    g=np.zeros(50),     # isotropic
    mu0=0.5,            # 60° solar zenith angle
    flux_toa=1.9,       # W/m²/nm
    surface_albedo=0.1,
)

print(f"Direct flux at surface: {result.flux_direct[-1]:.4f} W/m²/nm")
print(f"Diffuse flux at surface: {result.flux_down[-1]:.4f} W/m²/nm")
```

### Mie Scattering for Aerosols

```python
from raf_tran.scattering import MieScattering

# Create Mie calculator for dust aerosol
mie = MieScattering(refractive_index=1.55 + 0.003j)

# Calculate optical properties
wavelengths = np.array([0.4, 0.55, 0.7])  # μm
radius = 0.5  # μm

sigma_ext, sigma_sca, sigma_abs = mie.cross_sections(wavelengths, radius)
print(f"Extinction cross section at 550 nm: {sigma_ext[1]:.3f} μm²")
```

### Thermal Radiation

```python
from raf_tran.utils.spectral import planck_function, stefan_boltzmann_flux

# Planck blackbody emission
T = 300  # K
wavelength = np.linspace(4e-6, 20e-6, 100)  # 4-20 μm
B = planck_function(wavelength, T)

# Total blackbody flux
flux = stefan_boltzmann_flux(T)
print(f"Blackbody flux at {T} K: {flux:.1f} W/m²")
```

## Examples

The `examples/` directory contains 10 comprehensive, CLI-enabled examples demonstrating RAF-tran capabilities. Each example includes:
- Command-line arguments for customization
- Detailed console output with explanations
- Generated plots (requires matplotlib)

Run any example with `--help` to see available options:

```bash
cd examples
python 01_solar_zenith_angle_study.py --help
```

### Available Examples

| # | Example | Description |
|---|---------|-------------|
| 01 | `solar_zenith_angle_study.py` | Effect of sun angle on radiation |
| 02 | `spectral_transmission.py` | Why the sky is blue (Rayleigh scattering) |
| 03 | `aerosol_types_comparison.py` | Optical properties of different aerosols |
| 04 | `atmospheric_profiles.py` | Compare standard atmosphere models |
| 05 | `greenhouse_effect.py` | Demonstrate atmospheric warming mechanism |
| 06 | `surface_albedo_effects.py` | Ice-albedo feedback and surface types |
| 07 | `cloud_radiative_effects.py` | How clouds affect climate (warming vs cooling) |
| 08 | `ozone_uv_absorption.py` | Ozone layer and UV protection |
| 09 | `radiative_heating_rates.py` | Atmospheric heating/cooling calculations |
| 10 | `satellite_observation.py` | Simulate satellite remote sensing |

### Example Usage

```bash
# Study solar zenith angle effects
python 01_solar_zenith_angle_study.py --wavelength 0.55 --albedo 0.3

# Compare aerosol types at specific wavelength
python 03_aerosol_types_comparison.py --wavelength 0.55 --radius 0.5

# Greenhouse effect with different optical depths
python 05_greenhouse_effect.py --tau 2.5 --albedo 0.3

# Cloud effects at high altitude
python 07_cloud_radiative_effects.py --cloud-type Cirrus --sza 30

# Ozone depletion impact on UV
python 08_ozone_uv_absorption.py --ozone-column 200 --sza 45
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=raf_tran

# Run specific test file
pytest tests/test_atmosphere.py
```

## Architecture

```
raf_tran/
├── atmosphere/       # Atmospheric profile models
│   └── profiles.py   # Standard atmosphere implementations
├── gas_optics/       # Gas absorption calculations
│   └── ckd.py        # Correlated-k distribution
├── scattering/       # Scattering calculations
│   ├── rayleigh.py   # Rayleigh scattering
│   └── mie.py        # Mie scattering
├── rte_solver/       # RTE solvers
│   ├── two_stream.py # Two-stream approximation
│   └── disort.py     # Discrete ordinates
└── utils/            # Utilities and constants
    ├── constants.py  # Physical constants
    └── spectral.py   # Spectral functions
```

## Scientific Background

RAF-tran implements algorithms based on established scientific literature:

- **Rayleigh scattering**: Bodhaine et al. (1999), J. Atmos. Oceanic Technol.
- **Mie theory**: Bohren & Huffman (1983), Absorption and Scattering of Light by Small Particles
- **Two-stream**: Meador & Weaver (1980), J. Atmos. Sci.; Toon et al. (1989), J. Geophys. Res.
- **Correlated-k**: Lacis & Oinas (1991), J. Geophys. Res.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This project was developed based on the comprehensive literature review from the Undermind research report on "High-performance Python-accessible atmospheric radiative transfer algorithms and benchmarks."
