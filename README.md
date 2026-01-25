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
- Support for major absorbing gases (H2O, CO2, O3, etc.)
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

### Atmospheric Turbulence (NEW)
- **Cn2 profiles**: Hufnagel-Valley, SLC day/night models
- **Beam propagation**: Fried parameter, scintillation index, Rytov variance
- **Optical effects**: Beam wander, Strehl ratio, coherence time
- **Turbulence spectra**: Kolmogorov and von Karman power spectra
- Applications: Free-space optical communications, LIDAR, adaptive optics

### Air Mass Calculations (NEW)
- Chapman function for curved-Earth geometry at high solar zenith angles
- Kasten-Young empirical formula accurate to 90 deg SZA
- Automatic method selection based on conditions

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
wavelength = np.linspace(4e-6, 20e-6, 100)  # 4-20 um
B = planck_function(wavelength, T)

# Total blackbody flux
flux = stefan_boltzmann_flux(T)
print(f"Blackbody flux at {T} K: {flux:.1f} W/m^2")
```

### Atmospheric Turbulence (NEW)

```python
from raf_tran.turbulence import (
    hufnagel_valley_cn2, slc_day_cn2,
    fried_parameter, scintillation_index, rytov_variance
)

# Cn2 profile at different altitudes
altitudes = [0, 100, 1000, 5000, 10000]
for h in altitudes:
    cn2 = hufnagel_valley_cn2(h)
    print(f"Altitude {h}m: Cn2 = {cn2:.2e} m^(-2/3)")

# Beam propagation parameters for laser link
wavelength = 1.55e-6  # 1.55 um telecom laser
cn2_avg = 1e-15       # Path-averaged Cn2
path_length = 10000   # 10 km

# Path-integrated Cn2
cn2_integrated = cn2_avg * path_length

# Fried parameter (atmospheric coherence length)
r0 = fried_parameter(wavelength, cn2_integrated)
print(f"Fried parameter: {r0*100:.1f} cm")

# Scintillation index
si = scintillation_index(wavelength, cn2_avg, path_length)
print(f"Scintillation index: {si:.3f}")

# Rytov variance (turbulence strength)
sigma_r2 = rytov_variance(wavelength, cn2_avg, path_length)
if sigma_r2 < 0.3:
    print("Weak fluctuation regime")
elif sigma_r2 < 5:
    print("Moderate fluctuation regime")
else:
    print("Strong fluctuation (saturation)")
```

## Examples

The `examples/` directory contains 31 comprehensive, CLI-enabled examples demonstrating RAF-tran capabilities. Each example includes:
- Command-line arguments for customization
- Detailed console output with explanations
- Generated plots (requires matplotlib)

Run any example with `--help` to see available options:

```bash
cd examples
python 01_solar_zenith_angle_study.py --help
```

### Core Examples (Demonstration)

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
| 11 | `atmospheric_turbulence.py` | Cn2 profiles and beam propagation |

### Validation Examples (Physics Verification)

| # | Example | Description |
|---|---------|-------------|
| 12 | `beer_lambert_validation.py` | Validate Beer-Lambert law implementation |
| 13 | `planck_blackbody_validation.py` | Stefan-Boltzmann and Wien's law tests |
| 14 | `rayleigh_scattering_validation.py` | Lambda^-4 dependence and literature comparison |
| 15 | `mie_scattering_validation.py` | Rayleigh/geometric limits and energy conservation |
| 16 | `two_stream_benchmarks.py` | Solver accuracy vs analytical solutions |
| 17 | `solar_spectrum_analysis.py` | Solar irradiance and atmospheric windows |
| 18 | `thermal_emission_validation.py` | Kirchhoff's law and OLR validation |
| 19 | `path_radiance_remote_sensing.py` | Atmospheric correction for satellite imagery |
| 20 | `visibility_contrast.py` | Koschmieder's law and contrast reduction |
| 21 | `laser_propagation.py` | Combined absorption + turbulence effects |

### Advanced Applications (NEW)

| # | Example | Description |
|---|---------|-------------|
| 22 | `atmospheric_polarization.py` | Rayleigh/Mie polarization, sky polarization patterns |
| 23 | `infrared_atmospheric_windows.py` | MWIR/LWIR transmission, EO-IR sensor design |
| 24 | `volcanic_aerosol_forcing.py` | Pinatubo-type cooling, stratospheric sulfate |
| 25 | `water_vapor_feedback.py` | Clausius-Clapeyron, climate sensitivity |
| 26 | `high_altitude_solar.py` | Aviation/HAPS dosimetry, altitude effects |
| 27 | `twilight_spectra.py` | Sunset/sunrise colors, Chappuis band |
| 28 | `multi_layer_cloud.py` | Cloud overlap schemes (max, random, max-random) |
| 29 | `aod_retrieval_visibility.py` | Langley calibration, AERONET-style AOD |
| 30 | `spectral_surface_albedo.py` | Snow/ice, vegetation red edge, NDVI |
| 31 | `limb_viewing_geometry.py` | Satellite limb sounding, onion-peeling retrieval |
| 32 | `config_file_demo.py` | YAML/JSON configuration file usage |
| 33 | `validation_visualization.py` | Physics validation plots and comparisons |

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

# Advanced examples (NEW):
# IR atmospheric windows for EO sensor design
python 23_infrared_atmospheric_windows.py --altitude 5 --water-vapor 1.0

# Twilight spectra analysis
python 27_twilight_spectra.py --sza 92

# AOD retrieval and visibility
python 29_aod_retrieval_visibility.py --aod 0.3 --angstrom 1.4

# Limb viewing geometry for satellite missions
python 31_limb_viewing_geometry.py --tangent-height 30

# Configuration file usage
python 32_config_file_demo.py --config configs/sample_simulation.yaml
```

## Configuration Files

RAF-tran supports YAML and JSON configuration files for reproducible simulation setups:

```python
from raf_tran.utils import load_config, validate_config, create_default_config

# Load configuration from file
config = load_config("simulation.yaml")

# Validate configuration
issues = validate_config(config)
if issues:
    print("Configuration issues:", issues)

# Create default config template
create_default_config("my_config.yaml")
```

Sample configuration files are provided in `examples/configs/`:
- `sample_simulation.yaml` - Comprehensive example with all options
- `cloudy_atmosphere.yaml` - Stratocumulus cloud scenario
- `volcanic_scenario.json` - Post-volcanic eruption atmosphere (JSON format)

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
├── turbulence/       # Atmospheric optical turbulence (NEW)
│   ├── cn2_profiles.py  # Hufnagel-Valley, SLC models
│   ├── propagation.py   # Fried parameter, scintillation
│   └── kolmogorov.py    # Turbulence spectra
└── utils/            # Utilities and constants
    ├── constants.py  # Physical constants
    ├── spectral.py   # Spectral functions
    ├── air_mass.py   # Chapman function, air mass
    └── config.py     # YAML/JSON configuration support (NEW)
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
