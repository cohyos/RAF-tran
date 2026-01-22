# RAF-Tran

**Open-source Atmospheric Radiative Transfer Simulation System**

RAF-Tran is a Python library for spectral radiative transfer calculations, providing MODTRAN-like functionality with Line-by-Line (LBL) accuracy. It is designed for offline/air-gapped operation with full support for local data storage.

## Features

- **Line-by-Line (LBL) molecular absorption** using HITRAN spectral database
- **Mie scattering** for aerosols (Rural, Urban, Maritime, Desert models)
- **Rayleigh scattering** for molecular scattering
- **Standard atmosphere models**: US Standard 1976, Tropical, Mid-Latitude Summer/Winter, Sub-Arctic Summer/Winter
- **Multiple path geometries**: Horizontal, Slant Path, Vertical
- **Earth curvature correction** for long paths
- **Full offline operation** with local HDF5 spectral database
- **GPU acceleration support** (optional, via CuPy)
- **Docker containerization** for portable deployment

## Installation

### From Source

```bash
git clone https://github.com/cohyos/RAF-tran.git
cd RAF-tran
pip install -e .
```

### With Optional Dependencies

```bash
# Full installation with all dependencies
pip install -e ".[full]"

# With GPU support
pip install -e ".[gpu]"

# Development tools
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from raf_tran import Simulation

# Configure simulation
config = {
    "atmosphere": {
        "model": "US_STANDARD_1976",
        "aerosols": {"type": "RURAL", "visibility_km": 23.0}
    },
    "geometry": {
        "path_type": "HORIZONTAL",
        "h1_km": 0,
        "path_length_km": 1.0
    },
    "spectral": {
        "min_wavenumber": 2000,  # 5 um
        "max_wavenumber": 3333,  # 3 um
        "resolution": 0.1
    }
}

# Run simulation
sim = Simulation(config)
result = sim.run()

# Access results
print(f"Mean transmittance: {result.transmittance.mean():.3f}")
print(f"Spectral range: {result.wavelength_um[0]:.2f}-{result.wavelength_um[-1]:.2f} um")
```

### Quick Transmittance Calculation

```python
from raf_tran import Simulation

# One-liner for quick calculations
wavenumber, transmittance = Simulation.quick_transmittance(
    wavenumber_range=(2000, 3333),
    atmosphere_model="MID_LATITUDE_SUMMER",
    path_length_km=1.0
)
```

### Command Line Interface

```bash
# Run simulation with config file
raf-tran --config simulation.json --output results.json

# Quick simulation
raf-tran --atmosphere US_STANDARD_1976 --wn-min 2000 --wn-max 2500 -o result.csv

# Generate spectral database
generate-db --output data/hitran_lines.h5 --synthetic
```

## Configuration

RAF-Tran uses JSON configuration files:

```json
{
  "simulation_config": {
    "offline_mode": true,
    "database_path": "./data/spectral_db/hitran_lines.h5",
    "use_gpu": false
  },
  "atmosphere": {
    "model": "MID_LATITUDE_SUMMER",
    "aerosols": {"type": "RURAL", "visibility_km": 23.0},
    "custom_concentrations": {"CO2": 420.0}
  },
  "geometry": {
    "path_type": "SLANT",
    "h1_km": 0,
    "h2_km": 10,
    "angle_deg": 45
  },
  "spectral": {
    "min_wavenumber": 2000,
    "max_wavenumber": 2500,
    "resolution": 0.01
  }
}
```

### Atmosphere Models (FR-01)

| Model | Description |
|-------|-------------|
| `US_STANDARD_1976` | US Standard Atmosphere 1976 |
| `TROPICAL` | Tropical atmosphere (warm, humid) |
| `MID_LATITUDE_SUMMER` | Mid-latitude summer |
| `MID_LATITUDE_WINTER` | Mid-latitude winter |
| `SUB_ARCTIC_SUMMER` | Sub-arctic summer |
| `SUB_ARCTIC_WINTER` | Sub-arctic winter |

### Aerosol Types (FR-04)

| Type | Description |
|------|-------------|
| `NONE` | No aerosols (gas absorption only) |
| `RURAL` | Continental rural aerosol |
| `URBAN` | Urban/industrial aerosol |
| `MARITIME` | Maritime aerosol |
| `DESERT` | Desert dust aerosol |

### Path Geometries (FR-07)

| Type | Description |
|------|-------------|
| `HORIZONTAL` | Horizontal path at constant altitude |
| `SLANT` | Slant path between two altitudes |
| `VERTICAL` | Vertical (nadir/zenith) path |

## Offline Operation

RAF-Tran is designed for air-gapped environments:

1. **Generate spectral database** (requires internet):
   ```bash
   generate-db --output data/hitran_lines.h5 --molecules H2O CO2 O3 CH4 N2O
   ```

2. **Run simulations offline**:
   ```python
   config = {
       "simulation_config": {
           "offline_mode": True,
           "database_path": "./data/hitran_lines.h5"
       },
       # ... rest of config
   }
   sim = Simulation(config)
   result = sim.run()
   ```

## Docker

### Build and Run

```bash
# Build image
docker-compose build

# Run simulation
docker-compose run raf-tran --config /app/data/config.json -o /app/output/result.json

# Generate database
docker-compose run generate-db
```

### Docker Volumes

- `/app/data/spectral_db`: Mount spectral database (read-only)
- `/app/data`: Configuration files (read-only)
- `/app/output`: Output directory (read-write)

## Output Formats

RAF-Tran supports multiple output formats:

- **JSON**: Full structured output with metadata
- **CSV**: Simple tabular format
- **NetCDF**: Scientific data format (CF-compliant)

```python
# Save results
sim.save_result(result, "output.json", format="json")
sim.save_result(result, "output.csv", format="csv")
sim.save_result(result, "output.nc", format="netcdf")
```

## Architecture

```
RAF-Tran
├── Configuration Manager    # Load atmosphere profiles and settings
├── Data Ingestor (ETL)     # HITRAN to HDF5 conversion
├── Physics Core
│   ├── Gas Engine          # Line-by-Line absorption (Numba-accelerated)
│   └── Scattering Engine   # Mie and Rayleigh scattering
├── RTE Solver              # Radiative transfer integration
└── Output Formatter        # Export to CSV/JSON/NetCDF
```

## Performance

Target performance (NFR-01, NFR-02, NFR-03):

| Metric | Target | Notes |
|--------|--------|-------|
| Standard simulation | < 5 seconds | 3-5 um, horizontal path, modern CPU |
| Database cold start | < 10 seconds | Loading spectral data to memory |
| Memory usage | < 4 GB | Full resolution calculation |

## Validation

RAF-Tran includes a comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run benchmark tests
pytest tests/benchmarks/
```

### Validation Scenarios

1. **Baseline**: Horizontal path, sea level, no aerosols
2. **Aerosol Load**: Horizontal path, visibility 5 km
3. **Full Scene**: Slant path, humid atmosphere, aerosols

Target accuracy: RMSE < 1% vs MODTRAN reference data (NFR-04)

## References

RAF-Tran is based on established atmospheric radiative transfer methods:

- **HITRAN**: Spectral line database
- **AFGL**: Standard atmosphere profiles (Anderson et al., 1986)
- **Bohren & Huffman**: Mie scattering theory
- **Koschmieder**: Visibility-extinction relationship

See `Raf_Tran_Doc/` for the complete literature review.

## License

MIT License

## Contributing

Contributions are welcome! Please see the issues page for current needs.

## Citation

If you use RAF-Tran in your research, please cite:

```
RAF-Tran: Open-source Atmospheric Radiative Transfer Simulation System
https://github.com/cohyos/RAF-tran
```
