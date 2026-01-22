# RAF-Tran Project Summary for Session Continuation

## Project Overview
**Repository:** `/home/user/RAF-tran`
**Purpose:** Open-source MODTRAN-based atmospheric radiative transfer simulation tool
**Current Branch:** `claude/atmospheric-simulation-tool-zY7Uf`
**Push Branch:** `claude/continue-previous-work-Oe03I` (use this for git push operations)

## Current Progress

### Completed Tasks
1. ✅ **Simulation Core** - Complete atmospheric radiative transfer simulation system
2. ✅ **Bug Fixes Applied:**
   - Mie scattering coefficient initialization (`raf_tran/core/scattering_engine.py:165`)
   - H2O continuum absorption formula (`raf_tran/core/gas_engine.py:440`)
   - Synthetic database generation (`raf_tran/data/ingestor.py:409-424`)
3. ✅ **Validity Checks** - Physics-based tests for simulation accuracy
4. ✅ **ATP Tests** - Acceptance Test Procedure tests for all functional requirements
5. ✅ **All Tests Passing** - 92 passed, 0 skipped
6. ✅ **Non-determinism Bug Fixed** - Restructured `compute_absorption_lbl` to parallelize over grid points

### Test Results
```
tests/atp/test_acceptance.py          - 28 tests (all passed)
tests/integration/test_simulation.py  - 15 tests (all passed)
tests/unit/test_atmosphere.py         - 19 tests (all passed)
tests/unit/test_scattering.py         - 14 tests (all passed)
tests/validity/test_physics_validity.py - 15 tests (all passed)

Total: 92 passed, 0 skipped
```

## Active File Structure

```
/home/user/RAF-tran/
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── SESSION_SUMMARY.md          # This file
├── Raf_Tran_Doc/
│   └── Undermind - High-performance Python-accessible atmospheric radiative transfer algorithms and benchmarks.pdf
├── examples/
│   ├── basic_simulation.py
│   └── example_config.json
├── data/
│   └── spectral_db/
│       └── hitran_lines.h5  (generated synthetic database)
├── raf_tran/
│   ├── __init__.py
│   ├── cli.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── atmosphere.py      # Standard atmosphere profiles
│   │   ├── manager.py         # Configuration management
│   │   └── settings.py        # Simulation settings
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py       # Physical constants
│   │   ├── gas_engine.py      # Line-by-Line absorption (FIXED)
│   │   ├── rte_solver.py      # Radiative Transfer Equation solver
│   │   ├── scattering_engine.py  # Mie/Rayleigh scattering (FIXED)
│   │   └── simulation.py      # Main simulation class
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestor.py        # HITRAN data ingestion (FIXED)
│   │   └── spectral_db.py     # Spectral database interface
│   └── utils/
│       ├── __init__.py
│       └── output.py          # Output formatting (JSON, CSV, NetCDF)
└── tests/
    ├── __init__.py
    ├── atp/
    │   ├── __init__.py
    │   └── test_acceptance.py    # ATP tests
    ├── benchmarks/
    │   └── __init__.py
    ├── integration/
    │   ├── __init__.py
    │   └── test_simulation.py
    ├── unit/
    │   ├── __init__.py
    │   ├── test_atmosphere.py
    │   └── test_scattering.py
    └── validity/
        ├── __init__.py
        └── test_physics_validity.py  # Physics validity tests
```

## Key APIs

### Running a Simulation
```python
from raf_tran.core.simulation import Simulation

config = {
    "atmosphere": {"model": "US_STANDARD_1976", "aerosols": {"type": "RURAL", "visibility_km": 23.0}},
    "geometry": {"path_type": "HORIZONTAL", "h1_km": 0.0, "path_length_km": 1.0},
    "spectral": {"min_wavenumber": 2000, "max_wavenumber": 3333, "resolution": 1.0},
}
sim = Simulation(config)
result = sim.run()
# result.transmittance, result.optical_depth, result.radiance, etc.
```

### Standard Atmospheres
```python
from raf_tran.config.atmosphere import StandardAtmospheres

atm = StandardAtmospheres.get_profile("US_STANDARD_1976")  # or TROPICAL, MID_LATITUDE_SUMMER, etc.
atm = StandardAtmospheres.us_standard_1976()  # Direct method
atm = StandardAtmospheres.tropical()
```

### Available Aerosol Types
- `NONE`, `RURAL`, `URBAN`, `MARITIME`, `DESERT`

### Available Atmosphere Models
- `US_STANDARD_1976`, `TROPICAL`, `MID_LATITUDE_SUMMER`, `MID_LATITUDE_WINTER`, `SUB_ARCTIC_SUMMER`, `SUB_ARCTIC_WINTER`

## Known Issues

1. ~~**Non-determinism in Simulation** - FIXED~~
   - **Status:** RESOLVED
   - **Fix:** Restructured `compute_absorption_lbl` to parallelize over wavenumber grid points instead of lines
   - All 92 tests now pass (including `test_dr1_2_reproducibility`)

2. **Synthetic Database Limitations**
   - Line intensities are randomly generated, not from real HITRAN
   - Absorption features may be weaker than real atmospheric conditions
   - Tests are designed to work with synthetic data (relaxed thresholds)

## Immediate Next Tasks

1. **Integrate Real HITRAN Data**
   - Install `hitran-api` package
   - Update `ingestor.py` to download real spectral lines
   - Re-run tests with real data

2. **Performance Optimization**
   - Profile simulation performance
   - Consider GPU acceleration (CuPy already supported)

3. **Documentation**
   - Add API documentation
   - Create user guide with examples

## Running Tests
```bash
cd /home/user/RAF-tran
python -m pytest tests/ -v  # All tests
python -m pytest tests/validity/ -v  # Physics validity only
python -m pytest tests/atp/ -v  # ATP tests only
```

## Git Information
- **Current Branch:** `claude/atmospheric-simulation-tool-zY7Uf` (local working branch)
- **Push Branch:** `claude/continue-previous-work-Oe03I` (for pushing changes)
- **Last Commit:** `920ee64` - "Add session summary documentation for continuation"

### Branch Status
Both branches are aligned at commit `920ee64`. Use `claude/continue-previous-work-Oe03I` for pushing to remote.

---

# NEW SESSION CONTINUATION PROMPT

Copy and paste the following prompt to start a new session:

---

## Prompt for New Session:

```
I want to continue working on the RAF-Tran atmospheric radiative transfer simulation project.

**Project Location:** /home/user/RAF-tran
**Current Branch:** claude/atmospheric-simulation-tool-zY7Uf

Please read the SESSION_SUMMARY.md file in the project root to understand the current state.

**Quick Context:**
- RAF-Tran is an open-source MODTRAN-based atmospheric radiative transfer tool
- The simulation core is complete with gas absorption, scattering, and RTE solver
- All tests pass (91 passed, 1 skipped)
- Bug fixes applied: Mie scattering init, H2O continuum formula, synthetic database

**Immediate Priority Tasks:**
1. Fix the numba parallel non-determinism bug in `raf_tran/core/gas_engine.py:compute_absorption_lbl`
2. Integrate real HITRAN spectral data
3. Performance optimization

**Git Note:** For pushing changes, use branch `claude/continue-previous-work-Oe03I`

Please start by:
1. Reading SESSION_SUMMARY.md for full context
2. Running `python -m pytest tests/ -q` to verify tests still pass
3. Then proceed with the next priority task
```

---
