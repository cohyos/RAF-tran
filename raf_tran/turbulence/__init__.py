"""
Atmospheric Turbulence Module
=============================

This module provides functions for modeling atmospheric optical turbulence,
which is critical for electro-optics applications such as:

- Laser beam propagation
- Free-space optical communications
- Astronomical seeing
- Remote sensing through turbulent atmosphere
- Adaptive optics system design

OFFLINE OPERATION
-----------------
All turbulence models work fully offline. External data sources (NOAA, etc.)
are optional enhancements for site-specific accuracy.

Key Parameters
--------------
Cn2 : Refractive index structure constant (m^(-2/3))
    Characterizes the strength of optical turbulence
r0 : Fried parameter (m)
    Atmospheric coherence length - larger = less turbulence
theta0 : Isoplanatic angle (rad)
    Angular extent over which wavefront is coherent
sigma_I^2 : Scintillation index (dimensionless)
    Normalized variance of intensity fluctuations

Models
------
- Hufnagel-Valley (HV) model for Cn2 profiles
- SLC (Submarine Laser Communication) day/night models
- Kolmogorov spectrum for turbulence statistics
- Rytov theory for weak scintillation
- Adaptive optics performance modeling
- Real Cn2 data integration (optional online sources)

References
----------
- Andrews, L.C. & Phillips, R.L. (2005). Laser Beam Propagation through
  Random Media. SPIE Press.
- Tatarskii, V.I. (1971). The Effects of the Turbulent Atmosphere on
  Wave Propagation. Israel Program for Scientific Translations.
- Hufnagel, R.E. (1974). Variations of atmospheric turbulence. OSA
  Topical Meeting on Optical Propagation through Turbulence.
- Hardy, J.W. (1998). Adaptive Optics for Astronomical Telescopes. Oxford.
"""

from raf_tran.turbulence.cn2_profiles import (
    hufnagel_valley_cn2,
    slc_day_cn2,
    slc_night_cn2,
    cn2_from_weather,
)
from raf_tran.turbulence.propagation import (
    fried_parameter,
    isoplanatic_angle,
    scintillation_index,
    rytov_variance,
    greenwood_frequency,
    beam_wander_variance,
    log_amplitude_variance,
    coherence_time,
    strehl_ratio,
)
from raf_tran.turbulence.kolmogorov import (
    kolmogorov_spectrum,
    von_karman_spectrum,
    structure_function,
    # Phase-aware functions
    phase_structure_function,
    coherence_function,
    phase_variance,
    tilt_removed_phase_variance,
    angle_of_arrival_variance,
    zernike_variance,
    phase_power_spectrum,
    residual_phase_variance_ao,
    long_exposure_strehl,
)

# Adaptive optics simulation
from raf_tran.turbulence.adaptive_optics import (
    AOSystemConfig,
    AOPerformance,
    fitting_error,
    temporal_error,
    wfs_noise_propagation,
    angular_anisoplanatism,
    focus_anisoplanatism,
    compute_ao_performance,
    optimal_actuator_count,
    zernike_temporal_psd,
    ShackHartmannWFS,
    strehl_from_variance,
    variance_from_strehl,
    multi_conjugate_fitting_error,
)

# Real Cn2 data integration
from raf_tran.turbulence.real_cn2_data import (
    Cn2Profile,
    get_climatological_profile,
    hufnagel_valley_57_profile,
    bufton_wind_profile,
    load_profile_from_file,
    save_profile_to_file,
    interpolate_profile,
    combine_profiles,
    add_turbulent_layer,
    estimate_cn2_from_weather,
    get_profile,
)

__all__ = [
    # Cn2 profiles
    "hufnagel_valley_cn2",
    "slc_day_cn2",
    "slc_night_cn2",
    "cn2_from_weather",
    # Propagation parameters
    "fried_parameter",
    "isoplanatic_angle",
    "scintillation_index",
    "rytov_variance",
    "greenwood_frequency",
    "beam_wander_variance",
    "log_amplitude_variance",
    "coherence_time",
    "strehl_ratio",
    # Turbulence spectra
    "kolmogorov_spectrum",
    "von_karman_spectrum",
    "structure_function",
    # Phase-aware functions
    "phase_structure_function",
    "coherence_function",
    "phase_variance",
    "tilt_removed_phase_variance",
    "angle_of_arrival_variance",
    "zernike_variance",
    "phase_power_spectrum",
    "residual_phase_variance_ao",
    "long_exposure_strehl",
    # Adaptive optics
    "AOSystemConfig",
    "AOPerformance",
    "fitting_error",
    "temporal_error",
    "wfs_noise_propagation",
    "angular_anisoplanatism",
    "focus_anisoplanatism",
    "compute_ao_performance",
    "optimal_actuator_count",
    "zernike_temporal_psd",
    "ShackHartmannWFS",
    "strehl_from_variance",
    "variance_from_strehl",
    "multi_conjugate_fitting_error",
    # Real Cn2 data
    "Cn2Profile",
    "get_climatological_profile",
    "hufnagel_valley_57_profile",
    "bufton_wind_profile",
    "load_profile_from_file",
    "save_profile_to_file",
    "interpolate_profile",
    "combine_profiles",
    "add_turbulent_layer",
    "estimate_cn2_from_weather",
    "get_profile",
]
