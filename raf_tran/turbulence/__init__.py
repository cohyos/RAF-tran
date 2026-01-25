"""
Atmospheric Turbulence Module
=============================

This module provides functions for modeling atmospheric optical turbulence,
which is critical for electro-optics applications such as:

- Laser beam propagation
- Free-space optical communications
- Astronomical seeing
- Remote sensing through turbulent atmosphere

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

References
----------
- Andrews, L.C. & Phillips, R.L. (2005). Laser Beam Propagation through
  Random Media. SPIE Press.
- Tatarskii, V.I. (1971). The Effects of the Turbulent Atmosphere on
  Wave Propagation. Israel Program for Scientific Translations.
- Hufnagel, R.E. (1974). Variations of atmospheric turbulence. OSA
  Topical Meeting on Optical Propagation through Turbulence.
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
]
