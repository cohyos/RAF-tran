"""
Focal Plane Array (FPA) Detector Models
=======================================

This module provides models for infrared FPA detectors used in
thermal imaging and target detection applications.

Detector Types:
- InSb (Indium Antimonide): MWIR 3-5 um, requires cooling to ~77K
- MCT (HgCdTe): MWIR or LWIR, wavelength tunable by composition

Key Parameters:
- NETD: Noise Equivalent Temperature Difference (mK)
- D*: Specific Detectivity (cm*sqrt(Hz)/W)
- Pixel pitch: Detector element spacing (um)
- Integration time: Signal integration period (ms)

References:
- Rogalski, A. (2012). Infrared Detectors, CRC Press
- Dereniak & Boreman (1996). Infrared Detectors and Systems
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from raf_tran.utils.constants import (
    PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
)


@dataclass
class FPADetector:
    """
    Focal Plane Array detector model.

    Parameters
    ----------
    name : str
        Detector name/identifier
    spectral_band : Tuple[float, float]
        Wavelength range (um) as (min, max)
    d_star : float
        Specific detectivity D* (cm*sqrt(Hz)/W)
    pixel_pitch : float
        Pixel spacing (um)
    netd : float
        Noise equivalent temperature difference (mK)
    quantum_efficiency : float
        Quantum efficiency (0-1)
    fill_factor : float
        Active area fraction (0-1)
    well_capacity : float
        Full well capacity (electrons)
    read_noise : float
        Read noise (electrons RMS)
    dark_current : float
        Dark current density (A/cm^2)
    operating_temp : float
        Operating temperature (K)
    f_number : float
        Optical system f-number
    focal_length : float
        Focal length (mm)
    integration_time : float
        Integration time (ms)
    """
    name: str
    spectral_band: Tuple[float, float]  # (min_um, max_um)
    d_star: float  # cm*sqrt(Hz)/W
    pixel_pitch: float  # um
    netd: float = 20.0  # mK
    quantum_efficiency: float = 0.7
    fill_factor: float = 0.8
    well_capacity: float = 1e7  # electrons
    read_noise: float = 50.0  # electrons RMS
    dark_current: float = 1e-9  # A/cm^2
    operating_temp: float = 77.0  # K
    f_number: float = 2.0
    focal_length: float = 100.0  # mm
    integration_time: float = 10.0  # ms

    @property
    def pixel_area(self) -> float:
        """Pixel area in cm^2."""
        return (self.pixel_pitch * 1e-4) ** 2  # um -> cm

    @property
    def bandwidth(self) -> float:
        """Spectral bandwidth in um."""
        return self.spectral_band[1] - self.spectral_band[0]

    @property
    def center_wavelength(self) -> float:
        """Center wavelength in um."""
        return (self.spectral_band[0] + self.spectral_band[1]) / 2

    @property
    def electrical_bandwidth(self) -> float:
        """Electrical bandwidth in Hz (1/2*integration_time)."""
        return 1.0 / (2.0 * self.integration_time * 1e-3)

    @property
    def optics_solid_angle(self) -> float:
        """Solid angle subtended by optics (sr)."""
        # Omega = pi / (4 * f_number^2) for circular aperture
        return np.pi / (4.0 * self.f_number ** 2)

    @property
    def ifov(self) -> float:
        """Instantaneous field of view (mrad)."""
        # IFOV = pixel_pitch / focal_length
        return self.pixel_pitch / self.focal_length  # um/mm = mrad

    def noise_equivalent_irradiance(self) -> float:
        """
        Calculate noise equivalent irradiance (NEI).

        Returns
        -------
        nei : float
            NEI in W/cm^2
        """
        # NEI = sqrt(A * delta_f) / D*
        # where A is detector area, delta_f is bandwidth
        area = self.pixel_area
        bandwidth = self.electrical_bandwidth
        nei = np.sqrt(area * bandwidth) / self.d_star
        return nei

    def noise_equivalent_power(self) -> float:
        """
        Calculate noise equivalent power (NEP).

        Returns
        -------
        nep : float
            NEP in W
        """
        # NEP = sqrt(A * delta_f) / D*
        return self.noise_equivalent_irradiance() * self.pixel_area

    def signal_to_noise(self, irradiance: float) -> float:
        """
        Calculate signal-to-noise ratio for given irradiance.

        Parameters
        ----------
        irradiance : float
            Target irradiance at detector (W/cm^2)

        Returns
        -------
        snr : float
            Signal-to-noise ratio
        """
        nei = self.noise_equivalent_irradiance()
        return irradiance / nei

    def signal_electrons(self, irradiance: float) -> float:
        """
        Calculate signal in electrons for given irradiance.

        Parameters
        ----------
        irradiance : float
            Irradiance at detector (W/cm^2)

        Returns
        -------
        electrons : float
            Number of signal electrons
        """
        # Energy per photon at center wavelength
        wavelength_m = self.center_wavelength * 1e-6
        photon_energy = PLANCK_CONSTANT * SPEED_OF_LIGHT / wavelength_m

        # Power on pixel
        power = irradiance * self.pixel_area  # W

        # Photon flux
        photon_rate = power / photon_energy  # photons/s

        # Electrons
        electrons = (photon_rate * self.quantum_efficiency *
                    self.fill_factor * self.integration_time * 1e-3)

        return electrons

    def noise_electrons(self) -> float:
        """
        Calculate total noise in electrons.

        Returns
        -------
        noise : float
            Total noise (electrons RMS)
        """
        # Dark current noise
        dark_electrons = (self.dark_current * self.pixel_area *
                         self.integration_time * 1e-3 / 1.6e-19)
        dark_noise = np.sqrt(dark_electrons)

        # Total noise (quadrature sum)
        total_noise = np.sqrt(self.read_noise**2 + dark_noise**2)

        return total_noise

    def minimum_detectable_irradiance(self, snr_threshold: float = 1.0) -> float:
        """
        Calculate minimum detectable irradiance for given SNR.

        Parameters
        ----------
        snr_threshold : float
            Required SNR for detection

        Returns
        -------
        irradiance : float
            Minimum detectable irradiance (W/cm^2)
        """
        return self.noise_equivalent_irradiance() * snr_threshold


def InSbDetector(
    name: str = "InSb MWIR",
    pixel_pitch: float = 15.0,
    f_number: float = 2.0,
    integration_time: float = 10.0,
) -> FPADetector:
    """
    Create an InSb (Indium Antimonide) MWIR detector.

    InSb detectors are the standard for MWIR (3-5 um) applications.
    Requires cryogenic cooling to ~77K.

    Parameters
    ----------
    name : str
        Detector name
    pixel_pitch : float
        Pixel spacing in um (typical: 15-30 um)
    f_number : float
        Optical system f-number
    integration_time : float
        Integration time in ms

    Returns
    -------
    detector : FPADetector
        Configured InSb detector
    """
    return FPADetector(
        name=name,
        spectral_band=(3.0, 5.0),  # MWIR
        d_star=1e11,  # Typical D* for InSb at 77K
        pixel_pitch=pixel_pitch,
        netd=20.0,  # mK
        quantum_efficiency=0.85,
        fill_factor=0.80,
        well_capacity=1.2e7,
        read_noise=40.0,
        dark_current=1e-10,  # Low due to cooling
        operating_temp=77.0,
        f_number=f_number,
        integration_time=integration_time,
    )


def MCTDetector(
    name: str = "MCT LWIR",
    spectral_band: Tuple[float, float] = (8.0, 12.0),
    pixel_pitch: float = 20.0,
    f_number: float = 2.0,
    integration_time: float = 10.0,
) -> FPADetector:
    """
    Create an MCT (HgCdTe / Mercury Cadmium Telluride) detector.

    MCT detectors can be tuned for MWIR or LWIR by varying
    the Cd/Hg ratio. LWIR MCT requires cooling to 77K or below.

    Parameters
    ----------
    name : str
        Detector name
    spectral_band : Tuple[float, float]
        Wavelength range (um), typical LWIR: (8, 12)
    pixel_pitch : float
        Pixel spacing in um (typical: 15-30 um)
    f_number : float
        Optical system f-number
    integration_time : float
        Integration time in ms

    Returns
    -------
    detector : FPADetector
        Configured MCT detector
    """
    # D* depends on cutoff wavelength
    center_wl = (spectral_band[0] + spectral_band[1]) / 2
    if center_wl < 5.0:
        # MWIR MCT
        d_star = 5e10
        netd = 25.0
    else:
        # LWIR MCT
        d_star = 2e10
        netd = 30.0

    return FPADetector(
        name=name,
        spectral_band=spectral_band,
        d_star=d_star,
        pixel_pitch=pixel_pitch,
        netd=netd,
        quantum_efficiency=0.70,
        fill_factor=0.75,
        well_capacity=1e7,
        read_noise=60.0,
        dark_current=5e-9,
        operating_temp=77.0,
        f_number=f_number,
        integration_time=integration_time,
    )


def detector_from_type(
    detector_type: str,
    **kwargs,
) -> FPADetector:
    """
    Create a detector from type string.

    Parameters
    ----------
    detector_type : str
        One of: 'insb', 'mct_mwir', 'mct_lwir'
    **kwargs
        Additional parameters passed to detector constructor

    Returns
    -------
    detector : FPADetector
        Configured detector
    """
    detector_type = detector_type.lower()

    if detector_type == 'insb':
        return InSbDetector(**kwargs)
    elif detector_type == 'mct_mwir':
        return MCTDetector(
            name=kwargs.pop('name', 'MCT MWIR'),
            spectral_band=(3.0, 5.0),
            **kwargs
        )
    elif detector_type == 'mct_lwir':
        return MCTDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
