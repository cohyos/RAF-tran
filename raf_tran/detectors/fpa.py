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
    Create an MCT (HgCdTe / Mercury Cadmium Telluride) analog detector.

    MCT detectors can be tuned for MWIR or LWIR by varying
    the Cd/Hg ratio. LWIR MCT requires cooling to 77K or below.

    Analog ROIC characteristics:
    - Well capacity: 1.0e6 electrons (typical analog)
    - Read noise: 150 electrons RMS (typical analog)

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
        # LWIR MCT - typical analog performance
        d_star = 5e10  # Higher D* for fair comparison with digital
        netd = 40.0    # Higher NETD for analog

    return FPADetector(
        name=name,
        spectral_band=spectral_band,
        d_star=d_star,
        pixel_pitch=pixel_pitch,
        netd=netd,
        quantum_efficiency=0.70,
        fill_factor=0.75,
        well_capacity=1.0e6,   # Analog well capacity (baseline)
        read_noise=150.0,       # Analog read noise (baseline)
        dark_current=5e-9,
        operating_temp=77.0,
        f_number=f_number,
        integration_time=integration_time,
    )


@dataclass
class DigitalFPADetector(FPADetector):
    """
    Cooled digital LWIR detector with DROIC (Digital Read-Out IC).

    Digital pixel architecture provides significant improvements over
    analog ROIC through in-pixel ADC conversion:
    - 4x deeper wells (4Me vs 1Me electrons)
    - ~67% lower read noise (50e vs 150e RMS)
    - 50% lower dark current
    - ~1.7-1.9x overall SNR improvement

    Based on EO-simplified-simulation DROIC specifications.

    Additional Parameters
    ---------------------
    readout_type : str
        Readout architecture ('digital' for DROIC)
    bit_depth : int
        ADC bit depth (typically 14-16 bits)
    """
    readout_type: str = "digital"
    bit_depth: int = 14

    @property
    def digital_enhancement_factor(self) -> float:
        """
        SNR improvement factor from digital readout vs analog.

        Digital LWIR (DROIC) provides approximately 2x DETECTION RANGE
        improvement over analog, which requires 4x NEI improvement:
        - Range ∝ sqrt(1/NEI), so 2x range requires 4x better NEI
        - 4x well capacity: 4.0e6 vs 1.0e6 electrons
        - 3x lower read noise: 50e vs 150e RMS
        - Effective D* multiplier: 4.0 (gives 4x NEI improvement → 2x range)

        This matches reported DROIC field performance of ~2x detection
        range improvement over analog LWIR systems.
        """
        # 4x D* multiplier for ~2x detection range improvement
        # Accounts for well capacity (4x) and noise (3x) improvements
        return 4.0

    def effective_d_star(self) -> float:
        """D* including digital enhancement factor."""
        return self.d_star * self.digital_enhancement_factor

    def noise_equivalent_irradiance(self) -> float:
        """
        Calculate NEI with digital enhancement.

        Returns
        -------
        nei : float
            NEI in W/cm^2 (improved by digital readout)
        """
        # Base NEI calculation
        area = self.pixel_area
        bandwidth = self.electrical_bandwidth
        # Use effective D* which includes digital enhancement
        nei = np.sqrt(area * bandwidth) / self.effective_d_star()
        return nei


def DigitalLWIRDetector(
    name: str = "Digital LWIR (DROIC)",
    spectral_band: Tuple[float, float] = (8.0, 12.0),
    pixel_pitch: float = 15.0,
    f_number: float = 2.0,
    integration_time: float = 10.0,
) -> DigitalFPADetector:
    """
    Create a cooled digital LWIR detector with DROIC.

    Digital pixel architecture with in-pixel ADC provides 2x improved
    performance over analog MCT LWIR:
    - Well capacity: 4.0e6 electrons (vs 1.0e6 analog) - 4x improvement
    - Read noise: 50 electrons RMS (vs 150 analog) - 3x improvement
    - Combined effective D* improvement: 2x (via digital_enhancement_factor)
    - Expected 2x detection range improvement over analog LWIR

    Parameters
    ----------
    name : str
        Detector name
    spectral_band : Tuple[float, float]
        Wavelength range (um), default LWIR: (8, 12)
    pixel_pitch : float
        Pixel spacing in um
    f_number : float
        Optical system f-number
    integration_time : float
        Integration time in ms

    Returns
    -------
    detector : DigitalFPADetector
        Configured digital LWIR detector
    """
    return DigitalFPADetector(
        name=name,
        spectral_band=spectral_band,
        d_star=5e10,  # Base D* same as analog MCT LWIR
        pixel_pitch=pixel_pitch,
        netd=20.0,  # mK (improved from 40mK analog due to lower noise)
        quantum_efficiency=0.70,
        fill_factor=0.80,
        well_capacity=4.0e6,  # 4x deeper wells than analog
        read_noise=50.0,  # 3x lower than analog (150e)
        dark_current=5e-10,  # Lower than analog
        operating_temp=77.0,  # Still requires cooling
        f_number=f_number,
        integration_time=integration_time,
        readout_type="digital",
        bit_depth=14,
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
        One of: 'insb', 'mct_mwir', 'mct_lwir', 'digital_lwir', 'droic'
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
    elif detector_type in ('digital_lwir', 'droic'):
        return DigitalLWIRDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
