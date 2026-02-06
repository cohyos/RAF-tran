"""
Aircraft Infrared Signature Models
==================================

This module provides models for aircraft thermal signatures in
the infrared bands (MWIR and LWIR).

Signature Components:
- Exhaust plume: Hot gases from engine(s), dominant in MWIR
- Exhaust nozzle: Metal at elevated temperature
- Aerodynamically heated skin: Kinetic heating at high speed
- Sun-heated surfaces: Solar irradiance contribution

Aspect Dependence:
- Rear aspect: Maximum exhaust visibility
- Side aspect: Reduced exhaust, more skin area
- Front aspect: Minimum signature, some inlet radiation

References:
- Mahulikar et al. (2007). Infrared signature studies of aerospace vehicles
- Hudson (1969). Infrared System Engineering
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

from raf_tran.utils.constants import STEFAN_BOLTZMANN
from raf_tran.utils.spectral import planck_function


@dataclass
class AircraftSignature:
    """
    Aircraft infrared signature model.

    Parameters
    ----------
    name : str
        Aircraft identifier
    exhaust_temp : float
        Exhaust plume temperature (K)
    exhaust_area : float
        Effective exhaust emitting area (m^2)
    exhaust_emissivity : float
        Exhaust emissivity (0-1)
    nozzle_temp : float
        Exhaust nozzle temperature (K)
    nozzle_area : float
        Nozzle visible area (m^2)
    nozzle_emissivity : float
        Nozzle emissivity (0-1)
    skin_temp : float
        Aircraft skin temperature (K)
    skin_area : float
        Visible skin area (m^2)
    skin_emissivity : float
        Skin emissivity (0-1)
    aspect : str
        Viewing aspect ('rear', 'side', 'front')
    characteristic_dimension_m : float
        Characteristic dimension for Johnson criteria (m)
        Typically the aircraft length or wingspan
    """
    name: str
    exhaust_temp: float = 700.0  # K
    exhaust_area: float = 0.5  # m^2
    exhaust_emissivity: float = 0.9
    nozzle_temp: float = 600.0  # K
    nozzle_area: float = 0.3  # m^2
    nozzle_emissivity: float = 0.85
    skin_temp: float = 300.0  # K (ambient or heated)
    skin_area: float = 20.0  # m^2
    skin_emissivity: float = 0.9
    aspect: str = "rear"
    characteristic_dimension_m: float = 15.0  # m (for Johnson criteria)

    def total_radiant_intensity_band(
        self,
        wavelength_min: float,
        wavelength_max: float,
        n_points: int = 100,
    ) -> float:
        """
        Calculate total radiant intensity in a spectral band.

        Parameters
        ----------
        wavelength_min : float
            Minimum wavelength (um)
        wavelength_max : float
            Maximum wavelength (um)
        n_points : int
            Number of integration points

        Returns
        -------
        intensity : float
            Radiant intensity (W/sr)
        """
        wavelengths = np.linspace(wavelength_min, wavelength_max, n_points) * 1e-6  # m

        # Integrate radiance over band for each component
        total_intensity = 0.0

        # Exhaust contribution
        if self.exhaust_area > 0 and self.exhaust_temp > 0:
            B_exhaust = planck_function(wavelengths, self.exhaust_temp)
            # Integrate: W/m^2/sr/m * m * m^2 -> W/sr
            radiance_exhaust = np.trapezoid(B_exhaust, wavelengths)
            total_intensity += self.exhaust_emissivity * self.exhaust_area * radiance_exhaust / np.pi

        # Nozzle contribution
        if self.nozzle_area > 0 and self.nozzle_temp > 0:
            B_nozzle = planck_function(wavelengths, self.nozzle_temp)
            radiance_nozzle = np.trapezoid(B_nozzle, wavelengths)
            total_intensity += self.nozzle_emissivity * self.nozzle_area * radiance_nozzle / np.pi

        # Skin contribution
        if self.skin_area > 0 and self.skin_temp > 0:
            B_skin = planck_function(wavelengths, self.skin_temp)
            radiance_skin = np.trapezoid(B_skin, wavelengths)
            total_intensity += self.skin_emissivity * self.skin_area * radiance_skin / np.pi

        return total_intensity

    def radiant_intensity_mwir(self) -> float:
        """
        Calculate radiant intensity in MWIR band (3-5 um).

        Returns
        -------
        intensity : float
            Radiant intensity (W/sr)
        """
        return self.total_radiant_intensity_band(3.0, 5.0)

    def radiant_intensity_lwir(self) -> float:
        """
        Calculate radiant intensity in LWIR band (8-12 um).

        Returns
        -------
        intensity : float
            Radiant intensity (W/sr)
        """
        return self.total_radiant_intensity_band(8.0, 12.0)

    def irradiance_at_range(
        self,
        range_m: float,
        wavelength_min: float,
        wavelength_max: float,
        atmospheric_transmission: float = 1.0,
    ) -> float:
        """
        Calculate irradiance at detector from target at given range.

        Parameters
        ----------
        range_m : float
            Range to target (m)
        wavelength_min : float
            Minimum wavelength (um)
        wavelength_max : float
            Maximum wavelength (um)
        atmospheric_transmission : float
            Atmospheric transmission (0-1)

        Returns
        -------
        irradiance : float
            Irradiance at detector (W/m^2)
        """
        intensity = self.total_radiant_intensity_band(wavelength_min, wavelength_max)

        # Inverse square law: E = I / R^2
        irradiance = intensity / (range_m ** 2)

        # Apply atmospheric transmission
        irradiance *= atmospheric_transmission

        return irradiance

    def contrast_temperature(
        self,
        background_temp: float,
        wavelength_min: float,
        wavelength_max: float,
    ) -> float:
        """
        Calculate effective contrast temperature against background.

        Parameters
        ----------
        background_temp : float
            Background temperature (K)
        wavelength_min : float
            Minimum wavelength (um)
        wavelength_max : float
            Maximum wavelength (um)

        Returns
        -------
        delta_t : float
            Contrast temperature (K)
        """
        # Simplified: use area-weighted average temperature
        total_area = self.exhaust_area + self.nozzle_area + self.skin_area
        if total_area == 0:
            return 0.0

        avg_temp = (
            self.exhaust_area * self.exhaust_temp +
            self.nozzle_area * self.nozzle_temp +
            self.skin_area * self.skin_temp
        ) / total_area

        return avg_temp - background_temp

    @classmethod
    def with_aspect(cls, base_signature: 'AircraftSignature', aspect: str) -> 'AircraftSignature':
        """
        Create signature with different viewing aspect.

        Parameters
        ----------
        base_signature : AircraftSignature
            Base aircraft signature
        aspect : str
            New viewing aspect ('rear', 'side', 'front')

        Returns
        -------
        signature : AircraftSignature
            Adjusted signature for aspect
        """
        # Aspect factors for different components
        aspect_factors = {
            'rear': {'exhaust': 1.0, 'nozzle': 1.0, 'skin': 0.3},
            'side': {'exhaust': 0.3, 'nozzle': 0.5, 'skin': 1.0},
            'front': {'exhaust': 0.05, 'nozzle': 0.1, 'skin': 0.4},
        }

        factors = aspect_factors.get(aspect.lower(), aspect_factors['rear'])

        return cls(
            name=f"{base_signature.name} ({aspect})",
            exhaust_temp=base_signature.exhaust_temp,
            exhaust_area=base_signature.exhaust_area * factors['exhaust'],
            exhaust_emissivity=base_signature.exhaust_emissivity,
            nozzle_temp=base_signature.nozzle_temp,
            nozzle_area=base_signature.nozzle_area * factors['nozzle'],
            nozzle_emissivity=base_signature.nozzle_emissivity,
            skin_temp=base_signature.skin_temp,
            skin_area=base_signature.skin_area * factors['skin'],
            skin_emissivity=base_signature.skin_emissivity,
            aspect=aspect,
            characteristic_dimension_m=base_signature.characteristic_dimension_m,
        )


def generic_fighter(
    aspect: str = "rear",
    afterburner: bool = False,
    mach: float = 0.9,
) -> AircraftSignature:
    """
    Create generic fighter aircraft signature.

    Parameters
    ----------
    aspect : str
        Viewing aspect ('rear', 'side', 'front')
    afterburner : bool
        Whether afterburner is engaged
    mach : float
        Flight Mach number (affects skin temperature)

    Returns
    -------
    signature : AircraftSignature
        Fighter aircraft signature
    """
    # Skin heating due to aerodynamic friction
    # T_recovery = T_ambient * (1 + 0.2 * M^2) approximately
    t_ambient = 220.0  # K at altitude
    t_skin = t_ambient * (1 + 0.2 * mach ** 2)

    if afterburner:
        exhaust_temp = 1800.0  # K (afterburner)
        exhaust_area = 1.5  # m^2 (larger plume)
    else:
        exhaust_temp = 700.0  # K (military power)
        exhaust_area = 0.5  # m^2

    base = AircraftSignature(
        name="Generic Fighter",
        exhaust_temp=exhaust_temp,
        exhaust_area=exhaust_area,
        exhaust_emissivity=0.9,
        nozzle_temp=exhaust_temp * 0.7,  # Nozzle cooler than exhaust
        nozzle_area=0.4,
        nozzle_emissivity=0.85,
        skin_temp=t_skin,
        skin_area=25.0,  # m^2 total visible area
        skin_emissivity=0.9,
        aspect=aspect,
        characteristic_dimension_m=15.0,  # Fighter length ~15m
    )

    return AircraftSignature.with_aspect(base, aspect)


def generic_transport(
    aspect: str = "rear",
    mach: float = 0.8,
) -> AircraftSignature:
    """
    Create generic transport aircraft signature.

    Parameters
    ----------
    aspect : str
        Viewing aspect ('rear', 'side', 'front')
    mach : float
        Flight Mach number

    Returns
    -------
    signature : AircraftSignature
        Transport aircraft signature
    """
    t_ambient = 220.0  # K at altitude
    t_skin = t_ambient * (1 + 0.2 * mach ** 2)

    base = AircraftSignature(
        name="Generic Transport",
        exhaust_temp=550.0,  # K (turbofan, lower than fighter)
        exhaust_area=1.0,  # m^2 (4 engines, but shielded)
        exhaust_emissivity=0.85,
        nozzle_temp=450.0,
        nozzle_area=0.8,
        nozzle_emissivity=0.8,
        skin_temp=t_skin,
        skin_area=150.0,  # m^2 (larger aircraft)
        skin_emissivity=0.9,
        aspect=aspect,
        characteristic_dimension_m=60.0,  # Large transport ~60m
    )

    return AircraftSignature.with_aspect(base, aspect)


def generic_uav(
    aspect: str = "rear",
    mach: float = 0.4,
) -> AircraftSignature:
    """
    Create generic UAV/drone signature.

    Parameters
    ----------
    aspect : str
        Viewing aspect ('rear', 'side', 'front')
    mach : float
        Flight Mach number

    Returns
    -------
    signature : AircraftSignature
        UAV signature
    """
    t_ambient = 250.0  # K (lower altitude typically)
    t_skin = t_ambient * (1 + 0.2 * mach ** 2)

    base = AircraftSignature(
        name="Generic UAV",
        exhaust_temp=450.0,  # K (small turbine or piston)
        exhaust_area=0.05,  # m^2 (small engine)
        exhaust_emissivity=0.85,
        nozzle_temp=400.0,
        nozzle_area=0.02,
        nozzle_emissivity=0.8,
        skin_temp=t_skin,
        skin_area=5.0,  # m^2 (small platform)
        skin_emissivity=0.9,
        aspect=aspect,
        characteristic_dimension_m=3.0,  # Medium UAV wingspan ~3m
    )

    return AircraftSignature.with_aspect(base, aspect)
