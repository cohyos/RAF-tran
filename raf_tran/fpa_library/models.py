"""
FPA Library Data Models
=======================

Core data models for Focal Plane Arrays (FPAs), Readout Integrated Circuits
(ROICs), detector materials, and vendor specifications.

This module is designed to be modular and can be used independently of RAF-tran.

References
----------
- SCD Product Portfolio (2024)
- Teledyne FLIR Product Catalog (2024)
- L3Harris Electro-Optical Systems
- Raytheon Vision Systems 3rd Generation FLIR
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import json


# =============================================================================
# Enumerations
# =============================================================================

class SpectralBand(Enum):
    """Infrared spectral band classification."""
    SWIR = "SWIR"       # 0.9 - 1.7 um (Short-Wave IR)
    MWIR = "MWIR"       # 3.0 - 5.0 um (Mid-Wave IR)
    LWIR = "LWIR"       # 8.0 - 14.0 um (Long-Wave IR)
    VLWIR = "VLWIR"     # 14.0 - 30.0 um (Very Long-Wave IR)
    DUAL_MW_LW = "DUAL_MW_LW"  # Dual-band MW/LW
    VISIBLE = "VIS"     # 0.4 - 0.9 um (Visible)
    NIR = "NIR"         # 0.7 - 1.0 um (Near IR)


class DetectorType(Enum):
    """Detector material / technology type."""
    InSb = "InSb"
    MCT = "MCT"                   # Mercury Cadmium Telluride (HgCdTe)
    T2SL = "T2SL"                 # Type-II Superlattice
    XBn = "XBn"                   # nBn barrier (InAsSb)
    HFM = "HFM"                   # Hot FPA MWIR (SCD proprietary)
    VOx = "VOx"                   # Vanadium Oxide microbolometer
    aSi = "aSi"                   # Amorphous Silicon microbolometer
    InGaAs = "InGaAs"             # Indium Gallium Arsenide
    QWIP = "QWIP"                 # Quantum Well Infrared Photodetector
    SLS = "SLS"                   # Strained Layer Superlattice
    DUAL_BAND = "DUAL_BAND"       # Dual-band structure


class CoolingType(Enum):
    """Cooling technology."""
    COOLED_STIRLING = "Cooled (Stirling)"
    COOLED_JT = "Cooled (Joule-Thomson)"
    COOLED_PULSE_TUBE = "Cooled (Pulse Tube)"
    COOLED_GENERIC = "Cooled"
    UNCOOLED = "Uncooled"
    TEC = "TEC"  # Thermoelectric cooler


class IntegrationMode(Enum):
    """ROIC integration modes."""
    ITR = "ITR"   # Integrate-Then-Read
    IWR = "IWR"   # Integrate-While-Read
    CTIA = "CTIA"  # Capacitive Transimpedance Amplifier
    DI = "DI"      # Direct Injection
    SNAPSHOT = "Snapshot"


class InterfaceType(Enum):
    """Digital output interface types."""
    CAMERA_LINK = "Camera Link"
    CAMERA_LINK_HS = "Camera Link HS"
    USB2 = "USB 2.0"
    USB3 = "USB 3.0"
    GIGE = "GigE Vision"
    COAXPRESS = "CoaXPress"
    MIPI_CSI2 = "MIPI CSI-2"
    CMOS_PARALLEL = "CMOS Parallel"
    SPI = "SPI"
    UART = "UART"
    RS232 = "RS-232"
    RS422 = "RS-422"
    SDI = "SDI"
    ANALOG = "Analog Video"
    LVDS = "LVDS"
    I2C = "I2C"


class ApplicationDomain(Enum):
    """Primary application domains."""
    SURVEILLANCE = "Surveillance"
    TARGETING = "Targeting/Fire Control"
    MISSILE_WARNING = "Missile Warning System"
    DRONE_PAYLOAD = "Drone/UAV Payload"
    HANDHELD = "Handheld/Weapon Sight"
    NAVAL = "Naval/Maritime"
    DVE = "Driver Vision Enhancement"
    SCIENTIFIC = "Scientific/Research"
    INDUSTRIAL = "Industrial Thermography"
    SPACE = "Space/Satellite"
    SEARCH_RESCUE = "Search and Rescue"
    ISR = "Intelligence, Surveillance, Reconnaissance"
    WIDE_AREA = "Wide Area Persistent Surveillance"


class Vendor(Enum):
    """FPA and ROIC manufacturers."""
    SCD = "Semi Conductor Devices (SCD)"
    TELEDYNE_FLIR = "Teledyne FLIR"
    L3HARRIS = "L3Harris"
    RAYTHEON = "Raytheon Vision Systems"
    DRS = "DRS / Leonardo DRS"
    XENICS = "Exosens (Xenics)"
    AXIOM = "Axiom Optics"
    LIGHTPATH = "LightPath Technologies"
    SIERRA_OLYMPIA = "Sierra-Olympia Technologies"
    SOFRADIR = "Sofradir / Lynred"


# =============================================================================
# Core Data Models
# =============================================================================

@dataclass
class SpectralRange:
    """Spectral wavelength range in micrometers."""
    min_um: float
    max_um: float

    @property
    def center_um(self) -> float:
        """Center wavelength in micrometers."""
        return (self.min_um + self.max_um) / 2.0

    @property
    def bandwidth_um(self) -> float:
        """Bandwidth in micrometers."""
        return self.max_um - self.min_um

    def __str__(self) -> str:
        return f"{self.min_um:.1f}-{self.max_um:.1f} um"


@dataclass
class ArrayFormat:
    """FPA array dimensions."""
    columns: int  # Horizontal pixels
    rows: int     # Vertical pixels

    @property
    def total_pixels(self) -> int:
        return self.columns * self.rows

    @property
    def megapixels(self) -> float:
        return self.total_pixels / 1e6

    @property
    def aspect_ratio(self) -> float:
        return self.columns / self.rows

    def active_area_mm(self, pitch_um: float) -> Tuple[float, float]:
        """Calculate active area in mm for a given pixel pitch."""
        width_mm = self.columns * pitch_um / 1000.0
        height_mm = self.rows * pitch_um / 1000.0
        return (width_mm, height_mm)

    def diagonal_mm(self, pitch_um: float) -> float:
        """Calculate diagonal of active area in mm."""
        w, h = self.active_area_mm(pitch_um)
        return (w**2 + h**2)**0.5

    def __str__(self) -> str:
        return f"{self.columns} x {self.rows}"


@dataclass
class ROICSpec:
    """Readout Integrated Circuit specification."""
    name: str
    vendor: Vendor
    array_format: ArrayFormat
    pixel_pitch_um: float
    well_capacity_e: Optional[float] = None        # electrons
    adc_bits: Optional[int] = None                  # ADC resolution in bits
    integration_modes: List[IntegrationMode] = field(default_factory=list)
    max_frame_rate_hz: Optional[float] = None
    power_mw: Optional[float] = None                # Power in mW
    process_node_um: Optional[float] = None         # CMOS process node
    outputs: Optional[int] = None                   # Number of output channels
    interfaces: List[InterfaceType] = field(default_factory=list)
    dual_polarity: bool = False
    two_color: bool = False
    windowing: bool = True
    notes: str = ""

    @property
    def well_capacity_Me(self) -> Optional[float]:
        """Well capacity in mega-electrons."""
        if self.well_capacity_e is not None:
            return self.well_capacity_e / 1e6
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        d['name'] = self.name
        d['vendor'] = self.vendor.value
        d['array_format'] = str(self.array_format)
        d['pixel_pitch_um'] = self.pixel_pitch_um
        d['well_capacity_e'] = self.well_capacity_e
        d['adc_bits'] = self.adc_bits
        d['integration_modes'] = [m.value for m in self.integration_modes]
        d['max_frame_rate_hz'] = self.max_frame_rate_hz
        d['power_mw'] = self.power_mw
        d['process_node_um'] = self.process_node_um
        d['interfaces'] = [i.value for i in self.interfaces]
        d['dual_polarity'] = self.dual_polarity
        d['two_color'] = self.two_color
        d['notes'] = self.notes
        return d


@dataclass
class FPASpec:
    """
    Focal Plane Array specification.

    This is the central data model for the FPA library. It captures all
    relevant parameters of a focal plane array sensor including detector
    material, array format, performance metrics, and integration details.
    """
    # Identity
    name: str
    vendor: Vendor
    product_family: str = ""

    # Detector
    detector_type: DetectorType = DetectorType.InSb
    spectral_band: SpectralBand = SpectralBand.MWIR
    spectral_range: Optional[SpectralRange] = None
    cooling: CoolingType = CoolingType.COOLED_GENERIC
    operating_temp_k: Optional[float] = None

    # Array
    array_format: Optional[ArrayFormat] = None
    pixel_pitch_um: float = 15.0

    # Performance
    netd_mk: Optional[float] = None           # NETD in milliKelvin
    detectivity_jones: Optional[float] = None  # D* in Jones (cm*Hz^0.5/W)
    dark_current_na: Optional[float] = None    # Dark current in nA
    quantum_efficiency: Optional[float] = None # QE (0-1)
    operability_pct: Optional[float] = None    # Operability percentage

    # Readout
    roic: Optional[ROICSpec] = None
    adc_bits: Optional[int] = None
    well_capacity_e: Optional[float] = None
    max_frame_rate_hz: Optional[float] = None
    integration_modes: List[IntegrationMode] = field(default_factory=list)
    interfaces: List[InterfaceType] = field(default_factory=list)

    # SWaP-C
    weight_g: Optional[float] = None
    power_w: Optional[float] = None
    power_steady_w: Optional[float] = None
    dimensions_mm: Optional[Tuple[float, float, float]] = None  # W x H x D
    cooldown_time_min: Optional[float] = None
    mttf_hours: Optional[float] = None

    # Application
    applications: List[ApplicationDomain] = field(default_factory=list)
    is_itar: bool = False
    is_export_controlled: bool = True
    notes: str = ""

    # Windowing
    windowed_frame_rate_hz: Optional[float] = None
    window_step_rows: Optional[int] = None

    @property
    def resolution_str(self) -> str:
        if self.array_format:
            return str(self.array_format)
        return "Unknown"

    @property
    def total_pixels(self) -> Optional[int]:
        if self.array_format:
            return self.array_format.total_pixels
        return None

    @property
    def megapixels(self) -> Optional[float]:
        if self.array_format:
            return self.array_format.megapixels
        return None

    @property
    def active_area_mm(self) -> Optional[Tuple[float, float]]:
        if self.array_format:
            return self.array_format.active_area_mm(self.pixel_pitch_um)
        return None

    @property
    def f_number_limited_ifov_urad(self) -> Optional[float]:
        """Diffraction-limited IFOV in microradians (at center wavelength)."""
        if self.spectral_range and self.pixel_pitch_um:
            wl_um = self.spectral_range.center_um
            return 2.44 * wl_um / self.pixel_pitch_um
        return None

    @property
    def is_cooled(self) -> bool:
        return self.cooling not in (CoolingType.UNCOOLED, CoolingType.TEC)

    @property
    def is_hot(self) -> bool:
        """Check if this is a High Operating Temperature detector."""
        if self.operating_temp_k is not None:
            return self.operating_temp_k > 100  # Above ~100K = HOT
        return self.detector_type in (DetectorType.XBn, DetectorType.HFM)

    def ifov_at_focal_length(self, focal_length_mm: float) -> float:
        """Calculate IFOV in milliradians for a given focal length."""
        return self.pixel_pitch_um / focal_length_mm

    def fov_at_focal_length(self, focal_length_mm: float) -> Optional[Tuple[float, float]]:
        """Calculate full FOV in degrees for a given focal length."""
        if self.array_format is None:
            return None
        import math
        w_mm, h_mm = self.array_format.active_area_mm(self.pixel_pitch_um)
        fov_h = 2 * math.degrees(math.atan(w_mm / (2 * focal_length_mm)))
        fov_v = 2 * math.degrees(math.atan(h_mm / (2 * focal_length_mm)))
        return (fov_h, fov_v)

    def johnson_criteria_range(self, target_size_m: float,
                                focal_length_mm: float,
                                cycles: float = 6.0) -> float:
        """
        Estimate detection/recognition/identification range using Johnson
        criteria.

        Parameters
        ----------
        target_size_m : float
            Critical target dimension in meters
        focal_length_mm : float
            Lens focal length in mm
        cycles : float
            Required line pairs (Detection=1, Recognition=3, ID=6)

        Returns
        -------
        range_m : float
            Estimated range in meters
        """
        ifov_rad = self.pixel_pitch_um * 1e-3 / focal_length_mm
        pixels_on_target = 2 * cycles
        range_m = target_size_m / (pixels_on_target * ifov_rad)
        return range_m

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {}
        d['name'] = self.name
        d['vendor'] = self.vendor.value
        d['product_family'] = self.product_family
        d['detector_type'] = self.detector_type.value
        d['spectral_band'] = self.spectral_band.value
        if self.spectral_range:
            d['spectral_range'] = {
                'min_um': self.spectral_range.min_um,
                'max_um': self.spectral_range.max_um,
            }
        d['cooling'] = self.cooling.value
        d['operating_temp_k'] = self.operating_temp_k
        if self.array_format:
            d['array_format'] = {
                'columns': self.array_format.columns,
                'rows': self.array_format.rows,
            }
        d['pixel_pitch_um'] = self.pixel_pitch_um
        d['netd_mk'] = self.netd_mk
        d['detectivity_jones'] = self.detectivity_jones
        d['quantum_efficiency'] = self.quantum_efficiency
        d['adc_bits'] = self.adc_bits
        d['well_capacity_e'] = self.well_capacity_e
        d['max_frame_rate_hz'] = self.max_frame_rate_hz
        d['integration_modes'] = [m.value for m in self.integration_modes]
        d['interfaces'] = [i.value for i in self.interfaces]
        d['weight_g'] = self.weight_g
        d['power_w'] = self.power_w
        d['power_steady_w'] = self.power_steady_w
        d['cooldown_time_min'] = self.cooldown_time_min
        d['mttf_hours'] = self.mttf_hours
        d['applications'] = [a.value for a in self.applications]
        d['notes'] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FPASpec':
        """Create FPASpec from dictionary."""
        kwargs = {}
        kwargs['name'] = d['name']
        kwargs['vendor'] = Vendor(d['vendor'])
        kwargs['product_family'] = d.get('product_family', '')
        if 'detector_type' in d:
            kwargs['detector_type'] = DetectorType(d['detector_type'])
        if 'spectral_band' in d:
            kwargs['spectral_band'] = SpectralBand(d['spectral_band'])
        if 'spectral_range' in d and d['spectral_range']:
            sr = d['spectral_range']
            kwargs['spectral_range'] = SpectralRange(sr['min_um'], sr['max_um'])
        if 'cooling' in d:
            kwargs['cooling'] = CoolingType(d['cooling'])
        kwargs['operating_temp_k'] = d.get('operating_temp_k')
        if 'array_format' in d and d['array_format']:
            af = d['array_format']
            kwargs['array_format'] = ArrayFormat(af['columns'], af['rows'])
        kwargs['pixel_pitch_um'] = d.get('pixel_pitch_um', 15.0)
        kwargs['netd_mk'] = d.get('netd_mk')
        kwargs['detectivity_jones'] = d.get('detectivity_jones')
        kwargs['quantum_efficiency'] = d.get('quantum_efficiency')
        kwargs['adc_bits'] = d.get('adc_bits')
        kwargs['well_capacity_e'] = d.get('well_capacity_e')
        kwargs['max_frame_rate_hz'] = d.get('max_frame_rate_hz')
        if 'integration_modes' in d:
            kwargs['integration_modes'] = [IntegrationMode(m) for m in d['integration_modes']]
        if 'interfaces' in d:
            kwargs['interfaces'] = [InterfaceType(i) for i in d['interfaces']]
        kwargs['weight_g'] = d.get('weight_g')
        kwargs['power_w'] = d.get('power_w')
        kwargs['power_steady_w'] = d.get('power_steady_w')
        kwargs['cooldown_time_min'] = d.get('cooldown_time_min')
        kwargs['mttf_hours'] = d.get('mttf_hours')
        if 'applications' in d:
            kwargs['applications'] = [ApplicationDomain(a) for a in d['applications']]
        kwargs['notes'] = d.get('notes', '')
        return cls(**kwargs)

    def __str__(self) -> str:
        parts = [f"{self.name} ({self.vendor.value})"]
        if self.array_format:
            parts.append(f"  Format: {self.array_format} @ {self.pixel_pitch_um} um")
        parts.append(f"  Band: {self.spectral_band.value}, Material: {self.detector_type.value}")
        if self.netd_mk:
            parts.append(f"  NETD: {self.netd_mk} mK")
        if self.cooling:
            parts.append(f"  Cooling: {self.cooling.value}")
        return "\n".join(parts)
