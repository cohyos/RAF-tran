"""
FPA Database
============

Built-in database of Focal Plane Arrays, ROICs, and detector specifications
from major vendors: SCD, Teledyne FLIR, L3Harris, Raytheon Vision Systems,
and others.

Data sourced from publicly available vendor documentation and comparative
analysis reports.
"""

from typing import List, Dict, Optional
from raf_tran.fpa_library.models import (
    FPASpec, ROICSpec, ArrayFormat, SpectralRange,
    SpectralBand, DetectorType, CoolingType, IntegrationMode,
    InterfaceType, ApplicationDomain, Vendor,
)


# =============================================================================
# ROICs Database
# =============================================================================

def _build_roic_database() -> Dict[str, ROICSpec]:
    """Build the ROIC database."""
    roics = {}

    # --- Teledyne FLIR ISC Series ---
    roics['ISC1601'] = ROICSpec(
        name='ISC1601',
        vendor=Vendor.TELEDYNE_FLIR,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=8.0,
        well_capacity_e=2.6e6,
        integration_modes=[IntegrationMode.SNAPSHOT, IntegrationMode.IWR],
        interfaces=[InterfaceType.USB2, InterfaceType.RS232],
        notes="Used in Neutrino SX8. Prog. integration 0.01-16.6ms.",
    )

    roics['ISC0403'] = ROICSpec(
        name='ISC0403',
        vendor=Vendor.TELEDYNE_FLIR,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=15.0,
        well_capacity_e=7.0e6,
        integration_modes=[IntegrationMode.SNAPSHOT, IntegrationMode.ITR, IntegrationMode.IWR],
        interfaces=[InterfaceType.USB2, InterfaceType.UART],
        notes="Used in Neutrino LC. High flux capacity. Multi-output.",
    )

    roics['ISC1308'] = ROICSpec(
        name='ISC1308',
        vendor=Vendor.TELEDYNE_FLIR,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=12.0,
        well_capacity_e=3.0e6,
        dual_polarity=True,
        two_color=True,
        integration_modes=[IntegrationMode.SNAPSHOT],
        notes="Dual-polarity, two-color ROIC for simultaneous dual-band detection.",
    )

    roics['ISC1901'] = ROICSpec(
        name='ISC1901',
        vendor=Vendor.TELEDYNE_FLIR,
        array_format=ArrayFormat(2048, 1536),
        pixel_pitch_um=10.0,
        well_capacity_e=3.0e6,
        integration_modes=[IntegrationMode.SNAPSHOT, IntegrationMode.IWR],
        notes="Large-format, high-speed scientific/surveillance ROIC.",
    )

    roics['ISC1404'] = ROICSpec(
        name='ISC1404',
        vendor=Vendor.TELEDYNE_FLIR,
        array_format=ArrayFormat(2048, 2048),
        pixel_pitch_um=10.0,
        integration_modes=[IntegrationMode.SNAPSHOT, IntegrationMode.IWR],
        notes="Optimized for InSb and SLS. Large well capacity.",
    )

    # --- SCD ROICs ---
    roics['Pelican-D_LW_ROIC'] = ROICSpec(
        name='Pelican-D LW ROIC',
        vendor=Vendor.SCD,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=15.0,
        well_capacity_e=6.0e6,
        adc_bits=13,
        process_node_um=0.18,
        max_frame_rate_hz=360.0,
        power_mw=120.0,
        integration_modes=[IntegrationMode.ITR, IntegrationMode.IWR],
        interfaces=[InterfaceType.CAMERA_LINK],
        windowing=True,
        notes="0.18um CMOS. Bi-directional readout. 2-row windowing steps.",
    )

    # --- Raytheon ROICs ---
    roics['SB-350'] = ROICSpec(
        name='SB-350',
        vendor=Vendor.RAYTHEON,
        array_format=ArrayFormat(1280, 720),
        pixel_pitch_um=20.0,
        integration_modes=[IntegrationMode.IWR],
        notes="Dual-band MW/LW ROIC. TDMI architecture. 3rd Gen FLIR.",
    )

    roics['SB-275'] = ROICSpec(
        name='SB-275',
        vendor=Vendor.RAYTHEON,
        array_format=ArrayFormat(640, 480),
        pixel_pitch_um=20.0,
        integration_modes=[IntegrationMode.IWR],
        notes="Dual-band MW/LW ROIC for legacy systems.",
    )

    return roics


# =============================================================================
# FPA Database
# =============================================================================

def _build_fpa_database() -> Dict[str, FPASpec]:
    """Build the comprehensive FPA database."""
    fpas = {}

    # =========================================================================
    # SCD MWIR Products
    # =========================================================================

    fpas['SCD_Crane'] = FPASpec(
        name='Crane',
        vendor=Vendor.SCD,
        product_family='Crane',
        detector_type=DetectorType.XBn,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.6, 5.1),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=150.0,
        array_format=ArrayFormat(2560, 2048),
        pixel_pitch_um=5.0,
        applications=[
            ApplicationDomain.SURVEILLANCE,
            ApplicationDomain.ISR,
        ],
        notes="5-megapixel MWIR. 5um pitch enables miniaturized optics. XBn/HFM technology.",
    )

    fpas['SCD_Blackbird_1920'] = FPASpec(
        name='Blackbird 1920',
        vendor=Vendor.SCD,
        product_family='Blackbird',
        detector_type=DetectorType.InSb,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.6, 4.9),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=77.0,
        array_format=ArrayFormat(1920, 1536),
        pixel_pitch_um=10.0,
        applications=[
            ApplicationDomain.SURVEILLANCE,
            ApplicationDomain.ISR,
        ],
        notes="Digital High-Definition Imaging. InSb detector.",
    )

    fpas['SCD_Sparrow_HD'] = FPASpec(
        name='Sparrow-HD',
        vendor=Vendor.SCD,
        product_family='Sparrow',
        detector_type=DetectorType.XBn,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.6, 5.1),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=150.0,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=5.0,
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="XBn/HFM. Optimized for drones and miniature payloads.",
    )

    fpas['SCD_Mini_Blackbird_1280'] = FPASpec(
        name='Mini Blackbird 1280',
        vendor=Vendor.SCD,
        product_family='Mini Blackbird',
        detector_type=DetectorType.XBn,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.6, 5.1),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=150.0,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=10.0,
        applications=[
            ApplicationDomain.SURVEILLANCE,
            ApplicationDomain.DRONE_PAYLOAD,
        ],
        notes="HOT XBn. Low SWaP-C video cores.",
    )

    fpas['SCD_Hercules_1280'] = FPASpec(
        name='Hercules 1280',
        vendor=Vendor.SCD,
        product_family='Hercules',
        detector_type=DetectorType.InSb,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.6, 4.9),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=77.0,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=15.0,
        applications=[
            ApplicationDomain.SURVEILLANCE,
            ApplicationDomain.NAVAL,
        ],
        notes="InSb/XBn. Land and Naval Surveillance.",
    )

    fpas['SCD_Sundra'] = FPASpec(
        name='Sundra',
        vendor=Vendor.SCD,
        product_family='Sundra',
        detector_type=DetectorType.InSb,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.6, 4.9),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=77.0,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=15.0,
        applications=[
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="Long Range Thermal Surveillance.",
    )

    # =========================================================================
    # SCD LWIR Products
    # =========================================================================

    fpas['SCD_Pelican_D_LW'] = FPASpec(
        name='Pelican-D LW',
        vendor=Vendor.SCD,
        product_family='Pelican',
        detector_type=DetectorType.T2SL,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(7.5, 9.3),
        cooling=CoolingType.COOLED_STIRLING,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=15.0,
        netd_mk=15.0,
        adc_bits=13,
        well_capacity_e=6.0e6,
        max_frame_rate_hz=360.0,
        power_w=18.0,
        integration_modes=[IntegrationMode.ITR, IntegrationMode.IWR],
        interfaces=[InterfaceType.CAMERA_LINK],
        windowed_frame_rate_hz=360.0,
        window_step_rows=2,
        applications=[
            ApplicationDomain.TARGETING,
            ApplicationDomain.MISSILE_WARNING,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="IDDCA with digital ROIC. T2SL for uniformity and lower cost vs MCT. "
              "15mK NETD at 30Hz, 65% well fill. Bi-directional readout.",
    )

    # SCD Uncooled LWIR
    fpas['SCD_Bird_XGA'] = FPASpec(
        name='Bird XGA',
        vendor=Vendor.SCD,
        product_family='Bird',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(8.0, 14.0),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(1024, 768),
        pixel_pitch_um=17.0,
        applications=[
            ApplicationDomain.DVE,
            ApplicationDomain.HANDHELD,
        ],
        notes="VOx microbolometer. Ruggedized ceramic package.",
    )

    fpas['SCD_Bird_640'] = FPASpec(
        name='Bird 640',
        vendor=Vendor.SCD,
        product_family='Bird',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(8.0, 14.0),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(640, 480),
        pixel_pitch_um=17.0,
        applications=[
            ApplicationDomain.DVE,
            ApplicationDomain.HANDHELD,
        ],
        notes="VOx. Low SWaP-C, ceramic package.",
    )

    # =========================================================================
    # Teledyne FLIR Cooled Products
    # =========================================================================

    fpas['FLIR_Neutrino_SX8'] = FPASpec(
        name='Neutrino SX8',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='Neutrino',
        detector_type=DetectorType.XBn,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.4, 5.1),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=150.0,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=8.0,
        netd_mk=38.0,
        weight_g=420.0,
        power_steady_w=8.0,
        interfaces=[InterfaceType.USB2, InterfaceType.RS232, InterfaceType.RS422],
        applications=[
            ApplicationDomain.SURVEILLANCE,
            ApplicationDomain.ISR,
            ApplicationDomain.TARGETING,
        ],
        notes="Performance Series. HOT MWIR. SXGA format for long-range target ID.",
    )

    fpas['FLIR_Neutrino_LC'] = FPASpec(
        name='Neutrino LC',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='Neutrino',
        detector_type=DetectorType.XBn,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.4, 5.1),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=150.0,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=15.0,
        netd_mk=25.0,
        weight_g=370.0,
        power_steady_w=4.0,
        interfaces=[InterfaceType.USB2, InterfaceType.UART],
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.HANDHELD,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="SWaP Series. 370g, <4W steady state. Industry leader in miniaturization.",
    )

    # =========================================================================
    # Teledyne FLIR Uncooled Products
    # =========================================================================

    fpas['FLIR_Boson_Plus_640'] = FPASpec(
        name='Boson+ 640',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='Boson',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(7.5, 13.5),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=12.0,
        netd_mk=20.0,
        max_frame_rate_hz=60.0,
        weight_g=7.5,
        dimensions_mm=(21, 21, 11),
        interfaces=[InterfaceType.CMOS_PARALLEL, InterfaceType.USB2, InterfaceType.MIPI_CSI2],
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.HANDHELD,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="Gold standard for 12um uncooled. <=20mK NETD. 7.5g weight.",
    )

    fpas['FLIR_Boson_640'] = FPASpec(
        name='Boson 640',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='Boson',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(7.5, 13.5),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=12.0,
        netd_mk=40.0,
        max_frame_rate_hz=60.0,
        weight_g=7.5,
        dimensions_mm=(21, 21, 11),
        interfaces=[InterfaceType.CMOS_PARALLEL, InterfaceType.USB2, InterfaceType.UART],
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.HANDHELD,
        ],
        notes="Standard Boson. 12um VOx. <=40mK NETD.",
    )

    fpas['FLIR_Lepton_35'] = FPASpec(
        name='Lepton 3.5',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='Lepton',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(8.0, 14.0),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(160, 120),
        pixel_pitch_um=12.0,
        netd_mk=50.0,
        interfaces=[InterfaceType.SPI, InterfaceType.I2C],
        applications=[
            ApplicationDomain.INDUSTRIAL,
            ApplicationDomain.DRONE_PAYLOAD,
        ],
        notes="Ultra-compact thermal module. <50mK NETD. SPI/I2C interface.",
    )

    fpas['FLIR_Hadron_640'] = FPASpec(
        name='Hadron 640',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='Hadron',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(8.0, 14.0),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=12.0,
        interfaces=[InterfaceType.USB3, InterfaceType.MIPI_CSI2],
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="Dual-sensor: 640x512 radiometric LWIR + 64MP visible. AI-ready. "
              "Designed for sUAS and loitering munitions.",
    )

    fpas['FLIR_A6781_SLS'] = FPASpec(
        name='A6781 SLS',
        vendor=Vendor.TELEDYNE_FLIR,
        product_family='A6700',
        detector_type=DetectorType.SLS,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(7.5, 11.0),
        cooling=CoolingType.COOLED_STIRLING,
        array_format=ArrayFormat(640, 512),
        pixel_pitch_um=15.0,
        netd_mk=40.0,
        adc_bits=14,
        power_w=24.0,
        windowed_frame_rate_hz=4130.0,
        interfaces=[InterfaceType.GIGE],
        applications=[
            ApplicationDomain.SCIENTIFIC,
            ApplicationDomain.INDUSTRIAL,
        ],
        notes="SLS cooled LWIR. 4130Hz windowed mode. GigE interface.",
    )

    # =========================================================================
    # L3Harris Products
    # =========================================================================

    fpas['L3H_2K'] = FPASpec(
        name='2K x 2K Sensor Engine',
        vendor=Vendor.L3HARRIS,
        product_family='Large-Format IR',
        detector_type=DetectorType.InSb,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.0, 5.0),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=77.0,
        array_format=ArrayFormat(2048, 2048),
        pixel_pitch_um=15.0,
        max_frame_rate_hz=30.0,
        applications=[
            ApplicationDomain.WIDE_AREA,
            ApplicationDomain.SPACE,
            ApplicationDomain.ISR,
        ],
        notes="Reticulated MWIR InSb. 30.7x30.7mm active area. 4x14-bit parallel outputs.",
    )

    fpas['L3H_4K'] = FPASpec(
        name='4K x 4K Sensor Engine',
        vendor=Vendor.L3HARRIS,
        product_family='Large-Format IR',
        detector_type=DetectorType.InSb,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.0, 5.0),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=77.0,
        array_format=ArrayFormat(4096, 4096),
        pixel_pitch_um=15.0,
        max_frame_rate_hz=15.0,
        applications=[
            ApplicationDomain.WIDE_AREA,
            ApplicationDomain.SPACE,
        ],
        notes="16-megapixel InSb. 61.4x61.4mm active area (86.9mm diagonal). "
              "4x14-bit parallel digital interfaces.",
    )

    fpas['L3H_Onyx_Micro'] = FPASpec(
        name='Onyx Micro',
        vendor=Vendor.L3HARRIS,
        product_family='Onyx',
        detector_type=DetectorType.XBn,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.4, 5.1),
        cooling=CoolingType.COOLED_STIRLING,
        operating_temp_k=150.0,
        array_format=ArrayFormat(1280, 720),
        pixel_pitch_um=8.0,
        max_frame_rate_hz=60.0,
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="Small SWaP HOT MWIR. 8um pixel pitch.",
    )

    fpas['L3H_Sensor_XP'] = FPASpec(
        name='Sensor XP',
        vendor=Vendor.L3HARRIS,
        product_family='Sensor XP',
        detector_type=DetectorType.InSb,
        spectral_band=SpectralBand.MWIR,
        spectral_range=SpectralRange(3.0, 5.0),
        cooling=CoolingType.COOLED_STIRLING,
        array_format=ArrayFormat(1280, 720),
        pixel_pitch_um=10.0,
        max_frame_rate_hz=1.3,
        applications=[
            ApplicationDomain.NAVAL,
        ],
        notes="Maritime optimized. Spinning sensor (1.3 fps).",
    )

    # =========================================================================
    # Raytheon Vision Systems Products
    # =========================================================================

    fpas['RVS_DualBand_1280'] = FPASpec(
        name='Dual-Band 1280x720',
        vendor=Vendor.RAYTHEON,
        product_family='3rd Gen FLIR',
        detector_type=DetectorType.DUAL_BAND,
        spectral_band=SpectralBand.DUAL_MW_LW,
        spectral_range=SpectralRange(3.0, 12.0),
        cooling=CoolingType.COOLED_STIRLING,
        array_format=ArrayFormat(1280, 720),
        pixel_pitch_um=20.0,
        netd_mk=20.0,
        applications=[
            ApplicationDomain.TARGETING,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="SB-350 ROIC. TDMI architecture. MWIR NETD <20mK, LWIR <30mK. "
              "Single-bump detector: MW junction below LW junction.",
    )

    fpas['RVS_DualBand_640'] = FPASpec(
        name='Dual-Band 640x480',
        vendor=Vendor.RAYTHEON,
        product_family='3rd Gen FLIR',
        detector_type=DetectorType.DUAL_BAND,
        spectral_band=SpectralBand.DUAL_MW_LW,
        spectral_range=SpectralRange(3.0, 12.0),
        cooling=CoolingType.COOLED_STIRLING,
        array_format=ArrayFormat(640, 480),
        pixel_pitch_um=20.0,
        netd_mk=25.0,
        applications=[
            ApplicationDomain.TARGETING,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="SB-275 ROIC. Dual-band MW/LW. <25mK both bands.",
    )

    # =========================================================================
    # DRS / Leonardo DRS
    # =========================================================================

    fpas['DRS_Tenum_1280'] = FPASpec(
        name='Tenum 1280',
        vendor=Vendor.DRS,
        product_family='Tenum',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(8.0, 14.0),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=10.0,
        netd_mk=30.0,
        max_frame_rate_hz=30.0,
        weight_g=28.0,
        applications=[
            ApplicationDomain.DRONE_PAYLOAD,
            ApplicationDomain.SURVEILLANCE,
        ],
        notes="10um VOx. Starting at 28g. <30mK NETD.",
    )

    # =========================================================================
    # Axiom Optics
    # =========================================================================

    fpas['Axiom_Niels_12'] = FPASpec(
        name='Niels 12',
        vendor=Vendor.AXIOM,
        product_family='Niels',
        detector_type=DetectorType.VOx,
        spectral_band=SpectralBand.LWIR,
        spectral_range=SpectralRange(8.0, 14.0),
        cooling=CoolingType.UNCOOLED,
        array_format=ArrayFormat(1280, 1024),
        pixel_pitch_um=12.0,
        netd_mk=35.0,
        max_frame_rate_hz=60.0,
        interfaces=[InterfaceType.USB2],
        applications=[
            ApplicationDomain.INDUSTRIAL,
            ApplicationDomain.SCIENTIFIC,
        ],
        notes="12um uncooled core. USB connectivity. UVC protocol support.",
    )

    return fpas


# =============================================================================
# Database Access Functions
# =============================================================================

# Lazy-initialized caches
_fpa_cache: Optional[Dict[str, FPASpec]] = None
_roic_cache: Optional[Dict[str, ROICSpec]] = None


def get_fpa_database() -> Dict[str, FPASpec]:
    """Get the complete FPA database (cached)."""
    global _fpa_cache
    if _fpa_cache is None:
        _fpa_cache = _build_fpa_database()
    return _fpa_cache


def get_roic_database() -> Dict[str, ROICSpec]:
    """Get the complete ROIC database (cached)."""
    global _roic_cache
    if _roic_cache is None:
        _roic_cache = _build_roic_database()
    return _roic_cache


def get_fpa(name: str) -> Optional[FPASpec]:
    """Look up an FPA by its database key."""
    db = get_fpa_database()
    return db.get(name)


def get_roic(name: str) -> Optional[ROICSpec]:
    """Look up an ROIC by its database key."""
    db = get_roic_database()
    return db.get(name)


def list_fpas() -> List[str]:
    """List all FPA keys in the database."""
    return list(get_fpa_database().keys())


def list_roics() -> List[str]:
    """List all ROIC keys in the database."""
    return list(get_roic_database().keys())


def search_fpas(
    vendor: Optional[Vendor] = None,
    spectral_band: Optional[SpectralBand] = None,
    detector_type: Optional[DetectorType] = None,
    cooling: Optional[CoolingType] = None,
    min_resolution: Optional[int] = None,
    max_pitch_um: Optional[float] = None,
    max_netd_mk: Optional[float] = None,
    max_weight_g: Optional[float] = None,
    max_power_w: Optional[float] = None,
    application: Optional[ApplicationDomain] = None,
) -> List[FPASpec]:
    """
    Search the FPA database with filters.

    Parameters
    ----------
    vendor : Vendor, optional
        Filter by manufacturer
    spectral_band : SpectralBand, optional
        Filter by spectral band
    detector_type : DetectorType, optional
        Filter by detector technology
    cooling : CoolingType, optional
        Filter by cooling type
    min_resolution : int, optional
        Minimum total pixel count
    max_pitch_um : float, optional
        Maximum pixel pitch in micrometers
    max_netd_mk : float, optional
        Maximum NETD in milliKelvin
    max_weight_g : float, optional
        Maximum weight in grams
    max_power_w : float, optional
        Maximum power in watts
    application : ApplicationDomain, optional
        Filter by application domain

    Returns
    -------
    results : list of FPASpec
        Matching FPAs
    """
    results = []
    for fpa in get_fpa_database().values():
        if vendor and fpa.vendor != vendor:
            continue
        if spectral_band and fpa.spectral_band != spectral_band:
            continue
        if detector_type and fpa.detector_type != detector_type:
            continue
        if cooling and fpa.cooling != cooling:
            continue
        if min_resolution and (fpa.total_pixels is None or fpa.total_pixels < min_resolution):
            continue
        if max_pitch_um and fpa.pixel_pitch_um > max_pitch_um:
            continue
        if max_netd_mk and (fpa.netd_mk is None or fpa.netd_mk > max_netd_mk):
            continue
        if max_weight_g and (fpa.weight_g is None or fpa.weight_g > max_weight_g):
            continue
        if max_power_w:
            power = fpa.power_steady_w or fpa.power_w
            if power is None or power > max_power_w:
                continue
        if application and application not in fpa.applications:
            continue
        results.append(fpa)
    return results


def get_vendor_portfolio(vendor: Vendor) -> List[FPASpec]:
    """Get all FPAs from a specific vendor."""
    return search_fpas(vendor=vendor)


def get_band_options(band: SpectralBand) -> List[FPASpec]:
    """Get all FPAs for a specific spectral band."""
    return search_fpas(spectral_band=band)
