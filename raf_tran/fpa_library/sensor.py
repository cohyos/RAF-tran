"""
Sensor Assembly Module
======================

Define a complete sensor by combining an FPA with optics and operating
parameters. Validates parameter compatibility before applying, and provides
derived performance metrics for the assembled system.

A SensorAssembly is the combination of:
- FPA (detector + ROIC)
- Optics (focal length, F-number, aperture, transmission)
- Operating parameters (integration time, frame rate, windowing)

Conflict Validation
-------------------
Before applying parameters, the assembly checks for physical conflicts:
- Integration time vs frame rate
- F-number vs diffraction limit at pixel pitch
- Well capacity vs photon flux at integration time
- Spectral range consistency between FPA and optics
- Frame rate vs maximum ROIC rate
- FOV consistency with array format and focal length
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import math
import warnings

from raf_tran.fpa_library.models import (
    FPASpec, ArrayFormat, SpectralRange, SpectralBand, CoolingType,
)


# =============================================================================
# Optics Configuration
# =============================================================================

@dataclass
class OpticsConfig:
    """
    Optical system configuration.

    Parameters
    ----------
    focal_length_mm : float
        Effective focal length in mm
    f_number : float
        F-number (focal ratio)
    transmission : float
        End-to-end optical transmission (0-1), including all lens elements,
        filters, and windows
    spectral_filter : SpectralRange, optional
        Bandpass filter range in micrometers. If None, uses full FPA range.
    num_elements : int
        Number of optical elements (for stray light estimation)
    obscuration_ratio : float
        Central obscuration ratio (0-1) for reflective systems. 0 = none.
    """
    focal_length_mm: float
    f_number: float
    transmission: float = 0.85
    spectral_filter: Optional[SpectralRange] = None
    num_elements: int = 4
    obscuration_ratio: float = 0.0

    @property
    def aperture_mm(self) -> float:
        """Clear aperture diameter in mm."""
        return self.focal_length_mm / self.f_number

    @property
    def aperture_area_mm2(self) -> float:
        """Effective collecting area in mm^2, accounting for obscuration."""
        r = self.aperture_mm / 2.0
        area = math.pi * r**2
        if self.obscuration_ratio > 0:
            area *= (1.0 - self.obscuration_ratio**2)
        return area

    def diffraction_limit_urad(self, wavelength_um: float) -> float:
        """Diffraction-limited angular resolution (Airy disk) in urad."""
        wavelength_mm = wavelength_um / 1000.0
        return 2.44 * wavelength_mm / self.aperture_mm * 1e6

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'focal_length_mm': self.focal_length_mm,
            'f_number': self.f_number,
            'transmission': self.transmission,
            'num_elements': self.num_elements,
            'obscuration_ratio': self.obscuration_ratio,
        }
        if self.spectral_filter:
            d['spectral_filter'] = {
                'min_um': self.spectral_filter.min_um,
                'max_um': self.spectral_filter.max_um,
            }
        return d


# =============================================================================
# Operating Parameters
# =============================================================================

@dataclass
class OperatingParams:
    """
    Sensor operating parameters.

    Parameters
    ----------
    integration_time_ms : float
        Detector integration time in milliseconds
    frame_rate_hz : float
        Output frame rate in Hz
    gain : float
        System gain (electrons per DN). 1.0 = unity gain.
    window : tuple of (x, y, width, height), optional
        Sub-window region of interest (pixels). None = full frame.
    tdi_stages : int
        Time Delay Integration stages. 1 = no TDI.
    num_frames_avg : int
        Number of frames averaged for temporal noise reduction.
    nuc_enabled : bool
        Whether Non-Uniformity Correction is active.
    """
    integration_time_ms: float = 10.0
    frame_rate_hz: float = 30.0
    gain: float = 1.0
    window: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    tdi_stages: int = 1
    num_frames_avg: int = 1
    nuc_enabled: bool = True

    @property
    def integration_time_s(self) -> float:
        return self.integration_time_ms / 1000.0

    @property
    def frame_period_ms(self) -> float:
        return 1000.0 / self.frame_rate_hz

    @property
    def duty_cycle(self) -> float:
        """Integration duty cycle (0-1)."""
        return min(self.integration_time_ms / self.frame_period_ms, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'integration_time_ms': self.integration_time_ms,
            'frame_rate_hz': self.frame_rate_hz,
            'gain': self.gain,
            'tdi_stages': self.tdi_stages,
            'num_frames_avg': self.num_frames_avg,
            'nuc_enabled': self.nuc_enabled,
        }
        if self.window:
            d['window'] = list(self.window)
        return d


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue found during conflict checking."""
    severity: str          # 'error', 'warning', 'info'
    parameter: str         # Which parameter is involved
    message: str           # Human-readable description
    suggestion: str = ""   # Suggested fix

    def __str__(self) -> str:
        prefix = {'error': 'ERROR', 'warning': 'WARNING', 'info': 'INFO'}
        s = f"[{prefix.get(self.severity, '?')}] {self.parameter}: {self.message}"
        if self.suggestion:
            s += f"\n         Suggestion: {self.suggestion}"
        return s


@dataclass
class ValidationResult:
    """Result of parameter conflict validation."""
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings/info are acceptable)."""
        return not any(i.severity == 'error' for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == 'warning' for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'error')

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == 'warning')

    def __str__(self) -> str:
        if not self.issues:
            return "Validation: PASS (no issues)"
        lines = [f"Validation: {'FAIL' if not self.is_valid else 'PASS'} "
                 f"({self.error_count} errors, {self.warning_count} warnings)"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


# =============================================================================
# Sensor Assembly
# =============================================================================

class SensorAssembly:
    """
    Complete sensor system: FPA + Optics + Operating Parameters.

    Validates parameter compatibility and computes system-level performance.

    Parameters
    ----------
    fpa : FPASpec
        The focal plane array
    optics : OpticsConfig
        Optical system configuration
    operating : OperatingParams
        Operating parameters
    name : str
        Human-readable name for this assembly
    validate_on_init : bool
        If True, validate immediately on construction

    Examples
    --------
    >>> from raf_tran.fpa_library import get_fpa
    >>> from raf_tran.fpa_library.sensor import SensorAssembly, OpticsConfig, OperatingParams
    >>> fpa = get_fpa('SCD_Pelican_D_LW')
    >>> optics = OpticsConfig(focal_length_mm=100, f_number=2.0)
    >>> params = OperatingParams(integration_time_ms=10, frame_rate_hz=30)
    >>> sensor = SensorAssembly(fpa, optics, params, name="My LWIR Sensor")
    >>> print(sensor.validation)
    """

    def __init__(self, fpa: FPASpec, optics: OpticsConfig,
                 operating: OperatingParams,
                 name: str = "",
                 validate_on_init: bool = True):
        self._fpa = fpa
        self._optics = optics
        self._operating = operating
        self.name = name or f"{fpa.name} Assembly"
        self._validation: Optional[ValidationResult] = None

        if validate_on_init:
            self._validation = self.validate()

    # --- Properties ---

    @property
    def fpa(self) -> FPASpec:
        return self._fpa

    @property
    def optics(self) -> OpticsConfig:
        return self._optics

    @property
    def operating(self) -> OperatingParams:
        return self._operating

    @property
    def validation(self) -> ValidationResult:
        if self._validation is None:
            self._validation = self.validate()
        return self._validation

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid

    # --- Setters with re-validation ---

    def set_optics(self, optics: OpticsConfig) -> ValidationResult:
        """Update optics and re-validate."""
        self._optics = optics
        self._validation = self.validate()
        return self._validation

    def set_operating(self, operating: OperatingParams) -> ValidationResult:
        """Update operating parameters and re-validate."""
        self._operating = operating
        self._validation = self.validate()
        return self._validation

    def set_fpa(self, fpa: FPASpec) -> ValidationResult:
        """Swap the FPA and re-validate."""
        self._fpa = fpa
        self.name = f"{fpa.name} Assembly"
        self._validation = self.validate()
        return self._validation

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) -> ValidationResult:
        """
        Check for conflicts between FPA, optics, and operating parameters.

        Returns
        -------
        result : ValidationResult
        """
        issues = []

        self._check_integration_vs_frame_rate(issues)
        self._check_diffraction_limit(issues)
        self._check_well_fill(issues)
        self._check_spectral_consistency(issues)
        self._check_frame_rate_limit(issues)
        self._check_window_bounds(issues)
        self._check_f_number_feasibility(issues)
        self._check_nyquist_sampling(issues)

        return ValidationResult(issues=issues)

    def _check_integration_vs_frame_rate(self, issues: List[ValidationIssue]):
        """Integration time must fit within frame period."""
        frame_period_ms = 1000.0 / self._operating.frame_rate_hz
        if self._operating.integration_time_ms > frame_period_ms:
            issues.append(ValidationIssue(
                severity='error',
                parameter='integration_time_ms',
                message=(f"Integration time ({self._operating.integration_time_ms:.2f}ms) "
                         f"exceeds frame period ({frame_period_ms:.2f}ms at "
                         f"{self._operating.frame_rate_hz}Hz)"),
                suggestion=(f"Reduce integration time to <{frame_period_ms:.2f}ms "
                            f"or reduce frame rate"),
            ))
        elif self._operating.integration_time_ms > 0.95 * frame_period_ms:
            issues.append(ValidationIssue(
                severity='warning',
                parameter='integration_time_ms',
                message=(f"Integration time ({self._operating.integration_time_ms:.2f}ms) "
                         f"is >95% of frame period - very high duty cycle"),
                suggestion="Consider IWR mode or slightly reduce integration time",
            ))

    def _check_diffraction_limit(self, issues: List[ValidationIssue]):
        """Check if pixel pitch is compatible with diffraction limit."""
        sr = self._effective_spectral_range
        if sr is None:
            return

        diff_urad = self._optics.diffraction_limit_urad(sr.center_um)
        pixel_ifov_urad = self._optics_ifov_urad

        # Pixel IFOV should be >= 0.5 * diffraction limit (Nyquist)
        if pixel_ifov_urad < 0.5 * diff_urad:
            issues.append(ValidationIssue(
                severity='warning',
                parameter='f_number / pixel_pitch',
                message=(f"System is over-sampled: pixel IFOV ({pixel_ifov_urad:.1f} urad) "
                         f"< 0.5 * diffraction limit ({diff_urad:.1f} urad). "
                         f"Pixels are smaller than needed for this aperture."),
                suggestion="Increase F-number or use larger pixel pitch FPA",
            ))

        if pixel_ifov_urad > 2.5 * diff_urad:
            issues.append(ValidationIssue(
                severity='info',
                parameter='f_number / pixel_pitch',
                message=(f"System is under-sampled: pixel IFOV ({pixel_ifov_urad:.1f} urad) "
                         f"> 2.5 * diffraction limit ({diff_urad:.1f} urad)"),
                suggestion="Decrease F-number or use smaller pixel pitch FPA",
            ))

    def _check_well_fill(self, issues: List[ValidationIssue]):
        """Estimate well fill and check for saturation risk."""
        well = self._fpa.well_capacity_e
        if well is None:
            return

        # Rough background photon flux estimate for LWIR at ~300K scene
        sr = self._effective_spectral_range
        if sr is None:
            return

        # Photon flux depends heavily on band; use rough estimates
        band = self._fpa.spectral_band
        if band == SpectralBand.LWIR:
            photon_flux_per_s = 1e11  # photons/s/pixel approx for f/2 LWIR
        elif band == SpectralBand.MWIR:
            photon_flux_per_s = 1e10
        elif band == SpectralBand.SWIR:
            photon_flux_per_s = 1e8
        else:
            photon_flux_per_s = 1e10

        # Scale by F-number (flux proportional to 1/f_number^2)
        photon_flux_per_s *= (2.0 / self._optics.f_number) ** 2
        photon_flux_per_s *= self._optics.transmission

        # Electrons collected
        qe = self._fpa.quantum_efficiency or 0.7
        electrons = photon_flux_per_s * qe * self._operating.integration_time_s
        fill_fraction = electrons / well

        if fill_fraction > 1.0:
            issues.append(ValidationIssue(
                severity='error',
                parameter='well_capacity',
                message=(f"Estimated well fill {fill_fraction*100:.0f}% - SATURATION. "
                         f"~{electrons:.1e} e- vs {well:.1e} e- capacity"),
                suggestion=(f"Reduce integration time to <"
                            f"{self._operating.integration_time_ms / fill_fraction:.2f}ms "
                            f"or increase F-number"),
            ))
        elif fill_fraction > 0.85:
            issues.append(ValidationIssue(
                severity='warning',
                parameter='well_capacity',
                message=f"Estimated well fill {fill_fraction*100:.0f}% - near saturation",
                suggestion="Monitor for saturation on hot targets",
            ))
        elif fill_fraction < 0.1:
            issues.append(ValidationIssue(
                severity='info',
                parameter='well_capacity',
                message=f"Estimated well fill {fill_fraction*100:.1f}% - low signal utilization",
                suggestion="Consider longer integration time or faster optics",
            ))

    def _check_spectral_consistency(self, issues: List[ValidationIssue]):
        """Check that optics spectral filter is within FPA range."""
        fpa_range = self._fpa.spectral_range
        opt_filter = self._optics.spectral_filter

        if fpa_range is None or opt_filter is None:
            return

        if opt_filter.min_um < fpa_range.min_um or opt_filter.max_um > fpa_range.max_um:
            issues.append(ValidationIssue(
                severity='error',
                parameter='spectral_filter',
                message=(f"Filter range ({opt_filter}) extends beyond FPA response "
                         f"({fpa_range})"),
                suggestion="Adjust filter to be within FPA spectral range",
            ))

    def _check_frame_rate_limit(self, issues: List[ValidationIssue]):
        """Check frame rate against FPA/ROIC maximum."""
        max_rate = self._fpa.max_frame_rate_hz
        if max_rate is None:
            return

        if self._operating.frame_rate_hz > max_rate:
            issues.append(ValidationIssue(
                severity='error',
                parameter='frame_rate_hz',
                message=(f"Requested {self._operating.frame_rate_hz}Hz exceeds FPA maximum "
                         f"{max_rate}Hz"),
                suggestion=f"Reduce frame rate to <={max_rate}Hz or use windowing",
            ))

    def _check_window_bounds(self, issues: List[ValidationIssue]):
        """Check that ROI window is within array bounds."""
        win = self._operating.window
        fmt = self._fpa.array_format
        if win is None or fmt is None:
            return

        x, y, w, h = win
        if x < 0 or y < 0 or x + w > fmt.columns or y + h > fmt.rows:
            issues.append(ValidationIssue(
                severity='error',
                parameter='window',
                message=(f"Window ({x},{y},{w},{h}) exceeds array bounds "
                         f"({fmt.columns}x{fmt.rows})"),
                suggestion=f"Adjust window to fit within {fmt}",
            ))

    def _check_f_number_feasibility(self, issues: List[ValidationIssue]):
        """Check for physically unrealistic F-numbers."""
        if self._optics.f_number < 0.7:
            issues.append(ValidationIssue(
                severity='error',
                parameter='f_number',
                message=f"F/{self._optics.f_number:.1f} is physically unrealizable",
                suggestion="Use F/1.0 or higher for practical systems",
            ))
        elif self._optics.f_number < 1.0:
            issues.append(ValidationIssue(
                severity='warning',
                parameter='f_number',
                message=f"F/{self._optics.f_number:.1f} requires exotic optics design",
            ))

    def _check_nyquist_sampling(self, issues: List[ValidationIssue]):
        """Check spatial Nyquist sampling criterion."""
        sr = self._effective_spectral_range
        if sr is None:
            return

        # Cutoff frequency of optics: 1 / (lambda * f_number)
        lambda_mm = sr.center_um / 1000.0
        cutoff_lp_per_mm = 1.0 / (lambda_mm * self._optics.f_number)

        # Nyquist frequency of detector
        pitch_mm = self._fpa.pixel_pitch_um / 1000.0
        nyquist_lp_per_mm = 1.0 / (2.0 * pitch_mm)

        ratio = nyquist_lp_per_mm / cutoff_lp_per_mm
        if ratio < 0.5:
            issues.append(ValidationIssue(
                severity='warning',
                parameter='sampling',
                message=(f"Nyquist ratio = {ratio:.2f} (severely under-sampled). "
                         f"Detector Nyquist: {nyquist_lp_per_mm:.0f} lp/mm, "
                         f"Optics cutoff: {cutoff_lp_per_mm:.0f} lp/mm"),
                suggestion="Use smaller pixel pitch or increase F-number",
            ))

    # =========================================================================
    # Derived Performance Metrics
    # =========================================================================

    @property
    def _effective_spectral_range(self) -> Optional[SpectralRange]:
        """Effective spectral range (intersection of FPA and filter)."""
        if self._optics.spectral_filter:
            return self._optics.spectral_filter
        return self._fpa.spectral_range

    @property
    def _optics_ifov_urad(self) -> float:
        """Single-pixel IFOV in microradians."""
        return self._fpa.pixel_pitch_um / self._optics.focal_length_mm * 1000.0

    @property
    def ifov_urad(self) -> float:
        """Instantaneous Field of View per pixel in microradians."""
        return self._optics_ifov_urad

    @property
    def ifov_mrad(self) -> float:
        """IFOV in milliradians."""
        return self.ifov_urad / 1000.0

    @property
    def fov_deg(self) -> Optional[Tuple[float, float]]:
        """Full field of view (horizontal, vertical) in degrees."""
        return self._fpa.fov_at_focal_length(self._optics.focal_length_mm)

    @property
    def aperture_mm(self) -> float:
        """Clear aperture diameter in mm."""
        return self._optics.aperture_mm

    @property
    def ground_sample_distance_m(self) -> Optional[float]:
        """Not applicable without slant range; returns None."""
        return None

    def gsd_at_range(self, range_m: float) -> float:
        """Ground Sample Distance at given range in meters."""
        return range_m * self.ifov_urad * 1e-6

    def dri_ranges_m(self, target_size_m: float = 2.3) -> Dict[str, float]:
        """Detection, Recognition, Identification ranges in meters."""
        return {
            'detection': self._fpa.johnson_criteria_range(
                target_size_m, self._optics.focal_length_mm, 1.0),
            'recognition': self._fpa.johnson_criteria_range(
                target_size_m, self._optics.focal_length_mm, 3.0),
            'identification': self._fpa.johnson_criteria_range(
                target_size_m, self._optics.focal_length_mm, 6.0),
        }

    @property
    def estimated_nep_w(self) -> Optional[float]:
        """
        Estimate Noise Equivalent Power in Watts.

        NEP = A_pixel * delta_f / D*
        where A_pixel is pixel area and delta_f is noise bandwidth.
        """
        d_star = self._fpa.detectivity_jones
        if d_star is None:
            return None
        a_pixel_cm2 = (self._fpa.pixel_pitch_um * 1e-4) ** 2
        bandwidth = self._operating.frame_rate_hz / 2.0
        return math.sqrt(a_pixel_cm2 * bandwidth) / d_star

    @property
    def estimated_netd_mk(self) -> Optional[float]:
        """
        Return the FPA NETD, adjusted for integration time and frame averaging.

        If TDI or frame averaging is used, NETD improves by sqrt(N).
        """
        base_netd = self._fpa.netd_mk
        if base_netd is None:
            return None
        n_effective = self._operating.tdi_stages * self._operating.num_frames_avg
        return base_netd / math.sqrt(n_effective)

    @property
    def pixels_on_target(self) -> Dict[str, float]:
        """Pixels on a 2.3m target at various ranges."""
        result = {}
        for range_km in [1, 2, 5, 10, 20]:
            range_m = range_km * 1000
            gsd = self.gsd_at_range(range_m)
            if gsd > 0:
                result[f'{range_km}km'] = 2.3 / gsd
        return result

    @property
    def data_rate_mbps(self) -> Optional[float]:
        """Estimated data output rate in Mbps."""
        fmt = self._fpa.array_format
        if fmt is None:
            return None
        bits = self._fpa.adc_bits or 14
        if self._operating.window:
            _, _, w, h = self._operating.window
            pixels = w * h
        else:
            pixels = fmt.total_pixels
        return pixels * bits * self._operating.frame_rate_hz / 1e6

    def performance_summary(self) -> Dict[str, Any]:
        """
        Generate a full performance summary dictionary.

        Returns
        -------
        summary : dict
            All computed metrics in a flat dictionary
        """
        dri = self.dri_ranges_m()
        fov = self.fov_deg

        summary = {
            'name': self.name,
            'fpa': self._fpa.name,
            'vendor': self._fpa.vendor.value,
            'resolution': self._fpa.resolution_str,
            'pixel_pitch_um': self._fpa.pixel_pitch_um,
            'spectral_band': self._fpa.spectral_band.value,
            'focal_length_mm': self._optics.focal_length_mm,
            'f_number': self._optics.f_number,
            'aperture_mm': self.aperture_mm,
            'ifov_urad': self.ifov_urad,
            'ifov_mrad': self.ifov_mrad,
            'fov_h_deg': fov[0] if fov else None,
            'fov_v_deg': fov[1] if fov else None,
            'integration_time_ms': self._operating.integration_time_ms,
            'frame_rate_hz': self._operating.frame_rate_hz,
            'duty_cycle': self._operating.duty_cycle,
            'detection_range_km': dri['detection'] / 1000,
            'recognition_range_km': dri['recognition'] / 1000,
            'identification_range_km': dri['identification'] / 1000,
            'estimated_netd_mk': self.estimated_netd_mk,
            'data_rate_mbps': self.data_rate_mbps,
            'is_valid': self.is_valid,
            'validation_errors': self.validation.error_count,
            'validation_warnings': self.validation.warning_count,
        }
        return summary

    def __str__(self) -> str:
        lines = [f"=== {self.name} ==="]
        lines.append(f"FPA: {self._fpa.name} ({self._fpa.vendor.value})")
        lines.append(f"  {self._fpa.resolution_str} @ {self._fpa.pixel_pitch_um}um, "
                     f"{self._fpa.spectral_band.value}")
        lines.append(f"Optics: f={self._optics.focal_length_mm}mm, "
                     f"F/{self._optics.f_number}, "
                     f"D={self.aperture_mm:.1f}mm")
        lines.append(f"Operating: tint={self._operating.integration_time_ms}ms, "
                     f"{self._operating.frame_rate_hz}Hz")
        lines.append(f"IFOV: {self.ifov_urad:.1f} urad ({self.ifov_mrad:.3f} mrad)")
        fov = self.fov_deg
        if fov:
            lines.append(f"FOV: {fov[0]:.2f} x {fov[1]:.2f} deg")
        dri = self.dri_ranges_m()
        lines.append(f"DRI (2.3m target): D={dri['detection']/1000:.1f}km, "
                     f"R={dri['recognition']/1000:.1f}km, "
                     f"I={dri['identification']/1000:.1f}km")
        netd = self.estimated_netd_mk
        if netd:
            lines.append(f"Est. NETD: {netd:.1f} mK")
        lines.append(str(self.validation))
        return "\n".join(lines)
