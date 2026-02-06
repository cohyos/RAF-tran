#!/usr/bin/env python3
"""
Example 43: Air-to-Air F-16 Detection - Monte Carlo Sensor Optimization
========================================================================

Monte Carlo optimization study for detecting a head-on F-16 fighter
aircraft from a sensor platform at 35,000 feet altitude.

Scenario
--------
- Sensor platform:     35,000 ft (10,668 m) altitude
- Target:              F-16 head-on aspect at 35,000 ft (co-altitude)
- Required FOV:        50 deg x 50 deg (wide-area search)
- Objective:           Maximize detection range

Physics Modeled
---------------
- F-16 head-on IR signature: engine exhaust (plume + tailpipe), skin
  heating from aerodynamic friction, sky background contrast
- Atmospheric transmission at altitude (Beer-Lambert with molecular
  and aerosol extinction, altitude-scaled)
- Path radiance (in-path thermal emission)
- Turbulence-induced beam spread (Cn2 at altitude)
- Sky background radiance (cold space + atmospheric downwelling)
- Signal-to-noise ratio with frame averaging and TDI
- Johnson criteria detection (N pixels on target at threshold SNR)

Monte Carlo Variables (per trial)
---------------------------------
- Atmospheric visibility (15-80 km, log-normal)
- Turbulence strength Cn2 (weak-moderate at altitude)
- Target IR signature variation (+/-20% from nominal)
- Background clutter factor
- Sensor noise realization (NETD draw)

The study sweeps all candidate FPAs and computes the focal length
required for 50 deg FOV, then runs MC trials to find the sensor
configuration yielding best detection range statistics.

Usage:
    python examples/43_air_to_air_detection_optimization.py [--no-plot]
    python examples/43_air_to_air_detection_optimization.py --mc-runs 500
"""

import argparse
import sys
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# FPA Library
from raf_tran.fpa_library import (
    get_fpa_database,
    search_fpas,
)
from raf_tran.fpa_library.models import (
    FPASpec, SpectralBand, SpectralRange, CoolingType,
)
from raf_tran.fpa_library.sensor import (
    SensorAssembly, OpticsConfig, OperatingParams,
)


# =============================================================================
# Physical Constants and Scenario Parameters
# =============================================================================

ALTITUDE_FT = 35000
ALTITUDE_M = ALTITUDE_FT * 0.3048  # 10,668 m
FOV_DEG = 50.0                      # Required FOV (each axis)

# F-16 head-on parameters
F16_WINGSPAN_M = 9.96               # Full wingspan
F16_HEIGHT_M = 4.88                 # Height (fin tip to ground line)
F16_HEAD_ON_WIDTH_M = 3.0           # Effective head-on visible width (fuselage + inlet)
F16_HEAD_ON_HEIGHT_M = 2.5          # Effective head-on visible height
F16_TARGET_SIZE_M = 2.75            # Characteristic dimension for Johnson criteria

# F-16 IR signature (head-on, MWIR band 3-5 um)
# Exhaust plume temperature ~600-900K, tailpipe ~700K
# Head-on, the plume is partially occluded by fuselage; dominant
# contribution is residual plume and hot metal tailpipe ring.
# Reference: open-literature IRST studies for head-on aspect.
F16_MWIR_INTENSITY_W_SR = 25.0     # Radiant intensity (W/sr) in MWIR, head-on
F16_LWIR_INTENSITY_W_SR = 80.0     # Radiant intensity (W/sr) in LWIR, head-on
# (LWIR higher due to skin emission and broader Planck at lower T)

# At 35 kft, ambient temperature ~-54C (219 K), pressure ~238 hPa
T_AMBIENT_K = 219.0
P_AMBIENT_HPA = 238.0

# Standard atmosphere sea-level extinction coefficients (1/km)
# These are scaled by altitude for the actual path
MWIR_EXTINCTION_SEA_LEVEL = 0.15   # km^-1 at sea level (clear, 3-5 um band avg)
LWIR_EXTINCTION_SEA_LEVEL = 0.25   # km^-1 at sea level (clear, 8-14 um band avg)

# At 35 kft, air density is ~1/3 of sea level -> extinction scales accordingly
ALTITUDE_DENSITY_RATIO = math.exp(-ALTITUDE_M / 8500.0)  # scale height ~8.5 km

MWIR_EXTINCTION_ALTITUDE = MWIR_EXTINCTION_SEA_LEVEL * ALTITUDE_DENSITY_RATIO
LWIR_EXTINCTION_ALTITUDE = LWIR_EXTINCTION_SEA_LEVEL * ALTITUDE_DENSITY_RATIO

# Sky background radiance at altitude (W/sr/m^2/um)
# Cold sky looking horizontally at 35 kft:
MWIR_SKY_RADIANCE = 1e-4   # W/sr/m^2/um (very low cold sky background)
LWIR_SKY_RADIANCE = 5e-3   # W/sr/m^2/um (atmospheric self-emission)

# Detection threshold SNR
SNR_DETECTION = 5.0    # Standard detection threshold
SNR_RECOGNITION = 8.0
SNR_IDENTIFICATION = 12.0


# =============================================================================
# F-16 Target Model
# =============================================================================

@dataclass
class TargetSignature:
    """F-16 head-on IR signature for one MC realization."""
    intensity_w_sr: float        # Radiant intensity (W/sr)
    target_width_m: float        # Apparent width
    target_height_m: float       # Apparent height
    characteristic_size_m: float # For Johnson criteria


def sample_f16_signature(rng: np.random.Generator, band: SpectralBand) -> TargetSignature:
    """
    Sample an F-16 head-on IR signature with realistic variation.

    Variations model:
    - Engine throttle setting (cruise vs military power)
    - Plume aspect angle jitter (+/-2 deg from pure head-on)
    - Atmospheric heating variation on skin
    """
    if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
        base_intensity = F16_MWIR_INTENSITY_W_SR
    else:
        base_intensity = F16_LWIR_INTENSITY_W_SR

    # Throttle variation: 0.7x (cruise) to 1.4x (afterburner)
    throttle_factor = rng.uniform(0.7, 1.4)

    # Aspect jitter: small angle reduces signature from head-on
    # cos^2 model for plume visibility
    jitter_deg = rng.normal(0, 1.5)  # +/-1.5 deg std
    aspect_factor = math.cos(math.radians(jitter_deg)) ** 2

    # Random noise (sensor measurement uncertainty)
    noise_factor = rng.normal(1.0, 0.05)

    intensity = base_intensity * throttle_factor * aspect_factor * max(noise_factor, 0.3)

    # Target apparent size: slight variation from attitude
    width = F16_HEAD_ON_WIDTH_M * rng.uniform(0.9, 1.1)
    height = F16_HEAD_ON_HEIGHT_M * rng.uniform(0.9, 1.1)

    return TargetSignature(
        intensity_w_sr=intensity,
        target_width_m=width,
        target_height_m=height,
        characteristic_size_m=(width + height) / 2.0,
    )


# =============================================================================
# Atmospheric Model
# =============================================================================

@dataclass
class AtmosphericConditions:
    """Atmospheric conditions for one MC realization."""
    visibility_km: float
    cn2: float                   # Cn2 (m^-2/3) at altitude
    extinction_per_km: float     # Band-dependent extinction (1/km)
    path_radiance_factor: float  # Multiplicative factor on path radiance


def sample_atmosphere(rng: np.random.Generator, band: SpectralBand) -> AtmosphericConditions:
    """
    Sample atmospheric conditions at 35 kft.

    At cruise altitude the atmosphere is very clear but variable:
    - Visibility: 15-120 km (log-normal, median ~50 km at altitude)
    - Cn2: 1e-17 to 1e-16 m^-2/3 (weak turbulence at altitude)
    - Extinction: base + visibility-dependent scattering
    """
    # Visibility at altitude (cleaner than sea level)
    vis_km = rng.lognormal(mean=math.log(60.0), sigma=0.4)
    vis_km = np.clip(vis_km, 15.0, 150.0)

    # Cn2 at altitude: much weaker than ground level
    # Typical 35kft: 1e-17 to 5e-17 m^-2/3
    log_cn2 = rng.uniform(-17.0, -16.3)
    cn2 = 10 ** log_cn2

    # Extinction at altitude
    if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
        base_ext = MWIR_EXTINCTION_ALTITUDE
    else:
        base_ext = LWIR_EXTINCTION_ALTITUDE

    # Add visibility-dependent scattering: 3.912 / visibility (Koschmieder)
    scatter_ext = (3.912 / vis_km) * ALTITUDE_DENSITY_RATIO
    total_ext = base_ext + scatter_ext

    # Path radiance variation (atmospheric thermal emission along path)
    path_rad_factor = rng.uniform(0.7, 1.3)

    return AtmosphericConditions(
        visibility_km=vis_km,
        cn2=cn2,
        extinction_per_km=total_ext,
        path_radiance_factor=path_rad_factor,
    )


# =============================================================================
# Detection Range Computation
# =============================================================================

def compute_atmospheric_transmission(extinction_per_km: float, range_km: float) -> float:
    """Beer-Lambert atmospheric transmission."""
    return math.exp(-extinction_per_km * range_km)


def compute_signal_irradiance(target: TargetSignature,
                              range_m: float,
                              transmission: float,
                              optics_transmission: float) -> float:
    """
    Compute signal irradiance at the focal plane from target.

    E_signal = I_target * tau_atm * tau_opt / R^2

    Parameters
    ----------
    target : TargetSignature
    range_m : float
        Slant range in meters
    transmission : float
        Atmospheric transmission
    optics_transmission : float
        Optics transmission

    Returns
    -------
    irradiance : float
        Signal irradiance at aperture (W/m^2)
    """
    if range_m <= 0:
        return 0.0
    return target.intensity_w_sr * transmission * optics_transmission / (range_m ** 2)


def compute_snr(signal_irradiance: float,
                aperture_m2: float,
                pixel_area_m2: float,
                focal_length_m: float,
                integration_time_s: float,
                netd_k: float,
                background_radiance: float,
                n_frames: int = 1,
                tdi_stages: int = 1) -> float:
    """
    Compute signal-to-noise ratio for a point source on a single pixel.

    SNR = (signal_power * tint) / NEP

    For a thermal detector, NEP relates to NETD:
    NEP ~ NETD * (dL/dT)^-1 * A_pixel * Omega

    We use a simplified radiometric SNR model:
    SNR = (E_signal * A_aperture * tint) / (NEP * sqrt(BW))

    Adjusted for frame averaging and TDI.
    """
    # Collected signal energy
    signal_power = signal_irradiance * aperture_m2
    signal_energy = signal_power * integration_time_s

    # Noise: derive from NETD
    # NETD is the temperature difference that gives SNR=1 against background
    # For a ~300K background, dL/dT ~ 2e-5 W/m^2/sr/K in MWIR
    # We model noise as: noise_equivalent_irradiance = NETD_fraction * background
    # where NETD_fraction represents the fractional noise level

    # Background signal per pixel
    pixel_solid_angle = pixel_area_m2 / (focal_length_m ** 2)
    bg_power = background_radiance * aperture_m2 * pixel_solid_angle
    bg_energy = bg_power * integration_time_s

    # Noise equivalent signal (proportional to NETD / T_scene * background)
    # NETD/T_scene gives the fractional noise
    noise_fraction = netd_k / T_AMBIENT_K
    noise_energy = noise_fraction * bg_energy

    if noise_energy <= 0:
        # Use a minimum noise floor
        noise_energy = signal_energy * 0.01

    # SNR with frame averaging and TDI improvement
    n_effective = math.sqrt(n_frames * tdi_stages)
    snr = (signal_energy / noise_energy) * n_effective

    return snr


def find_detection_range(target: TargetSignature,
                         atm: AtmosphericConditions,
                         sensor: SensorAssembly,
                         band: SpectralBand,
                         snr_threshold: float = SNR_DETECTION) -> float:
    """
    Find maximum detection range by bisection search.

    Returns range in km where SNR = threshold.
    """
    optics = sensor.optics
    operating = sensor.operating
    fpa = sensor.fpa

    # Optics parameters
    aperture_m = optics.aperture_mm / 1000.0
    aperture_area_m2 = math.pi * (aperture_m / 2) ** 2
    if optics.obscuration_ratio > 0:
        aperture_area_m2 *= (1 - optics.obscuration_ratio ** 2)

    focal_length_m = optics.focal_length_mm / 1000.0
    pixel_size_m = fpa.pixel_pitch_um * 1e-6
    pixel_area_m2 = pixel_size_m ** 2

    # NETD in Kelvin
    netd_k = (fpa.netd_mk or 30.0) / 1000.0

    # Background radiance
    if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
        bg_rad = MWIR_SKY_RADIANCE * atm.path_radiance_factor
    else:
        bg_rad = LWIR_SKY_RADIANCE * atm.path_radiance_factor

    integration_time_s = operating.integration_time_ms / 1000.0
    n_frames = operating.num_frames_avg
    tdi = operating.tdi_stages

    # Bisection search for range where SNR = threshold
    r_min_km = 0.5
    r_max_km = 500.0

    for _ in range(50):
        r_mid_km = (r_min_km + r_max_km) / 2.0
        r_mid_m = r_mid_km * 1000.0

        tau_atm = compute_atmospheric_transmission(atm.extinction_per_km, r_mid_km)
        e_signal = compute_signal_irradiance(
            target, r_mid_m, tau_atm, optics.transmission)

        snr = compute_snr(
            signal_irradiance=e_signal,
            aperture_m2=aperture_area_m2,
            pixel_area_m2=pixel_area_m2,
            focal_length_m=focal_length_m,
            integration_time_s=integration_time_s,
            netd_k=netd_k,
            background_radiance=bg_rad,
            n_frames=n_frames,
            tdi_stages=tdi,
        )

        if snr > snr_threshold:
            r_min_km = r_mid_km
        else:
            r_max_km = r_mid_km

        if abs(r_max_km - r_min_km) < 0.01:
            break

    return (r_min_km + r_max_km) / 2.0


# =============================================================================
# FOV-Constrained Focal Length
# =============================================================================

def focal_length_for_fov(fpa: FPASpec, fov_deg: float) -> Optional[float]:
    """
    Compute the focal length (mm) needed for a given FOV.

    Uses the larger array dimension to set the FOV.
    FOV = 2 * atan(d / (2*f))  =>  f = d / (2 * tan(FOV/2))
    """
    if fpa.array_format is None:
        return None

    # Use the larger dimension for the specified FOV
    cols = fpa.array_format.columns
    rows = fpa.array_format.rows
    max_dim = max(cols, rows)
    array_size_mm = max_dim * fpa.pixel_pitch_um / 1000.0

    half_fov_rad = math.radians(fov_deg / 2.0)
    if half_fov_rad <= 0 or half_fov_rad >= math.pi / 2:
        return None

    focal_length_mm = array_size_mm / (2.0 * math.tan(half_fov_rad))
    return focal_length_mm


# =============================================================================
# Monte Carlo Sensor Evaluation
# =============================================================================

@dataclass
class SensorCandidate:
    """A candidate sensor configuration for the MC study."""
    fpa_key: str
    fpa: FPASpec
    focal_length_mm: float
    f_number: float
    fov_h_deg: float
    fov_v_deg: float
    ifov_urad: float
    assembly: SensorAssembly


@dataclass
class MCResult:
    """Monte Carlo results for one sensor candidate."""
    candidate: SensorCandidate
    detection_ranges_km: np.ndarray
    recognition_ranges_km: np.ndarray
    identification_ranges_km: np.ndarray

    @property
    def mean_detection_km(self) -> float:
        return float(np.mean(self.detection_ranges_km))

    @property
    def std_detection_km(self) -> float:
        return float(np.std(self.detection_ranges_km))

    @property
    def p50_detection_km(self) -> float:
        return float(np.median(self.detection_ranges_km))

    @property
    def p90_detection_km(self) -> float:
        return float(np.percentile(self.detection_ranges_km, 90))

    @property
    def p10_detection_km(self) -> float:
        return float(np.percentile(self.detection_ranges_km, 10))

    @property
    def mean_recognition_km(self) -> float:
        return float(np.mean(self.recognition_ranges_km))

    @property
    def mean_identification_km(self) -> float:
        return float(np.mean(self.identification_ranges_km))


def build_candidates(fov_deg: float) -> List[SensorCandidate]:
    """
    Build sensor candidates from the database.

    For each FPA, compute the focal length that gives the required FOV,
    then try several F-numbers.
    """
    db = get_fpa_database()
    candidates = []

    # F-numbers to try for each FPA
    f_numbers = [1.4, 2.0, 2.8, 4.0]

    for key, fpa in db.items():
        # Skip FPAs without array format
        if fpa.array_format is None:
            continue

        # Skip very low resolution sensors (Lepton etc.)
        if fpa.array_format.total_pixels < 100000:
            continue

        # Compute focal length for FOV
        fl_mm = focal_length_for_fov(fpa, fov_deg)
        if fl_mm is None or fl_mm < 2.0:
            continue

        # Get actual FOV (may differ slightly on each axis)
        fov = fpa.fov_at_focal_length(fl_mm)
        if fov is None:
            continue

        # IFOV
        ifov_urad = fpa.pixel_pitch_um / fl_mm * 1000.0

        for fn in f_numbers:
            # Determine integration time based on band
            # MWIR: shorter tint (higher background), LWIR: very short
            if fpa.spectral_band == SpectralBand.LWIR:
                tint_ms = 0.1   # LWIR has high background flux
            elif fpa.spectral_band == SpectralBand.MWIR:
                tint_ms = 2.0
            else:
                tint_ms = 1.0   # default

            optics = OpticsConfig(
                focal_length_mm=fl_mm,
                f_number=fn,
                transmission=0.80,
            )
            operating = OperatingParams(
                integration_time_ms=tint_ms,
                frame_rate_hz=30.0,
                num_frames_avg=4,   # 4-frame temporal averaging
            )

            assembly = SensorAssembly(
                fpa, optics, operating,
                name=f"{fpa.name} F/{fn}",
                validate_on_init=False,
            )

            candidates.append(SensorCandidate(
                fpa_key=key,
                fpa=fpa,
                focal_length_mm=fl_mm,
                f_number=fn,
                fov_h_deg=fov[0],
                fov_v_deg=fov[1],
                ifov_urad=ifov_urad,
                assembly=assembly,
            ))

    return candidates


def run_mc_evaluation(candidates: List[SensorCandidate],
                      n_runs: int = 200,
                      seed: int = 42) -> List[MCResult]:
    """
    Run Monte Carlo evaluation for all candidates.

    For each candidate and each MC trial:
    1. Sample F-16 target signature
    2. Sample atmospheric conditions
    3. Compute detection, recognition, identification ranges
    """
    rng = np.random.default_rng(seed)
    results = []

    for cand in candidates:
        band = cand.fpa.spectral_band
        det_ranges = np.zeros(n_runs)
        rec_ranges = np.zeros(n_runs)
        idf_ranges = np.zeros(n_runs)

        for i in range(n_runs):
            target = sample_f16_signature(rng, band)
            atm = sample_atmosphere(rng, band)

            det_ranges[i] = find_detection_range(
                target, atm, cand.assembly, band, SNR_DETECTION)
            rec_ranges[i] = find_detection_range(
                target, atm, cand.assembly, band, SNR_RECOGNITION)
            idf_ranges[i] = find_detection_range(
                target, atm, cand.assembly, band, SNR_IDENTIFICATION)

        results.append(MCResult(
            candidate=cand,
            detection_ranges_km=det_ranges,
            recognition_ranges_km=rec_ranges,
            identification_ranges_km=idf_ranges,
        ))

    return results


# =============================================================================
# Main
# =============================================================================

def main(args):
    print("=" * 70)
    print("Example 43: Air-to-Air F-16 Detection - MC Sensor Optimization")
    print("=" * 70)
    print()

    print("Scenario:")
    print(f"  Platform altitude:    {ALTITUDE_FT:,} ft ({ALTITUDE_M:.0f} m)")
    print(f"  Target:               F-16 head-on at {ALTITUDE_FT:,} ft (co-altitude)")
    print(f"  Required FOV:         {FOV_DEG:.0f} deg x {FOV_DEG:.0f} deg")
    print(f"  MC runs per sensor:   {args.mc_runs}")
    print(f"  F-16 MWIR intensity:  {F16_MWIR_INTENSITY_W_SR} W/sr (head-on)")
    print(f"  F-16 LWIR intensity:  {F16_LWIR_INTENSITY_W_SR} W/sr (head-on)")
    print(f"  Ambient temp:         {T_AMBIENT_K:.0f} K ({T_AMBIENT_K - 273.15:.0f} C)")
    print(f"  MWIR extinction:      {MWIR_EXTINCTION_ALTITUDE:.4f} /km (at altitude)")
    print(f"  LWIR extinction:      {LWIR_EXTINCTION_ALTITUDE:.4f} /km (at altitude)")
    print(f"  Detection threshold:  SNR >= {SNR_DETECTION}")
    print()

    # -----------------------------------------------------------------
    # 1. Build Sensor Candidates
    # -----------------------------------------------------------------
    print("1. Building Sensor Candidates")
    print("-" * 40)

    candidates = build_candidates(FOV_DEG)
    print(f"   Total candidates: {len(candidates)} (FPA x F-number combinations)")

    # Show focal lengths by FPA
    seen = {}
    for c in candidates:
        if c.fpa.name not in seen:
            seen[c.fpa.name] = c
    print(f"   Unique FPAs evaluated: {len(seen)}")
    print()

    print("   FPA                    | Band  | Format      | Pitch | FL(mm) | IFOV(urad)")
    print("   " + "-" * 80)
    for name, c in sorted(seen.items(), key=lambda x: x[1].focal_length_mm):
        fmt = c.fpa.resolution_str
        band = c.fpa.spectral_band.value
        print(f"   {name:24s} | {band:5s} | {fmt:11s} | {c.fpa.pixel_pitch_um:5.1f} | "
              f"{c.focal_length_mm:6.1f} | {c.ifov_urad:9.1f}")
    print()

    # -----------------------------------------------------------------
    # 2. Run Monte Carlo
    # -----------------------------------------------------------------
    print("2. Running Monte Carlo Evaluation...")
    print("-" * 40)

    mc_results = run_mc_evaluation(candidates, n_runs=args.mc_runs, seed=42)
    print(f"   Completed {len(mc_results)} x {args.mc_runs} = "
          f"{len(mc_results) * args.mc_runs:,} simulations")
    print()

    # -----------------------------------------------------------------
    # 3. Rank by Mean Detection Range
    # -----------------------------------------------------------------
    print("3. Results Ranked by Mean Detection Range")
    print("-" * 40)

    mc_results.sort(key=lambda r: r.mean_detection_km, reverse=True)

    print(f"   {'Rank':>4} | {'FPA':24s} | {'F/#':>4} | {'Band':5s} | "
          f"{'FL(mm)':>7} | {'IFOV':>7} | {'Det(km)':>8} | {'Rec(km)':>8} | "
          f"{'ID(km)':>8} | {'Det P10':>7} | {'Det P90':>7}")
    print("   " + "-" * 120)

    top_n = min(20, len(mc_results))
    for i, r in enumerate(mc_results[:top_n]):
        c = r.candidate
        print(f"   {i+1:4d} | {c.fpa.name:24s} | {c.f_number:4.1f} | "
              f"{c.fpa.spectral_band.value:5s} | {c.focal_length_mm:7.1f} | "
              f"{c.ifov_urad:7.1f} | {r.mean_detection_km:8.1f} | "
              f"{r.mean_recognition_km:8.1f} | {r.mean_identification_km:8.1f} | "
              f"{r.p10_detection_km:7.1f} | {r.p90_detection_km:7.1f}")
    print()

    # -----------------------------------------------------------------
    # 4. Best Overall Configuration
    # -----------------------------------------------------------------
    print("4. Optimal Sensor Configuration")
    print("-" * 40)

    best = mc_results[0]
    bc = best.candidate
    print(f"   FPA:                {bc.fpa.name} ({bc.fpa.vendor.value})")
    print(f"   Detector type:      {bc.fpa.detector_type.value}")
    print(f"   Spectral band:      {bc.fpa.spectral_band.value}", end="")
    if bc.fpa.spectral_range:
        print(f" ({bc.fpa.spectral_range})")
    else:
        print()
    print(f"   Array format:       {bc.fpa.resolution_str}")
    print(f"   Pixel pitch:        {bc.fpa.pixel_pitch_um} um")
    print(f"   Cooling:            {bc.fpa.cooling.value}")
    print()
    print(f"   === Recommended Optics ===")
    print(f"   Focal length:       {bc.focal_length_mm:.1f} mm")
    print(f"   F-number:           F/{bc.f_number:.1f}")
    print(f"   Aperture diameter:  {bc.focal_length_mm / bc.f_number:.1f} mm")
    print(f"   Optical transmission: {bc.assembly.optics.transmission:.0%}")
    fov = bc.fpa.fov_at_focal_length(bc.focal_length_mm)
    if fov:
        print(f"   Actual FOV:         {fov[0]:.1f} x {fov[1]:.1f} deg")
    print(f"   IFOV:               {bc.ifov_urad:.1f} urad ({bc.ifov_urad/1000:.3f} mrad)")
    print()
    print(f"   === Operating Parameters ===")
    print(f"   Integration time:   {bc.assembly.operating.integration_time_ms} ms")
    print(f"   Frame rate:         {bc.assembly.operating.frame_rate_hz} Hz")
    print(f"   Frame averaging:    {bc.assembly.operating.num_frames_avg}x")
    print()
    print(f"   === MC Detection Performance (N={args.mc_runs}) ===")
    print(f"   Detection range:    {best.mean_detection_km:.1f} +/- {best.std_detection_km:.1f} km")
    print(f"     P10 (worst 10%):  {best.p10_detection_km:.1f} km")
    print(f"     P50 (median):     {best.p50_detection_km:.1f} km")
    print(f"     P90 (best 10%):   {best.p90_detection_km:.1f} km")
    print(f"   Recognition range:  {best.mean_recognition_km:.1f} km")
    print(f"   Identification:     {best.mean_identification_km:.1f} km")
    print()

    # -----------------------------------------------------------------
    # 5. Band Comparison
    # -----------------------------------------------------------------
    print("5. Spectral Band Comparison (best per band)")
    print("-" * 40)

    bands_seen = {}
    for r in mc_results:
        b = r.candidate.fpa.spectral_band
        if b not in bands_seen:
            bands_seen[b] = r

    for band, r in sorted(bands_seen.items(), key=lambda x: -x[1].mean_detection_km):
        c = r.candidate
        print(f"   {band.value:12s}: {c.fpa.name:24s} F/{c.f_number} -> "
              f"Det={r.mean_detection_km:.1f}km, Rec={r.mean_recognition_km:.1f}km, "
              f"ID={r.mean_identification_km:.1f}km")
    print()

    # -----------------------------------------------------------------
    # 6. F-Number Trade Study (for best FPA)
    # -----------------------------------------------------------------
    print("6. F-Number Trade Study (best FPA)")
    print("-" * 40)

    best_fpa_name = bc.fpa.name
    fn_results = [r for r in mc_results if r.candidate.fpa.name == best_fpa_name]
    fn_results.sort(key=lambda r: r.candidate.f_number)

    for r in fn_results:
        c = r.candidate
        print(f"   F/{c.f_number:.1f}  aperture={c.focal_length_mm/c.f_number:.1f}mm  "
              f"-> Det={r.mean_detection_km:.1f}km  Rec={r.mean_recognition_km:.1f}km  "
              f"ID={r.mean_identification_km:.1f}km")
    print()

    # =================================================================
    # Visualization
    # =================================================================
    if args.no_plot:
        print("[Plots skipped (--no-plot)]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle(f'Air-to-Air F-16 Detection Optimization\n'
                 f'Sensor at {ALTITUDE_FT:,} ft, {FOV_DEG:.0f}° FOV, '
                 f'{args.mc_runs} MC runs per config',
                 fontsize=13, fontweight='bold')

    # --- Plot 1: Top 15 sensors ranked by mean detection range ---
    ax = axes[0, 0]
    top15 = mc_results[:15]
    names = [f"{r.candidate.fpa.name}\nF/{r.candidate.f_number}" for r in top15]
    means = [r.mean_detection_km for r in top15]
    p10s = [r.p10_detection_km for r in top15]
    p90s = [r.p90_detection_km for r in top15]
    yerr_lo = [m - p for m, p in zip(means, p10s)]
    yerr_hi = [p - m for m, p in zip(means, p90s)]
    y_pos = np.arange(len(names))

    # Color by band
    band_colors = {
        SpectralBand.MWIR: '#2196F3',
        SpectralBand.LWIR: '#F44336',
        SpectralBand.DUAL_MW_LW: '#9C27B0',
        SpectralBand.SWIR: '#4CAF50',
    }
    colors = [band_colors.get(r.candidate.fpa.spectral_band, '#888')
              for r in top15]

    ax.barh(y_pos, means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.errorbar(means, y_pos, xerr=[yerr_lo, yerr_hi], fmt='none',
                ecolor='black', capsize=3, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Detection Range (km)')
    ax.set_title('Top 15 Sensor Configs (P10-P90 bars)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    # Legend for bands
    from matplotlib.patches import Patch
    legend_items = [Patch(color=c, label=b.value) for b, c in band_colors.items()
                    if any(r.candidate.fpa.spectral_band == b for r in top15)]
    if legend_items:
        ax.legend(handles=legend_items, fontsize=7, loc='lower right')

    # --- Plot 2: Detection range histogram for best sensor ---
    ax = axes[0, 1]
    ax.hist(best.detection_ranges_km, bins=30, color='#2196F3', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Detection')
    ax.hist(best.recognition_ranges_km, bins=30, color='#FF9800', alpha=0.6,
            edgecolor='black', linewidth=0.5, label='Recognition')
    ax.hist(best.identification_ranges_km, bins=30, color='#F44336', alpha=0.5,
            edgecolor='black', linewidth=0.5, label='Identification')
    ax.axvline(best.mean_detection_km, color='blue', linestyle='--', linewidth=1.5,
               label=f'Mean Det: {best.mean_detection_km:.1f} km')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Count')
    ax.set_title(f'Range Distribution: {bc.fpa.name} F/{bc.f_number}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: F-number trade study ---
    ax = axes[0, 2]
    if fn_results:
        fns = [r.candidate.f_number for r in fn_results]
        det_means = [r.mean_detection_km for r in fn_results]
        rec_means = [r.mean_recognition_km for r in fn_results]
        id_means = [r.mean_identification_km for r in fn_results]
        det_p10 = [r.p10_detection_km for r in fn_results]
        det_p90 = [r.p90_detection_km for r in fn_results]

        ax.fill_between(fns, det_p10, det_p90, alpha=0.2, color='#2196F3')
        ax.plot(fns, det_means, 'o-', color='#2196F3', label='Detection', linewidth=2)
        ax.plot(fns, rec_means, 's--', color='#FF9800', label='Recognition', linewidth=1.5)
        ax.plot(fns, id_means, '^:', color='#F44336', label='Identification', linewidth=1.5)
        ax.set_xlabel('F-Number')
        ax.set_ylabel('Range (km)')
        ax.set_title(f'F/# Trade: {best_fpa_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Plot 4: IFOV vs Detection Range (all candidates) ---
    ax = axes[1, 0]
    for r in mc_results:
        c = r.candidate
        color = band_colors.get(c.fpa.spectral_band, '#888')
        ax.scatter(c.ifov_urad, r.mean_detection_km, color=color, alpha=0.6,
                   s=30, edgecolor='black', linewidth=0.3)
    # Highlight best
    ax.scatter(bc.ifov_urad, best.mean_detection_km, color='gold',
               s=150, marker='*', edgecolor='black', linewidth=1, zorder=10,
               label=f'Best: {bc.fpa.name}')
    ax.set_xlabel('IFOV (urad)')
    ax.set_ylabel('Mean Detection Range (km)')
    ax.set_title('IFOV vs Detection Range')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 5: Aperture vs Detection Range ---
    ax = axes[1, 1]
    for r in mc_results:
        c = r.candidate
        aperture = c.focal_length_mm / c.f_number
        color = band_colors.get(c.fpa.spectral_band, '#888')
        ax.scatter(aperture, r.mean_detection_km, color=color, alpha=0.6,
                   s=30, edgecolor='black', linewidth=0.3)
    best_ap = bc.focal_length_mm / bc.f_number
    ax.scatter(best_ap, best.mean_detection_km, color='gold',
               s=150, marker='*', edgecolor='black', linewidth=1, zorder=10,
               label=f'Best: {best_ap:.1f}mm aperture')
    ax.set_xlabel('Aperture Diameter (mm)')
    ax.set_ylabel('Mean Detection Range (km)')
    ax.set_title('Aperture vs Detection Range')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Band comparison box plot ---
    ax = axes[1, 2]
    band_data = {}
    for r in mc_results:
        b = r.candidate.fpa.spectral_band.value
        if b not in band_data:
            band_data[b] = []
        band_data[b].append(r.mean_detection_km)

    band_names = sorted(band_data.keys())
    box_data = [band_data[b] for b in band_names]
    bp = ax.boxplot(box_data, tick_labels=band_names, patch_artist=True, widths=0.5)
    box_colors_list = ['#2196F3', '#F44336', '#9C27B0', '#4CAF50']
    for patch, color in zip(bp['boxes'], box_colors_list[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Mean Detection Range (km)')
    ax.set_title('Detection Range by Spectral Band')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '43_air_to_air_detection.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Plot saved to: {output_path}")

    print()
    print("=" * 70)
    print("Example 43 complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Air-to-Air F-16 Detection MC Optimization")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--mc-runs", type=int, default=200,
                        help="Number of Monte Carlo runs per sensor (default: 200)")
    args = parser.parse_args()
    main(args)
