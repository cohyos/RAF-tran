#!/usr/bin/env python3
"""
Example 45: Ballistic Missile Post-Boost Detection - MC Sensor Optimization
============================================================================

Monte Carlo optimization study for detecting a ballistic missile after
its boost phase from a space-based or high-altitude sensor platform.

Scenario
--------
- Sensor platform:     Space-based (LEO, 500 km altitude)
- Target:              Ballistic missile post-boost (midcourse phase)
                       at 200-400 km altitude
- Required FOV:        50 deg x 50 deg (wide-area staring)
- Objective:           Maximize detection range

Target Characteristics (Post-Boost)
------------------------------------
After boost motor burnout, the post-boost vehicle (PBV) and deployed
reentry vehicles (RVs) coast in the upper atmosphere / exo-atmosphere:

- RV body temperature: 250-350 K (cooling after boost heating)
  Thermal inertia keeps RV warm for minutes after burnout
- PBV (bus): 280-400 K (attitude thrusters, electronics waste heat)
- Residual plume: fading rapidly, ~300-500 K dissipating gas cloud
- Characteristic size: RV cone ~2 m length, ~0.5 m base diameter
  PBV: ~3 m length, ~1.5 m diameter
- No atmospheric obscuration above 100 km (exo-atmospheric)

IR Signature:
- MWIR (3-5 um): dominated by warm body emission against cold space
  Intensity: ~50-200 W/sr (PBV with thrusters), ~5-30 W/sr (cold RV)
- LWIR (8-14 um): strong thermal emission from warm bodies
  Intensity: ~200-800 W/sr (PBV), ~30-100 W/sr (RV)
- Cold space background: ~3K CMB -> essentially zero in IR bands
  Negligible background -> detection is sensor-noise limited

Atmospheric Path:
- Sensor in LEO looking down/across: slant path through upper
  atmosphere has very low extinction (target above 100 km)
- If looking through atmosphere: path from 500 km to target at
  200-400 km crosses only the rarefied upper atmosphere

Usage:
    python examples/45_ballistic_missile_detection_mc.py [--no-plot]
    python examples/45_ballistic_missile_detection_mc.py --mc-runs 500
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
# Scenario Parameters
# =============================================================================

SENSOR_ALTITUDE_KM = 500.0          # LEO orbit altitude
TARGET_ALTITUDE_KM = 300.0          # Post-boost midcourse, ~300 km
FOV_DEG = 50.0

# Post-boost vehicle (PBV / bus) parameters
PBV_LENGTH_M = 3.0
PBV_DIAMETER_M = 1.5
PBV_TARGET_SIZE_M = 2.25            # Characteristic dimension for Johnson

# Reentry vehicle (RV) parameters
RV_LENGTH_M = 2.0
RV_BASE_DIAMETER_M = 0.5
RV_TARGET_SIZE_M = 1.0

# We model the aggregate PBV + deployed RVs cluster.
# Post-boost phase: the PBV is still maneuvering (attitude thrusters on),
# deploying RVs sequentially. The cluster is the primary detection target.
CLUSTER_SIZE_M = 3.0                # Effective characteristic size

# IR Signature (PBV with active thrusters, post-boost)
# MWIR: warm body against cold space background
# Attitude control thrusters produce small hot spots (~600-1000 K)
# PBV bus body: ~300-400 K from electronics, thermal inertia
PBV_MWIR_INTENSITY_W_SR = 120.0     # W/sr MWIR (PBV with thrusters)
PBV_LWIR_INTENSITY_W_SR = 500.0     # W/sr LWIR (warm body + thrusters)

# Cold RV (deployed, coasting): much dimmer
RV_MWIR_INTENSITY_W_SR = 15.0       # W/sr MWIR (warm cone cooling)
RV_LWIR_INTENSITY_W_SR = 60.0       # W/sr LWIR (thermal emission of warm body)

# Background: cold space
# At LEO looking at target above 100 km, the background is essentially
# deep space (~3K CMB) plus Earth limb/earthshine if in FOV.
# We model only the cold space background.
MWIR_SPACE_RADIANCE = 1e-8          # W/sr/m^2/um (cold space, negligible)
LWIR_SPACE_RADIANCE = 1e-7          # W/sr/m^2/um (cold space, negligible)

# Earth limb radiance (if target is near Earth's limb in FOV)
# This is the dominant background: warm Earth atmosphere at the limb
MWIR_EARTHLIMB_RADIANCE = 1e-4      # W/sr/m^2/um
LWIR_EARTHLIMB_RADIANCE = 5e-3      # W/sr/m^2/um

# Atmospheric extinction: near-zero for exo-atmospheric targets
# Small residual from upper atmosphere (above 80 km: essentially vacuum)
# We model a very small extinction for the slant path
EXOATM_EXTINCTION_PER_KM = 1e-5     # Nearly transparent

# For paths that graze the upper atmosphere:
UPPER_ATM_EXTINCTION_PER_KM = 5e-4  # Grazing path through mesosphere

# Detection thresholds (space-based: lower clutter, can use lower threshold)
SNR_DETECTION = 4.0
SNR_RECOGNITION = 7.0
SNR_IDENTIFICATION = 12.0

T_AMBIENT_K = 3.0                   # Cold space background temperature


# =============================================================================
# Target Model
# =============================================================================

@dataclass
class TargetSignature:
    intensity_w_sr: float
    target_width_m: float
    target_height_m: float
    characteristic_size_m: float


def sample_target_signature(rng: np.random.Generator,
                            band: SpectralBand,
                            target_type: str = 'pbv') -> TargetSignature:
    """
    Sample post-boost target signature.

    target_type: 'pbv' (post-boost vehicle) or 'rv' (reentry vehicle)

    Variations:
    - Time since burnout (PBV cooling curve)
    - Thruster activity (PBV attitude control)
    - RV thermal state (cooling from boost heating)
    - Aspect angle (random orientation in space)
    """
    if target_type == 'pbv':
        if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
            base = PBV_MWIR_INTENSITY_W_SR
        else:
            base = PBV_LWIR_INTENSITY_W_SR

        # Time since burnout: PBV cools over time
        # 0-60s post-burnout: hot, 60-300s: cooling
        time_factor = rng.uniform(0.6, 1.5)

        # Thruster activity: periodic attitude control firings
        # Thruster on: 2-5x boost, off: baseline
        thruster_active = rng.random() < 0.3  # 30% chance thrusters firing
        if thruster_active:
            thruster_factor = rng.uniform(2.0, 5.0)
        else:
            thruster_factor = 1.0

        # Aspect angle in space (tumble/spin)
        aspect_factor = rng.uniform(0.5, 1.5)

        intensity = base * time_factor * thruster_factor * aspect_factor
        width = PBV_DIAMETER_M * rng.uniform(0.8, 1.2)
        height = PBV_LENGTH_M * rng.uniform(0.8, 1.2)
        char_size = CLUSTER_SIZE_M

    else:  # RV
        if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
            base = RV_MWIR_INTENSITY_W_SR
        else:
            base = RV_LWIR_INTENSITY_W_SR

        # RV cooling from boost phase heating
        cooling_factor = rng.uniform(0.4, 1.2)
        aspect_factor = rng.uniform(0.6, 1.4)

        intensity = base * cooling_factor * aspect_factor
        width = RV_BASE_DIAMETER_M * rng.uniform(0.8, 1.2)
        height = RV_LENGTH_M * rng.uniform(0.8, 1.2)
        char_size = RV_TARGET_SIZE_M

    intensity = max(intensity, 0.5)

    return TargetSignature(
        intensity_w_sr=intensity,
        target_width_m=width,
        target_height_m=height,
        characteristic_size_m=char_size,
    )


# =============================================================================
# Atmospheric / Path Model
# =============================================================================

@dataclass
class PathConditions:
    """Path conditions for one MC realization."""
    slant_range_factor: float       # Geometry variation
    extinction_per_km: float
    background_radiance: float      # W/sr/m^2/um
    earthlimb_in_fov: bool          # Whether Earth limb is behind target


def sample_path_conditions(rng: np.random.Generator,
                           band: SpectralBand) -> PathConditions:
    """
    Sample space-to-space path conditions.

    At orbital altitudes, the path is nearly vacuum with negligible
    extinction. The main background variable is whether the target
    is silhouetted against cold space or against Earth's limb.
    """
    # Slant range geometry variation
    # Target could be directly below (short range) or at angle (long range)
    slant_factor = rng.uniform(0.8, 1.5)

    # Extinction: essentially zero for exo-atmospheric paths
    # Occasional grazing of upper atmosphere adds small extinction
    grazing = rng.random() < 0.2  # 20% chance of atmosphere-grazing path
    if grazing:
        ext = UPPER_ATM_EXTINCTION_PER_KM * rng.uniform(0.5, 2.0)
    else:
        ext = EXOATM_EXTINCTION_PER_KM * rng.uniform(0.5, 2.0)

    # Background: cold space vs Earth limb
    earthlimb = rng.random() < 0.4  # 40% of time Earth limb is behind target

    if earthlimb:
        if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
            bg = MWIR_EARTHLIMB_RADIANCE * rng.uniform(0.5, 2.0)
        else:
            bg = LWIR_EARTHLIMB_RADIANCE * rng.uniform(0.5, 2.0)
    else:
        if band in (SpectralBand.MWIR, SpectralBand.DUAL_MW_LW):
            bg = MWIR_SPACE_RADIANCE * rng.uniform(0.5, 2.0)
        else:
            bg = LWIR_SPACE_RADIANCE * rng.uniform(0.5, 2.0)

    return PathConditions(
        slant_range_factor=slant_factor,
        extinction_per_km=ext,
        background_radiance=bg,
        earthlimb_in_fov=earthlimb,
    )


# =============================================================================
# Detection Range Computation
# =============================================================================

def compute_atmospheric_transmission(ext_per_km: float, range_km: float) -> float:
    return math.exp(-ext_per_km * range_km)


def compute_signal_irradiance(target: TargetSignature, range_m: float,
                              transmission: float,
                              optics_transmission: float) -> float:
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
    SNR for space-based sensor.

    In space, the background is either cold space (very low noise)
    or Earth limb (moderate). The detection is primarily limited
    by sensor self-noise (dark current, read noise).
    """
    signal_power = signal_irradiance * aperture_m2
    signal_energy = signal_power * integration_time_s

    pixel_solid_angle = pixel_area_m2 / (focal_length_m ** 2)
    bg_power = background_radiance * aperture_m2 * pixel_solid_angle
    bg_energy = bg_power * integration_time_s

    # In space with cold background, sensor dark current dominates.
    # Model noise floor from NETD at some reference background temp.
    # For cooled sensors: dark current is very low.
    # NETD-based noise scaled to actual background level.
    if bg_energy > 0:
        noise_fraction = netd_k / max(T_AMBIENT_K, 50.0)
        noise_energy = noise_fraction * bg_energy
    else:
        noise_energy = 0.0

    # Sensor self-noise floor (read noise, dark current)
    # Even with zero background, there is a minimum noise.
    # Model as NETD-equivalent at 300K scene with f/2 optics.
    min_noise = netd_k / 300.0 * 1e-12 * aperture_m2 * integration_time_s
    noise_energy = max(noise_energy, min_noise)

    if noise_energy <= 0:
        noise_energy = signal_energy * 0.001

    n_effective = math.sqrt(n_frames * tdi_stages)
    return (signal_energy / noise_energy) * n_effective


def find_detection_range(target: TargetSignature,
                         path: PathConditions,
                         sensor: SensorAssembly,
                         band: SpectralBand,
                         snr_threshold: float = SNR_DETECTION) -> float:
    """Find max detection range (km) by bisection."""
    optics = sensor.optics
    operating = sensor.operating
    fpa = sensor.fpa

    aperture_m = optics.aperture_mm / 1000.0
    aperture_area_m2 = math.pi * (aperture_m / 2) ** 2
    if optics.obscuration_ratio > 0:
        aperture_area_m2 *= (1 - optics.obscuration_ratio ** 2)

    focal_length_m = optics.focal_length_mm / 1000.0
    pixel_size_m = fpa.pixel_pitch_um * 1e-6
    pixel_area_m2 = pixel_size_m ** 2
    netd_k = (fpa.netd_mk or 30.0) / 1000.0

    bg_rad = path.background_radiance
    integration_time_s = operating.integration_time_ms / 1000.0
    n_frames = operating.num_frames_avg
    tdi = operating.tdi_stages

    # Base range: distance between sensor (500 km) and target (~300 km)
    # in same orbital plane. Varies from ~200 km (nadir) to ~2000 km (limb).
    r_min_km = 10.0
    r_max_km = 5000.0

    for _ in range(50):
        r_mid_km = (r_min_km + r_max_km) / 2.0
        r_mid_m = r_mid_km * 1000.0

        tau = compute_atmospheric_transmission(path.extinction_per_km,
                                               r_mid_km)
        e_signal = compute_signal_irradiance(
            target, r_mid_m, tau, optics.transmission)

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

        if abs(r_max_km - r_min_km) < 0.1:
            break

    return (r_min_km + r_max_km) / 2.0


# =============================================================================
# FOV-Constrained Focal Length
# =============================================================================

def focal_length_for_fov(fpa: FPASpec, fov_deg: float) -> Optional[float]:
    if fpa.array_format is None:
        return None
    max_dim = max(fpa.array_format.columns, fpa.array_format.rows)
    array_size_mm = max_dim * fpa.pixel_pitch_um / 1000.0
    half_fov_rad = math.radians(fov_deg / 2.0)
    if half_fov_rad <= 0 or half_fov_rad >= math.pi / 2:
        return None
    return array_size_mm / (2.0 * math.tan(half_fov_rad))


# =============================================================================
# Monte Carlo Evaluation
# =============================================================================

@dataclass
class SensorCandidate:
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
    db = get_fpa_database()
    candidates = []
    f_numbers = [1.4, 2.0, 2.8, 4.0]

    for key, fpa in db.items():
        if fpa.array_format is None:
            continue
        if fpa.array_format.total_pixels < 100000:
            continue

        fl_mm = focal_length_for_fov(fpa, fov_deg)
        if fl_mm is None or fl_mm < 2.0:
            continue

        fov = fpa.fov_at_focal_length(fl_mm)
        if fov is None:
            continue

        ifov_urad = fpa.pixel_pitch_um / fl_mm * 1000.0

        for fn in f_numbers:
            # Space-based: can use longer integration (stable platform)
            if fpa.spectral_band == SpectralBand.LWIR:
                tint_ms = 5.0
            elif fpa.spectral_band == SpectralBand.MWIR:
                tint_ms = 10.0
            else:
                tint_ms = 8.0

            optics = OpticsConfig(focal_length_mm=fl_mm, f_number=fn,
                                  transmission=0.85)
            operating = OperatingParams(integration_time_ms=tint_ms,
                                        frame_rate_hz=10.0,  # Space: lower rate OK
                                        num_frames_avg=8)    # More averaging in space
            assembly = SensorAssembly(fpa, optics, operating,
                                      name=f"{fpa.name} F/{fn}",
                                      validate_on_init=False)

            candidates.append(SensorCandidate(
                fpa_key=key, fpa=fpa, focal_length_mm=fl_mm, f_number=fn,
                fov_h_deg=fov[0], fov_v_deg=fov[1], ifov_urad=ifov_urad,
                assembly=assembly))

    return candidates


def run_mc_evaluation(candidates: List[SensorCandidate],
                      n_runs: int = 200, seed: int = 45,
                      target_type: str = 'pbv') -> List[MCResult]:
    rng = np.random.default_rng(seed)
    results = []

    for cand in candidates:
        band = cand.fpa.spectral_band
        det_ranges = np.zeros(n_runs)
        rec_ranges = np.zeros(n_runs)
        idf_ranges = np.zeros(n_runs)

        for i in range(n_runs):
            target = sample_target_signature(rng, band, target_type)
            path = sample_path_conditions(rng, band)

            det_ranges[i] = find_detection_range(
                target, path, cand.assembly, band, SNR_DETECTION)
            rec_ranges[i] = find_detection_range(
                target, path, cand.assembly, band, SNR_RECOGNITION)
            idf_ranges[i] = find_detection_range(
                target, path, cand.assembly, band, SNR_IDENTIFICATION)

        results.append(MCResult(
            candidate=cand,
            detection_ranges_km=det_ranges,
            recognition_ranges_km=rec_ranges,
            identification_ranges_km=idf_ranges))

    return results


# =============================================================================
# Main
# =============================================================================

def main(args):
    print("=" * 70)
    print("Example 45: Ballistic Missile Post-Boost Detection - MC Optimization")
    print("=" * 70)
    print()

    print("Scenario:")
    print(f"  Sensor platform:     LEO ({SENSOR_ALTITUDE_KM:.0f} km)")
    print(f"  Target:              Post-boost vehicle (PBV) at ~{TARGET_ALTITUDE_KM:.0f} km")
    print(f"  Target type:         {args.target_type.upper()}")
    print(f"  Required FOV:        {FOV_DEG:.0f} x {FOV_DEG:.0f} deg")
    print(f"  MC runs per sensor:  {args.mc_runs}")
    if args.target_type == 'pbv':
        print(f"  PBV MWIR intensity:  {PBV_MWIR_INTENSITY_W_SR} W/sr")
        print(f"  PBV LWIR intensity:  {PBV_LWIR_INTENSITY_W_SR} W/sr")
    else:
        print(f"  RV MWIR intensity:   {RV_MWIR_INTENSITY_W_SR} W/sr")
        print(f"  RV LWIR intensity:   {RV_LWIR_INTENSITY_W_SR} W/sr")
    print(f"  Background:          Cold space + Earth limb (40% probability)")
    print(f"  Exo-atm extinction:  ~{EXOATM_EXTINCTION_PER_KM:.0e} /km")
    print(f"  Detection threshold: SNR >= {SNR_DETECTION}")
    print()

    # -----------------------------------------------------------------
    # 1. Build candidates
    # -----------------------------------------------------------------
    print("1. Building Sensor Candidates")
    print("-" * 40)

    candidates = build_candidates(FOV_DEG)
    print(f"   Total candidates: {len(candidates)} (FPA x F-number)")

    seen = {}
    for c in candidates:
        if c.fpa.name not in seen:
            seen[c.fpa.name] = c
    print(f"   Unique FPAs: {len(seen)}")
    print()

    print("   FPA                    | Band  | Format      | Pitch | FL(mm) | IFOV(urad)")
    print("   " + "-" * 80)
    for name, c in sorted(seen.items(), key=lambda x: x[1].focal_length_mm):
        print(f"   {name:24s} | {c.fpa.spectral_band.value:5s} | "
              f"{c.fpa.resolution_str:11s} | {c.fpa.pixel_pitch_um:5.1f} | "
              f"{c.focal_length_mm:6.1f} | {c.ifov_urad:9.1f}")
    print()

    # -----------------------------------------------------------------
    # 2. Run MC (PBV)
    # -----------------------------------------------------------------
    print(f"2. Running Monte Carlo ({args.target_type.upper()})...")
    print("-" * 40)

    mc_results = run_mc_evaluation(candidates, n_runs=args.mc_runs, seed=45,
                                    target_type=args.target_type)
    print(f"   Completed {len(mc_results)} x {args.mc_runs} = "
          f"{len(mc_results) * args.mc_runs:,} simulations")
    print()

    # -----------------------------------------------------------------
    # 3. Top 20
    # -----------------------------------------------------------------
    mc_results.sort(key=lambda r: r.mean_detection_km, reverse=True)

    print("3. Top 20 Results by Mean Detection Range")
    print("-" * 40)
    print(f"   {'Rank':>4} | {'FPA':24s} | {'F/#':>4} | {'Band':5s} | "
          f"{'FL(mm)':>7} | {'IFOV':>7} | {'Det(km)':>9} | {'Rec(km)':>9} | "
          f"{'ID(km)':>9} | {'P10':>8} | {'P90':>8}")
    print("   " + "-" * 125)

    for i, r in enumerate(mc_results[:20]):
        c = r.candidate
        print(f"   {i+1:4d} | {c.fpa.name:24s} | {c.f_number:4.1f} | "
              f"{c.fpa.spectral_band.value:5s} | {c.focal_length_mm:7.1f} | "
              f"{c.ifov_urad:7.1f} | {r.mean_detection_km:9.1f} | "
              f"{r.mean_recognition_km:9.1f} | {r.mean_identification_km:9.1f} | "
              f"{r.p10_detection_km:8.1f} | {r.p90_detection_km:8.1f}")
    print()

    # -----------------------------------------------------------------
    # 4. Best configuration
    # -----------------------------------------------------------------
    best = mc_results[0]
    bc = best.candidate

    print("4. Optimal Sensor Configuration")
    print("-" * 40)
    print(f"   FPA:                {bc.fpa.name} ({bc.fpa.vendor.value})")
    print(f"   Detector:           {bc.fpa.detector_type.value}")
    print(f"   Spectral band:      {bc.fpa.spectral_band.value}", end="")
    if bc.fpa.spectral_range:
        print(f" ({bc.fpa.spectral_range})")
    else:
        print()
    print(f"   Array:              {bc.fpa.resolution_str}")
    print(f"   Pixel pitch:        {bc.fpa.pixel_pitch_um} um")
    print(f"   Cooling:            {bc.fpa.cooling.value}")
    print()
    print(f"   === Recommended Optics ===")
    print(f"   Focal length:       {bc.focal_length_mm:.1f} mm")
    print(f"   F-number:           F/{bc.f_number:.1f}")
    print(f"   Aperture:           {bc.focal_length_mm / bc.f_number:.1f} mm")
    fov = bc.fpa.fov_at_focal_length(bc.focal_length_mm)
    if fov:
        print(f"   Actual FOV:         {fov[0]:.1f} x {fov[1]:.1f} deg")
    print(f"   IFOV:               {bc.ifov_urad:.1f} urad")
    print()
    print(f"   === MC Performance (N={args.mc_runs}) ===")
    print(f"   Detection:          {best.mean_detection_km:.1f} +/- {best.std_detection_km:.1f} km")
    print(f"     P10:              {best.p10_detection_km:.1f} km")
    print(f"     P50:              {best.p50_detection_km:.1f} km")
    print(f"     P90:              {best.p90_detection_km:.1f} km")
    print(f"   Recognition:        {best.mean_recognition_km:.1f} km")
    print(f"   Identification:     {best.mean_identification_km:.1f} km")
    print()

    # -----------------------------------------------------------------
    # 5. Band comparison
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
              f"Det={r.mean_detection_km:.1f}km")
    print()

    # -----------------------------------------------------------------
    # 6. FULL FPA RANKING (high to low)
    # -----------------------------------------------------------------
    print("6. Complete FPA Ranking (best F/# per FPA, high to low)")
    print("=" * 100)

    fpa_best = {}
    for r in mc_results:
        name = r.candidate.fpa.name
        if name not in fpa_best or r.mean_detection_km > fpa_best[name].mean_detection_km:
            fpa_best[name] = r

    ranking = sorted(fpa_best.values(),
                     key=lambda r: r.mean_detection_km, reverse=True)

    print(f"   {'Rank':>4} | {'FPA':24s} | {'Vendor':20s} | {'Band':5s} | "
          f"{'F/#':>4} | {'Det(km)':>9} | {'Rec(km)':>9} | {'ID(km)':>9} | "
          f"{'P10':>8} | {'P90':>8}")
    print("   " + "-" * 130)

    for i, r in enumerate(ranking):
        c = r.candidate
        vendor = c.fpa.vendor.value.split('(')[0].strip()[:20]
        print(f"   {i+1:4d} | {c.fpa.name:24s} | {vendor:20s} | "
              f"{c.fpa.spectral_band.value:5s} | {c.f_number:4.1f} | "
              f"{r.mean_detection_km:9.1f} | {r.mean_recognition_km:9.1f} | "
              f"{r.mean_identification_km:9.1f} | "
              f"{r.p10_detection_km:8.1f} | {r.p90_detection_km:8.1f}")

    print()
    print(f"   Total FPAs ranked: {len(ranking)}")
    print(f"   Best:  {ranking[0].candidate.fpa.name} "
          f"({ranking[0].mean_detection_km:.1f} km)")
    print(f"   Worst: {ranking[-1].candidate.fpa.name} "
          f"({ranking[-1].mean_detection_km:.1f} km)")
    print()

    # =================================================================
    # Visualization
    # =================================================================
    if args.no_plot:
        print("[Plots skipped (--no-plot)]")
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    target_label = 'PBV' if args.target_type == 'pbv' else 'RV'
    fig.suptitle(f'Ballistic Missile Post-Boost ({target_label}) Detection\n'
                 f'LEO sensor ({SENSOR_ALTITUDE_KM:.0f} km), '
                 f'{FOV_DEG:.0f}\u00b0 FOV, {args.mc_runs} MC runs',
                 fontsize=13, fontweight='bold')

    band_colors = {
        SpectralBand.MWIR: '#2196F3',
        SpectralBand.LWIR: '#F44336',
        SpectralBand.DUAL_MW_LW: '#9C27B0',
        SpectralBand.SWIR: '#4CAF50',
    }

    # --- Plot 1: Full FPA ranking ---
    ax = axes[0, 0]
    rank_names = [r.candidate.fpa.name for r in ranking]
    rank_det = [r.mean_detection_km for r in ranking]
    rank_colors = [band_colors.get(r.candidate.fpa.spectral_band, '#888')
                   for r in ranking]
    y_pos = np.arange(len(rank_names))
    ax.barh(y_pos, rank_det, color=rank_colors, alpha=0.8,
            edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rank_names, fontsize=6)
    ax.set_xlabel('Mean Detection Range (km)')
    ax.set_title('FPA Ranking (best F/# each)')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    from matplotlib.patches import Patch
    legend_items = [Patch(color=c, label=b.value) for b, c in band_colors.items()
                    if any(r.candidate.fpa.spectral_band == b for r in ranking)]
    if legend_items:
        ax.legend(handles=legend_items, fontsize=7, loc='lower right')

    # --- Plot 2: Detection histogram ---
    ax = axes[0, 1]
    ax.hist(best.detection_ranges_km, bins=30, color='#2196F3', alpha=0.7,
            edgecolor='black', linewidth=0.5, label='Detection')
    ax.hist(best.recognition_ranges_km, bins=30, color='#FF9800', alpha=0.6,
            edgecolor='black', linewidth=0.5, label='Recognition')
    ax.hist(best.identification_ranges_km, bins=30, color='#F44336', alpha=0.5,
            edgecolor='black', linewidth=0.5, label='Identification')
    ax.axvline(best.mean_detection_km, color='blue', linestyle='--',
               linewidth=1.5, label=f'Mean: {best.mean_detection_km:.0f} km')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Count')
    ax.set_title(f'Range Distribution: {bc.fpa.name} F/{bc.f_number}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: F/# trade ---
    ax = axes[0, 2]
    best_fpa_name = bc.fpa.name
    fn_results = sorted([r for r in mc_results
                         if r.candidate.fpa.name == best_fpa_name],
                        key=lambda r: r.candidate.f_number)
    if fn_results:
        fns = [r.candidate.f_number for r in fn_results]
        det_m = [r.mean_detection_km for r in fn_results]
        rec_m = [r.mean_recognition_km for r in fn_results]
        id_m = [r.mean_identification_km for r in fn_results]
        p10 = [r.p10_detection_km for r in fn_results]
        p90 = [r.p90_detection_km for r in fn_results]
        ax.fill_between(fns, p10, p90, alpha=0.2, color='#2196F3')
        ax.plot(fns, det_m, 'o-', color='#2196F3', label='Detection', linewidth=2)
        ax.plot(fns, rec_m, 's--', color='#FF9800', label='Recognition')
        ax.plot(fns, id_m, '^:', color='#F44336', label='Identification')
        ax.set_xlabel('F-Number')
        ax.set_ylabel('Range (km)')
        ax.set_title(f'F/# Trade: {best_fpa_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # --- Plot 4: IFOV vs detection ---
    ax = axes[1, 0]
    for r in mc_results:
        c = r.candidate
        color = band_colors.get(c.fpa.spectral_band, '#888')
        ax.scatter(c.ifov_urad, r.mean_detection_km, color=color,
                   alpha=0.5, s=25, edgecolor='black', linewidth=0.3)
    ax.scatter(bc.ifov_urad, best.mean_detection_km, color='gold',
               s=150, marker='*', edgecolor='black', linewidth=1, zorder=10,
               label=f'Best: {bc.fpa.name}')
    ax.set_xlabel('IFOV (urad)')
    ax.set_ylabel('Detection Range (km)')
    ax.set_title('IFOV vs Detection')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 5: Aperture vs detection ---
    ax = axes[1, 1]
    for r in mc_results:
        c = r.candidate
        aperture = c.focal_length_mm / c.f_number
        color = band_colors.get(c.fpa.spectral_band, '#888')
        ax.scatter(aperture, r.mean_detection_km, color=color,
                   alpha=0.5, s=25, edgecolor='black', linewidth=0.3)
    best_ap = bc.focal_length_mm / bc.f_number
    ax.scatter(best_ap, best.mean_detection_km, color='gold',
               s=150, marker='*', edgecolor='black', linewidth=1, zorder=10,
               label=f'Best: {best_ap:.1f}mm')
    ax.set_xlabel('Aperture (mm)')
    ax.set_ylabel('Detection Range (km)')
    ax.set_title('Aperture vs Detection')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Band box plot ---
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
    box_cl = ['#2196F3', '#F44336', '#9C27B0', '#4CAF50']
    for patch, color in zip(bp['boxes'], box_cl[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Detection Range (km)')
    ax.set_title('Detection by Spectral Band')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '45_ballistic_missile_detection.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Plot saved to: {output_path}")

    print()
    print("=" * 70)
    print("Example 45 complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ballistic Missile Post-Boost Detection MC Optimization")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--mc-runs", type=int, default=200)
    parser.add_argument("--target-type", choices=['pbv', 'rv'], default='pbv',
                        help="Target: pbv (post-boost vehicle) or rv (reentry vehicle)")
    args = parser.parse_args()
    main(args)
