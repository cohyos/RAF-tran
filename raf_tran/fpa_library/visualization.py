"""
FPA Library Visualization
=========================

Plotting functions for FPA comparison charts, trade studies,
and performance analysis.

All functions return matplotlib figure objects for flexibility.
"""

from typing import List, Dict, Optional, Tuple

from raf_tran.fpa_library.models import (
    FPASpec, SpectralBand, DetectorType, CoolingType, Vendor,
)
from raf_tran.fpa_library.analysis import (
    compare_fpas, compute_dri_ranges, compute_swap_score,
    compute_sensitivity_score, compute_ifov_urad,
    pitch_miniaturization_factor, hot_reliability_gain,
)


def _get_vendor_color(vendor: Vendor) -> str:
    """Get a consistent color for each vendor."""
    colors = {
        Vendor.SCD: '#1f77b4',
        Vendor.TELEDYNE_FLIR: '#ff7f0e',
        Vendor.L3HARRIS: '#2ca02c',
        Vendor.RAYTHEON: '#d62728',
        Vendor.DRS: '#9467bd',
        Vendor.XENICS: '#8c564b',
        Vendor.AXIOM: '#e377c2',
        Vendor.LIGHTPATH: '#7f7f7f',
        Vendor.SIERRA_OLYMPIA: '#bcbd22',
        Vendor.SOFRADIR: '#17becf',
    }
    return colors.get(vendor, '#333333')


def _get_band_color(band: SpectralBand) -> str:
    """Get a consistent color for each spectral band."""
    colors = {
        SpectralBand.SWIR: '#FFD700',
        SpectralBand.MWIR: '#FF6B35',
        SpectralBand.LWIR: '#C70039',
        SpectralBand.VLWIR: '#900C3F',
        SpectralBand.DUAL_MW_LW: '#581845',
        SpectralBand.VISIBLE: '#44AA44',
    }
    return colors.get(band, '#666666')


def plot_resolution_vs_pitch(fpas: List[FPASpec], ax=None):
    """
    Scatter plot of total resolution vs pixel pitch, colored by vendor.

    Parameters
    ----------
    fpas : list of FPASpec
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    for fpa in fpas:
        if fpa.total_pixels is None:
            continue
        color = _get_vendor_color(fpa.vendor)
        mp = fpa.megapixels
        ax.scatter(fpa.pixel_pitch_um, mp, c=color, s=100, alpha=0.8,
                   edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(fpa.name, (fpa.pixel_pitch_um, mp),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Add vendor legend
    vendors_seen = set(fpa.vendor for fpa in fpas if fpa.total_pixels)
    for v in sorted(vendors_seen, key=lambda x: x.value):
        ax.scatter([], [], c=_get_vendor_color(v), s=80, label=v.value,
                   edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Pixel Pitch (um)')
    ax.set_ylabel('Resolution (Megapixels)')
    ax.set_title('FPA Resolution vs Pixel Pitch')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    return ax


def plot_netd_comparison(fpas: List[FPASpec], ax=None):
    """
    Horizontal bar chart comparing NETD sensitivity across FPAs.

    Parameters
    ----------
    fpas : list of FPASpec
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Filter FPAs with NETD data
    with_netd = [(fpa, fpa.netd_mk) for fpa in fpas if fpa.netd_mk is not None]
    with_netd.sort(key=lambda x: x[1])

    names = [f"{fpa.name}\n({fpa.vendor.value.split('(')[0].strip()})" for fpa, _ in with_netd]
    netds = [n for _, n in with_netd]
    colors = [_get_vendor_color(fpa.vendor) for fpa, _ in with_netd]

    bars = ax.barh(range(len(names)), netds, color=colors, edgecolor='black',
                   linewidth=0.5, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('NETD (mK) - Lower is Better')
    ax.set_title('Thermal Sensitivity Comparison')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, netds):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} mK', va='center', fontsize=8)

    ax.grid(True, axis='x', alpha=0.3)
    return ax


def plot_dri_ranges(fpas: List[FPASpec],
                    focal_length_mm: float = 100.0,
                    target_size_m: float = 2.3,
                    ax=None):
    """
    Grouped bar chart of Detection/Recognition/Identification ranges.

    Parameters
    ----------
    fpas : list of FPASpec
    focal_length_mm : float
    target_size_m : float
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    names = []
    det_ranges = []
    rec_ranges = []
    id_ranges = []

    for fpa in fpas:
        if fpa.total_pixels is None:
            continue
        dri = compute_dri_ranges(fpa, focal_length_mm, target_size_m)
        names.append(f"{fpa.name}")
        det_ranges.append(dri['detection_m'] / 1000)
        rec_ranges.append(dri['recognition_m'] / 1000)
        id_ranges.append(dri['identification_m'] / 1000)

    x = np.arange(len(names))
    width = 0.25

    ax.bar(x - width, det_ranges, width, label='Detection (1 lp)', color='#2196F3', alpha=0.8)
    ax.bar(x, rec_ranges, width, label='Recognition (3 lp)', color='#FF9800', alpha=0.8)
    ax.bar(x + width, id_ranges, width, label='Identification (6 lp)', color='#F44336', alpha=0.8)

    ax.set_ylabel('Range (km)')
    ax.set_title(f'Johnson Criteria DRI Ranges (f={focal_length_mm}mm, target={target_size_m}m)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def plot_swap_analysis(fpas: List[FPASpec], ax=None):
    """
    Scatter plot of weight vs power, with bubble size proportional to resolution.

    Parameters
    ----------
    fpas : list of FPASpec
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    for fpa in fpas:
        weight = fpa.weight_g
        power = fpa.power_steady_w or fpa.power_w
        if weight is None or power is None or fpa.total_pixels is None:
            continue

        mp = fpa.megapixels
        size = max(mp * 100, 30)
        color = _get_vendor_color(fpa.vendor)

        ax.scatter(weight, power, s=size, c=color, alpha=0.7,
                   edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(fpa.name, (weight, power),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    ax.set_xlabel('Weight (g)')
    ax.set_ylabel('Power (W)')
    ax.set_title('SWaP Analysis (bubble size = resolution)')
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectral_coverage(fpas: List[FPASpec], ax=None):
    """
    Horizontal range chart showing spectral coverage of each FPA.

    Parameters
    ----------
    fpas : list of FPASpec
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    with_range = [(fpa, fpa.spectral_range) for fpa in fpas
                  if fpa.spectral_range is not None]

    for i, (fpa, sr) in enumerate(with_range):
        color = _get_band_color(fpa.spectral_band)
        ax.barh(i, sr.bandwidth_um, left=sr.min_um, height=0.6,
                color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.text(sr.max_um + 0.1, i, fpa.name, va='center', fontsize=8)

    # Draw atmospheric windows
    for wl_min, wl_max, label in [(3.0, 5.0, 'MWIR Window'), (8.0, 14.0, 'LWIR Window')]:
        ax.axvspan(wl_min, wl_max, alpha=0.08, color='cyan')
        ax.text((wl_min + wl_max) / 2, len(with_range) + 0.3, label,
                ha='center', fontsize=8, style='italic', color='teal')

    ax.set_yticks(range(len(with_range)))
    ax.set_yticklabels([f"{fpa.vendor.value.split('(')[0].strip()}"
                        for fpa, _ in with_range], fontsize=8)
    ax.set_xlabel('Wavelength (um)')
    ax.set_title('Spectral Coverage by FPA')
    ax.set_xlim(0, 16)
    ax.grid(True, axis='x', alpha=0.3)
    return ax


def plot_technology_landscape(fpas: List[FPASpec], ax=None):
    """
    Scatter plot of pixel pitch vs NETD, with shapes for detector type.

    Parameters
    ----------
    fpas : list of FPASpec
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    markers = {
        DetectorType.InSb: 'o',
        DetectorType.XBn: 's',
        DetectorType.HFM: 's',
        DetectorType.T2SL: '^',
        DetectorType.SLS: '^',
        DetectorType.VOx: 'D',
        DetectorType.MCT: 'P',
        DetectorType.DUAL_BAND: '*',
    }

    for fpa in fpas:
        if fpa.netd_mk is None:
            continue
        marker = markers.get(fpa.detector_type, 'o')
        color = _get_vendor_color(fpa.vendor)
        cooled = fpa.is_cooled

        ax.scatter(fpa.pixel_pitch_um, fpa.netd_mk, marker=marker, c=color,
                   s=120 if cooled else 80,
                   alpha=0.8, edgecolors='black', linewidths=0.5 if cooled else 1.5,
                   zorder=5)
        ax.annotate(fpa.name, (fpa.pixel_pitch_um, fpa.netd_mk),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, alpha=0.8)

    # Legends for detector types
    for dt, m in markers.items():
        relevant = [f for f in fpas if f.detector_type == dt and f.netd_mk]
        if relevant:
            ax.scatter([], [], marker=m, c='gray', s=80, label=dt.value,
                       edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Pixel Pitch (um)')
    ax.set_ylabel('NETD (mK) - Lower is Better')
    ax.set_title('Technology Landscape: Pitch vs Sensitivity')
    ax.legend(fontsize=8, title='Detector Type', loc='upper right')
    ax.grid(True, alpha=0.3)
    return ax


def plot_vendor_portfolio_summary(fpas: List[FPASpec], ax=None):
    """
    Stacked bar chart showing each vendor's product count by spectral band.

    Parameters
    ----------
    fpas : list of FPASpec
    ax : matplotlib axis, optional
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    vendors = sorted(set(fpa.vendor for fpa in fpas), key=lambda v: v.value)
    bands = [SpectralBand.MWIR, SpectralBand.LWIR, SpectralBand.DUAL_MW_LW, SpectralBand.SWIR]
    band_labels = [b.value for b in bands]

    data = {}
    for band in bands:
        data[band.value] = []
        for vendor in vendors:
            count = sum(1 for f in fpas if f.vendor == vendor and f.spectral_band == band)
            data[band.value].append(count)

    x = np.arange(len(vendors))
    width = 0.6
    bottom = np.zeros(len(vendors))

    for band in bands:
        vals = np.array(data[band.value])
        color = _get_band_color(band)
        ax.bar(x, vals, width, label=band.value, bottom=bottom,
               color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([v.value.split('(')[0].strip() for v in vendors],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Number of Products')
    ax.set_title('Vendor Portfolio by Spectral Band')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    return ax
