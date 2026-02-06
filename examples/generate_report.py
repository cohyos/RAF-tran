#!/usr/bin/env python3
"""
Generate PDF Report from All Examples
=====================================

This script runs all RAF-tran examples and generates a comprehensive PDF report
containing:
- Console output from each example
- All generated plots
- Timestamp in filename

Requirements:
    pip install matplotlib reportlab Pillow

Usage:
    python generate_report.py
    python generate_report.py --output-dir ./reports
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from io import StringIO
import tempfile
import shutil

# Check for required packages
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
        Preformatted, Table, TableStyle
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. Install with: pip install reportlab Pillow")


EXAMPLES = [
    # Core Examples (Demonstration)
    ("01_solar_zenith_angle_study.py", "Solar Zenith Angle Effects"),
    ("02_spectral_transmission.py", "Spectral Transmission (Sky Color)"),
    ("03_aerosol_types_comparison.py", "Aerosol Types Comparison"),
    ("04_atmospheric_profiles.py", "Atmospheric Profiles"),
    ("05_greenhouse_effect.py", "Greenhouse Effect"),
    ("06_surface_albedo_effects.py", "Surface Albedo Effects"),
    ("07_cloud_radiative_effects.py", "Cloud Radiative Effects"),
    ("08_ozone_uv_absorption.py", "Ozone UV Absorption"),
    ("09_radiative_heating_rates.py", "Radiative Heating Rates"),
    ("10_satellite_observation.py", "Satellite Observation Simulation"),
    ("11_atmospheric_turbulence.py", "Atmospheric Turbulence (Cn2, Fried)"),
    # Validation Examples (Physics Verification)
    ("12_beer_lambert_validation.py", "Beer-Lambert Law Validation"),
    ("13_planck_blackbody_validation.py", "Planck Blackbody Validation"),
    ("14_rayleigh_scattering_validation.py", "Rayleigh Scattering Validation"),
    ("15_mie_scattering_validation.py", "Mie Scattering Validation"),
    ("16_two_stream_benchmarks.py", "Two-Stream Solver Benchmarks"),
    ("17_solar_spectrum_analysis.py", "Solar Spectrum Analysis"),
    ("18_thermal_emission_validation.py", "Thermal Emission Validation"),
    ("19_path_radiance_remote_sensing.py", "Path Radiance Remote Sensing"),
    ("20_visibility_contrast.py", "Visibility and Contrast"),
    ("21_laser_propagation.py", "Laser Propagation"),
    # Advanced Applications
    ("22_atmospheric_polarization.py", "Atmospheric Polarization"),
    ("23_infrared_atmospheric_windows.py", "IR Atmospheric Windows"),
    ("24_volcanic_aerosol_forcing.py", "Volcanic Aerosol Forcing"),
    ("25_water_vapor_feedback.py", "Water Vapor Feedback"),
    ("26_high_altitude_solar.py", "High Altitude Solar Radiation"),
    ("27_twilight_spectra.py", "Twilight Spectra"),
    ("28_multi_layer_cloud.py", "Multi-Layer Cloud Overlap"),
    ("29_aod_retrieval_visibility.py", "AOD Retrieval and Visibility"),
    ("30_spectral_surface_albedo.py", "Spectral Surface Albedo"),
    ("31_limb_viewing_geometry.py", "Limb Viewing Geometry"),
    ("32_config_file_demo.py", "Configuration File Usage"),
    ("33_validation_visualization.py", "Physics Validation Visualization"),
    # Detection Applications
    ("34_fpa_detection_comparison.py", "FPA Detection Range Comparison"),
    ("35_fpa_altitude_detection_study.py", "FPA Altitude Detection Study"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all examples and generate PDF report",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory for output PDF (default: current directory)"
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="Don't run examples, just collect existing outputs"
    )
    parser.add_argument(
        "--max-lines", type=int, default=0,
        help="Max output lines per example (0 = unlimited, default: unlimited)"
    )
    return parser.parse_args()


def run_example(filename, work_dir):
    """Run an example and capture its output."""
    cmd = [sys.executable, filename]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=120  # 2 minute timeout per example
        )
        output = result.stdout
        if result.stderr:
            output += "\n\nSTDERR:\n" + result.stderr
        success = result.returncode == 0
        return output, success
    except subprocess.TimeoutExpired:
        return "ERROR: Example timed out after 120 seconds", False
    except Exception as e:
        return f"ERROR: {str(e)}", False


def parse_readme_sections(readme_path):
    """Parse README.md into sections for the report."""
    sections = {}
    current_section = None
    current_content = []

    try:
        with open(readme_path, encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            # Detect section headers (## or ###)
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = ''.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            elif line.startswith('### '):
                # Subsections - append to current section
                if current_section:
                    current_content.append(f"\n**{line[4:].strip()}**\n")
            else:
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = ''.join(current_content).strip()

    except Exception as e:
        sections['Error'] = f"Could not parse README: {e}"

    return sections


def get_scientific_references():
    """Return list of scientific references for the report."""
    return [
        # Atmospheric radiative transfer
        ("Bodhaine, B. A., Wood, N. B., Dutton, E. G., & Slusser, J. R. (1999). "
         "On Rayleigh optical depth calculations. J. Atmos. Oceanic Technol., 16(11), 1854-1861."),

        ("Bohren, C. F., & Huffman, D. R. (1983). Absorption and Scattering of Light "
         "by Small Particles. Wiley."),

        ("Meador, W. E., & Weaver, W. R. (1980). Two-stream approximations to radiative "
         "transfer in planetary atmospheres: A unified description of existing methods. "
         "J. Atmos. Sci., 37(3), 630-643."),

        ("Toon, O. B., McKay, C. P., Ackerman, T. P., & Santhanam, K. (1989). Rapid calculation "
         "of radiative heating rates and photodissociation rates in inhomogeneous multiple "
         "scattering atmospheres. J. Geophys. Res., 94(D13), 16287-16301."),

        ("Lacis, A. A., & Oinas, V. (1991). A description of the correlated k-distribution "
         "method for modeling nongray gaseous absorption. J. Geophys. Res., 96(D5), 9027-9063."),

        # IR detection and FPA
        ("Johnson, J. (1958). Analysis of image forming systems. Image Intensifier Symposium, "
         "Fort Belvoir, VA: Army Engineer Research and Development Laboratories."),

        ("Rogalski, A. (2011). Infrared Detectors (2nd ed.). CRC Press."),

        ("Hudson, R. D. (1969). Infrared System Engineering. Wiley."),

        # Digital ROIC technology
        ("Sizov, F. (2018). Brief history of THz and IR technologies. Semiconductor Physics, "
         "Quantum Electronics & Optoelectronics, 21(1), 6-28."),

        # Atmospheric turbulence
        ("Hufnagel, R. E., & Stanley, N. R. (1964). Modulation transfer function associated "
         "with image transmission through turbulent media. JOSA, 54(1), 52-61."),

        ("Andrews, L. C., & Phillips, R. L. (2005). Laser Beam Propagation through Random Media "
         "(2nd ed.). SPIE Press."),
    ]


def find_plot_for_example(filename, work_dir):
    """Find the plot file generated by an example."""
    # Common plot naming patterns
    base = filename.replace(".py", "")
    patterns = [
        f"{base}.png",
        f"{base.split('_', 1)[1]}.png" if '_' in base else None,
    ]

    # Also check for plots mentioned in the example files
    plot_mappings = {
        "01_solar_zenith_angle_study.py": "solar_zenith_study.png",
        "02_spectral_transmission.py": "spectral_transmission.png",
        "03_aerosol_types_comparison.py": "aerosol_types_comparison.png",
        "04_atmospheric_profiles.py": "atmospheric_profiles.png",
        "05_greenhouse_effect.py": "greenhouse_effect.png",
        "06_surface_albedo_effects.py": "surface_albedo_effects.png",
        "07_cloud_radiative_effects.py": "cloud_radiative_effects.png",
        "08_ozone_uv_absorption.py": "ozone_uv_absorption.png",
        "09_radiative_heating_rates.py": "radiative_heating_rates.png",
        "10_satellite_observation.py": "satellite_observation.png",
        "11_atmospheric_turbulence.py": "atmospheric_turbulence.png",
        "12_beer_lambert_validation.py": "beer_lambert_validation.png",
        "13_planck_blackbody_validation.py": "planck_blackbody_validation.png",
        "14_rayleigh_scattering_validation.py": "rayleigh_scattering_validation.png",
        "15_mie_scattering_validation.py": "mie_scattering_validation.png",
        "16_two_stream_benchmarks.py": "two_stream_benchmarks.png",
        "17_solar_spectrum_analysis.py": "solar_spectrum_analysis.png",
        "18_thermal_emission_validation.py": "thermal_emission_validation.png",
        "19_path_radiance_remote_sensing.py": "path_radiance_remote_sensing.png",
        "20_visibility_contrast.py": "visibility_contrast.png",
        "21_laser_propagation.py": "laser_propagation.png",
        "22_atmospheric_polarization.py": "atmospheric_polarization.png",
        "23_infrared_atmospheric_windows.py": "infrared_atmospheric_windows.png",
        "24_volcanic_aerosol_forcing.py": "volcanic_aerosol_forcing.png",
        "25_water_vapor_feedback.py": "water_vapor_feedback.png",
        "26_high_altitude_solar.py": "high_altitude_solar.png",
        "27_twilight_spectra.py": "twilight_spectra.png",
        "28_multi_layer_cloud.py": "multi_layer_cloud.png",
        "29_aod_retrieval_visibility.py": "aod_visibility.png",
        "30_spectral_surface_albedo.py": "spectral_albedo.png",
        "31_limb_viewing_geometry.py": "limb_viewing.png",
        "32_config_file_demo.py": "config_file_demo.png",
        "33_validation_visualization.py": "validation_visualization.png",
        "34_fpa_detection_comparison.py": "fpa_detection.png",
        "35_fpa_altitude_detection_study.py": "fpa_altitude_study.png",
    }

    if filename in plot_mappings:
        plot_path = work_dir / plot_mappings[filename]
        if plot_path.exists():
            return plot_path

    # Search for any matching PNG
    for pattern in patterns:
        if pattern:
            plot_path = work_dir / pattern
            if plot_path.exists():
                return plot_path

    return None


def create_pdf_report(results, output_path, max_lines=0):
    """Create PDF report from results.

    Parameters
    ----------
    results : list
        List of (filename, description, output, success, plot_path) tuples
    output_path : Path
        Output PDF path
    max_lines : int
        Maximum lines per example output (0 = unlimited)
    """
    if not REPORTLAB_AVAILABLE:
        print("ERROR: reportlab not available. Cannot create PDF.")
        print("Install with: pip install reportlab Pillow")
        return False

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.darkblue
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=7,
        leading=9,
        fontName='Courier',
        backColor=colors.Color(0.95, 0.95, 0.95)
    )

    story = []

    # Title page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("RAF-tran Examples Report", title_style))
    story.append(Spacer(1, 0.5*inch))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated: {timestamp}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Summary table
    summary_data = [["#", "Example", "Status"]]
    for filename, description, output, success, plot_path in results:
        status = "[OK] PASS" if success else "[X] FAIL"
        num = filename.split("_")[0]
        summary_data.append([num, description[:40], status])

    summary_table = Table(summary_data, colWidths=[0.5*inch, 4*inch, 1*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    story.append(summary_table)
    story.append(PageBreak())

    # README Chapter - Add overview from README.md
    readme_path = Path(__file__).parent.parent / 'README.md'
    print(f"Looking for README at: {readme_path}")
    if readme_path.exists():
        story.append(Paragraph("RAF-tran Overview", heading_style))
        story.append(Spacer(1, 0.1*inch))

        readme_sections = parse_readme_sections(readme_path)
        priority_sections = ['Overview', 'Features', 'Quick Start', 'Architecture']

        print(f"Found sections: {list(readme_sections.keys())}")

        for section_name in priority_sections:
            if section_name in readme_sections:
                story.append(Paragraph(section_name, styles['Heading3']))
                # Truncate very long sections
                content = readme_sections[section_name]
                print(f"  {section_name}: {len(content)} chars")
                if len(content) > 2000:
                    content = content[:2000] + "\n\n[... truncated for brevity ...]"
                # Clean up markdown syntax for PDF
                content = content.replace('```python', '').replace('```bash', '').replace('```', '')
                content = content.replace('**', '').replace('`', '')
                # Escape special characters for XML
                content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

                # Use Paragraph instead of Preformatted for better rendering
                # Split into paragraphs for better formatting
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Replace newlines with <br/> for single line breaks
                        para_text = para.strip().replace('\n', '<br/>')
                        try:
                            story.append(Paragraph(para_text, styles['Normal']))
                            story.append(Spacer(1, 0.05*inch))
                        except Exception as e:
                            print(f"    Error rendering paragraph: {e}")
                            # Fallback to preformatted
                            story.append(Preformatted(para.strip(), code_style))

                story.append(Spacer(1, 0.15*inch))
            else:
                print(f"  {section_name}: NOT FOUND")

        story.append(PageBreak())
    else:
        print(f"README not found at: {readme_path}")

    # Each example
    for filename, description, output, success, plot_path in results:
        num = filename.split("_")[0]

        # Header
        story.append(Paragraph(f"Example {num}: {description}", heading_style))
        story.append(Paragraph(f"File: {filename}", styles['Normal']))

        status_color = colors.green if success else colors.red
        status_text = "PASSED" if success else "FAILED"
        story.append(Paragraph(
            f"<font color='{status_color}'>{status_text}</font>",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))

        # Plot (if exists)
        if plot_path and plot_path.exists():
            try:
                img = Image(str(plot_path), width=6.5*inch, height=4.5*inch)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"[Could not load plot: {e}]", styles['Normal']))

        # Console output (truncated if limit set)
        story.append(Paragraph("Console Output:", styles['Heading4']))

        # Limit output length (0 = unlimited)
        output_lines = output.split('\n')
        if max_lines > 0 and len(output_lines) > max_lines:
            truncated_output = '\n'.join(output_lines[:max_lines])
            truncated_output += f"\n\n... [{len(output_lines) - max_lines} more lines truncated]"
        else:
            truncated_output = output

        # Escape special characters for reportlab
        safe_output = truncated_output.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        story.append(Preformatted(safe_output, code_style))
        story.append(PageBreak())

    # References Chapter - Add scientific references at the end
    story.append(Paragraph("References", heading_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "The following scientific publications form the theoretical foundation for RAF-tran:",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.15*inch))

    references = get_scientific_references()
    for i, ref in enumerate(references, 1):
        ref_text = f"[{i}] {ref}"
        # Escape special characters
        ref_text = ref_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(ref_text, styles['Normal']))
        story.append(Spacer(1, 0.05*inch))

    # Build PDF
    try:
        doc.build(story)
        return True
    except Exception as e:
        print(f"Error building PDF: {e}")
        return False


def main():
    args = parse_args()

    # Setup paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"raf_tran_examples_report_{timestamp}.pdf"
    pdf_path = output_dir / pdf_filename

    print("=" * 70)
    print("RAF-TRAN EXAMPLES PDF REPORT GENERATOR")
    print("=" * 70)
    print(f"Output: {pdf_path}")
    print()

    results = []

    for i, (filename, description) in enumerate(EXAMPLES, 1):
        print(f"[{i}/{len(EXAMPLES)}] Running {filename}...", end=" ", flush=True)

        if not args.no_run:
            output, success = run_example(filename, script_dir)
        else:
            output = "[Not run - using existing outputs]"
            success = True

        plot_path = find_plot_for_example(filename, script_dir)

        status = "[OK]" if success else "[X]"
        plot_status = "" if plot_path else "  "
        print(f"{status} {plot_status}")

        results.append((filename, description, output, success, plot_path))

    print()
    print("-" * 70)
    print("Generating PDF report...")

    if create_pdf_report(results, pdf_path, max_lines=args.max_lines):
        print(f"[OK] Report saved to: {pdf_path}")
    else:
        print("[X] Failed to create PDF report")

        # Fallback: save text report
        txt_path = output_dir / f"raf_tran_examples_report_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write("RAF-TRAN EXAMPLES REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")

            for filename, description, output, success, plot_path in results:
                f.write(f"\n{'='*70}\n")
                f.write(f"Example: {filename}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Status: {'PASS' if success else 'FAIL'}\n")
                if plot_path:
                    f.write(f"Plot: {plot_path}\n")
                f.write("-" * 70 + "\n")
                f.write(output)
                f.write("\n")

        print(f"[OK] Text report saved to: {txt_path}")

    # List generated plots
    plots = list(script_dir.glob("*.png"))
    if plots:
        print(f"\nGenerated plots ({len(plots)}):")
        for p in sorted(plots):
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
