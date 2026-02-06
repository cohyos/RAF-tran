"""
RAF-tran Streamlit GUI Application
==================================

Interactive web interface for atmospheric radiative transfer simulations.

Run with:
    streamlit run streamlit_app/app.py

Or:
    python -m streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import streamlit as st
    import matplotlib.pyplot as plt
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAF-tran",
        page_icon="sun",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RAF-tran: Atmospheric Radiative Transfer")
    st.markdown("*Open source atmospheric radiative transfer library*")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Module",
        [
            "Home",
            "Atmospheric Profiles",
            "Radiative Transfer",
            "IR Detection",
            "Turbulence",
            "Validation",
        ]
    )

    if page == "Home":
        show_home()
    elif page == "Atmospheric Profiles":
        show_atmospheric_profiles()
    elif page == "Radiative Transfer":
        show_radiative_transfer()
    elif page == "IR Detection":
        show_ir_detection()
    elif page == "Turbulence":
        show_turbulence()
    elif page == "Validation":
        show_validation()


def show_home():
    """Home page with overview."""
    st.header("Welcome to RAF-tran")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Features")
        st.markdown("""
        - **Atmospheric Models**: US Standard Atmosphere, CIRA-86, AFGL profiles
        - **Gas Optics**: Correlated-k method, optional HITRAN line-by-line
        - **Scattering**: Rayleigh (molecular) and Mie (aerosol/cloud)
        - **Radiative Transfer**: Two-stream and discrete ordinate solvers
        - **IR Detection**: FPA detector models, range equation
        - **Turbulence**: Cn2 profiles, Fried parameter, adaptive optics
        - **3D Geometry**: Spherical Earth, limb viewing, refraction
        """)

    with col2:
        st.subheader("Quick Start")
        st.code("""
# Python usage
from raf_tran.atmosphere import StandardAtmosphere
from raf_tran.rte_solver import TwoStreamSolver

atm = StandardAtmosphere()
solver = TwoStreamSolver(atm)
flux = solver.solve(wavelength=550e-9)
        """, language="python")

    st.info("This GUI works fully offline using built-in data.")


def show_atmospheric_profiles():
    """Atmospheric profiles page."""
    st.header("Atmospheric Profiles")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        profile_type = st.selectbox(
            "Profile Type",
            ["US Standard 1976", "Tropical", "Midlatitude Summer",
             "Midlatitude Winter", "Subarctic Summer", "Subarctic Winter"]
        )

        max_altitude = st.slider(
            "Maximum Altitude (km)",
            min_value=10,
            max_value=100,
            value=50,
        )

        show_humidity = st.checkbox("Show Humidity", value=True)

    with col2:
        st.subheader("Profile Visualization")

        # Generate profile
        from raf_tran.weather import (
            us_standard_atmosphere,
            tropical_atmosphere,
            midlatitude_summer,
            midlatitude_winter,
            subarctic_summer,
            subarctic_winter,
        )

        profile_funcs = {
            "US Standard 1976": us_standard_atmosphere,
            "Tropical": tropical_atmosphere,
            "Midlatitude Summer": midlatitude_summer,
            "Midlatitude Winter": midlatitude_winter,
            "Subarctic Summer": subarctic_summer,
            "Subarctic Winter": subarctic_winter,
        }

        altitudes = np.linspace(0, max_altitude * 1000, 100)
        profile = profile_funcs[profile_type](altitudes)

        # Create plot
        fig, axes = plt.subplots(1, 3 if show_humidity else 2, figsize=(12, 6))

        # Temperature
        axes[0].plot(profile.temperature, profile.altitudes / 1000, 'b-', linewidth=2)
        axes[0].set_xlabel("Temperature (K)")
        axes[0].set_ylabel("Altitude (km)")
        axes[0].set_title("Temperature Profile")
        axes[0].grid(True, alpha=0.3)

        # Pressure (log scale)
        axes[1].semilogx(profile.pressure / 100, profile.altitudes / 1000, 'r-', linewidth=2)
        axes[1].set_xlabel("Pressure (hPa)")
        axes[1].set_title("Pressure Profile")
        axes[1].grid(True, alpha=0.3)

        if show_humidity and profile.humidity is not None:
            axes[2].plot(profile.humidity * 100, profile.altitudes / 1000, 'g-', linewidth=2)
            axes[2].set_xlabel("Relative Humidity (%)")
            axes[2].set_title("Humidity Profile")
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Show statistics
        st.markdown(f"""
        **Profile Statistics:**
        - Surface Temperature: {profile.surface_temperature:.1f} K
        - Surface Pressure: {profile.surface_pressure/100:.1f} hPa
        - Scale Height: {profile.scale_height/1000:.1f} km
        """)


def show_radiative_transfer():
    """Radiative transfer calculations page."""
    st.header("Radiative Transfer")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        wavelength = st.slider(
            "Wavelength (nm)",
            min_value=300,
            max_value=2000,
            value=550,
        )

        sza = st.slider(
            "Solar Zenith Angle (deg)",
            min_value=0,
            max_value=85,
            value=30,
        )

        surface_albedo = st.slider(
            "Surface Albedo",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
        )

        show_spectrum = st.checkbox("Show Full Spectrum", value=False)

    with col2:
        st.subheader("Results")

        if show_spectrum:
            # Calculate for range of wavelengths
            wavelengths = np.linspace(300, 2000, 50)
            transmission = np.zeros_like(wavelengths)
            rayleigh_od = np.zeros_like(wavelengths)

            for i, wl in enumerate(wavelengths):
                # Simplified Rayleigh optical depth
                tau = 0.008569 * (wl / 1000)**(-4) * (1 + 0.0113 * (wl / 1000)**(-2))
                rayleigh_od[i] = tau

                # Transmission with air mass
                air_mass = 1 / np.cos(np.radians(sza))
                transmission[i] = np.exp(-tau * air_mass)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(wavelengths, rayleigh_od, 'b-', linewidth=2)
            axes[0].set_xlabel("Wavelength (nm)")
            axes[0].set_ylabel("Optical Depth")
            axes[0].set_title("Rayleigh Optical Depth")
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(wavelengths, transmission * 100, 'g-', linewidth=2)
            axes[1].set_xlabel("Wavelength (nm)")
            axes[1].set_ylabel("Transmission (%)")
            axes[1].set_title(f"Atmospheric Transmission (SZA={sza} deg)")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            # Single wavelength calculation
            wl = wavelength
            tau = 0.008569 * (wl / 1000)**(-4) * (1 + 0.0113 * (wl / 1000)**(-2))
            air_mass = 1 / np.cos(np.radians(sza))
            transmission = np.exp(-tau * air_mass)

            st.metric("Rayleigh Optical Depth", f"{tau:.4f}")
            st.metric("Air Mass", f"{air_mass:.2f}")
            st.metric("Direct Transmission", f"{transmission*100:.1f}%")

            # Flux calculations
            solar_constant = 1361  # W/m^2
            direct_flux = solar_constant * transmission * np.cos(np.radians(sza))
            diffuse_fraction = 0.1 * tau * air_mass  # Simplified

            st.metric("Direct Flux", f"{direct_flux:.0f} W/m^2")
            st.metric("Diffuse Fraction", f"{diffuse_fraction*100:.1f}%")


def show_ir_detection():
    """IR detection simulation page."""
    st.header("IR Detection Simulation")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Target Parameters")

        target_type = st.selectbox(
            "Target Type",
            ["Fighter (Rear)", "Fighter (Front)", "Transport", "Custom"]
        )

        if target_type == "Custom":
            target_temp = st.slider("Target Temperature (K)", 250, 1000, 400)
            target_area = st.slider("Target Area (m^2)", 0.1, 10.0, 1.0)
        else:
            target_configs = {
                "Fighter (Rear)": (700, 2.0),
                "Fighter (Front)": (280, 5.0),
                "Transport": (400, 10.0),
            }
            target_temp, target_area = target_configs[target_type]

        st.subheader("Detector Parameters")

        detector_type = st.selectbox(
            "Detector Type",
            ["InSb MWIR", "MCT LWIR Analog", "Digital LWIR"]
        )

        altitude = st.slider("Target Altitude (km)", 0, 20, 10)
        humidity = st.slider("Relative Humidity", 0.0, 1.0, 0.5)

    with col2:
        st.subheader("Detection Range Analysis")

        # Simplified detection range calculation
        detector_configs = {
            "InSb MWIR": {"d_star": 3e11, "wavelength": 4e-6, "band": "MWIR"},
            "MCT LWIR Analog": {"d_star": 5e10, "wavelength": 10e-6, "band": "LWIR"},
            "Digital LWIR": {"d_star": 2e11, "wavelength": 10e-6, "band": "LWIR"},
        }

        config = detector_configs[detector_type]

        # Stefan-Boltzmann for target radiance
        sigma = 5.67e-8
        target_radiance = sigma * target_temp**4 * target_area

        # Atmospheric transmission (simplified)
        if config["band"] == "MWIR":
            beta = 0.06 + 0.1 * humidity
        else:
            beta = 0.05 + 0.08 * humidity

        # Detection range (simplified)
        nei = 1e-12 / config["d_star"]
        ranges = np.linspace(1, 200, 100)  # km

        snr = np.zeros_like(ranges)
        for i, r in enumerate(ranges):
            tau = np.exp(-beta * r)
            irradiance = target_radiance * tau / (4 * np.pi * (r * 1000)**2)
            snr[i] = irradiance / nei

        # Find detection range (SNR = 5)
        detect_idx = np.where(snr >= 5)[0]
        if len(detect_idx) > 0:
            detection_range = ranges[detect_idx[-1]]
        else:
            detection_range = 0

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogy(ranges, snr, 'b-', linewidth=2, label='SNR')
        ax.axhline(y=5, color='r', linestyle='--', label='Detection Threshold (SNR=5)')
        if detection_range > 0:
            ax.axvline(x=detection_range, color='g', linestyle=':', label=f'Max Range: {detection_range:.1f} km')

        ax.set_xlabel("Range (km)")
        ax.set_ylabel("Signal-to-Noise Ratio")
        ax.set_title(f"Detection Range: {detector_type} vs {target_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.1, 1000)

        st.pyplot(fig)
        plt.close()

        # Display metrics
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Detection Range", f"{detection_range:.1f} km")
        col_b.metric("Target Radiance", f"{target_radiance:.1e} W")
        col_c.metric("Atm. Coefficient", f"{beta:.3f} /km")


def show_turbulence():
    """Atmospheric turbulence page."""
    st.header("Atmospheric Turbulence")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")

        cn2_ground = st.select_slider(
            "Ground Cn2 (m^-2/3)",
            options=[1e-16, 5e-16, 1e-15, 5e-15, 1e-14, 5e-14, 1e-13],
            value=1e-14,
            format_func=lambda x: f"{x:.0e}"
        )

        wind_speed = st.slider("Wind Speed (m/s)", 1, 50, 10)

        wavelength = st.slider("Wavelength (um)", 0.5, 10.0, 1.0)

        aperture = st.slider("Aperture Diameter (m)", 0.1, 4.0, 1.0)

    with col2:
        st.subheader("Turbulence Profile & Parameters")

        # Generate Cn2 profile
        from raf_tran.turbulence import hufnagel_valley_cn2

        altitudes = np.linspace(0, 20000, 100)
        cn2_profile = np.array([hufnagel_valley_cn2(h, cn2_ground) for h in altitudes])

        # Calculate parameters
        wavelength_m = wavelength * 1e-6
        k = 2 * np.pi / wavelength_m

        # Integrated Cn2
        cn2_integrated = np.trapezoid(cn2_profile, altitudes)

        # Fried parameter
        r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

        # Isoplanatic angle
        h53_integral = np.trapezoid(cn2_profile * altitudes**(5/3), altitudes)
        theta0 = (2.91 * k**2 * h53_integral)**(-3/5)

        # Greenwood frequency
        v_integral = np.trapezoid(cn2_profile * wind_speed**(5/3), altitudes)
        f_g = 0.102 * k**(6/5) * v_integral**(3/5)

        # Seeing
        seeing = 0.98 * wavelength_m / r0 * 206265

        # Strehl ratio (diffraction-limited if r0 > D)
        if aperture < r0:
            strehl = 1 - (aperture / r0)**(5/3)
        else:
            strehl = (r0 / aperture)**2

        # Plot Cn2 profile
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].semilogy(cn2_profile, altitudes / 1000, 'b-', linewidth=2)
        axes[0].set_xlabel("Cn2 (m^-2/3)")
        axes[0].set_ylabel("Altitude (km)")
        axes[0].set_title("Cn2 Profile (Hufnagel-Valley)")
        axes[0].grid(True, alpha=0.3)

        # Phase structure function
        separations = np.linspace(0.01, aperture, 50)
        D_phi = 6.88 * (separations / r0)**(5/3)

        axes[1].plot(separations * 100, D_phi, 'r-', linewidth=2)
        axes[1].set_xlabel("Separation (cm)")
        axes[1].set_ylabel("Phase Structure Function (rad^2)")
        axes[1].set_title("Wavefront Phase Statistics")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Display metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Fried Parameter (r0)", f"{r0*100:.1f} cm")
            st.metric("Seeing FWHM", f"{seeing:.2f} arcsec")
            st.metric("Integrated Cn2", f"{cn2_integrated:.2e}")

        with col_b:
            st.metric("Isoplanatic Angle", f"{theta0*206265:.1f} arcsec")
            st.metric("Greenwood Frequency", f"{f_g:.1f} Hz")
            st.metric("Long-Exp Strehl", f"{strehl:.3f}")


def show_validation():
    """Validation results page."""
    st.header("Validation Suite")

    st.markdown("""
    Run validation tests comparing RAF-tran outputs against benchmark data
    from MODTRAN, literature, and analytical solutions.
    """)

    if st.button("Run All Validations"):
        with st.spinner("Running validation tests..."):
            try:
                from raf_tran.validation import run_all_validations

                results = run_all_validations(verbose=False)

                # Display results
                st.subheader("Validation Results")

                if results.all_passed:
                    st.success(f"All {results.n_tests} tests PASSED")
                else:
                    st.warning(f"{results.n_passed}/{results.n_tests} tests passed")

                # Results table
                data = []
                for r in results.results:
                    data.append({
                        "Test": r.test_name,
                        "Status": "PASS" if r.passed else "FAIL",
                        "Max Error": f"{r.max_error:.4g}",
                        "Tolerance": f"{r.tolerance:.4g}",
                        "Rel. Error (%)": f"{r.relative_error:.2f}",
                        "Source": r.benchmark_source,
                    })

                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Validation failed: {e}")

    st.markdown("---")
    st.markdown("""
    **Benchmark Sources:**
    - Rayleigh scattering: Bodhaine et al. (1999)
    - Atmospheric profiles: US Standard Atmosphere 1976
    - Solar irradiance: Gueymard (2004), ASTM E490
    - Mie scattering: Bohren & Huffman (1983)
    - Turbulence: Andrews & Phillips (2005)
    """)


if __name__ == "__main__":
    main()
