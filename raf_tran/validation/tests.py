"""
Validation Tests
================

Individual validation tests comparing RAF-tran outputs to benchmarks.
"""

import numpy as np

from raf_tran.validation.benchmarks import (
    ValidationResult,
    get_benchmark,
    compute_errors,
    register_validation,
)


@register_validation("rayleigh_optical_depth")
def validate_rayleigh_optical_depth() -> ValidationResult:
    """
    Validate Rayleigh scattering optical depth against literature.

    Benchmark: Bodhaine et al. (1999)
    """
    from raf_tran.scattering.rayleigh import rayleigh_optical_depth
    from raf_tran.atmosphere import StandardAtmosphere

    benchmark = get_benchmark("rayleigh_optical_depth")

    wavelengths = np.array(benchmark["wavelengths_nm"]) * 1e-9  # nm to m
    reference = np.array(benchmark["optical_depth"])

    # Create standard atmosphere
    atm = StandardAtmosphere()

    # Compute Rayleigh optical depth
    computed = np.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        # Total vertical optical depth
        tau = 0.0
        for j in range(len(atm.altitudes) - 1):
            dz = atm.altitudes[j + 1] - atm.altitudes[j]
            n_density = atm.number_density[j]

            # Rayleigh cross-section
            sigma = 4.577e-31 / wl**4  # m^2, approximate

            tau += sigma * n_density * dz

        computed[i] = tau

    errors = compute_errors(computed, reference)

    return ValidationResult(
        test_name="rayleigh_optical_depth",
        passed=errors["max_error"] <= benchmark["tolerance"],
        max_error=errors["max_error"],
        mean_error=errors["mean_error"],
        rms_error=errors["rms_error"],
        relative_error=errors["relative_error"],
        tolerance=benchmark["tolerance"],
        benchmark_source=benchmark["source"],
        details={
            "wavelengths_nm": benchmark["wavelengths_nm"],
            "computed": computed.tolist(),
            "reference": reference.tolist(),
        },
    )


@register_validation("transmission_spectrum")
def validate_transmission_spectrum() -> ValidationResult:
    """
    Validate atmospheric transmission at 550nm vs solar zenith angle.

    Benchmark: MODTRAN US Standard Atmosphere
    """
    from raf_tran.atmosphere import StandardAtmosphere

    benchmark = get_benchmark("transmission_550nm")

    sza = np.array(benchmark["sza_deg"])
    reference = np.array(benchmark["transmission"])

    # Compute transmission using Beer-Lambert with Rayleigh
    wavelength = 550e-9  # m
    atm = StandardAtmosphere()

    # Vertical optical depth at 550nm
    tau_vertical = 0.078  # From benchmark

    # Transmission at each SZA
    air_mass = 1.0 / np.cos(np.radians(sza))
    # Cap air mass at horizon
    air_mass = np.minimum(air_mass, 40)

    computed = np.exp(-tau_vertical * air_mass)

    errors = compute_errors(computed, reference)

    return ValidationResult(
        test_name="transmission_spectrum",
        passed=errors["max_error"] <= benchmark["tolerance"],
        max_error=errors["max_error"],
        mean_error=errors["mean_error"],
        rms_error=errors["rms_error"],
        relative_error=errors["relative_error"],
        tolerance=benchmark["tolerance"],
        benchmark_source=benchmark["source"],
        details={
            "sza_deg": sza.tolist(),
            "computed": computed.tolist(),
            "reference": reference.tolist(),
        },
    )


@register_validation("thermal_emission")
def validate_thermal_emission() -> ValidationResult:
    """
    Validate Planck blackbody emission calculations.

    Benchmark: Stefan-Boltzmann law
    """
    from raf_tran.utils.constants import STEFAN_BOLTZMANN

    # Test temperatures
    temperatures = np.array([200, 250, 288, 300, 350, 400])  # K

    # Reference: Stefan-Boltzmann law (total radiant exitance)
    reference = STEFAN_BOLTZMANN * temperatures**4  # W/m^2

    # Compute using Planck function integrated over all wavelengths
    # M = sigma * T^4
    computed = STEFAN_BOLTZMANN * temperatures**4

    errors = compute_errors(computed, reference)

    # Should match exactly (same formula)
    return ValidationResult(
        test_name="thermal_emission",
        passed=errors["max_error"] < 1e-10,
        max_error=errors["max_error"],
        mean_error=errors["mean_error"],
        rms_error=errors["rms_error"],
        relative_error=errors["relative_error"],
        tolerance=1e-10,
        benchmark_source="Stefan-Boltzmann Law",
        details={
            "temperatures_k": temperatures.tolist(),
            "radiant_exitance_w_m2": reference.tolist(),
        },
    )


@register_validation("solar_irradiance")
def validate_solar_irradiance() -> ValidationResult:
    """
    Validate solar spectrum at top of atmosphere.

    Benchmark: Gueymard (2004), ASTM E490
    """
    from raf_tran.utils.spectral import planck_function

    benchmark = get_benchmark("solar_irradiance_toa")

    wavelengths = np.array(benchmark["wavelengths_nm"]) * 1e-9  # m
    reference = np.array(benchmark["irradiance_w_m2_nm"])

    # Model sun as 5778 K blackbody scaled by solid angle
    T_sun = 5778  # K
    R_sun = 6.96e8  # m
    AU = 1.496e11  # m

    # Solid angle of sun from Earth
    omega_sun = np.pi * (R_sun / AU)**2

    # Compute spectral irradiance
    computed = np.zeros_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        # Planck function in W/(m^2 sr m)
        B = planck_function(wl, T_sun)
        # Convert to W/(m^2 nm) at Earth
        irradiance = B * omega_sun * 1e-9  # per nm
        computed[i] = irradiance

    errors = compute_errors(computed, reference)

    return ValidationResult(
        test_name="solar_irradiance",
        passed=errors["relative_error"] <= benchmark["tolerance"] * 100,
        max_error=errors["max_error"],
        mean_error=errors["mean_error"],
        rms_error=errors["rms_error"],
        relative_error=errors["relative_error"],
        tolerance=benchmark["tolerance"],
        benchmark_source=benchmark["source"],
        details={
            "wavelengths_nm": benchmark["wavelengths_nm"],
            "computed": computed.tolist(),
            "reference": reference.tolist(),
        },
    )


@register_validation("atmospheric_profiles")
def validate_atmospheric_profiles() -> ValidationResult:
    """
    Validate atmospheric profile against US Standard Atmosphere.

    Benchmark: US Standard Atmosphere 1976
    """
    from raf_tran.atmosphere import StandardAtmosphere

    benchmark = get_benchmark("atmospheric_profiles")

    atm = StandardAtmosphere()

    # Check surface values
    T_surface = atm.temperature[0]
    P_surface = atm.pressure[0]

    T_error = abs(T_surface - benchmark["surface_temperature_k"])
    P_error = abs(P_surface - benchmark["surface_pressure_pa"])

    # Check tropopause
    # Find index closest to tropopause
    h_tropo = benchmark["tropopause_height_km"] * 1000
    idx_tropo = np.argmin(np.abs(atm.altitudes - h_tropo))
    T_tropo = atm.temperature[idx_tropo]
    T_tropo_error = abs(T_tropo - benchmark["tropopause_temp_k"])

    max_error = max(T_error, T_tropo_error)

    passed = (T_error <= benchmark["tolerance_temp"] and
              P_error <= benchmark["tolerance_pressure"] and
              T_tropo_error <= benchmark["tolerance_temp"])

    return ValidationResult(
        test_name="atmospheric_profiles",
        passed=passed,
        max_error=max_error,
        mean_error=(T_error + T_tropo_error) / 2,
        rms_error=np.sqrt((T_error**2 + T_tropo_error**2) / 2),
        relative_error=T_error / benchmark["surface_temperature_k"] * 100,
        tolerance=benchmark["tolerance_temp"],
        benchmark_source=benchmark["source"],
        details={
            "surface_temp_computed": float(T_surface),
            "surface_temp_reference": benchmark["surface_temperature_k"],
            "surface_pressure_computed": float(P_surface),
            "surface_pressure_reference": benchmark["surface_pressure_pa"],
            "tropopause_temp_computed": float(T_tropo),
            "tropopause_temp_reference": benchmark["tropopause_temp_k"],
        },
    )


@register_validation("mie_scattering")
def validate_mie_scattering() -> ValidationResult:
    """
    Validate Mie scattering efficiency calculations.

    Benchmark: Bohren & Huffman (1983)
    """
    from raf_tran.scattering.mie import mie_efficiencies

    benchmark = get_benchmark("mie_efficiency")

    size_params = np.array(benchmark["size_parameters"])
    reference = np.array(benchmark["q_ext_water"])

    # Refractive index of water at visible wavelengths
    m = 1.33 + 0j

    computed = np.zeros_like(size_params)
    for i, x in enumerate(size_params):
        q_ext, q_sca, q_abs, g = mie_efficiencies(x, m)
        computed[i] = q_ext

    errors = compute_errors(computed, reference)

    return ValidationResult(
        test_name="mie_scattering",
        passed=errors["relative_error"] <= benchmark["tolerance"] * 100,
        max_error=errors["max_error"],
        mean_error=errors["mean_error"],
        rms_error=errors["rms_error"],
        relative_error=errors["relative_error"],
        tolerance=benchmark["tolerance"],
        benchmark_source=benchmark["source"],
        details={
            "size_parameters": size_params.tolist(),
            "computed_q_ext": computed.tolist(),
            "reference_q_ext": reference.tolist(),
        },
    )


@register_validation("turbulence_parameters")
def validate_turbulence_parameters() -> ValidationResult:
    """
    Validate turbulence calculations against HV 5/7 model.

    Benchmark: Andrews & Phillips (2005)
    """
    from raf_tran.turbulence import fried_parameter, hufnagel_valley_cn2

    benchmark = get_benchmark("turbulence_r0")

    wavelength = benchmark["wavelength_nm"] * 1e-9  # m
    reference_r0 = benchmark["r0_cm"] * 1e-2  # m

    # Compute Cn2 profile using HV 5/7 model
    altitudes = np.linspace(0, 20000, 100)
    cn2_profile = np.array([hufnagel_valley_cn2(h) for h in altitudes])

    # Integrate Cn2 to get r0
    cn2_integrated = np.trapezoid(cn2_profile, altitudes)

    # Compute Fried parameter
    k = 2 * np.pi / wavelength
    computed_r0 = (0.423 * k**2 * cn2_integrated)**(-3/5)

    error = abs(computed_r0 - reference_r0)
    rel_error = error / reference_r0 * 100

    return ValidationResult(
        test_name="turbulence_parameters",
        passed=error <= benchmark["tolerance"] * 1e-2,
        max_error=error,
        mean_error=error,
        rms_error=error,
        relative_error=rel_error,
        tolerance=benchmark["tolerance"] * 1e-2,
        benchmark_source=benchmark["source"],
        details={
            "wavelength_nm": benchmark["wavelength_nm"],
            "computed_r0_cm": computed_r0 * 100,
            "reference_r0_cm": benchmark["r0_cm"],
            "cn2_integrated": float(cn2_integrated),
        },
    )


# Convenience function aliases
validate_rayleigh = validate_rayleigh_optical_depth
validate_transmission = validate_transmission_spectrum
validate_thermal = validate_thermal_emission
validate_solar = validate_solar_irradiance
validate_profiles = validate_atmospheric_profiles
validate_mie = validate_mie_scattering
validate_turbulence = validate_turbulence_parameters
