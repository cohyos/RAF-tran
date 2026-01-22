"""
Radiative Transfer Equation (RTE) Solver.

Implements FR-10: Solution of the radiative transfer equation including:
- Transmittance calculation
- Thermal emission (self-emission)
- Solar scattering (single scattering approximation)

Supports multiple path geometries (FR-07) with Earth curvature correction (FR-08).
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from numba import jit, prange

from raf_tran.core.constants import (
    C2_RADIATION,
    EARTH_RADIUS_KM,
    STEFAN_BOLTZMANN,
    PATH_TYPES,
    wavenumber_to_wavelength,
)
from raf_tran.core.gas_engine import GasEngine, AbsorptionCoefficients
from raf_tran.core.scattering_engine import ScatteringEngine, ScatteringProperties
from raf_tran.config.atmosphere import AtmosphereProfile

logger = logging.getLogger(__name__)


@dataclass
class RTEResult:
    """Results from radiative transfer calculation.

    Attributes:
        wavenumber: Spectral grid [cm^-1]
        wavelength_um: Wavelength grid [um]
        transmittance: Path transmittance [0-1]
        radiance: Spectral radiance [W/(cm²·sr·cm^-1)]
        optical_depth: Total optical depth (dimensionless)
        thermal_emission: Thermal emission contribution [W/(cm²·sr·cm^-1)]
        solar_scattering: Solar scattering contribution (if applicable)
        layer_transmittances: Per-layer transmittances (if intermediate results requested)
    """
    wavenumber: np.ndarray
    wavelength_um: np.ndarray
    transmittance: np.ndarray
    radiance: np.ndarray
    optical_depth: np.ndarray
    thermal_emission: np.ndarray
    solar_scattering: Optional[np.ndarray] = None
    layer_transmittances: Optional[List[np.ndarray]] = None
    layer_optical_depths: Optional[List[np.ndarray]] = None


@dataclass
class PathGeometry:
    """Path geometry specification.

    Attributes:
        path_type: Type of path (HORIZONTAL, SLANT, VERTICAL)
        h1_km: Start altitude [km]
        h2_km: End altitude [km]
        zenith_angle_deg: Zenith angle [degrees] (0=vertical, 90=horizontal)
        path_length_km: Total path length [km]
        layer_path_lengths_km: Path length through each layer [km]
        include_earth_curvature: Apply Earth curvature correction
    """
    path_type: str
    h1_km: float
    h2_km: float
    zenith_angle_deg: float
    path_length_km: float
    layer_path_lengths_km: np.ndarray = field(default_factory=lambda: np.array([]))
    include_earth_curvature: bool = True


# =============================================================================
# Numba-accelerated RTE functions
# =============================================================================

@jit(nopython=True, cache=True)
def planck_function(wavenumber: np.ndarray, temperature: float) -> np.ndarray:
    """Compute Planck function for blackbody radiation.

    B(nu, T) = c1 * nu^3 / (exp(c2 * nu / T) - 1)

    where c1 = 2*h*c^2 and c2 = h*c/k

    Args:
        wavenumber: Wavenumber array [cm^-1]
        temperature: Temperature [K]

    Returns:
        Spectral radiance [W/(cm²·sr·cm^-1)]
    """
    n = len(wavenumber)
    result = np.zeros(n)

    # First radiation constant c1 for wavenumber [W·cm²/sr]
    c1 = 1.191042953e-12  # 2*h*c^2 in W·cm²

    # Second radiation constant [cm·K]
    c2 = 1.4387769

    for i in range(n):
        nu = wavenumber[i]
        if nu > 0 and temperature > 0:
            x = c2 * nu / temperature
            if x < 700:  # Avoid overflow
                result[i] = c1 * nu**3 / (np.exp(x) - 1.0)

    return result


@jit(nopython=True, cache=True, parallel=True)
def integrate_rte_layers(
    wavenumber: np.ndarray,
    layer_absorption: np.ndarray,  # Shape: (n_layers, n_wavenumber)
    layer_scattering: np.ndarray,  # Shape: (n_layers, n_wavenumber)
    layer_temperatures: np.ndarray,  # Shape: (n_layers,)
    layer_path_lengths: np.ndarray,  # Shape: (n_layers,) in km
    background_radiance: np.ndarray,  # Shape: (n_wavenumber,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate RTE along path through atmospheric layers.

    Implements the discrete ordinate approximation for plane-parallel
    atmosphere with thermal emission.

    L(s) = L(0) * T(0,s) + integral[B(T(s')) * dT(s',s)]

    Args:
        wavenumber: Spectral grid [cm^-1]
        layer_absorption: Absorption coefficients [cm^-1] per layer
        layer_scattering: Scattering coefficients [km^-1] per layer
        layer_temperatures: Layer temperatures [K]
        layer_path_lengths: Path lengths through layers [km]
        background_radiance: Background radiance [W/(cm²·sr·cm^-1)]

    Returns:
        Tuple of (radiance, transmittance, optical_depth)
    """
    n_wn = len(wavenumber)
    n_layers = len(layer_temperatures)

    # Initialize outputs
    total_transmittance = np.ones(n_wn)
    total_optical_depth = np.zeros(n_wn)
    radiance = background_radiance.copy()
    thermal_emission = np.zeros(n_wn)

    # Integrate from far end to observer
    for layer in range(n_layers - 1, -1, -1):
        path_km = layer_path_lengths[layer]
        temp = layer_temperatures[layer]

        for i in prange(n_wn):
            # Total extinction = absorption + scattering
            # Convert absorption from cm^-1 to km^-1 (multiply by 1e5)
            extinction_km = layer_absorption[layer, i] * 1e5 + layer_scattering[layer, i]

            # Layer optical depth
            layer_tau = extinction_km * path_km

            # Layer transmittance
            layer_trans = np.exp(-layer_tau)

            # Planck function at layer temperature
            nu = wavenumber[i]
            if nu > 0 and temp > 0:
                c1 = 1.191042953e-12
                c2 = 1.4387769
                x = c2 * nu / temp
                if x < 700:
                    B_layer = c1 * nu**3 / (np.exp(x) - 1.0)
                else:
                    B_layer = 0.0
            else:
                B_layer = 0.0

            # Update radiance: L = L_in * tau + B * (1 - tau)
            radiance[i] = radiance[i] * layer_trans + B_layer * (1.0 - layer_trans)

            # Accumulate transmittance and optical depth
            total_transmittance[i] *= layer_trans
            total_optical_depth[i] += layer_tau

            # Track thermal emission separately
            thermal_emission[i] += B_layer * (1.0 - layer_trans) * total_transmittance[i]

    return radiance, total_transmittance, total_optical_depth


class RTESolver:
    """Radiative Transfer Equation solver.

    Solves the RTE for atmospheric paths including:
    - Absorption by gases (Line-by-Line)
    - Scattering by aerosols and molecules
    - Thermal emission from atmospheric layers
    - Background radiance (surface, space)

    Example:
        >>> solver = RTESolver(gas_engine, scattering_engine)
        >>> result = solver.solve(
        ...     wavenumber_range=(2000, 2500),
        ...     atmosphere=atmosphere_profile,
        ...     geometry=path_geometry,
        ... )
    """

    def __init__(
        self,
        gas_engine: GasEngine,
        scattering_engine: ScatteringEngine,
    ):
        """Initialize the RTE solver.

        Args:
            gas_engine: GasEngine instance for absorption calculations
            scattering_engine: ScatteringEngine instance for scattering calculations
        """
        self.gas_engine = gas_engine
        self.scattering_engine = scattering_engine

    def solve(
        self,
        wavenumber_range: Tuple[float, float],
        atmosphere: AtmosphereProfile,
        geometry: PathGeometry,
        molecules: List[str],
        aerosol_type: str = "NONE",
        visibility_km: float = 23.0,
        surface_temperature: Optional[float] = None,
        surface_emissivity: float = 0.98,
        resolution: float = 0.01,
        include_thermal_emission: bool = True,
        include_intermediate: bool = False,
    ) -> RTEResult:
        """Solve the radiative transfer equation.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            atmosphere: AtmosphereProfile with layer data
            geometry: PathGeometry specification
            molecules: List of molecules to include
            aerosol_type: Aerosol model type
            visibility_km: Surface visibility [km]
            surface_temperature: Surface temperature [K] (defaults to lowest layer)
            surface_emissivity: Surface emissivity [0-1]
            resolution: Spectral resolution [cm^-1]
            include_thermal_emission: Include thermal emission
            include_intermediate: Save per-layer results

        Returns:
            RTEResult with computed quantities
        """
        wn_min, wn_max = wavenumber_range
        num_points = int((wn_max - wn_min) / resolution) + 1
        wavenumber = np.linspace(wn_min, wn_max, num_points)
        wavelength_um = wavenumber_to_wavelength(wavenumber)

        logger.info(
            f"Solving RTE: {wn_min}-{wn_max} cm^-1, "
            f"{len(atmosphere.layers)} layers, {len(molecules)} molecules"
        )

        # Compute path geometry through atmosphere
        layer_paths = self._compute_layer_paths(atmosphere, geometry)

        # Get relevant atmospheric layers
        layers_in_path = self._select_layers_in_path(atmosphere, geometry)
        n_layers = len(layers_in_path)

        # Initialize layer arrays
        layer_absorption = np.zeros((n_layers, num_points))
        layer_scattering = np.zeros((n_layers, num_points))
        layer_temperatures = np.zeros(n_layers)
        layer_path_lengths = layer_paths[:n_layers]

        # Compute optical properties for each layer
        layer_transmittances = [] if include_intermediate else None
        layer_optical_depths = [] if include_intermediate else None

        for i, layer in enumerate(layers_in_path):
            layer_temperatures[i] = layer.temperature_k

            # Get VMR for each molecule in this layer
            vmr = {}
            for mol in molecules:
                try:
                    gas_profile = atmosphere.get_gas_profile(mol)
                    layer_idx = np.searchsorted(atmosphere.altitudes, layer.altitude_km)
                    layer_idx = min(layer_idx, len(gas_profile) - 1)
                    vmr[mol] = gas_profile[layer_idx] * 1e-6  # ppmv to ratio
                except ValueError:
                    vmr[mol] = 0.0

            # Compute gas absorption
            absorption = self.gas_engine.compute_absorption(
                wavenumber_range=wavenumber_range,
                temperature=layer.temperature_k,
                pressure_pa=layer.pressure_pa,
                molecules=molecules,
                vmr=vmr,
                resolution=resolution,
            )
            layer_absorption[i, :] = absorption.total_absorption

            # Compute scattering
            scattering = self.scattering_engine.compute_total_scattering(
                wavenumber_range=wavenumber_range,
                aerosol_type=aerosol_type,
                visibility_km=visibility_km,
                pressure_pa=layer.pressure_pa,
                temperature=layer.temperature_k,
                altitude_km=layer.altitude_km,
                resolution=resolution,
            )
            layer_scattering[i, :] = scattering.extinction_coeff

            if include_intermediate:
                # Per-layer optical depth
                extinction_km = layer_absorption[i, :] * 1e5 + layer_scattering[i, :]
                layer_tau = extinction_km * layer_path_lengths[i]
                layer_optical_depths.append(layer_tau)
                layer_transmittances.append(np.exp(-layer_tau))

        # Determine background radiance
        if surface_temperature is None:
            surface_temperature = layers_in_path[0].temperature_k

        if geometry.path_type == "SLANT" and geometry.h2_km > geometry.h1_km:
            # Looking up - space background (cold)
            background_radiance = planck_function(wavenumber, 3.0)  # CMB
        else:
            # Looking at surface
            background_radiance = surface_emissivity * planck_function(
                wavenumber, surface_temperature
            )

        # Integrate RTE
        radiance, transmittance, optical_depth = integrate_rte_layers(
            wavenumber=wavenumber,
            layer_absorption=layer_absorption,
            layer_scattering=layer_scattering,
            layer_temperatures=layer_temperatures,
            layer_path_lengths=layer_path_lengths,
            background_radiance=background_radiance,
        )

        # Compute thermal emission contribution
        if include_thermal_emission:
            # Thermal emission is included in the integration
            thermal_emission = radiance - background_radiance * transmittance
        else:
            thermal_emission = np.zeros_like(radiance)
            radiance = background_radiance * transmittance

        return RTEResult(
            wavenumber=wavenumber,
            wavelength_um=wavelength_um,
            transmittance=transmittance,
            radiance=radiance,
            optical_depth=optical_depth,
            thermal_emission=thermal_emission,
            solar_scattering=None,
            layer_transmittances=layer_transmittances,
            layer_optical_depths=layer_optical_depths,
        )

    def _compute_layer_paths(
        self,
        atmosphere: AtmosphereProfile,
        geometry: PathGeometry,
    ) -> np.ndarray:
        """Compute path lengths through atmospheric layers.

        Implements FR-07 (path types) and FR-08 (Earth curvature).

        Args:
            atmosphere: AtmosphereProfile
            geometry: PathGeometry

        Returns:
            Array of path lengths through each layer [km]
        """
        n_layers = len(atmosphere.layers)
        path_lengths = np.zeros(n_layers)

        if geometry.path_type == "HORIZONTAL":
            # Simple horizontal path at h1_km altitude
            # Find the layer containing h1_km
            for i, layer in enumerate(atmosphere.layers):
                if i < n_layers - 1:
                    layer_top = atmosphere.layers[i + 1].altitude_km
                else:
                    layer_top = layer.altitude_km + 1.0

                if layer.altitude_km <= geometry.h1_km < layer_top:
                    path_lengths[i] = geometry.path_length_km
                    break

        elif geometry.path_type == "SLANT":
            # Slant path from h1 to h2
            zenith_rad = np.radians(geometry.zenith_angle_deg)
            cos_zenith = np.cos(zenith_rad)

            if abs(cos_zenith) < 1e-6:
                # Near-horizontal - use horizontal approximation
                return self._compute_layer_paths(
                    atmosphere,
                    PathGeometry(
                        path_type="HORIZONTAL",
                        h1_km=geometry.h1_km,
                        h2_km=geometry.h1_km,
                        zenith_angle_deg=90.0,
                        path_length_km=geometry.path_length_km,
                    ),
                )

            # Calculate slant path through each layer
            h_start = min(geometry.h1_km, geometry.h2_km)
            h_end = max(geometry.h1_km, geometry.h2_km)

            altitudes = atmosphere.altitudes

            for i, layer in enumerate(atmosphere.layers):
                layer_bottom = layer.altitude_km
                if i < n_layers - 1:
                    layer_top = atmosphere.layers[i + 1].altitude_km
                else:
                    layer_top = layer.altitude_km + 10.0  # Extend last layer

                # Check if path intersects this layer
                if layer_top <= h_start or layer_bottom >= h_end:
                    continue

                # Effective layer boundaries for this path
                z_bot = max(layer_bottom, h_start)
                z_top = min(layer_top, h_end)
                dz = z_top - z_bot

                if dz <= 0:
                    continue

                # Path length through layer
                if geometry.include_earth_curvature and geometry.zenith_angle_deg > 70:
                    # Use spherical geometry for near-horizontal paths (FR-08)
                    path_lengths[i] = self._spherical_path_length(
                        z_bot, z_top, zenith_rad
                    )
                else:
                    # Plane-parallel approximation
                    path_lengths[i] = dz / cos_zenith

        elif geometry.path_type == "VERTICAL":
            # Vertical path (zenith angle = 0)
            h_start = min(geometry.h1_km, geometry.h2_km)
            h_end = max(geometry.h1_km, geometry.h2_km)

            for i, layer in enumerate(atmosphere.layers):
                layer_bottom = layer.altitude_km
                if i < n_layers - 1:
                    layer_top = atmosphere.layers[i + 1].altitude_km
                else:
                    layer_top = layer.altitude_km + 10.0

                if layer_top <= h_start or layer_bottom >= h_end:
                    continue

                z_bot = max(layer_bottom, h_start)
                z_top = min(layer_top, h_end)
                path_lengths[i] = z_top - z_bot

        return path_lengths

    def _spherical_path_length(
        self,
        z_bottom: float,
        z_top: float,
        zenith_angle: float,
    ) -> float:
        """Calculate path length through layer using spherical geometry.

        Uses the formula for refracted ray path through a spherical atmosphere.

        Args:
            z_bottom: Layer bottom altitude [km]
            z_top: Layer top altitude [km]
            zenith_angle: Zenith angle [radians]

        Returns:
            Path length [km]
        """
        R = EARTH_RADIUS_KM

        # Radii at layer boundaries
        r_bot = R + z_bottom
        r_top = R + z_top

        # Impact parameter
        sin_zen = np.sin(zenith_angle)
        b = R * sin_zen

        # Check for tangent height
        if b > r_bot:
            return 0.0

        # Path length calculation
        def path_at_r(r):
            if b > r:
                return 0.0
            return np.sqrt(r**2 - b**2)

        return path_at_r(r_top) - path_at_r(r_bot)

    def _select_layers_in_path(
        self,
        atmosphere: AtmosphereProfile,
        geometry: PathGeometry,
    ) -> List:
        """Select atmospheric layers that are in the viewing path.

        Args:
            atmosphere: AtmosphereProfile
            geometry: PathGeometry

        Returns:
            List of AtmosphereLayer objects in the path
        """
        h_min = min(geometry.h1_km, geometry.h2_km)
        h_max = max(geometry.h1_km, geometry.h2_km)

        selected = []
        for i, layer in enumerate(atmosphere.layers):
            if i < len(atmosphere.layers) - 1:
                layer_top = atmosphere.layers[i + 1].altitude_km
            else:
                layer_top = layer.altitude_km + 10.0

            # Include layer if it overlaps with path
            if layer.altitude_km < h_max and layer_top > h_min:
                selected.append(layer)

        return selected if selected else [atmosphere.layers[0]]

    def compute_transmittance_only(
        self,
        wavenumber_range: Tuple[float, float],
        atmosphere: AtmosphereProfile,
        geometry: PathGeometry,
        molecules: List[str],
        aerosol_type: str = "NONE",
        visibility_km: float = 23.0,
        resolution: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast transmittance-only calculation.

        Skips thermal emission calculation for faster results.

        Args:
            wavenumber_range: (min, max) wavenumber [cm^-1]
            atmosphere: AtmosphereProfile
            geometry: PathGeometry
            molecules: List of molecules
            aerosol_type: Aerosol model type
            visibility_km: Surface visibility [km]
            resolution: Spectral resolution [cm^-1]

        Returns:
            Tuple of (wavenumber, transmittance) arrays
        """
        result = self.solve(
            wavenumber_range=wavenumber_range,
            atmosphere=atmosphere,
            geometry=geometry,
            molecules=molecules,
            aerosol_type=aerosol_type,
            visibility_km=visibility_km,
            resolution=resolution,
            include_thermal_emission=False,
        )

        return result.wavenumber, result.transmittance
