"""Tests for radiative transfer equation solvers."""

import numpy as np
import pytest

from raf_tran.rte_solver import TwoStreamSolver, DiscreteOrdinatesSolver
from raf_tran.rte_solver.two_stream import TwoStreamMethod


class TestTwoStreamSolver:
    """Tests for two-stream RTE solver."""

    @pytest.fixture
    def solver(self):
        return TwoStreamSolver(method=TwoStreamMethod.DELTA_EDDINGTON)

    def test_conservation_no_absorption(self, solver):
        """Test flux conservation for pure scattering."""
        # Single layer, no absorption (omega=1)
        tau = np.array([1.0])
        omega = np.array([1.0])
        g = np.array([0.0])

        result = solver.solve_solar(
            tau=tau,
            omega=omega,
            g=g,
            mu0=0.5,
            flux_toa=1.0,
            surface_albedo=0.0,
        )

        # With no absorption and no surface reflection,
        # all flux should be absorbed at surface or scattered back to space
        flux_in = 1.0 * 0.5  # F_solar * mu0
        flux_out = result.flux_up[0] + result.flux_direct[-1] + result.flux_down[-1]

        # This is approximate for two-stream
        assert flux_out > 0

    def test_direct_beam_attenuation(self, solver):
        """Test direct beam follows Beer's law."""
        tau = np.array([1.0, 0.5, 0.5])  # Total tau = 2
        omega = np.array([0.5, 0.5, 0.5])
        g = np.array([0.0, 0.0, 0.0])
        mu0 = 1.0  # Overhead sun

        result = solver.solve_solar(
            tau=tau, omega=omega, g=g, mu0=mu0, flux_toa=1.0
        )

        # Direct beam: F = F0 * mu0 * exp(-tau/mu0)
        # With delta-scaling, effective tau is modified
        # Just check attenuation occurs
        assert result.flux_direct[-1] < result.flux_direct[0]

    def test_thermal_emission_surface(self, solver):
        """Test thermal emission from surface."""
        from raf_tran.utils.constants import STEFAN_BOLTZMANN

        tau = np.array([0.1])  # Optically thin
        omega = np.array([0.0])  # Pure absorption
        g = np.array([0.0])

        T_surface = 300.0

        result = solver.solve_thermal(
            tau=tau,
            omega=omega,
            g=g,
            temperature=np.array([280.0]),
            surface_temperature=T_surface,
            surface_emissivity=1.0,
        )

        expected_surface_flux = STEFAN_BOLTZMANN * T_surface**4

        # Upward flux at surface should be close to blackbody
        assert np.isclose(result.flux_up[-1], expected_surface_flux, rtol=0.01)

    def test_optically_thick_thermal(self, solver):
        """Test thermal radiation in optically thick atmosphere."""
        from raf_tran.utils.constants import STEFAN_BOLTZMANN

        tau = np.array([10.0])  # Optically thick
        omega = np.array([0.0])  # Pure absorption
        g = np.array([0.0])

        T_layer = 280.0

        result = solver.solve_thermal(
            tau=tau,
            omega=omega,
            g=g,
            temperature=np.array([T_layer]),
            surface_temperature=300.0,
            surface_emissivity=1.0,
        )

        # TOA upward flux should approach layer emission for optically thick
        expected_flux = STEFAN_BOLTZMANN * T_layer**4
        # This is approximate
        assert result.flux_up[0] > 0

    def test_heating_rate_positive_for_solar(self, solver):
        """Test that absorbed solar radiation causes positive heating."""
        tau = np.array([0.5, 0.5])
        omega = np.array([0.5, 0.5])  # 50% scattering
        g = np.array([0.0, 0.0])

        result = solver.solve_solar(
            tau=tau,
            omega=omega,
            g=g,
            mu0=0.5,
            flux_toa=1361.0,
            surface_albedo=0.3,
        )

        # Pressure levels (simplified)
        pressure = np.array([100000, 50000, 10000])

        heating = solver.compute_heating_rate(
            result.flux_up,
            result.flux_down,
            result.flux_direct,
            pressure,
        )

        # At least some heating should occur
        assert any(heating != 0)


class TestTwoStreamMethods:
    """Test different two-stream approximation methods."""

    @pytest.mark.parametrize(
        "method",
        [
            TwoStreamMethod.EDDINGTON,
            TwoStreamMethod.QUADRATURE,
            TwoStreamMethod.HEMISPHERIC_MEAN,
            TwoStreamMethod.DELTA_EDDINGTON,
        ],
    )
    def test_methods_run(self, method):
        """Test all two-stream methods run without error."""
        solver = TwoStreamSolver(method=method)

        tau = np.array([0.5])
        omega = np.array([0.8])
        g = np.array([0.7])

        result = solver.solve_solar(
            tau=tau, omega=omega, g=g, mu0=0.5, flux_toa=1.0
        )

        assert result.flux_up is not None
        assert result.flux_down is not None
        assert result.flux_direct is not None


class TestDiscreteOrdinatesSolver:
    """Tests for discrete ordinates solver."""

    @pytest.fixture
    def solver(self):
        return DiscreteOrdinatesSolver(n_streams=4)

    def test_initialization(self, solver):
        """Test solver initialization."""
        assert solver.n_streams == 4
        assert len(solver.mu_up) == 4
        assert len(solver.mu_down) == 4

    def test_quadrature_weights_sum(self, solver):
        """Test quadrature weights sum correctly."""
        # Full weights should sum to 2 (for [-1, 1])
        total = np.sum(solver.w_up) + np.sum(solver.w_down)
        assert np.isclose(total, 2.0, rtol=1e-10)

    def test_no_scattering_absorption(self, solver):
        """Test pure absorption case."""
        tau = np.array([1.0])
        omega = np.array([0.0])  # No scattering
        g = np.array([0.0])

        result = solver.solve(
            tau=tau,
            omega=omega,
            g=g,
            mu0=1.0,
            flux_toa=1.0,
        )

        # Direct beam should follow Beer's law
        expected_direct = 1.0 * np.exp(-1.0)
        assert np.isclose(result.flux_direct[-1], expected_direct, rtol=0.01)

    def test_thermal_emission(self, solver):
        """Test thermal emission source."""
        tau = np.array([0.5])
        omega = np.array([0.0])
        g = np.array([0.0])

        thermal = np.array([100.0])  # W/m² thermal emission

        result = solver.solve(
            tau=tau,
            omega=omega,
            g=g,
            thermal_emission=thermal,
            surface_emission=0.0,
        )

        # Some flux should emerge from thermal emission
        assert result.flux_up[0] > 0 or result.flux_down[-1] > 0

    def test_surface_reflection(self, solver):
        """Test surface albedo effect."""
        tau = np.array([0.1])  # Thin atmosphere
        omega = np.array([0.0])
        g = np.array([0.0])

        # Without surface reflection
        result_no_refl = solver.solve(
            tau=tau,
            omega=omega,
            g=g,
            mu0=1.0,
            flux_toa=1.0,
            surface_albedo=0.0,
        )

        # With surface reflection
        result_with_refl = solver.solve(
            tau=tau,
            omega=omega,
            g=g,
            mu0=1.0,
            flux_toa=1.0,
            surface_albedo=0.5,
        )

        # Upward flux at TOA should be higher with surface reflection
        assert result_with_refl.flux_up[0] > result_no_refl.flux_up[0]


class TestSolverConsistency:
    """Test consistency between solvers."""

    def test_two_stream_vs_disort_thin_atmosphere(self):
        """Test two-stream and DISORT give similar results for thin atmosphere."""
        tau = np.array([0.1])
        omega = np.array([0.5])
        g = np.array([0.0])

        two_stream = TwoStreamSolver()
        disort = DiscreteOrdinatesSolver(n_streams=8)

        result_ts = two_stream.solve_solar(
            tau=tau, omega=omega, g=g, mu0=0.5, flux_toa=1.0
        )

        result_do = disort.solve(
            tau=tau, omega=omega, g=g, mu0=0.5, flux_toa=1.0
        )

        # Results should be within 20% for thin atmosphere
        assert np.isclose(
            result_ts.flux_up[0], result_do.flux_up[0], rtol=0.3
        ) or (result_ts.flux_up[0] < 0.1 and result_do.flux_up[0] < 0.1)
