"""Tests for gas optics module."""

import numpy as np
import pytest

from raf_tran.gas_optics import CKDTable, GasOptics, compute_optical_depth
from raf_tran.gas_optics.ckd import create_simple_ckd_table, generate_gauss_legendre


class TestGaussLegendre:
    """Tests for Gauss-Legendre quadrature."""

    @pytest.mark.parametrize("n_points", [4, 8, 16])
    def test_weights_sum_to_one(self, n_points):
        """Test quadrature weights sum to 1 for [0,1] interval."""
        points, weights = generate_gauss_legendre(n_points)
        assert np.isclose(np.sum(weights), 1.0, rtol=1e-10)

    @pytest.mark.parametrize("n_points", [4, 8, 16])
    def test_points_in_unit_interval(self, n_points):
        """Test all quadrature points are in [0,1]."""
        points, weights = generate_gauss_legendre(n_points)
        assert all(0 <= p <= 1 for p in points)

    def test_quadrature_accuracy(self):
        """Test quadrature integrates polynomials exactly."""
        # Gauss-Legendre with n points integrates polynomials up to degree 2n-1 exactly
        n = 8
        points, weights = generate_gauss_legendre(n)

        # Test integration of x² over [0,1]: ∫x²dx = 1/3
        integral = np.sum(weights * points**2)
        assert np.isclose(integral, 1/3, rtol=1e-10)

        # Test integration of x⁴ over [0,1]: ∫x⁴dx = 1/5
        integral = np.sum(weights * points**4)
        assert np.isclose(integral, 1/5, rtol=1e-10)


class TestCKDTable:
    """Tests for CKD lookup table."""

    @pytest.fixture
    def ckd_table(self):
        return create_simple_ckd_table(
            gas_name="H2O",
            wavenumber_bounds=(500, 600),
            n_g_points=8,
            reference_k=1e-24,
        )

    def test_table_creation(self, ckd_table):
        """Test CKD table is created correctly."""
        assert ckd_table.gas_name == "H2O"
        assert ckd_table.wavenumber_bounds == (500, 600)
        assert ckd_table.n_g_points == 8
        assert len(ckd_table.g_points) == 8
        assert len(ckd_table.g_weights) == 8

    def test_k_coefficients_shape(self, ckd_table):
        """Test k-coefficient array has correct shape."""
        n_p = len(ckd_table.pressures)
        n_t = len(ckd_table.temperatures)
        n_g = ckd_table.n_g_points

        assert ckd_table.k_coefficients.shape == (n_p, n_t, n_g)

    def test_interpolation_at_reference_point(self, ckd_table):
        """Test interpolation returns exact values at reference points."""
        # Test at a reference pressure and temperature
        p_ref = ckd_table.pressures[0]
        T_ref = ckd_table.temperatures[0]

        k_interp = ckd_table.interpolate_k(
            np.array([p_ref]), np.array([T_ref])
        )

        expected = ckd_table.k_coefficients[0, 0, :]
        assert np.allclose(k_interp[0], expected, rtol=1e-6)

    def test_interpolation_between_points(self, ckd_table):
        """Test interpolation between reference points."""
        # Pressure between first two reference values
        p_mid = np.sqrt(ckd_table.pressures[0] * ckd_table.pressures[1])
        T_mid = (ckd_table.temperatures[0] + ckd_table.temperatures[1]) / 2

        k_interp = ckd_table.interpolate_k(
            np.array([p_mid]), np.array([T_mid])
        )

        # Result should be between the corner values
        k00 = ckd_table.k_coefficients[0, 0, :]
        k11 = ckd_table.k_coefficients[1, 1, :]

        assert all(
            (min(k00[i], k11[i]) <= k_interp[0, i] <= max(k00[i], k11[i]))
            or np.isclose(k_interp[0, i], k00[i], rtol=0.5)
            or np.isclose(k_interp[0, i], k11[i], rtol=0.5)
            for i in range(len(k00))
        )


class TestGasOptics:
    """Tests for GasOptics calculator."""

    @pytest.fixture
    def gas_optics(self):
        go = GasOptics()
        go.add_gas(create_simple_ckd_table("H2O", (500, 600), 8, 1e-24))
        go.add_gas(create_simple_ckd_table("CO2", (500, 600), 8, 1e-25))
        return go

    def test_add_gas(self, gas_optics):
        """Test adding gas species."""
        assert "H2O" in gas_optics.ckd_tables
        assert "CO2" in gas_optics.ckd_tables

    def test_optical_depth_calculation(self, gas_optics):
        """Test optical depth computation."""
        n_layers = 5
        pressure = np.linspace(1e5, 1e4, n_layers)
        temperature = np.linspace(300, 250, n_layers)
        dz = np.full(n_layers, 2000.0)  # 2 km layers
        number_density = pressure / (1.38e-23 * temperature)  # Ideal gas

        vmr = {
            "H2O": np.full(n_layers, 0.01),  # 1%
            "CO2": np.full(n_layers, 400e-6),  # 400 ppm
        }

        tau, g_weights = gas_optics.compute_optical_depth(
            pressure, temperature, vmr, dz, number_density
        )

        assert tau.shape == (n_layers, 8)
        assert len(g_weights) == 8
        assert all(tau.ravel() >= 0)

    def test_missing_gas_ignored(self, gas_optics):
        """Test that missing gases are silently ignored."""
        n_layers = 3
        pressure = np.array([1e5, 5e4, 2e4])
        temperature = np.array([300, 280, 260])
        dz = np.full(n_layers, 2000.0)
        number_density = pressure / (1.38e-23 * temperature)

        vmr = {"H2O": np.full(n_layers, 0.01)}  # Only H2O, no CO2

        # Should not raise an error
        tau, g_weights = gas_optics.compute_optical_depth(
            pressure, temperature, vmr, dz, number_density
        )

        assert tau.shape == (n_layers, 8)


class TestComputeOpticalDepth:
    """Tests for optical depth function."""

    def test_optical_depth_basic(self):
        """Test basic optical depth calculation."""
        k = np.array([[1e-24, 2e-24], [1.5e-24, 3e-24]])  # (n_layers, n_g)
        column = np.array([1e20, 2e20])  # mol/m²

        tau = compute_optical_depth(k, column)

        expected = np.array([
            [1e-24 * 1e20, 2e-24 * 1e20],
            [1.5e-24 * 2e20, 3e-24 * 2e20],
        ])
        assert np.allclose(tau, expected)

    def test_optical_depth_zero_column(self):
        """Test optical depth is zero for zero column amount."""
        k = np.array([[1e-24, 2e-24]])
        column = np.array([0.0])

        tau = compute_optical_depth(k, column)

        assert np.allclose(tau, 0)
