"""
Tests for the detection module (detectors, targets, detection).
"""

import pytest
import numpy as np

from raf_tran.detectors import FPADetector, InSbDetector, MCTDetector, detector_from_type
from raf_tran.targets import AircraftSignature, generic_fighter, generic_transport, generic_uav
from raf_tran.detection import (
    DetectionResult,
    calculate_detection_range,
    calculate_snr_vs_range,
    atmospheric_transmission_ir,
)


# =============================================================================
# FPA Detector Tests
# =============================================================================

class TestFPADetector:
    """Tests for FPADetector class."""

    def test_detector_creation(self):
        """Test basic detector creation."""
        detector = FPADetector(
            name="Test Detector",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=15.0,
        )
        assert detector.name == "Test Detector"
        assert detector.spectral_band == (3.0, 5.0)
        assert detector.d_star == 1e11

    def test_pixel_area(self):
        """Test pixel area calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=20.0,  # 20 um
        )
        # 20 um = 20e-4 cm, area = (20e-4)^2 = 4e-6 cm^2
        expected_area = (20e-4) ** 2
        assert np.isclose(detector.pixel_area, expected_area, rtol=1e-10)

    def test_bandwidth(self):
        """Test spectral bandwidth calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=15.0,
        )
        assert detector.bandwidth == 2.0

    def test_center_wavelength(self):
        """Test center wavelength calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(8.0, 12.0),
            d_star=1e10,
            pixel_pitch=20.0,
        )
        assert detector.center_wavelength == 10.0

    def test_electrical_bandwidth(self):
        """Test electrical bandwidth calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=15.0,
            integration_time=10.0,  # ms
        )
        # Bandwidth = 1 / (2 * 10ms) = 50 Hz
        assert detector.electrical_bandwidth == 50.0

    def test_noise_equivalent_irradiance(self):
        """Test NEI calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=15.0,
            integration_time=10.0,
        )
        nei = detector.noise_equivalent_irradiance()
        assert nei > 0
        # Higher D* should give lower NEI
        detector2 = FPADetector(
            name="Test2",
            spectral_band=(3.0, 5.0),
            d_star=2e11,  # 2x D*
            pixel_pitch=15.0,
            integration_time=10.0,
        )
        nei2 = detector2.noise_equivalent_irradiance()
        assert nei2 < nei  # Should be lower

    def test_signal_to_noise(self):
        """Test SNR calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=15.0,
        )
        nei = detector.noise_equivalent_irradiance()

        # SNR should be 1 at NEI
        snr = detector.signal_to_noise(nei)
        assert np.isclose(snr, 1.0, rtol=1e-10)

        # SNR should be 10 at 10*NEI
        snr10 = detector.signal_to_noise(10 * nei)
        assert np.isclose(snr10, 10.0, rtol=1e-10)

    def test_ifov(self):
        """Test IFOV calculation."""
        detector = FPADetector(
            name="Test",
            spectral_band=(3.0, 5.0),
            d_star=1e11,
            pixel_pitch=15.0,  # um
            focal_length=100.0,  # mm
        )
        # IFOV = 15 um / 100 mm = 0.15 mrad
        assert np.isclose(detector.ifov, 0.15, rtol=1e-10)


class TestDetectorFactories:
    """Tests for detector factory functions."""

    def test_insb_detector(self):
        """Test InSb detector creation."""
        detector = InSbDetector()
        assert detector.spectral_band == (3.0, 5.0)
        assert detector.d_star == 1e11
        assert detector.operating_temp == 77.0

    def test_mct_detector_lwir(self):
        """Test MCT LWIR detector creation."""
        detector = MCTDetector()
        assert detector.spectral_band == (8.0, 12.0)
        assert detector.operating_temp == 77.0

    def test_mct_detector_mwir(self):
        """Test MCT MWIR detector creation."""
        detector = MCTDetector(spectral_band=(3.0, 5.0))
        assert detector.spectral_band == (3.0, 5.0)

    def test_detector_from_type(self):
        """Test detector creation from type string."""
        insb = detector_from_type('insb')
        assert insb.spectral_band == (3.0, 5.0)

        mct_lwir = detector_from_type('mct_lwir')
        assert mct_lwir.spectral_band == (8.0, 12.0)

        mct_mwir = detector_from_type('mct_mwir')
        assert mct_mwir.spectral_band == (3.0, 5.0)

    def test_detector_from_type_invalid(self):
        """Test error for invalid detector type."""
        with pytest.raises(ValueError):
            detector_from_type('invalid_type')


# =============================================================================
# Aircraft Signature Tests
# =============================================================================

class TestAircraftSignature:
    """Tests for AircraftSignature class."""

    def test_signature_creation(self):
        """Test basic signature creation."""
        sig = AircraftSignature(
            name="Test Aircraft",
            exhaust_temp=700.0,
            exhaust_area=0.5,
        )
        assert sig.name == "Test Aircraft"
        assert sig.exhaust_temp == 700.0
        assert sig.exhaust_area == 0.5

    def test_radiant_intensity_positive(self):
        """Test that radiant intensity is positive."""
        sig = AircraftSignature(
            name="Test",
            exhaust_temp=700.0,
            exhaust_area=0.5,
            skin_temp=300.0,
            skin_area=20.0,
        )
        mwir = sig.radiant_intensity_mwir()
        lwir = sig.radiant_intensity_lwir()

        assert mwir > 0
        assert lwir > 0

    def test_hot_target_higher_mwir(self):
        """Test that hot targets have higher MWIR intensity."""
        # Hot target (afterburner)
        hot = AircraftSignature(
            name="Hot",
            exhaust_temp=1800.0,
            exhaust_area=1.5,
            skin_temp=300.0,
            skin_area=20.0,
        )
        # Cool target
        cool = AircraftSignature(
            name="Cool",
            exhaust_temp=500.0,
            exhaust_area=0.5,
            skin_temp=300.0,
            skin_area=20.0,
        )

        # Hot target should have much higher MWIR intensity
        assert hot.radiant_intensity_mwir() > cool.radiant_intensity_mwir()

    def test_irradiance_decreases_with_range(self):
        """Test inverse square law for irradiance."""
        sig = AircraftSignature(
            name="Test",
            exhaust_temp=700.0,
            exhaust_area=0.5,
        )

        irrad_1km = sig.irradiance_at_range(1000, 3.0, 5.0)
        irrad_2km = sig.irradiance_at_range(2000, 3.0, 5.0)

        # At 2x range, irradiance should be 1/4 (inverse square)
        assert np.isclose(irrad_2km / irrad_1km, 0.25, rtol=0.01)

    def test_aspect_modifies_signature(self):
        """Test that aspect affects visible areas."""
        base = AircraftSignature(
            name="Base",
            exhaust_area=1.0,
            nozzle_area=0.5,
            skin_area=20.0,
        )

        rear = AircraftSignature.with_aspect(base, "rear")
        front = AircraftSignature.with_aspect(base, "front")

        # Rear should have more exhaust visible
        assert rear.exhaust_area > front.exhaust_area
        # Front should have less skin visible
        assert front.skin_area < rear.skin_area


class TestAircraftFactories:
    """Tests for aircraft signature factory functions."""

    def test_generic_fighter(self):
        """Test generic fighter creation."""
        fighter = generic_fighter()
        assert "Fighter" in fighter.name
        assert fighter.exhaust_temp > 0
        assert fighter.exhaust_area > 0

    def test_generic_fighter_afterburner(self):
        """Test afterburner increases temperature."""
        mil_power = generic_fighter(afterburner=False)
        afterburner = generic_fighter(afterburner=True)

        assert afterburner.exhaust_temp > mil_power.exhaust_temp
        assert afterburner.exhaust_area > mil_power.exhaust_area

    def test_generic_transport(self):
        """Test generic transport creation."""
        transport = generic_transport()
        assert "Transport" in transport.name
        # Transport has lower exhaust temp than fighter
        fighter = generic_fighter()
        assert transport.exhaust_temp < fighter.exhaust_temp

    def test_generic_uav(self):
        """Test generic UAV creation."""
        uav = generic_uav()
        assert "UAV" in uav.name
        # UAV has smaller signature than fighter
        fighter = generic_fighter()
        assert uav.skin_area < fighter.skin_area


# =============================================================================
# Detection Range Tests
# =============================================================================

class TestAtmosphericTransmission:
    """Tests for atmospheric transmission function."""

    def test_transmission_bounds(self):
        """Test transmission is between 0 and 1."""
        for range_km in [1, 5, 10, 50]:
            trans = atmospheric_transmission_ir(range_km, 3.0, 5.0)
            assert 0 <= trans <= 1

    def test_transmission_decreases_with_range(self):
        """Test transmission decreases with range."""
        trans_1 = atmospheric_transmission_ir(1, 3.0, 5.0)
        trans_10 = atmospheric_transmission_ir(10, 3.0, 5.0)
        trans_50 = atmospheric_transmission_ir(50, 3.0, 5.0)

        assert trans_1 > trans_10 > trans_50

    def test_transmission_affected_by_humidity(self):
        """Test higher humidity reduces transmission."""
        trans_low = atmospheric_transmission_ir(10, 3.0, 5.0, humidity_percent=20)
        trans_high = atmospheric_transmission_ir(10, 3.0, 5.0, humidity_percent=80)

        assert trans_low > trans_high

    def test_transmission_affected_by_visibility(self):
        """Test lower visibility reduces transmission."""
        trans_clear = atmospheric_transmission_ir(10, 3.0, 5.0, visibility_km=50)
        trans_hazy = atmospheric_transmission_ir(10, 3.0, 5.0, visibility_km=5)

        assert trans_clear > trans_hazy


class TestDetectionRange:
    """Tests for detection range calculations."""

    def test_detection_range_positive(self):
        """Test detection range is positive for reasonable targets."""
        detector = InSbDetector()
        target = generic_fighter(afterburner=True)

        result = calculate_detection_range(detector, target)

        assert result.detection_range_m > 0
        assert result.detection_range_km > 0

    def test_higher_snr_shorter_range(self):
        """Test higher SNR threshold gives shorter range."""
        detector = InSbDetector()
        target = generic_fighter()

        result_5 = calculate_detection_range(detector, target, snr_threshold=5.0)
        result_10 = calculate_detection_range(detector, target, snr_threshold=10.0)

        assert result_5.detection_range_km > result_10.detection_range_km

    def test_hotter_target_longer_range(self):
        """Test hotter targets detected at longer range."""
        detector = InSbDetector()

        target_mil = generic_fighter(afterburner=False)
        target_ab = generic_fighter(afterburner=True)

        result_mil = calculate_detection_range(detector, target_mil)
        result_ab = calculate_detection_range(detector, target_ab)

        assert result_ab.detection_range_km > result_mil.detection_range_km

    def test_detection_result_properties(self):
        """Test DetectionResult has correct properties."""
        detector = InSbDetector()
        target = generic_fighter()

        result = calculate_detection_range(detector, target)

        assert hasattr(result, 'detection_range_m')
        assert hasattr(result, 'detection_range_km')
        assert hasattr(result, 'snr_at_range')
        assert hasattr(result, 'target_irradiance')
        assert hasattr(result, 'atmospheric_transmission')

        # km should be m/1000
        assert np.isclose(result.detection_range_km,
                         result.detection_range_m / 1000)


class TestSNRvsRange:
    """Tests for SNR vs range calculation."""

    def test_snr_vs_range_shapes(self):
        """Test output shapes match input."""
        detector = InSbDetector()
        target = generic_fighter()
        ranges = np.linspace(1, 50, 100)

        snr, irrad, trans = calculate_snr_vs_range(detector, target, ranges)

        assert snr.shape == ranges.shape
        assert irrad.shape == ranges.shape
        assert trans.shape == ranges.shape

    def test_snr_decreases_with_range(self):
        """Test SNR generally decreases with range."""
        detector = InSbDetector()
        target = generic_fighter()
        ranges = np.array([1, 5, 10, 20, 50])

        snr, _, _ = calculate_snr_vs_range(detector, target, ranges)

        # SNR should decrease monotonically
        for i in range(len(snr) - 1):
            assert snr[i] >= snr[i + 1]

    def test_snr_positive(self):
        """Test SNR is always positive."""
        detector = InSbDetector()
        target = generic_fighter()
        ranges = np.linspace(1, 100, 50)

        snr, _, _ = calculate_snr_vs_range(detector, target, ranges)

        assert np.all(snr > 0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete detection workflow."""

    def test_full_detection_workflow(self):
        """Test complete detection range calculation workflow."""
        # Create detector
        detector = InSbDetector(pixel_pitch=15.0, f_number=2.0)

        # Create target
        target = generic_fighter(aspect="rear", afterburner=False)

        # Calculate detection range
        result = calculate_detection_range(
            detector, target,
            snr_threshold=5.0,
            visibility_km=23.0,
            humidity_percent=50.0,
        )

        # Verify reasonable results
        assert 1 < result.detection_range_km < 100  # Reasonable range
        assert 0 < result.atmospheric_transmission < 1
        assert result.target_irradiance > 0

    def test_mwir_vs_lwir_comparison(self):
        """Test MWIR vs LWIR detector comparison."""
        insb = InSbDetector()  # MWIR
        mct = MCTDetector()    # LWIR

        # Hot target (should favor MWIR)
        hot_target = generic_fighter(afterburner=True)

        result_mwir = calculate_detection_range(insb, hot_target)
        result_lwir = calculate_detection_range(mct, hot_target)

        # Both should have positive range
        assert result_mwir.detection_range_km > 0
        assert result_lwir.detection_range_km > 0

    def test_different_aspects(self):
        """Test detection range varies with viewing aspect."""
        detector = InSbDetector()

        rear = generic_fighter(aspect="rear")
        side = generic_fighter(aspect="side")
        front = generic_fighter(aspect="front")

        result_rear = calculate_detection_range(detector, rear)
        result_side = calculate_detection_range(detector, side)
        result_front = calculate_detection_range(detector, front)

        # Rear aspect should have longest range (most exhaust visible)
        assert result_rear.detection_range_km > result_front.detection_range_km
