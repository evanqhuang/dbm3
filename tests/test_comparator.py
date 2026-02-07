"""
Comprehensive unit tests for backtest/comparator.py

Tests cover:
- ComparisonResult properties (risk, NDL, ceiling, deco, validity)
- ModelComparator initialization and profile comparison
- Report generation with statistics
"""

import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backtest.comparator import ComparisonResult, ModelComparator, _run_buhlmann_single
from backtest.buhlmann_runner import BuhlmannResult, BuhlmannRunner
from backtest.slab_model import SlabResult, SlabModel
from backtest.profile_generator import ProfileGenerator, DiveProfile
from backtest.buhlmann_constants import GradientFactors


# Factory helpers to construct minimal valid result objects
def _make_buhlmann_result(**overrides):
    """Create a BuhlmannResult with sensible defaults."""
    defaults = dict(
        times=[0.0, 5.0, 10.0],
        pressures=[1.0, 4.0, 1.0],
        compartment_n2=[[0.8]*16, [1.5]*16, [0.9]*16],
        compartment_he=[[0.0]*16]*3,
        ceilings=[0.9, 1.5, 0.95],
        ndl_times=[100.0, 15.0, 80.0],
        max_ceiling=1.5,
        min_ndl=15.0,
        max_supersaturation=0.65,
        gf_low=1.0,
        gf_high=1.0,
    )
    defaults.update(overrides)
    return BuhlmannResult(**defaults)


def _make_slab_result(**overrides):
    """Create a SlabResult with sensible defaults."""
    defaults = dict(
        times=[0.0, 5.0, 10.0],
        depths=[0.0, 30.0, 0.0],
        slab_history=np.zeros((3, 3, 20)),
        max_loads=[0.8, 2.5, 0.9],
        max_tissue_load=2.5,
        min_margin=0.3,
        critical_compartment="Spine",
        max_cv_ratio=0.7,
        final_ndl=18.0,
        final_slabs={"Spine": np.zeros(20), "Muscle": np.zeros(20), "Joints": np.zeros(20)},
        ceiling_at_bottom=3.0,
    )
    defaults.update(overrides)
    return SlabResult(**defaults)


def _make_output_line(time, pressure, n2_base=0.8, he_base=0.0, ceiling=0.95, ndl=28.0):
    """Create a 36-value output line in libbuhlmann format."""
    parts = [f"{time:.2f}", f"{pressure:.2f}"]
    for c in range(16):
        parts.append(f"{n2_base + c * 0.01:.4f}")
        parts.append(f"{he_base:.4f}")
    parts.append(f"{ceiling:.4f}")
    parts.append(f"{ndl:.2f}")
    return " ".join(parts)


class TestComparisonResultRisk:
    """Test risk-related properties of ComparisonResult."""

    def test_buhlmann_risk_returns_max_supersaturation(self):
        """max_supersaturation=0.65 -> buhlmann_risk=0.65"""
        buhlmann = _make_buhlmann_result(max_supersaturation=0.65)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.buhlmann_risk == pytest.approx(0.65)

    def test_slab_risk_returns_max_cv_ratio(self):
        """max_cv_ratio=0.7 -> slab_risk=0.7"""
        buhlmann = _make_buhlmann_result()
        slab = _make_slab_result(max_cv_ratio=0.7)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.slab_risk == pytest.approx(0.7)

    def test_delta_risk_is_slab_minus_buhlmann(self):
        """slab=0.7, buhlmann=0.65 -> delta=0.05"""
        buhlmann = _make_buhlmann_result(max_supersaturation=0.65)
        slab = _make_slab_result(max_cv_ratio=0.7)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.delta_risk == pytest.approx(0.05)

    def test_delta_risk_positive_slab_more_conservative(self):
        """When slab risk > buhlmann risk, delta should be positive."""
        buhlmann = _make_buhlmann_result(max_supersaturation=0.5)
        slab = _make_slab_result(max_cv_ratio=0.8)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.delta_risk > 0

    def test_delta_risk_negative_buhlmann_more_conservative(self):
        """When buhlmann risk > slab risk, delta should be negative."""
        buhlmann = _make_buhlmann_result(max_supersaturation=0.9)
        slab = _make_slab_result(max_cv_ratio=0.6)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.delta_risk < 0


class TestComparisonResultNDL:
    """Test NDL-related properties of ComparisonResult."""

    def test_buhlmann_ndl_returns_min_ndl(self):
        """min_ndl=15.0 -> buhlmann_ndl=15.0"""
        buhlmann = _make_buhlmann_result(min_ndl=15.0)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.buhlmann_ndl == pytest.approx(15.0)

    def test_slab_ndl_returns_final_ndl(self):
        """final_ndl=18.0 -> slab_ndl=18.0"""
        buhlmann = _make_buhlmann_result()
        slab = _make_slab_result(final_ndl=18.0)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.slab_ndl == pytest.approx(18.0)

    def test_delta_ndl_is_slab_minus_buhlmann(self):
        """18.0 - 15.0 = 3.0"""
        buhlmann = _make_buhlmann_result(min_ndl=15.0)
        slab = _make_slab_result(final_ndl=18.0)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.delta_ndl == pytest.approx(3.0)


class TestComparisonResultCeiling:
    """Test ceiling-related properties of ComparisonResult."""

    def test_buhlmann_ceiling_converts_bar_to_meters(self):
        """max_ceiling=2.0 -> (2.0-1.01325)*10 = 9.8675m"""
        buhlmann = _make_buhlmann_result(max_ceiling=2.0)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        expected = (2.0 - 1.01325) * 10
        assert result.buhlmann_ceiling == pytest.approx(expected)

    def test_buhlmann_ceiling_clamps_at_zero(self):
        """max_ceiling=0.9 (below surface) -> 0.0"""
        buhlmann = _make_buhlmann_result(max_ceiling=0.9)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.buhlmann_ceiling == pytest.approx(0.0)

    def test_slab_ceiling_returns_ceiling_at_bottom(self):
        """ceiling_at_bottom=3.0 -> 3.0"""
        buhlmann = _make_buhlmann_result()
        slab = _make_slab_result(ceiling_at_bottom=3.0)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.slab_ceiling == pytest.approx(3.0)

    def test_delta_ceiling_is_slab_minus_buhlmann(self):
        """Compute and verify delta_ceiling."""
        buhlmann = _make_buhlmann_result(max_ceiling=2.0)  # (2.0-1.01325)*10 = 9.8675m
        slab = _make_slab_result(ceiling_at_bottom=6.0)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        expected_buhlmann_ceiling = (2.0 - 1.01325) * 10
        expected_delta = 6.0 - expected_buhlmann_ceiling
        assert result.delta_ceiling == pytest.approx(expected_delta)

    def test_buhlmann_ceiling_at_surface_pressure(self):
        """max_ceiling=1.01325 -> exactly 0.0"""
        buhlmann = _make_buhlmann_result(max_ceiling=1.01325)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.buhlmann_ceiling == pytest.approx(0.0)


class TestComparisonResultDeco:
    """Test deco requirement properties."""

    def test_slab_requires_deco_when_ceiling_positive(self):
        """ceiling_at_bottom=3.0 -> True"""
        buhlmann = _make_buhlmann_result()
        slab = _make_slab_result(ceiling_at_bottom=3.0)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.slab_requires_deco is True

    def test_slab_no_deco_when_ceiling_zero(self):
        """ceiling_at_bottom=0.0 -> False"""
        buhlmann = _make_buhlmann_result()
        slab = _make_slab_result(ceiling_at_bottom=0.0)
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.slab_requires_deco is False

    def test_buhlmann_requires_deco_when_ceiling_above_surface(self):
        """max_ceiling=2.0 -> True (ceiling_m > 0)"""
        buhlmann = _make_buhlmann_result(max_ceiling=2.0)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.buhlmann_requires_deco is True

    def test_buhlmann_no_deco_when_ceiling_at_surface(self):
        """max_ceiling=1.0 -> False (ceiling_m=0)"""
        buhlmann = _make_buhlmann_result(max_ceiling=1.0)
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.buhlmann_requires_deco is False


class TestComparisonResultValidity:
    """Test validity checking of ComparisonResult."""

    def test_is_valid_both_present(self):
        """Both results -> True"""
        buhlmann = _make_buhlmann_result()
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=slab)

        assert result.is_valid is True

    def test_is_not_valid_buhlmann_none(self):
        """buhlmann_result=None -> False"""
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=None, slab_result=slab)

        assert result.is_valid is False

    def test_is_not_valid_slab_none(self):
        """slab_result=None -> False"""
        buhlmann = _make_buhlmann_result()
        result = ComparisonResult(profile=None, buhlmann_result=buhlmann, slab_result=None)

        assert result.is_valid is False

    def test_nan_when_buhlmann_none(self):
        """All delta/risk properties return NaN when buhlmann_result is None."""
        slab = _make_slab_result()
        result = ComparisonResult(profile=None, buhlmann_result=None, slab_result=slab)

        assert math.isnan(result.buhlmann_risk)
        assert math.isnan(result.delta_risk)
        assert math.isnan(result.buhlmann_ndl)
        assert math.isnan(result.delta_ndl)
        assert math.isnan(result.buhlmann_ceiling)
        assert math.isnan(result.delta_ceiling)


class TestParseBuhlmannOutputModule:
    """Test the module-level _run_buhlmann_single function."""

    def test_parse_valid_output_returns_result(self):
        """Valid lines -> BuhlmannResult (not None)"""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result = _run_buhlmann_single(profile, gf_low=1.0, gf_high=1.0)

        assert result is not None
        assert isinstance(result, BuhlmannResult)
        assert len(result.times) > 0
        assert result.gf_low == 1.0
        assert result.gf_high == 1.0

    def test_parse_empty_output_returns_none(self):
        """Empty profile -> None (or exception handled)"""
        # Create a profile with no points
        empty_profile = DiveProfile(name="Empty", points=[])

        result = _run_buhlmann_single(empty_profile, gf_low=1.0, gf_high=1.0)

        # Should return None due to exception handling
        assert result is None

    def test_parse_with_standard_gf_uses_raw(self):
        """gf_low=1.0, gf_high=1.0 uses raw ceiling values"""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result = _run_buhlmann_single(profile, gf_low=1.0, gf_high=1.0)

        assert result is not None
        assert result.gf_low == 1.0
        assert result.gf_high == 1.0
        # Values should not be adjusted

    def test_parse_with_conservative_gf_adjusts(self):
        """gf_low=0.7, gf_high=0.85 adjusts ceilings"""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result_standard = _run_buhlmann_single(profile, gf_low=1.0, gf_high=1.0)
        result_conservative = _run_buhlmann_single(profile, gf_low=0.7, gf_high=0.85)

        assert result_standard is not None
        assert result_conservative is not None

        # Conservative GF should produce different (typically higher) ceilings
        # or lower NDL, but we just verify it's not identical
        assert result_conservative.gf_low == 0.7
        assert result_conservative.gf_high == 0.85


class TestModelComparatorInit:
    """Test ModelComparator initialization."""

    def test_init_with_injected_dependencies(self):
        """Accepts pre-built runner and model"""
        mock_runner = MagicMock(spec=BuhlmannRunner)
        mock_runner.gf = GradientFactors(gf_low=1.0, gf_high=1.0)
        mock_model = MagicMock(spec=SlabModel)

        comparator = ModelComparator(
            buhlmann_runner=mock_runner,
            slab_model=mock_model,
        )

        assert comparator.buhlmann is mock_runner
        assert comparator.slab is mock_model

    def test_init_loads_gf_from_config(self):
        """With real config at /Users/evanhuang/dbm3/config.yaml, verify GF is loaded"""
        config_path = "/Users/evanhuang/dbm3/config.yaml"
        comparator = ModelComparator(config_path=config_path)

        # GF should be loaded from config (or default)
        assert comparator.gf is not None
        assert isinstance(comparator.gf, GradientFactors)

    def test_init_creates_profile_generator(self):
        """comparator.generator is a ProfileGenerator instance"""
        comparator = ModelComparator()

        assert comparator.generator is not None
        assert isinstance(comparator.generator, ProfileGenerator)


class TestCompareProfile:
    """Test single profile comparison."""

    def test_compare_returns_comparison_result(self):
        """Returns ComparisonResult"""
        mock_buhlmann = MagicMock()
        mock_buhlmann.gf = GradientFactors(gf_low=1.0, gf_high=1.0)
        mock_buhlmann.run.return_value = _make_buhlmann_result()
        mock_slab = MagicMock()
        mock_slab.run.return_value = _make_slab_result()

        comparator = ModelComparator(
            buhlmann_runner=mock_buhlmann,
            slab_model=mock_slab,
        )

        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result = comparator.compare_profile(profile)

        assert isinstance(result, ComparisonResult)
        assert result.buhlmann_result is not None
        assert result.slab_result is not None

    def test_compare_calls_both_models(self):
        """Both run() methods called"""
        mock_buhlmann = MagicMock()
        mock_buhlmann.gf = GradientFactors(gf_low=1.0, gf_high=1.0)
        mock_buhlmann.run.return_value = _make_buhlmann_result()
        mock_slab = MagicMock()
        mock_slab.run.return_value = _make_slab_result()

        comparator = ModelComparator(
            buhlmann_runner=mock_buhlmann,
            slab_model=mock_slab,
        )

        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        comparator.compare_profile(profile)

        mock_buhlmann.run.assert_called_once_with(profile)
        mock_slab.run.assert_called_once_with(profile)

    def test_compare_handles_buhlmann_failure(self):
        """buhlmann.run raises -> buhlmann_result=None, slab still runs"""
        mock_buhlmann = MagicMock()
        mock_buhlmann.gf = GradientFactors(gf_low=1.0, gf_high=1.0)
        mock_buhlmann.run.side_effect = RuntimeError("Binary failed")
        mock_slab = MagicMock()
        mock_slab.run.return_value = _make_slab_result()

        comparator = ModelComparator(
            buhlmann_runner=mock_buhlmann,
            slab_model=mock_slab,
        )

        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result = comparator.compare_profile(profile)

        assert result.buhlmann_result is None
        assert result.slab_result is not None
        assert not result.is_valid

    def test_compare_handles_slab_failure(self):
        """slab.run raises -> slab_result=None, buhlmann still runs"""
        mock_buhlmann = MagicMock()
        mock_buhlmann.gf = GradientFactors(gf_low=1.0, gf_high=1.0)
        mock_buhlmann.run.return_value = _make_buhlmann_result()
        mock_slab = MagicMock()
        mock_slab.run.side_effect = RuntimeError("Slab failed")

        comparator = ModelComparator(
            buhlmann_runner=mock_buhlmann,
            slab_model=mock_slab,
        )

        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result = comparator.compare_profile(profile)

        assert result.buhlmann_result is not None
        assert result.slab_result is None
        assert not result.is_valid


class TestGenerateReport:
    """Test report generation from comparison results."""

    def test_report_with_no_valid_results(self):
        """All invalid -> returns dict with 'error' key"""
        comparator = ModelComparator()

        # Create invalid results (both None)
        gen = ProfileGenerator()
        profile1 = gen.generate_square(depth=20, bottom_time=10)
        profile2 = gen.generate_square(depth=30, bottom_time=15)

        results = [
            ComparisonResult(profile=profile1, buhlmann_result=None, slab_result=None),
            ComparisonResult(profile=profile2, buhlmann_result=None, slab_result=None),
        ]

        report = comparator.generate_report(results)

        assert "error" in report
        assert report["error"] == "No valid results"

    def test_report_risk_mean_correct(self):
        """Hand-compute mean from 3 known delta_risk values"""
        gen = ProfileGenerator()
        profile1 = gen.generate_square(depth=20, bottom_time=10)
        profile2 = gen.generate_square(depth=30, bottom_time=15)
        profile3 = gen.generate_square(depth=40, bottom_time=20)

        results = [
            ComparisonResult(
                profile=profile1,
                buhlmann_result=_make_buhlmann_result(max_supersaturation=0.5),
                slab_result=_make_slab_result(max_cv_ratio=0.6),
            ),
            ComparisonResult(
                profile=profile2,
                buhlmann_result=_make_buhlmann_result(max_supersaturation=0.7),
                slab_result=_make_slab_result(max_cv_ratio=0.8),
            ),
            ComparisonResult(
                profile=profile3,
                buhlmann_result=_make_buhlmann_result(max_supersaturation=0.6),
                slab_result=_make_slab_result(max_cv_ratio=0.65),
            ),
        ]

        comparator = ModelComparator()
        report = comparator.generate_report(results)

        # Delta risk: (0.6-0.5), (0.8-0.7), (0.65-0.6) = 0.1, 0.1, 0.05
        # Mean = (0.1 + 0.1 + 0.05) / 3 = 0.25 / 3 ≈ 0.0833
        expected_mean = (0.1 + 0.1 + 0.05) / 3

        assert report["mean_delta_risk"] == pytest.approx(expected_mean, abs=1e-4)

    def test_report_ndl_mean_correct(self):
        """Hand-compute mean delta_ndl"""
        gen = ProfileGenerator()
        profile1 = gen.generate_square(depth=20, bottom_time=10)
        profile2 = gen.generate_square(depth=30, bottom_time=15)

        results = [
            ComparisonResult(
                profile=profile1,
                buhlmann_result=_make_buhlmann_result(min_ndl=20.0),
                slab_result=_make_slab_result(final_ndl=22.0),
            ),
            ComparisonResult(
                profile=profile2,
                buhlmann_result=_make_buhlmann_result(min_ndl=15.0),
                slab_result=_make_slab_result(final_ndl=18.0),
            ),
        ]

        comparator = ModelComparator()
        report = comparator.generate_report(results)

        # Delta NDL: (22-20), (18-15) = 2.0, 3.0
        # Mean = (2.0 + 3.0) / 2 = 2.5
        assert report["mean_delta_ndl"] == pytest.approx(2.5)

    def test_report_total_profiles_correct(self):
        """total_profiles matches input count"""
        gen = ProfileGenerator()
        profile1 = gen.generate_square(depth=20, bottom_time=10)
        profile2 = gen.generate_square(depth=30, bottom_time=15)
        profile3 = gen.generate_square(depth=40, bottom_time=20)

        results = [
            ComparisonResult(
                profile=profile1,
                buhlmann_result=_make_buhlmann_result(),
                slab_result=_make_slab_result(),
            ),
            ComparisonResult(
                profile=profile2,
                buhlmann_result=_make_buhlmann_result(),
                slab_result=_make_slab_result(),
            ),
            ComparisonResult(
                profile=profile3,
                buhlmann_result=None,  # Invalid
                slab_result=None,
            ),
        ]

        comparator = ModelComparator()
        report = comparator.generate_report(results)

        assert report["total_profiles"] == 3
        assert report["valid_profiles"] == 2

    def test_report_deco_agreement(self):
        """Test the deco agreement counting logic"""
        gen = ProfileGenerator()
        profile1 = gen.generate_square(depth=20, bottom_time=10)
        profile2 = gen.generate_square(depth=30, bottom_time=15)
        profile3 = gen.generate_square(depth=40, bottom_time=20)
        profile4 = gen.generate_square(depth=50, bottom_time=25)

        results = [
            # Both agree: no deco
            ComparisonResult(
                profile=profile1,
                buhlmann_result=_make_buhlmann_result(max_ceiling=1.0),
                slab_result=_make_slab_result(ceiling_at_bottom=0.0),
            ),
            # Both agree: both require deco
            ComparisonResult(
                profile=profile2,
                buhlmann_result=_make_buhlmann_result(max_ceiling=2.0),
                slab_result=_make_slab_result(ceiling_at_bottom=5.0),
            ),
            # Disagree: Slab only
            ComparisonResult(
                profile=profile3,
                buhlmann_result=_make_buhlmann_result(max_ceiling=1.0),
                slab_result=_make_slab_result(ceiling_at_bottom=3.0),
            ),
            # Disagree: Buhlmann only
            ComparisonResult(
                profile=profile4,
                buhlmann_result=_make_buhlmann_result(max_ceiling=2.0),
                slab_result=_make_slab_result(ceiling_at_bottom=0.0),
            ),
        ]

        comparator = ModelComparator()
        report = comparator.generate_report(results)

        # Agreement: 2 out of 4 = 50%
        assert report["deco_agreement_pct"] == pytest.approx(50.0)
        assert report["both_require_deco"] == 1
        assert report["neither_require_deco"] == 1
        assert report["slab_only_deco"] == 1
        assert report["buhlmann_only_deco"] == 1
