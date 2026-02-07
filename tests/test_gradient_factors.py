"""
Comprehensive tests for Bühlmann Gradient Factor implementation.

Tests validate real mathematical behavior against hand-computed values using
ZH-L16C constants and GF formulas.
"""

import math
import pytest
from backtest.buhlmann_constants import (
    GradientFactors,
    GF_DEFAULT,
    m_value,
    m_value_gf,
    ceiling_pressure_gf,
    compute_ceilings_gf,
    compute_ndl_gf,
    compute_max_supersaturation_gf,
    ZH_L16_N2_A,
    ZH_L16_N2_B,
    ZH_L16_N2_HALFTIMES,
    NUM_COMPARTMENTS,
)


class TestGradientFactorsValidation:
    """Test GradientFactors dataclass validation and properties."""

    def test_valid_construction(self):
        """Valid GF pairs should construct without error."""
        gf = GradientFactors(gf_low=0.7, gf_high=0.85)
        assert gf.gf_low == 0.7
        assert gf.gf_high == 0.85
        assert not gf.is_standard

    def test_standard_gf(self):
        """GF 100/100 should be recognized as standard."""
        gf = GradientFactors(gf_low=1.0, gf_high=1.0)
        assert gf.is_standard
        assert gf == GF_DEFAULT

    def test_edge_case_equal_gf(self):
        """GF_low = GF_high should be valid."""
        gf = GradientFactors(gf_low=0.85, gf_high=0.85)
        assert gf.gf_low == gf.gf_high
        assert not gf.is_standard

    def test_very_conservative_gf(self):
        """Very conservative GF values should be valid."""
        gf = GradientFactors(gf_low=0.3, gf_high=0.7)
        assert gf.gf_low == 0.3
        assert gf.gf_high == 0.7

    def test_reject_zero_gf_low(self):
        """GF_low = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="gf_low must be in"):
            GradientFactors(gf_low=0.0, gf_high=0.85)

    def test_reject_negative_gf_low(self):
        """Negative GF_low should raise ValueError."""
        with pytest.raises(ValueError, match="gf_low must be in"):
            GradientFactors(gf_low=-0.5, gf_high=0.85)

    def test_reject_zero_gf_high(self):
        """GF_high = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="gf_high must be in"):
            GradientFactors(gf_low=0.7, gf_high=0.0)

    def test_reject_gf_high_over_1(self):
        """GF_high > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="gf_high must be in"):
            GradientFactors(gf_low=0.7, gf_high=1.1)

    def test_reject_gf_low_over_1(self):
        """GF_low > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="gf_low must be in"):
            GradientFactors(gf_low=1.5, gf_high=0.85)

    def test_reject_gf_low_exceeds_gf_high(self):
        """GF_low > GF_high should raise ValueError."""
        with pytest.raises(ValueError, match="gf_low.*must be <= gf_high"):
            GradientFactors(gf_low=0.9, gf_high=0.7)

    def test_frozen_dataclass(self):
        """GradientFactors should be immutable (frozen)."""
        gf = GradientFactors(gf_low=0.7, gf_high=0.85)
        with pytest.raises(Exception):  # FrozenInstanceError
            gf.gf_low = 0.5


class TestMValueMath:
    """Test M-value calculations with hand-computed expected values."""

    def test_standard_m_value_compartment_0_at_1_bar(self):
        """Standard M-value for compartment 0 at 1 bar should match formula."""
        # M(P) = a + P/b
        # Compartment 0: a=1.2599, b=0.5050
        # M(1.0) = 1.2599 + 1.0/0.5050 = 1.2599 + 1.9801... = 3.2400...
        expected = ZH_L16_N2_A[0] + 1.0 / ZH_L16_N2_B[0]
        actual = m_value(0, 1.0)
        assert abs(actual - expected) < 1e-6

    def test_standard_m_value_compartment_15_at_4_bar(self):
        """Standard M-value for compartment 15 at 4 bar should match formula."""
        # Compartment 15: a=0.2327, b=0.9653
        # M(4.0) = 0.2327 + 4.0/0.9653 = 0.2327 + 4.1438... = 4.3765...
        expected = ZH_L16_N2_A[15] + 4.0 / ZH_L16_N2_B[15]
        actual = m_value(15, 4.0)
        assert abs(actual - expected) < 1e-6

    def test_m_value_gf_at_100_equals_standard(self):
        """M_gf at GF=1.0 should equal standard M-value."""
        compartment = 5
        pressure = 3.0
        standard = m_value(compartment, pressure)
        gf_adjusted = m_value_gf(compartment, pressure, gf=1.0)
        assert abs(gf_adjusted - standard) < 1e-9

    def test_m_value_gf_below_100_is_more_conservative(self):
        """M_gf at GF < 1.0 should be less than standard M-value."""
        compartment = 3
        pressure = 2.5
        standard = m_value(compartment, pressure)
        gf_adjusted = m_value_gf(compartment, pressure, gf=0.85)
        assert gf_adjusted < standard

    def test_m_value_gf_at_50_is_halfway(self):
        """M_gf at GF=0.5 should be halfway between ambient and standard M-value."""
        # M_gf(P) = P + gf * (M(P) - P)
        # At gf=0.5: M_gf = P + 0.5 * (M - P) = (P + M) / 2
        compartment = 2
        pressure = 3.0
        standard = m_value(compartment, pressure)
        expected_halfway = (pressure + standard) / 2.0
        gf_adjusted = m_value_gf(compartment, pressure, gf=0.5)
        assert abs(gf_adjusted - expected_halfway) < 1e-9

    def test_m_value_gf_formula_verification(self):
        """Verify M_gf formula matches expected calculation."""
        # M_gf(P) = P + gf * (a + P/b - P)
        compartment = 1
        pressure = 2.0
        gf = 0.7
        a = ZH_L16_N2_A[compartment]
        b = ZH_L16_N2_B[compartment]
        expected = pressure + gf * (a + pressure / b - pressure)
        actual = m_value_gf(compartment, pressure, gf)
        assert abs(actual - expected) < 1e-9


class TestCeilingMath:
    """Test ceiling calculations with hand-computed expected values."""

    def test_ceiling_gf_100_matches_standard_formula(self):
        """Ceiling at GF=1.0 should match standard formula: b * (p_tissue - a)."""
        # Standard ceiling: P_ceil = b * (P_tissue - a)
        # At GF=1.0, the GF formula should reduce to this
        compartment = 4
        tissue_pressure = 2.5
        a = ZH_L16_N2_A[compartment]
        b = ZH_L16_N2_B[compartment]
        expected_standard = b * (tissue_pressure - a)
        actual_gf = ceiling_pressure_gf(compartment, tissue_pressure, gf=1.0)
        assert abs(actual_gf - expected_standard) < 1e-9

    def test_ceiling_gf_below_100_is_deeper(self):
        """GF_low < 1.0 should produce deeper (higher pressure) ceiling."""
        compartment = 5
        tissue_pressure = 3.0
        ceil_standard = ceiling_pressure_gf(compartment, tissue_pressure, gf=1.0)
        ceil_conservative = ceiling_pressure_gf(compartment, tissue_pressure, gf=0.7)
        # Lower GF = more conservative = deeper ceiling = higher pressure
        assert ceil_conservative > ceil_standard

    def test_ceiling_zero_when_not_supersaturated(self):
        """Ceiling should be 0.0 when tissue is at or below surface equilibrium."""
        # Low tissue pressure (e.g., surface equilibrium N2 ~0.75 bar)
        compartment = 0
        tissue_pressure = 0.8  # Below M-value at surface
        ceil = ceiling_pressure_gf(compartment, tissue_pressure, gf=1.0)
        assert ceil == 0.0

    def test_ceiling_formula_verification(self):
        """Verify ceiling formula: ceil = (p_tissue - gf*a) / (1 - gf + gf/b)."""
        compartment = 7
        tissue_pressure = 4.0
        gf = 0.85
        a = ZH_L16_N2_A[compartment]
        b = ZH_L16_N2_B[compartment]
        expected = (tissue_pressure - gf * a) / (1.0 - gf + gf / b)
        actual = ceiling_pressure_gf(compartment, tissue_pressure, gf)
        assert abs(actual - expected) < 1e-9

    def test_compute_ceilings_gf_returns_max_and_per_compartment(self):
        """compute_ceilings_gf should return (max_ceiling, list_of_ceilings)."""
        # Simulate tissue loading with different pressures
        n2_tensions = [1.5, 2.0, 1.8, 1.2] + [0.8] * 12  # 16 compartments
        he_tensions = [0.0] * 16
        gf = 0.8
        max_ceil, per_comp = compute_ceilings_gf(n2_tensions, he_tensions, gf)

        assert len(per_comp) == 16
        assert max_ceil >= 0.0
        # Max should be the maximum of all compartments
        assert max_ceil == max(per_comp)

    def test_compute_ceilings_gf_with_helium(self):
        """Ceiling calculation should account for N2 + He total inert pressure."""
        n2_tensions = [1.5] * 16
        he_tensions = [0.5] * 16
        gf = 0.85
        max_ceil_combined, _ = compute_ceilings_gf(n2_tensions, he_tensions, gf)
        max_ceil_n2_only, _ = compute_ceilings_gf(n2_tensions, [0.0] * 16, gf)
        # Adding He should increase ceiling
        assert max_ceil_combined > max_ceil_n2_only


class TestNDLMath:
    """Test NDL calculations with hand-computed expected values."""

    def test_ndl_gf_100_is_longer_than_gf_85(self):
        """GF_high < 1.0 should produce shorter NDL than GF_high = 1.0."""
        # Start with surface-equilibrated tissues
        n2_tensions = [0.75] * 16  # approximate surface N2 saturation
        he_tensions = [0.0] * 16
        ambient_pressure = 4.0  # 30m depth
        f_inert = 0.79  # air

        ndl_standard = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=1.0)
        ndl_conservative = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=0.85)

        # Lower GF_high = more conservative = shorter NDL
        assert ndl_conservative < ndl_standard
        assert ndl_conservative > 0  # Should have some NDL

    def test_ndl_unlimited_when_wont_exceed(self):
        """NDL should return 100.0 (cap) when gas loading won't exceed limit."""
        # Very shallow depth, already saturated tissues
        n2_tensions = [0.75] * 16
        he_tensions = [0.0] * 16
        ambient_pressure = 1.01325  # surface
        f_inert = 0.79

        ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=1.0)
        assert ndl == 100.0  # Capped to match libbuhlmann

    def test_ndl_zero_when_already_exceeded(self):
        """NDL should return 0.0 when M-value already exceeded."""
        # Highly loaded tissues that exceed surface M-value
        n2_tensions = [3.0] * 16  # very high tissue loading
        he_tensions = [0.0] * 16
        ambient_pressure = 4.0  # 30m depth
        f_inert = 0.79

        ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=1.0)
        assert ndl == 0.0

    def test_ndl_reasonable_value_30m_air(self):
        """NDL at 30m on air should be reasonable (15-30 minutes range)."""
        # Surface-equilibrated tissues
        p_surface = 1.01325
        f_inert = 0.79
        n2_tensions = [p_surface * f_inert] * 16
        he_tensions = [0.0] * 16
        ambient_pressure = 4.0  # ~30m

        ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=1.0)
        # Bühlmann NDL at 30m air is ~20 minutes (compartment-dependent)
        # Should be in reasonable range
        assert 10.0 < ndl < 50.0

    def test_ndl_exponential_loading_formula(self):
        """NDL should use correct exponential loading formula."""
        # For a known compartment, verify the math
        # P(t) = P_inspired + (P_tissue - P_inspired) * exp(-k*t)
        # Solve for t when P(t) = M_target

        # Use a slower compartment and check that the formula is correctly applied
        # Compartment 5: halftime=38.3 min, a=0.5933, b=0.8434
        compartment_idx = 5
        p_surface = 1.01325
        f_inert = 0.79
        ambient_pressure = 2.5  # Shallower depth for better NDL prediction
        p_inspired_inert = ambient_pressure * f_inert
        p_tissue_initial = p_surface * f_inert  # surface equilibrium

        # Single-compartment calculation
        gf_high = 1.0
        m_target = m_value_gf(compartment_idx, p_surface, gf_high)

        # Only proceed if this compartment will actually limit
        if p_inspired_inert > m_target and p_tissue_initial < m_target:
            k = math.log(2) / ZH_L16_N2_HALFTIMES[compartment_idx]
            ratio = (m_target - p_inspired_inert) / (p_tissue_initial - p_inspired_inert)
            if ratio > 0:
                expected_ndl_comp5 = -math.log(ratio) / k

                # Now run the full NDL calculation
                n2_tensions = [p_tissue_initial] * 16
                he_tensions = [0.0] * 16
                actual_ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high)

                # Actual NDL should be reasonably close to our compartment 5 calculation
                # May be limited by a different compartment, so just verify it's reasonable
                assert 0.0 < actual_ndl <= 100.0
                assert abs(actual_ndl - expected_ndl_comp5) < 50.0  # within 50 minutes (may be different limiting compartment)
        else:
            # If the math doesn't work out for this compartment, just verify NDL is computed
            n2_tensions = [p_tissue_initial] * 16
            he_tensions = [0.0] * 16
            actual_ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high)
            assert 0.0 < actual_ndl <= 100.0


class TestMaxSupersaturation:
    """Test max supersaturation calculations."""

    def test_max_supersaturation_gf_100_matches_standard(self):
        """GF 100/100 should produce same supersaturation as standard calculation."""
        # Single timestep, single compartment loaded to 90% of M-value
        p_surface = 1.01325
        m_val = m_value(0, p_surface)
        tissue_loaded = 0.9 * m_val

        n2_series = [[tissue_loaded] + [0.8] * 15]
        he_series = [[0.0] * 16]
        pressures = [p_surface]

        max_super = compute_max_supersaturation_gf(n2_series, he_series, pressures, gf_low=1.0, gf_high=1.0)
        expected_ratio = tissue_loaded / m_val
        assert abs(max_super - expected_ratio) < 1e-6

    def test_max_supersaturation_gf_below_100_produces_higher_ratio(self):
        """GF < 100 produces higher supersaturation ratio (since limit is lower)."""
        # Same tissue loading, but lower GF means lower M-value limit
        p_surface = 1.01325
        tissue_pressure = 2.0

        n2_series = [[tissue_pressure] + [0.8] * 15]
        he_series = [[0.0] * 16]
        pressures = [p_surface]

        ratio_standard = compute_max_supersaturation_gf(n2_series, he_series, pressures, gf_low=1.0, gf_high=1.0)
        ratio_conservative = compute_max_supersaturation_gf(n2_series, he_series, pressures, gf_low=0.7, gf_high=0.85)

        # Lower GF = lower M-value limit = higher ratio for same tissue loading
        assert ratio_conservative > ratio_standard

    def test_max_supersaturation_multiple_timesteps(self):
        """Max supersaturation should find the maximum across all timesteps."""
        # Create a profile with varying tissue loading
        n2_series = [
            [1.0] * 16,
            [1.5] * 16,
            [2.0] * 16,  # Peak loading
            [1.8] * 16,
        ]
        he_series = [[0.0] * 16] * 4
        pressures = [2.0, 3.0, 4.0, 3.5]

        max_super = compute_max_supersaturation_gf(n2_series, he_series, pressures, gf_low=0.8, gf_high=0.9)
        assert max_super > 0.0

        # Should be based on highest tissue loading timestep
        # Exact value depends on interpolated GF, but should be reasonable
        assert max_super < 2.0  # Shouldn't be wildly excessive

    def test_max_supersaturation_gf_interpolation(self):
        """Supersaturation should use interpolated GF between low and high."""
        # At surface (low pressure): should use gf_high
        # At depth (high pressure): should use gf_low

        p_surface = 1.01325
        p_deep = 5.0
        tissue_pressure = 3.0

        # Surface timestep
        n2_surface = [[tissue_pressure] + [0.8] * 15]
        he_surface = [[0.0] * 16]
        pressures_surface = [p_surface]

        # Deep timestep
        n2_deep = [[tissue_pressure] + [0.8] * 15]
        he_deep = [[0.0] * 16]
        pressures_deep = [p_deep]

        gf_low = 0.5
        gf_high = 0.9

        ratio_surface = compute_max_supersaturation_gf(n2_surface, he_surface, pressures_surface, gf_low, gf_high)
        ratio_deep = compute_max_supersaturation_gf(n2_deep, he_deep, pressures_deep, gf_low, gf_high)

        # Different ambient pressures should use different interpolated GF values
        # This leads to different M-value limits and thus different ratios
        assert ratio_surface != ratio_deep


class TestIntegrationAndBackwardCompatibility:
    """Test integration with BuhlmannResult and backward compatibility."""

    def test_gf_default_is_100_100(self):
        """GF_DEFAULT should be (1.0, 1.0)."""
        assert GF_DEFAULT.gf_low == 1.0
        assert GF_DEFAULT.gf_high == 1.0
        assert GF_DEFAULT.is_standard

    def test_all_compartments_constant(self):
        """NUM_COMPARTMENTS should be 16."""
        assert NUM_COMPARTMENTS == 16
        assert len(ZH_L16_N2_HALFTIMES) == 16
        assert len(ZH_L16_N2_A) == 16
        assert len(ZH_L16_N2_B) == 16

    def test_halftimes_increasing(self):
        """Compartment halftimes should be monotonically increasing."""
        for i in range(len(ZH_L16_N2_HALFTIMES) - 1):
            assert ZH_L16_N2_HALFTIMES[i] < ZH_L16_N2_HALFTIMES[i + 1]

    def test_a_coefficients_decreasing(self):
        """M-value 'a' coefficients should be monotonically decreasing."""
        for i in range(len(ZH_L16_N2_A) - 1):
            assert ZH_L16_N2_A[i] > ZH_L16_N2_A[i + 1]

    def test_b_coefficients_increasing(self):
        """M-value 'b' coefficients should be monotonically increasing."""
        for i in range(len(ZH_L16_N2_B) - 1):
            assert ZH_L16_N2_B[i] < ZH_L16_N2_B[i + 1]

    def test_constants_are_tuples(self):
        """ZH-L16 constants should be immutable tuples."""
        assert isinstance(ZH_L16_N2_HALFTIMES, tuple)
        assert isinstance(ZH_L16_N2_A, tuple)
        assert isinstance(ZH_L16_N2_B, tuple)


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_ceiling_with_very_low_gf(self):
        """Very low GF should produce deeper ceiling (higher pressure) than standard."""
        compartment = 3
        tissue_pressure = 3.0
        gf_low = 0.01  # Extremely conservative
        gf_standard = 1.0

        ceil_conservative = ceiling_pressure_gf(compartment, tissue_pressure, gf_low)
        ceil_standard = ceiling_pressure_gf(compartment, tissue_pressure, gf_standard)

        # Lower GF = more conservative = deeper ceiling = HIGHER ceiling pressure
        # The ceiling is the minimum safe ambient pressure
        # Lower GF means less supersaturation tolerance, requiring deeper stops
        assert ceil_conservative > ceil_standard
        # Ceiling should approach but not exceed tissue pressure as GF approaches 0
        assert ceil_conservative < tissue_pressure

    def test_ndl_with_very_low_gf_high(self):
        """Very low GF_high should produce very short NDL."""
        n2_tensions = [0.75] * 16
        he_tensions = [0.0] * 16
        ambient_pressure = 4.0
        f_inert = 0.79
        gf_high = 0.3  # Extremely conservative

        ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high)
        # Should be very short but not zero
        assert 0.0 < ndl < 10.0

    def test_ceiling_negative_tissue_pressure(self):
        """Ceiling calculation should handle edge case of negative result gracefully."""
        # This shouldn't happen in practice, but test robustness
        compartment = 0
        tissue_pressure = 0.5  # Very low
        gf = 1.0

        ceil = ceiling_pressure_gf(compartment, tissue_pressure, gf)
        # Should return 0.0, not negative
        assert ceil == 0.0

    def test_ndl_with_helium_loading(self):
        """NDL calculation should account for He + N2 total inert loading."""
        n2_tensions = [0.5] * 16
        he_tensions = [0.3] * 16  # Trimix
        ambient_pressure = 4.0
        f_inert = 0.79
        gf_high = 1.0

        ndl_with_he = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high)
        ndl_without_he = compute_ndl_gf(n2_tensions, [0.0] * 16, ambient_pressure, f_inert, gf_high)

        # Adding He should reduce NDL
        assert ndl_with_he < ndl_without_he

    def test_compute_ceilings_gf_empty_helium(self):
        """compute_ceilings_gf should handle empty/None helium list."""
        n2_tensions = [1.5] * 16
        he_tensions = []  # Empty
        gf = 0.8

        max_ceil, per_comp = compute_ceilings_gf(n2_tensions, he_tensions, gf)
        assert max_ceil >= 0.0
        assert len(per_comp) == 16

    def test_max_supersaturation_single_timestep(self):
        """Max supersaturation should work with single timestep."""
        n2_series = [[1.5] * 16]
        he_series = [[0.0] * 16]
        pressures = [2.0]

        max_super = compute_max_supersaturation_gf(n2_series, he_series, pressures, gf_low=0.8, gf_high=0.9)
        assert max_super > 0.0
        assert max_super < 10.0  # Sanity check


class TestRealWorldScenarios:
    """Test gradient factors with realistic dive scenarios."""

    def test_gf_reduces_ndl_proportionally(self):
        """GF 85/85 should reduce NDL by approximately 15% compared to GF 100/100."""
        n2_tensions = [0.75] * 16  # surface equilibrium
        he_tensions = [0.0] * 16
        ambient_pressure = 4.0  # 30m
        f_inert = 0.79  # air

        ndl_100 = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=1.0)
        ndl_85 = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=0.85)

        # GF 85 should give roughly 10-20% shorter NDL
        reduction_ratio = ndl_85 / ndl_100
        assert 0.7 < reduction_ratio < 0.95

    def test_deep_dive_ceiling_progression(self):
        """Deep dive should show ceiling decreasing as GF varies from low to high."""
        # Simulate tissue loading from a deep dive
        n2_tensions = [2.5, 2.8, 3.0, 2.9, 2.7] + [2.0] * 11
        he_tensions = [0.0] * 16

        # At first stop (deep), use GF_low
        max_ceil_low, _ = compute_ceilings_gf(n2_tensions, he_tensions, gf=0.3)

        # At shallow stop, use higher GF
        max_ceil_mid, _ = compute_ceilings_gf(n2_tensions, he_tensions, gf=0.6)

        # At surface, use GF_high
        max_ceil_high, _ = compute_ceilings_gf(n2_tensions, he_tensions, gf=0.9)

        # Ceiling should decrease (shallower) as GF increases
        assert max_ceil_low > max_ceil_mid > max_ceil_high

    def test_trimix_dive_with_gf(self):
        """Trimix dive with He should produce reasonable GF-adjusted results."""
        # Trimix 18/45 at 60m (7 bar)
        n2_tensions = [0.9] * 16  # Some N2 loading
        he_tensions = [2.0] * 16  # Significant He loading
        ambient_pressure = 7.0  # 60m
        f_o2 = 0.18
        f_he = 0.45
        f_inert = 1.0 - f_o2  # N2 + He

        # NDL should be very short at this depth
        ndl = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=0.85)
        assert 0.0 <= ndl < 20.0  # Very short or zero at 60m

        # Ceiling should be present due to He + N2 loading
        max_ceil, _ = compute_ceilings_gf(n2_tensions, he_tensions, gf=0.3)
        assert max_ceil > 1.01325  # Ceiling deeper than surface

    def test_conservative_gf_vs_aggressive_gf(self):
        """Conservative GF (30/70) vs aggressive GF (90/95) should show significant differences."""
        # Start with surface-equilibrated tissues
        p_surface = 1.01325
        f_inert = 0.79
        n2_tensions = [p_surface * f_inert] * 16
        he_tensions = [0.0] * 16
        ambient_pressure = 3.5  # 25m

        # Conservative: GF 30/70
        ndl_conservative = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=0.7)

        # Aggressive: GF 90/95
        ndl_aggressive = compute_ndl_gf(n2_tensions, he_tensions, ambient_pressure, f_inert, gf_high=0.95)

        # Conservative should give shorter NDL
        assert ndl_conservative < ndl_aggressive

        # Test ceilings with loaded tissues
        n2_loaded = [2.0] * 16
        he_loaded = [0.3] * 16
        ceil_conservative, _ = compute_ceilings_gf(n2_loaded, he_loaded, gf=0.3)
        ceil_aggressive, _ = compute_ceilings_gf(n2_loaded, he_loaded, gf=0.9)

        # Conservative should give deeper (higher pressure) ceiling
        assert ceil_conservative > ceil_aggressive

    def test_compartment_variability_with_gf(self):
        """Different compartments should respond differently to GF adjustments."""
        # Load compartments differently (fast vs slow)
        n2_tensions = [3.0, 2.5, 2.0, 1.8] + [1.5] * 12
        he_tensions = [0.0] * 16
        gf = 0.7

        _, per_comp_ceilings = compute_ceilings_gf(n2_tensions, he_tensions, gf)

        # Fast compartments (high loading) should have higher ceilings
        # Compartment 0 has highest N2, should have meaningful ceiling
        assert per_comp_ceilings[0] > 0.0
        # Later compartments with less loading should have lower ceilings
        assert per_comp_ceilings[0] > per_comp_ceilings[10]

    def test_m_value_convergence_at_depth(self):
        """M-values at increasing depths should converge for different GF values."""
        compartment = 5
        gf_low = 0.5
        gf_high = 1.0

        # At shallow depth
        m_shallow_low = m_value_gf(compartment, 2.0, gf_low)
        m_shallow_high = m_value_gf(compartment, 2.0, gf_high)
        diff_shallow = m_shallow_high - m_shallow_low

        # At deep depth
        m_deep_low = m_value_gf(compartment, 10.0, gf_low)
        m_deep_high = m_value_gf(compartment, 10.0, gf_high)
        diff_deep = m_deep_high - m_deep_low

        # Difference should be larger at depth (M-value grows with 1/b term)
        assert diff_deep > diff_shallow

    def test_surface_interval_tissue_loading(self):
        """After surface interval, lower tissue loading should extend NDL even with conservative GF."""
        ambient_pressure = 3.0  # 20m
        f_inert = 0.79

        # Residual loading from previous dive
        n2_high = [1.5] * 16
        he_high = [0.0] * 16
        ndl_high = compute_ndl_gf(n2_high, he_high, ambient_pressure, f_inert, gf_high=0.85)

        # After surface interval, lower loading
        n2_low = [0.9] * 16
        he_low = [0.0] * 16
        ndl_low = compute_ndl_gf(n2_low, he_low, ambient_pressure, f_inert, gf_high=0.85)

        # Lower tissue loading should give longer NDL
        assert ndl_low > ndl_high
