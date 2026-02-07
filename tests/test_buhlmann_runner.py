"""
Comprehensive unit tests for backtest/buhlmann_runner.py.

Tests BuhlmannRunner interface, output parsing, GF adjustments,
and integration with the libbuhlmann binary.
"""

import os
import pytest
from backtest.buhlmann_runner import BuhlmannRunner, BuhlmannResult
from backtest.buhlmann_constants import GradientFactors, GF_DEFAULT
from backtest.profile_generator import ProfileGenerator


# Helper function to build synthetic output lines
def _make_output_line(time, pressure, n2_base=0.8, he_base=0.0, ceiling=0.95, ndl=28.0):
    """Build a single 36-value line matching libbuhlmann output format."""
    parts = [f"{time:.2f}", f"{pressure:.2f}"]
    for c in range(16):
        parts.append(f"{n2_base + c * 0.01:.4f}")
        parts.append(f"{he_base:.4f}")
    parts.append(f"{ceiling:.4f}")
    parts.append(f"{ndl:.2f}")
    return " ".join(parts)


# Skip marker for integration tests requiring the C binary
BINARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'libbuhlmann', 'src', 'dive')
skip_no_binary = pytest.mark.skipif(
    not os.path.exists(BINARY_PATH),
    reason="libbuhlmann binary not available"
)


# ============================================================================
# TestBuhlmannResult
# ============================================================================

class TestBuhlmannResult:
    """Test BuhlmannResult dataclass properties and methods."""

    def test_requires_deco_when_ceiling_above_surface(self):
        """requires_deco=True when max_ceiling > 1.0 bar."""
        result = BuhlmannResult(
            times=[0.0, 1.0],
            pressures=[3.0, 3.0],
            compartment_n2=[[0.8] * 16, [0.9] * 16],
            compartment_he=[[0.0] * 16, [0.0] * 16],
            ceilings=[1.0, 2.5],
            ndl_times=[20.0, 15.0],
            max_ceiling=2.5,
            min_ndl=15.0,
            max_supersaturation=0.75,
        )
        assert result.requires_deco is True

    def test_no_deco_when_ceiling_at_surface(self):
        """requires_deco=False when max_ceiling <= 1.0 bar (surface pressure)."""
        result = BuhlmannResult(
            times=[0.0, 1.0],
            pressures=[2.0, 2.0],
            compartment_n2=[[0.7] * 16, [0.75] * 16],
            compartment_he=[[0.0] * 16, [0.0] * 16],
            ceilings=[0.95, 1.0],
            ndl_times=[30.0, 28.0],
            max_ceiling=1.0,
            min_ndl=28.0,
            max_supersaturation=0.45,
        )
        assert result.requires_deco is False

    def test_risk_score_equals_max_supersaturation(self):
        """risk_score property returns max_supersaturation."""
        result = BuhlmannResult(
            times=[0.0],
            pressures=[2.0],
            compartment_n2=[[0.8] * 16],
            compartment_he=[[0.0] * 16],
            ceilings=[1.0],
            ndl_times=[25.0],
            max_ceiling=1.0,
            min_ndl=25.0,
            max_supersaturation=0.82,
        )
        assert result.risk_score == pytest.approx(0.82)

    def test_gf_defaults_to_standard(self):
        """Default gf_low=1.0, gf_high=1.0 (standard Bühlmann)."""
        result = BuhlmannResult(
            times=[0.0],
            pressures=[2.0],
            compartment_n2=[[0.8] * 16],
            compartment_he=[[0.0] * 16],
            ceilings=[1.0],
            ndl_times=[28.0],
            max_ceiling=1.0,
            min_ndl=28.0,
            max_supersaturation=0.5,
        )
        assert result.gf_low == pytest.approx(1.0)
        assert result.gf_high == pytest.approx(1.0)


# ============================================================================
# TestBuhlmannRunnerInit
# ============================================================================

class TestBuhlmannRunnerInit:
    """Test BuhlmannRunner initialization."""

    @skip_no_binary
    def test_auto_detect_binary(self):
        """Runner auto-detects libbuhlmann/src/dive binary."""
        runner = BuhlmannRunner(use_python_engine=False)
        assert runner.binary_path is not None
        assert os.path.exists(runner.binary_path)
        assert runner.binary_path.endswith('dive')

    def test_missing_binary_raises(self):
        """Non-existent binary path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BuhlmannRunner(binary_path='/nonexistent/path/dive', use_python_engine=False)

    def test_gf_defaults_to_standard(self):
        """No gf arg -> GF_DEFAULT (100/100)."""
        runner = BuhlmannRunner(use_python_engine=True)
        assert runner.gf.gf_low == pytest.approx(1.0)
        assert runner.gf.gf_high == pytest.approx(1.0)
        assert runner.gf == GF_DEFAULT

    def test_custom_gf_stored(self):
        """Custom GradientFactors stored correctly."""
        gf = GradientFactors(0.7, 0.85)
        runner = BuhlmannRunner(gf=gf, use_python_engine=True)
        assert runner.gf.gf_low == pytest.approx(0.7)
        assert runner.gf.gf_high == pytest.approx(0.85)


# ============================================================================
# TestParseOutput
# ============================================================================

class TestParseOutput:
    """Test _parse_output() method (private method accessed via instance)."""

    def test_parse_single_line_time_and_pressure(self):
        """Time and pressure extracted correctly from single line."""
        runner = BuhlmannRunner(use_python_engine=True)
        output = _make_output_line(time=5.2, pressure=3.01, n2_base=0.85)
        result = runner._parse_output(output)

        assert len(result.times) == 1
        assert result.times[0] == pytest.approx(5.2)
        assert result.pressures[0] == pytest.approx(3.01)

    def test_parse_compartment_n2_values(self):
        """16 N2 values extracted from correct indices (2, 4, 6, ..., 32)."""
        runner = BuhlmannRunner(use_python_engine=True)
        output = _make_output_line(time=1.0, pressure=2.0, n2_base=0.80)
        result = runner._parse_output(output)

        assert len(result.compartment_n2) == 1
        assert len(result.compartment_n2[0]) == 16
        # n2_base + c * 0.01 for c in [0, 1, ..., 15]
        for c in range(16):
            expected = 0.80 + c * 0.01
            assert result.compartment_n2[0][c] == pytest.approx(expected, abs=1e-4)

    def test_parse_compartment_he_values(self):
        """16 He values extracted from correct indices (3, 5, 7, ..., 33)."""
        runner = BuhlmannRunner(use_python_engine=True)
        output = _make_output_line(time=1.0, pressure=2.0, n2_base=0.8, he_base=0.15)
        result = runner._parse_output(output)

        assert len(result.compartment_he) == 1
        assert len(result.compartment_he[0]) == 16
        for c in range(16):
            assert result.compartment_he[0][c] == pytest.approx(0.15, abs=1e-4)

    def test_parse_ceiling_and_ndl(self):
        """Ceiling and NDL from last 2 values (indices -2, -1)."""
        runner = BuhlmannRunner(use_python_engine=True)
        output = _make_output_line(
            time=10.0, pressure=3.0, ceiling=1.25, ndl=18.5
        )
        result = runner._parse_output(output)

        assert len(result.ceilings) == 1
        assert result.ceilings[0] == pytest.approx(1.25)
        assert len(result.ndl_times) == 1
        assert result.ndl_times[0] == pytest.approx(18.5)

    def test_parse_multiple_lines_produces_multiple_timesteps(self):
        """5 lines -> len(times)==5."""
        runner = BuhlmannRunner(use_python_engine=True)
        lines = [
            _make_output_line(time=float(i), pressure=2.0 + i * 0.1)
            for i in range(5)
        ]
        output = "\n".join(lines)
        result = runner._parse_output(output)

        assert len(result.times) == 5
        assert len(result.pressures) == 5
        assert len(result.compartment_n2) == 5
        assert len(result.compartment_he) == 5
        assert len(result.ceilings) == 5
        assert len(result.ndl_times) == 5

    def test_parse_skips_empty_lines(self):
        """Empty lines between data lines still parse correctly."""
        runner = BuhlmannRunner(use_python_engine=True)
        lines = [
            _make_output_line(time=0.0, pressure=1.0),
            "",
            _make_output_line(time=1.0, pressure=2.0),
            "",
            "",
            _make_output_line(time=2.0, pressure=3.0),
        ]
        output = "\n".join(lines)
        result = runner._parse_output(output)

        assert len(result.times) == 3
        assert result.times == pytest.approx([0.0, 1.0, 2.0])

    def test_parse_skips_short_lines(self):
        """Lines with < 36 values skipped."""
        runner = BuhlmannRunner(use_python_engine=True)
        lines = [
            _make_output_line(time=0.0, pressure=1.0),
            "1.0 2.0 0.8",  # Only 3 values (should be skipped)
            _make_output_line(time=1.0, pressure=2.0),
        ]
        output = "\n".join(lines)
        result = runner._parse_output(output)

        assert len(result.times) == 2
        assert result.times == pytest.approx([0.0, 1.0])

    def test_parse_empty_output_raises_valueerror(self):
        """Empty output raises ValueError."""
        runner = BuhlmannRunner(use_python_engine=True)
        with pytest.raises(ValueError, match="No valid output from libbuhlmann"):
            runner._parse_output("")

    def test_parse_max_ceiling_is_maximum(self):
        """max_ceiling == max(all ceilings)."""
        runner = BuhlmannRunner(use_python_engine=True)
        lines = [
            _make_output_line(time=0.0, pressure=2.0, ceiling=1.0),
            _make_output_line(time=1.0, pressure=3.0, ceiling=1.5),
            _make_output_line(time=2.0, pressure=2.5, ceiling=1.2),
        ]
        output = "\n".join(lines)
        result = runner._parse_output(output)

        assert result.max_ceiling == pytest.approx(1.5)

    def test_parse_min_ndl_is_minimum(self):
        """min_ndl == min(all ndl_times)."""
        runner = BuhlmannRunner(use_python_engine=True)
        lines = [
            _make_output_line(time=0.0, pressure=2.0, ndl=30.0),
            _make_output_line(time=1.0, pressure=3.0, ndl=18.5),
            _make_output_line(time=2.0, pressure=2.5, ndl=25.0),
        ]
        output = "\n".join(lines)
        result = runner._parse_output(output)

        assert result.min_ndl == pytest.approx(18.5)


# ============================================================================
# TestParseOutputGFAdjustment
# ============================================================================

class TestParseOutputGFAdjustment:
    """Test GF adjustment in _build_result() via _parse_output()."""

    def test_gf_standard_uses_raw_ceilings(self):
        """GF 100/100 uses raw ceiling/ndl from output."""
        runner_standard = BuhlmannRunner(gf=GF_DEFAULT, use_python_engine=True)
        output = _make_output_line(time=10.0, pressure=3.0, ceiling=1.5, ndl=20.0)
        result = runner_standard._parse_output(output)

        # For GF 100/100, the ceiling and NDL should be close to raw values
        # (exact match may vary due to supersaturation calculation)
        assert result.max_ceiling == pytest.approx(1.5)
        assert result.min_ndl == pytest.approx(20.0)

    def test_gf_conservative_recalculates_ceilings(self):
        """GF 70/85 produces different ceilings than raw output."""
        runner_standard = BuhlmannRunner(gf=GF_DEFAULT, use_python_engine=True)
        runner_conservative = BuhlmannRunner(
            gf=GradientFactors(0.7, 0.85), use_python_engine=True
        )

        # Create realistic output with tissue loading
        output = _make_output_line(
            time=20.0, pressure=3.0, n2_base=1.2, ceiling=1.5, ndl=20.0
        )

        result_std = runner_standard._parse_output(output)
        result_cons = runner_conservative._parse_output(output)

        # Conservative GF should produce different ceiling/ndl
        # (We can't predict exact values, just verify they differ)
        assert result_std.max_ceiling != pytest.approx(result_cons.max_ceiling)

    def test_gf_conservative_deeper_ceiling(self):
        """GF <100 produces higher max_ceiling (deeper stops) when tissue is loaded."""
        runner_standard = BuhlmannRunner(gf=GF_DEFAULT, use_python_engine=True)
        runner_conservative = BuhlmannRunner(
            gf=GradientFactors(0.7, 0.85), use_python_engine=True
        )

        # High tissue loading to ensure ceiling differences
        # Use higher n2_base values to ensure supersaturation
        output = _make_output_line(
            time=25.0, pressure=2.5, n2_base=2.2, ceiling=2.0, ndl=10.0
        )

        result_std = runner_standard._parse_output(output)
        result_cons = runner_conservative._parse_output(output)

        # Conservative should have deeper (higher pressure) ceiling when tissue is loaded
        # Both should have ceilings, but conservative should be higher
        assert result_cons.max_ceiling >= result_std.max_ceiling

    def test_gf_conservative_shorter_ndl(self):
        """gf_high <1.0 produces shorter min_ndl when tissue is already loaded."""
        runner_standard = BuhlmannRunner(gf=GF_DEFAULT, use_python_engine=True)
        runner_conservative = BuhlmannRunner(
            gf=GradientFactors(0.7, 0.85), use_python_engine=True
        )

        # Higher tissue loading to ensure NDL is affected
        # At 3.0 bar (20m), with tissue tensions starting at 1.5 bar
        output = _make_output_line(
            time=15.0, pressure=3.0, n2_base=1.5, ceiling=1.2, ndl=25.0
        )

        result_std = runner_standard._parse_output(output)
        result_cons = runner_conservative._parse_output(output)

        # Conservative GF limits M-value more, so NDL should be shorter or equal
        assert result_cons.min_ndl <= result_std.min_ndl


# ============================================================================
# TestBuhlmannRunnerIntegration
# ============================================================================

class TestBuhlmannRunnerIntegration:
    """Integration tests with real libbuhlmann binary."""

    @skip_no_binary
    def test_run_square_30m_20min_valid_result(self):
        """Real binary produces valid result with 16 compartments."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=20)

        runner = BuhlmannRunner(use_python_engine=False)
        result = runner.run(profile)

        # Validate structure
        assert len(result.times) > 0
        assert len(result.pressures) > 0
        assert len(result.compartment_n2) > 0
        assert len(result.compartment_he) > 0
        assert len(result.compartment_n2[0]) == 16
        assert len(result.compartment_he[0]) == 16

        # Validate metrics are reasonable
        assert result.max_ceiling >= 1.0  # Should have some ceiling
        assert 0 <= result.min_ndl <= 100  # NDL capped at 100
        assert result.max_supersaturation >= 0

    @skip_no_binary
    def test_run_shallow_dive_no_deco(self):
        """10m/10min: requires_deco=False, risk_score < 1.0 (safe)."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=10, bottom_time=10)

        runner = BuhlmannRunner(use_python_engine=False)
        result = runner.run(profile)

        assert result.requires_deco is False
        # Risk score should be below 1.0 (not exceeding M-value)
        assert result.risk_score < 1.0

    @skip_no_binary
    def test_run_with_gf_more_conservative(self):
        """GF 70/85 produces lower NDL than 100/100."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=15)

        runner_std = BuhlmannRunner(gf=GF_DEFAULT, use_python_engine=False)
        runner_cons = BuhlmannRunner(
            gf=GradientFactors(0.7, 0.85), use_python_engine=False
        )

        result_std = runner_std.run(profile)
        result_cons = runner_cons.run(profile)

        # Conservative GF should produce shorter NDL
        assert result_cons.min_ndl <= result_std.min_ndl

    @skip_no_binary
    def test_run_batch_processes_all(self):
        """3 profiles -> 3 results."""
        gen = ProfileGenerator()
        profiles = [
            gen.generate_square(depth=15, bottom_time=10),
            gen.generate_square(depth=20, bottom_time=15),
            gen.generate_square(depth=25, bottom_time=20),
        ]

        runner = BuhlmannRunner(use_python_engine=False)
        results = runner.run_batch(profiles)

        assert len(results) == 3
        # All should succeed (no None values)
        assert all(r is not None for r in results)

    @skip_no_binary
    def test_run_result_compartment_count(self):
        """len(compartment_n2[0]) == 16."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=12)

        runner = BuhlmannRunner(use_python_engine=False)
        result = runner.run(profile)

        assert len(result.compartment_n2[0]) == 16
        assert len(result.compartment_he[0]) == 16
