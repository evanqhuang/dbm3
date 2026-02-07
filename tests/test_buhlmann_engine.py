"""Tests for the pure Python Bühlmann ZH-L16 tissue simulation engine."""

import numpy as np

from backtest.buhlmann_constants import (
    alveolar_pressure,
    schreiner_vec,
    haldane_vec,
    WATER_VAPOR_PRESSURE,
    SURFACE_N2_FRACTION,
    ZH_L16_N2_HALFTIMES,
)
from backtest.buhlmann_engine import BuhlmannEngine
from backtest.profile_generator import ProfileGenerator


class TestAlveolarPressure:
    """Verify ventilation formula against hand-computed values."""

    def test_air_at_surface(self):
        """N2 alveolar pressure in air at 1 bar, RQ=1.0."""
        # (1.0 - 0.0627) * 0.78084 = 0.9373 * 0.78084
        expected = (1.0 - WATER_VAPOR_PRESSURE) * SURFACE_N2_FRACTION
        result = alveolar_pressure(1.0, SURFACE_N2_FRACTION)
        assert abs(result - expected) < 1e-10
        assert abs(result - 0.7317) < 0.001  # sanity check

    def test_air_at_30m(self):
        """N2 alveolar pressure at 4 bar (30m)."""
        expected = (4.0 - WATER_VAPOR_PRESSURE) * SURFACE_N2_FRACTION
        result = alveolar_pressure(4.0, SURFACE_N2_FRACTION)
        assert abs(result - expected) < 1e-10

    def test_zero_fraction(self):
        """Zero gas fraction should give zero alveolar pressure."""
        result = alveolar_pressure(4.0, 0.0)
        assert result == 0.0

    def test_scales_linearly_with_pressure(self):
        """Doubling ambient pressure should roughly double alveolar pressure."""
        p1 = alveolar_pressure(2.0, 0.79)
        p2 = alveolar_pressure(4.0, 0.79)
        # Not exactly 2x because of WVP subtraction, but close
        ratio = p2 / p1
        assert 1.9 < ratio < 2.1

    def test_helium_fraction(self):
        """He alveolar pressure for trimix."""
        # Trimix 18/45: fHe=0.45
        result = alveolar_pressure(4.0, 0.45)
        expected = (4.0 - WATER_VAPOR_PRESSURE) * 0.45
        assert abs(result - expected) < 1e-10


class TestSchreinerEquation:
    """Verify Schreiner equation against known properties."""

    def test_zero_rate_equals_haldane(self):
        """When rate=0, Schreiner should reduce to Haldane."""
        pt0 = np.array([0.75, 0.80, 0.85])
        palv0 = 2.5
        t = 10.0
        k = np.log(2) / np.array([4.0, 8.0, 12.5])

        s_result = schreiner_vec(pt0, palv0, 0.0, t, k)
        h_result = haldane_vec(pt0, palv0, t, k)
        np.testing.assert_allclose(s_result, h_result, atol=1e-10)

    def test_tissue_approaches_ambient(self):
        """After long time at constant pressure, tissue should approach ambient."""
        pt0 = np.array([0.75] * 16)
        palv0 = 3.0  # ambient inert gas pressure
        t = 10000.0  # very long time
        k = np.log(2) / np.array(ZH_L16_N2_HALFTIMES)

        result = haldane_vec(pt0, palv0, t, k)
        # After ~10000 min (~15.7 half-lives for slowest 635-min compartment)
        np.testing.assert_allclose(result, palv0, atol=1e-4)

    def test_fast_compartment_loads_faster(self):
        """Fast compartment (short halftime) should load gas faster."""
        pt0 = np.array([0.75, 0.75])
        palv0 = 3.0
        t = 5.0
        k = np.log(2) / np.array([4.0, 635.0])  # fast vs slow

        result = haldane_vec(pt0, palv0, t, k)
        # Fast compartment should be closer to ambient
        assert result[0] > result[1]
        assert result[0] > pt0[0]  # both should increase
        assert result[1] > pt0[1]

    def test_descent_increases_tissue_loading(self):
        """During descent (positive rate), tissues should load more than at constant."""
        pt0 = np.array([0.75] * 16)
        palv0 = 2.0
        rate = 0.5  # bar/min descent rate
        t = 5.0
        k = np.log(2) / np.array(ZH_L16_N2_HALFTIMES)

        constant = haldane_vec(pt0, palv0, t, k)
        descending = schreiner_vec(pt0, palv0, rate, t, k)

        # Descending should produce higher tissue loading than constant
        # because ambient pressure is increasing during the interval
        assert np.all(descending >= constant)


class TestEngineSimulation:
    """Test BuhlmannEngine simulation with real dive profiles."""

    def test_square_profile_returns_correct_shape(self):
        """Engine output should have correct shape and keys."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        assert 'times' in raw
        assert 'pressures' in raw
        assert 'compartment_n2' in raw
        assert 'compartment_he' in raw
        assert 'ceilings' in raw
        assert 'ndl_times' in raw

        n_steps = len(raw['times'])
        assert n_steps > 0
        assert len(raw['pressures']) == n_steps
        assert len(raw['compartment_n2']) == n_steps
        assert len(raw['compartment_he']) == n_steps
        assert len(raw['ceilings']) == n_steps
        assert len(raw['ndl_times']) == n_steps

        # Each timestep should have 16 compartments
        assert len(raw['compartment_n2'][0]) == 16
        assert len(raw['compartment_he'][0]) == 16

    def test_tissue_loading_increases_at_depth(self):
        """N2 tissue loading should increase while at depth."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=20)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        n2 = raw['compartment_n2']
        # Find timesteps at bottom (pressure ~= 4.0 bar)
        bottom_indices = [
            i for i, p in enumerate(raw['pressures'])
            if abs(p - 4.0) < 0.1
        ]
        assert len(bottom_indices) > 2

        # Fast compartment (0) should show increasing N2 during bottom time
        first_bottom = n2[bottom_indices[0]][0]
        last_bottom = n2[bottom_indices[-1]][0]
        assert last_bottom > first_bottom, \
            f"Fast compartment should load gas: {first_bottom} -> {last_bottom}"

    def test_he_stays_zero_for_air(self):
        """He tissue loading should stay zero for air dives (fHe=0)."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=10)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        for he_step in raw['compartment_he']:
            for he_val in he_step:
                assert abs(he_val) < 1e-10, f"He should be zero for air: {he_val}"

    def test_ndl_decreases_at_depth(self):
        """NDL should decrease while spending time at depth."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=25)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        # Find NDL values at bottom
        bottom_indices = [
            i for i, p in enumerate(raw['pressures'])
            if abs(p - 4.0) < 0.1
        ]
        if len(bottom_indices) > 2:
            first_ndl = raw['ndl_times'][bottom_indices[0]]
            last_ndl = raw['ndl_times'][bottom_indices[-1]]
            assert last_ndl < first_ndl, \
                f"NDL should decrease at depth: {first_ndl} -> {last_ndl}"

    def test_initial_tissue_state(self):
        """Initial tissue should be at surface N2 equilibrium."""
        gen = ProfileGenerator()
        # Very short profile to check initial state
        profile = gen.generate_square(depth=0.1, bottom_time=0.01)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        expected_n2 = alveolar_pressure(1.0, SURFACE_N2_FRACTION)
        first_n2 = raw['compartment_n2'][0]
        for c in range(16):
            assert abs(first_n2[c] - expected_n2) < 0.01, \
                f"Compartment {c} initial N2 should be ~{expected_n2:.4f}, got {first_n2[c]:.4f}"

    def test_ceiling_reasonable_at_30m(self):
        """Ceiling after 30m/25min should be meaningful (requires deco)."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=25)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        max_ceil = max(raw['ceilings'])
        # After 25min at 30m, some ceiling is expected
        assert max_ceil > 0.5, f"Expected some ceiling after 30m/25min, got {max_ceil}"


class TestNDLReasonableness:
    """Verify NDL values are physically reasonable."""

    def test_ndl_30m_air(self):
        """NDL at 30m on air should be in the 15-30 minute range."""
        engine = BuhlmannEngine()
        # Simulate initial state (surface equilibrium)
        n2_p = np.full(16, alveolar_pressure(1.0, SURFACE_N2_FRACTION))
        he_p = np.zeros(16)
        ndl = engine._compute_ndl(n2_p, he_p, 4.0, 0.79)
        assert 10.0 < ndl < 40.0, f"NDL at 30m should be ~20min, got {ndl:.1f}"

    def test_ndl_at_surface_is_capped(self):
        """NDL at surface should be 100 (capped)."""
        engine = BuhlmannEngine()
        n2_p = np.full(16, alveolar_pressure(1.0, SURFACE_N2_FRACTION))
        he_p = np.zeros(16)
        ndl = engine._compute_ndl(n2_p, he_p, 1.01325, 0.79)
        assert ndl == 100.0

    def test_ndl_deeper_is_shorter(self):
        """NDL should be shorter at deeper depths."""
        engine = BuhlmannEngine()
        n2_p = np.full(16, alveolar_pressure(1.0, SURFACE_N2_FRACTION))
        he_p = np.zeros(16)
        ndl_20m = engine._compute_ndl(n2_p, he_p, 3.0, 0.79)
        ndl_30m = engine._compute_ndl(n2_p, he_p, 4.0, 0.79)
        ndl_40m = engine._compute_ndl(n2_p, he_p, 5.0, 0.79)
        assert ndl_20m > ndl_30m > ndl_40m

    def test_ndl_zero_when_exceeded(self):
        """NDL should be 0 when tissues already exceed M-value."""
        engine = BuhlmannEngine()
        n2_p = np.full(16, 4.0)  # heavily loaded
        he_p = np.zeros(16)
        ndl = engine._compute_ndl(n2_p, he_p, 4.0, 0.79)
        assert ndl == 0.0


class TestCeilingConsistency:
    """Verify ceiling calculations are consistent."""

    def test_ceiling_increases_with_tissue_loading(self):
        """Higher tissue loading should produce higher ceiling."""
        engine = BuhlmannEngine()
        low = engine._compute_ceiling(np.full(16, 1.0), np.zeros(16))
        high = engine._compute_ceiling(np.full(16, 2.5), np.zeros(16))
        assert high > low

    def test_ceiling_separate_n2_he(self):
        """Ceiling should consider N2 and He separately (matching C getCeiling)."""
        engine = BuhlmannEngine()
        # Only N2 loaded
        ceil_n2 = engine._compute_ceiling(np.full(16, 2.0), np.zeros(16))
        # Only He loaded (same pressure)
        ceil_he = engine._compute_ceiling(np.zeros(16), np.full(16, 2.0))
        # Both should produce meaningful ceilings
        assert ceil_n2 > 0
        assert ceil_he > 0


class TestTrimixProfile:
    """Test engine with non-zero helium fraction."""

    def test_trimix_he_loading(self):
        """He compartments should load when fHe > 0."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=50, bottom_time=15, fO2=0.18, fHe=0.45)
        engine = BuhlmannEngine()
        raw = engine.simulate(profile)

        # He should be non-zero at depth
        bottom_indices = [
            i for i, p in enumerate(raw['pressures'])
            if abs(p - 6.0) < 0.1
        ]
        if bottom_indices:
            last_he = raw['compartment_he'][bottom_indices[-1]]
            assert any(h > 0.01 for h in last_he), \
                f"He should be loaded for trimix: max={max(last_he)}"

    def test_trimix_reduces_n2_loading(self):
        """Trimix should have less N2 loading than air at same depth."""
        gen = ProfileGenerator()
        depth, time = 40, 15
        profile_air = gen.generate_square(depth=depth, bottom_time=time)
        profile_tmx = gen.generate_square(depth=depth, bottom_time=time,
                                           fO2=0.21, fHe=0.35)
        engine = BuhlmannEngine()
        raw_air = engine.simulate(profile_air)
        raw_tmx = engine.simulate(profile_tmx)

        # Compare N2 loading at end of bottom time
        last_air_n2 = raw_air['compartment_n2'][-1][0]  # fast compartment
        last_tmx_n2 = raw_tmx['compartment_n2'][-1][0]
        assert last_tmx_n2 < last_air_n2, \
            f"Trimix N2 ({last_tmx_n2:.3f}) should be less than air N2 ({last_air_n2:.3f})"
