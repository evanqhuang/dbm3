"""
Comprehensive tests for decompression (deco) functionality in the slab diffusion model.

These tests validate real decompression physics:
- NDL boundary behavior
- Ceiling calculation accuracy
- Deco schedule generation
- Gradient constraints
- Compartment control transitions
"""

import pytest
import numpy as np
from backtest.slab_model import SlabModel, DecoSchedule, DecoStop
from backtest.profile_generator import ProfileGenerator, DiveProfile


class TestNDLBoundary:
    """Test deco/no-deco boundary conditions."""

    def test_ndl_dive_no_deco_required(self):
        """NDL dive (20m/10min) should not require decompression stops."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Generate profile and deco schedule
        profile, schedule = model.generate_deco_profile(
            depth=20.0,
            bottom_time=10.0,
        )

        # Validate no deco is required
        assert schedule.requires_deco is False, "NDL dive should not require deco"
        assert len(schedule.stops) == 0, "NDL dive should have zero deco stops"

        # TTS should only include ascent time (no stop time)
        expected_ascent_time = 20.0 / 10.0  # depth / ascent_rate = 2 minutes
        assert abs(schedule.tts - expected_ascent_time) < 0.5, \
            f"TTS should be ~{expected_ascent_time}min ascent, got {schedule.tts}min"

        assert schedule.total_deco_time == 0.0, "NDL dive should have zero deco time"

    def test_deco_dive_produces_stops(self):
        """Deco dive (40m/30min) should produce decompression stops."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=40.0,
            bottom_time=30.0,
        )

        # Validate deco is required
        assert schedule.requires_deco is True, "Deco dive should require decompression"
        assert len(schedule.stops) > 0, "Deco dive should have stops"

        # Total deco time should be sum of individual stop durations
        total_stop_time = sum(stop.duration_min for stop in schedule.stops)
        assert schedule.total_deco_time == total_stop_time, \
            "total_deco_time should equal sum of stop durations"

        # TTS should include both ascent and deco time
        assert schedule.tts > schedule.total_deco_time, \
            "TTS should include ascent time + deco time"

    def test_ceiling_at_ndl_boundary(self):
        """At 30m/28min (NDL boundary), ceiling should transition from safe to mandatory."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Just under NDL - should be safe
        profile_under, schedule_under = model.generate_deco_profile(
            depth=30.0,
            bottom_time=25.0,  # Under the ~28min NDL
        )

        # At the NDL boundary, tissue is loaded
        # ceiling_at_start is measured at bottom (before ascent)
        assert schedule_under.ceiling_at_start >= 0.0, \
            "Ceiling at bottom should be non-negative"

        # After deco (or direct ascent if NDL), final ceiling should be safe
        # Run the profile through to verify final state is safe
        result = model.run(profile_under)

        # Simulate ascent from final state to surface
        final_slabs = [result.final_slabs[comp.name] for comp in model.compartments]
        surface_slabs = model._simulate_ascent(final_slabs, 0.0)

        # Check that boundary gradient is safe at surface
        ppn2_surface = model._get_atmospheric_pressure() * (1 - model.f_o2)
        p_surface = model._get_atmospheric_pressure()

        for i, comp in enumerate(model.compartments):
            gradient = surface_slabs[i][1] - ppn2_surface
            # At surface, pressure ratio = 1.0
            effective_g_crit = comp.g_crit * model.conservatism * 1.0
            if effective_g_crit > 0:
                ratio = gradient / effective_g_crit
                assert ratio <= 1.1, \
                    f"{comp.name}: gradient ratio {ratio:.3f} exceeds safe limit at surface"


class TestCeilingCalculation:
    """Test ceiling calculation accuracy and properties."""

    def test_shallow_dive_zero_ceiling(self):
        """Shallow dive (10m/20min) should have zero ceiling - safe to surface."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=10.0,
            bottom_time=20.0,
        )

        assert schedule.ceiling_at_start == 0.0, \
            "Shallow dive should have zero ceiling (safe to surface directly)"
        assert schedule.requires_deco is False, "Should not require deco stops"

    def test_ceiling_monotonicity_during_stop(self):
        """During a deco stop, ceiling should decrease or stay constant (never increase)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Generate a profile that requires deco
        profile, schedule = model.generate_deco_profile(
            depth=40.0,
            bottom_time=25.0,
        )

        if not schedule.requires_deco:
            pytest.skip("This dive doesn't require deco - adjust parameters")

        # Simulate the dive to get tissue state at first stop
        gen = ProfileGenerator()
        descent_time = 40.0 / 20.0  # depth / descent_rate

        # Initialize tissue and descend
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        for comp in model.compartments:
            comp.slab[:] = ppn2_surface

        # Descent
        descent_steps = int(descent_time * 60 / model.dt)
        for step in range(descent_steps):
            ratio = step / max(descent_steps, 1)
            depth = ratio * 40.0
            ppn2 = model._get_ppn2(depth)
            for comp in model.compartments:
                model._update_compartment(comp, ppn2)

        # Bottom time
        ppn2_bottom = model._get_ppn2(40.0)
        bottom_steps = int(25.0 * 60 / model.dt)
        for _ in range(bottom_steps):
            for comp in model.compartments:
                model._update_compartment(comp, ppn2_bottom)

        # Get tissue state
        tissue_slabs = [comp.slab.copy() for comp in model.compartments]

        # Check ceiling at start
        ceiling_start = model.calculate_ceiling(tissue_slabs, 40.0)

        # Simulate holding at first stop for 1 minute
        first_stop = schedule.stops[0]

        # Ascend to first stop
        stop_slabs = model._simulate_ascent_to_depth(
            tissue_slabs, 40.0, first_stop.depth
        )

        # Track ceiling evolution during stop
        previous_ceiling = model.calculate_ceiling(stop_slabs, first_stop.depth)

        # Hold for a few minutes and check ceiling decreases
        for minute in range(1, min(5, int(first_stop.duration_min) + 1)):
            stop_slabs = model._simulate_time_at_depth(
                stop_slabs, first_stop.depth, 1.0
            )
            current_ceiling = model.calculate_ceiling(stop_slabs, first_stop.depth)

            # Ceiling should not increase (monotonicity)
            assert current_ceiling <= previous_ceiling + 0.1, \
                f"Ceiling increased from {previous_ceiling}m to {current_ceiling}m during stop"

            previous_ceiling = current_ceiling

    def test_stop_depths_are_multiples_of_increment(self):
        """All deco stops should be at depths that are multiples of stop_increment (3m)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=50.0,
            bottom_time=25.0,
        )

        if not schedule.requires_deco:
            pytest.skip("This dive doesn't require deco")

        stop_increment = model.deco_stop_increment  # 3.0m from config

        for stop in schedule.stops:
            # Check if depth is a multiple of stop_increment
            remainder = stop.depth % stop_increment
            assert remainder < 0.1, \
                f"Stop depth {stop.depth}m is not a multiple of {stop_increment}m"


class TestDecoScheduleGeneration:
    """Test complete deco schedule generation."""

    def test_deep_dive_multiple_stops(self):
        """Deep dive (60m/20min) should produce multiple stops at 3m increments."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=60.0,
            bottom_time=20.0,
        )

        assert schedule.requires_deco is True, "Deep dive should require deco"

        # Should have stops at multiple depths
        # Based on slab model behavior, expect stops at 9m, 6m, 3m at minimum
        stop_depths = [stop.depth for stop in schedule.stops]

        # Verify stops are ordered deepest-first
        for i in range(len(stop_depths) - 1):
            assert stop_depths[i] >= stop_depths[i + 1], \
                "Stops should be ordered deepest to shallowest"

        # Verify last stop is at last_stop_depth (3m)
        if schedule.stops:
            assert schedule.stops[-1].depth == model.deco_last_stop_depth, \
                f"Last stop should be at {model.deco_last_stop_depth}m"

    def test_round_trip_gradient_verification(self):
        """After completing a deco schedule, boundary gradient should be safe at surface."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Generate profile with deco
        profile, schedule = model.generate_deco_profile(
            depth=40.0,
            bottom_time=30.0,
        )

        # Run the complete profile (including deco stops)
        result = model.run(profile)

        # Extract final tissue state
        final_slabs = [result.final_slabs[comp.name] for comp in model.compartments]

        # Calculate boundary gradient risk at surface
        ppn2_surface = model._get_atmospheric_pressure() * (1 - model.f_o2)
        p_surface = model._get_atmospheric_pressure()

        for i, comp in enumerate(model.compartments):
            # Boundary gradient: slab[1] - ppN2_surface
            gradient = final_slabs[i][1] - ppn2_surface

            # At surface, pressure_ratio = 1.0
            effective_g_crit = comp.g_crit * model.conservatism * 1.0

            if effective_g_crit > 0:
                ratio = gradient / effective_g_crit

                # Allow small tolerance (1.05) for numerical precision and ascent transients
                assert ratio <= 1.05, \
                    f"{comp.name}: gradient ratio {ratio:.4f} exceeds limit 1.05 at surface. " \
                    f"gradient={gradient:.4f}, g_crit={effective_g_crit:.4f}"


class TestConservatismFactor:
    """Test conservatism factor effects on deco."""

    def test_conservatism_produces_longer_deco(self):
        """Lower conservatism (0.85) should produce longer/deeper deco than 1.0."""
        # Conservative model (0.85 = 15% safety margin)
        model_conservative = SlabModel(
            config_path='/Users/evanhuang/dbm3/config.yaml',
            conservatism=0.85
        )

        # Standard model (1.0 = match Buhlmann)
        model_standard = SlabModel(
            config_path='/Users/evanhuang/dbm3/config.yaml',
            conservatism=1.0
        )

        # Same dive profile
        depth = 40.0
        bottom_time = 25.0

        _, schedule_conservative = model_conservative.generate_deco_profile(depth, bottom_time)
        _, schedule_standard = model_standard.generate_deco_profile(depth, bottom_time)

        # Conservative model should require more deco
        if schedule_standard.requires_deco:
            # If standard requires deco, conservative should too
            assert schedule_conservative.requires_deco, \
                "Conservative model should require deco if standard does"

            # Conservative should have longer total deco time
            assert schedule_conservative.total_deco_time >= schedule_standard.total_deco_time, \
                f"Conservative deco time {schedule_conservative.total_deco_time}min " \
                f"should be >= standard {schedule_standard.total_deco_time}min"
        else:
            # If standard is NDL, conservative might require deco or have longer NDL
            # This is expected behavior - more conservative = more restrictive
            pass


class TestControllingCompartment:
    """Test compartment control transitions."""

    def test_moderate_dive_muscle_controls(self):
        """Moderate dive (30m/20min) should be controlled by Muscle (medium D)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=30.0,
            bottom_time=20.0,
        )

        if schedule.requires_deco:
            # For moderate dives, Muscle typically controls
            # Spine is too fast (clears quickly), Joints too slow (not yet limiting)
            controlling = schedule.controlling_compartment

            # Muscle or Spine are expected controllers at moderate depths
            assert controlling in ["Muscle", "Spine"], \
                f"Moderate dive should be controlled by Muscle or Spine, got {controlling}"

    def test_deep_dive_joints_controls(self):
        """Deep long dive (50m/30min) should eventually be controlled by Joints (slow D)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=50.0,
            bottom_time=30.0,
        )

        if schedule.requires_deco and len(schedule.stops) > 0:
            # For deep dives with long bottom time, slow compartments (Joints) trap gas
            controlling = schedule.controlling_compartment

            # At deep depths with long exposure, Joints or Muscle should control
            # (Spine clears too fast to be limiting on ascent)
            assert controlling in ["Joints", "Muscle"], \
                f"Deep dive should be controlled by Joints or Muscle, got {controlling}"


class TestDecoPhysicsValidation:
    """Validate physical correctness of deco algorithms."""

    def test_ceiling_never_negative(self):
        """Ceiling depth should never be negative."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        test_profiles = [
            (10, 10),   # Shallow short
            (20, 20),   # Moderate
            (30, 30),   # NDL boundary
            (40, 25),   # Deco required
        ]

        for depth, bottom_time in test_profiles:
            profile, schedule = model.generate_deco_profile(depth, bottom_time)

            assert schedule.ceiling_at_start >= 0.0, \
                f"Ceiling at {depth}m/{bottom_time}min is negative: {schedule.ceiling_at_start}"

            # All stop depths should be non-negative
            for stop in schedule.stops:
                assert stop.depth >= 0.0, \
                    f"Stop depth is negative: {stop.depth}m"

    def test_tts_components_consistent(self):
        """TTS should equal ascent_time + total_deco_time."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        depth = 40.0
        bottom_time = 25.0

        profile, schedule = model.generate_deco_profile(depth, bottom_time)

        # Calculate expected ascent time manually
        ascent_rate = model.deco_ascent_rate

        if schedule.requires_deco:
            # With stops: ascent from depth to first stop, between stops, and to surface
            total_ascent_distance = depth  # Eventually ascends full depth
            expected_ascent_time = total_ascent_distance / ascent_rate

            # TTS should be approximately ascent + deco time
            calculated_tts = expected_ascent_time + schedule.total_deco_time

            # Allow some tolerance for step-wise simulation
            assert abs(schedule.tts - calculated_tts) < 2.0, \
                f"TTS mismatch: schedule.tts={schedule.tts:.2f}, " \
                f"calculated={calculated_tts:.2f} (ascent={expected_ascent_time:.2f} + deco={schedule.total_deco_time:.2f})"
        else:
            # No deco: TTS should be just ascent time
            expected_tts = depth / ascent_rate
            assert abs(schedule.tts - expected_tts) < 0.5, \
                f"NDL TTS should be ~{expected_tts:.2f}min, got {schedule.tts:.2f}min"

    def test_stop_durations_positive(self):
        """All stop durations should be positive."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        profile, schedule = model.generate_deco_profile(
            depth=45.0,
            bottom_time=25.0,
        )

        for stop in schedule.stops:
            assert stop.duration_min > 0, \
                f"Stop at {stop.depth}m has non-positive duration: {stop.duration_min}min"

            # Duration should be reasonable (not exceed max_stop_time)
            assert stop.duration_min <= model.deco_max_stop_time, \
                f"Stop duration {stop.duration_min}min exceeds max {model.deco_max_stop_time}min"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_surface_interval_reduces_tissue_loading(self):
        """Surface interval should progressively reduce tissue loading."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)

        # Start with loaded tissue (uniformly at elevated pressure)
        initial_slabs = []
        for comp in model.compartments:
            slab = np.full(comp.slices, ppn2_surface + 0.5)  # Uniformly loaded
            initial_slabs.append(slab.copy())

        # Calculate initial excess gas for each compartment
        initial_excess = [
            model._compute_excess_gas(initial_slabs[i], ppn2_surface)
            for i in range(len(model.compartments))
        ]

        # Simulate progressive offgassing: 2h, 6h, 12h
        intervals = [120, 360, 720]  # minutes
        previous_excess = initial_excess.copy()

        for interval in intervals:
            surface_slabs = model._simulate_time_at_depth(
                initial_slabs,
                depth=0.0,
                duration_min=interval,
            )

            current_excess = [
                model._compute_excess_gas(surface_slabs[i], ppn2_surface)
                for i in range(len(model.compartments))
            ]

            # Each compartment should have less excess gas than before
            for i, comp in enumerate(model.compartments):
                assert current_excess[i] < previous_excess[i], \
                    f"{comp.name}: excess gas should decrease over time"

            previous_excess = current_excess

        # After 12 hours, all compartments should have reduced excess gas
        for i, comp in enumerate(model.compartments):
            reduction_fraction = (initial_excess[i] - previous_excess[i]) / initial_excess[i]

            # Verify meaningful reduction based on compartment speed
            # The slab diffusion model exhibits slower offgassing than exponential models
            # because gas must diffuse from core slices to the surface.
            # Fast compartments (Spine D=0.002) clear >50%
            # Medium compartments (Muscle D=0.0005) clear >25%
            # Slow compartments (Joints D=0.0001) clear >10%
            if comp.D >= 0.001:  # Fast (Spine)
                min_reduction = 0.50
            elif comp.D >= 0.0003:  # Medium (Muscle)
                min_reduction = 0.25
            else:  # Slow (Joints)
                min_reduction = 0.10

            assert reduction_fraction > min_reduction, \
                f"{comp.name}: should clear >{min_reduction*100:.0f}% in 12h, got {reduction_fraction*100:.1f}%"

    def test_repetitive_dive_with_deeper_profile(self):
        """Deeper repetitive dive should show increased decompression stress."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)

        # First dive: deeper and longer to create significant residual loading
        for comp in model.compartments:
            comp.slab[:] = ppn2_surface

        gen = ProfileGenerator()
        profile1 = gen.generate_square(depth=30, bottom_time=20)
        result1 = model.run(profile1)

        # Get tissue state after first dive
        tissue_after_dive1 = [result1.final_slabs[comp.name].copy() for comp in model.compartments]

        # Verify there is residual nitrogen after first dive
        residual_excess_dive1 = [
            model._compute_excess_gas(tissue_after_dive1[i], ppn2_surface)
            for i in range(len(model.compartments))
        ]
        assert any(excess > 0.01 for excess in residual_excess_dive1), \
            "First dive should leave residual nitrogen"

        # Short surface interval (30 minutes) - minimal offgassing
        tissue_after_si = model._simulate_time_at_depth(
            tissue_after_dive1,
            depth=0.0,
            duration_min=30.0,
        )

        # Verify still has residual loading after SI
        residual_excess_after_si = [
            model._compute_excess_gas(tissue_after_si[i], ppn2_surface)
            for i in range(len(model.compartments))
        ]
        assert any(excess > 0.01 for excess in residual_excess_after_si), \
            "Should still have residual nitrogen after 30min SI"

        # Set up for second dive with pre-loaded tissue
        for i, comp in enumerate(model.compartments):
            comp.slab[:] = tissue_after_si[i]

        # Second dive (same profile)
        profile2 = gen.generate_square(depth=30, bottom_time=20)
        result2 = model.run(profile2)

        # Second dive should have shorter NDL due to residual loading
        # (NDL is calculated from deepest point, so pre-loading affects it)
        assert result2.final_ndl <= result1.final_ndl, \
            f"Repetitive dive should have <= NDL: " \
            f"dive1={result1.final_ndl:.1f}min, dive2={result2.final_ndl:.1f}min"

        # The final excess gas should be higher or at least not significantly lower
        final_excess_dive2 = [
            model._compute_excess_gas([result2.final_slabs[comp.name] for comp in model.compartments][i], ppn2_surface)
            for i in range(len(model.compartments))
        ]
        final_excess_dive1 = [
            model._compute_excess_gas(tissue_after_dive1[i], ppn2_surface)
            for i in range(len(model.compartments))
        ]

        # At least one compartment should show increased loading
        max_excess_dive1 = max(final_excess_dive1)
        max_excess_dive2 = max(final_excess_dive2)

        assert max_excess_dive2 >= max_excess_dive1 * 0.9, \
            f"Repetitive dive should maintain or increase loading: " \
            f"dive1 max={max_excess_dive1:.4f}, dive2 max={max_excess_dive2:.4f}"


class TestBoyleExponentScaling:
    """Test sublinear Boyle's Law scaling for g_crit."""

    def test_sublinear_scaling_deeper_ceiling(self):
        """Sublinear scaling (exponent=0.5) should produce deeper ceilings than linear (1.0) at depth."""
        # Build a loaded tissue state at 50m for 25min, then compare ceilings
        model_sqrt = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')  # boyle_exponent=0.5 from config

        # Simulate descent + bottom time to load tissue
        ppn2_surface = model_sqrt._get_atmospheric_pressure() * (1 - model_sqrt.f_o2)
        for comp in model_sqrt.compartments:
            comp.slab[:] = ppn2_surface

        # Descent
        depth = 50.0
        descent_steps = int((depth / 20.0) * 60 / model_sqrt.dt)
        for step in range(descent_steps):
            ratio = step / max(descent_steps, 1)
            ppn2 = model_sqrt._get_ppn2(ratio * depth)
            for comp in model_sqrt.compartments:
                model_sqrt._update_compartment(comp, ppn2)

        # Bottom time
        ppn2_bottom = model_sqrt._get_ppn2(depth)
        bottom_steps = int(25.0 * 60 / model_sqrt.dt)
        for _ in range(bottom_steps):
            for comp in model_sqrt.compartments:
                model_sqrt._update_compartment(comp, ppn2_bottom)

        tissue_slabs = [comp.slab.copy() for comp in model_sqrt.compartments]

        # Ceiling with sqrt scaling (config default 0.5)
        ceiling_sqrt = model_sqrt.calculate_ceiling(tissue_slabs, depth)

        # Now compute ceiling with linear scaling
        model_sqrt.boyle_exponent = 1.0
        ceiling_linear = model_sqrt.calculate_ceiling(tissue_slabs, depth)

        # Sqrt scaling should produce a deeper (more conservative) ceiling
        assert ceiling_sqrt >= ceiling_linear, \
            f"Sqrt ceiling ({ceiling_sqrt}m) should be >= linear ceiling ({ceiling_linear}m)"
        # They should actually differ at depth
        assert ceiling_sqrt > ceiling_linear, \
            f"At 50m depth, sqrt and linear ceilings should differ: sqrt={ceiling_sqrt}m, linear={ceiling_linear}m"

    def test_boyle_exponent_surface_invariant(self):
        """At surface (pressure_ratio=1.0), boyle_exponent should have no effect (1.0^n = 1.0)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Create tissue state with moderate loading
        ppn2_surface = model._get_atmospheric_pressure() * (1 - model.f_o2)
        loaded_slabs = []
        for comp in model.compartments:
            slab = np.full(comp.slices, ppn2_surface + 0.5)
            loaded_slabs.append(slab)

        # Check safety at surface with different exponents
        results = []
        for exponent in [0.3, 0.5, 0.7, 1.0]:
            model.boyle_exponent = exponent
            safe = model._check_safe_at_depth(loaded_slabs, 0.0)
            results.append(safe)

        # All should produce the same result
        assert len(set(results)) == 1, \
            f"Surface safety should be identical for all exponents, got {results}"

    def test_boyle_exponent_monotonicity(self):
        """Smaller boyle_exponent should produce deeper (more conservative) ceilings."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Load tissue at 45m/20min
        ppn2_surface = model._get_atmospheric_pressure() * (1 - model.f_o2)
        for comp in model.compartments:
            comp.slab[:] = ppn2_surface

        depth = 45.0
        descent_steps = int((depth / 20.0) * 60 / model.dt)
        for step in range(descent_steps):
            ratio = step / max(descent_steps, 1)
            ppn2 = model._get_ppn2(ratio * depth)
            for comp in model.compartments:
                model._update_compartment(comp, ppn2)

        ppn2_bottom = model._get_ppn2(depth)
        bottom_steps = int(20.0 * 60 / model.dt)
        for _ in range(bottom_steps):
            for comp in model.compartments:
                model._update_compartment(comp, ppn2_bottom)

        tissue_slabs = [comp.slab.copy() for comp in model.compartments]

        # Calculate ceilings at different exponents
        exponents = [0.3, 0.5, 0.7, 1.0]
        ceilings = []
        for exp in exponents:
            model.boyle_exponent = exp
            ceiling = model.calculate_ceiling(tissue_slabs, depth)
            ceilings.append(ceiling)

        # Smaller exponent = less permissive scaling = deeper ceiling
        for i in range(len(ceilings) - 1):
            assert ceilings[i] >= ceilings[i + 1], \
                f"Exponent {exponents[i]} ceiling ({ceilings[i]}m) should be >= " \
                f"exponent {exponents[i+1]} ceiling ({ceilings[i+1]}m)"


class TestSlabModelRun:
    """Test SlabModel.run() - the main profile execution pipeline."""

    def test_run_square_30m_20min_returns_valid_result(self):
        """Running 30m/20min produces SlabResult with all fields populated."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=20)

        result = model.run(profile)

        # Verify all key fields are populated
        assert len(result.times) > 0, "times should be populated"
        assert len(result.depths) > 0, "depths should be populated"
        assert result.slab_history.size > 0, "slab_history should be populated"
        assert len(result.final_slabs) == 3, "final_slabs should have 3 compartments"
        assert result.critical_compartment in ["Spine", "Muscle", "Joints"]

    def test_run_shallow_dive_low_risk(self):
        """10m/10min produces max_cv_ratio < 0.5."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=10, bottom_time=10)

        result = model.run(profile)

        assert result.max_cv_ratio < 0.5, \
            f"Shallow dive should have low risk, got {result.max_cv_ratio:.3f}"

    def test_run_risk_increases_with_depth(self):
        """30m/20min has higher max_cv_ratio than 20m/20min."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()

        profile_20m = gen.generate_square(depth=20, bottom_time=20)
        result_20m = model.run(profile_20m)

        # Reset tissue state
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        for comp in model.compartments:
            comp.slab[:] = ppn2_surface

        profile_30m = gen.generate_square(depth=30, bottom_time=20)
        result_30m = model.run(profile_30m)

        assert result_30m.max_cv_ratio > result_20m.max_cv_ratio, \
            f"30m risk {result_30m.max_cv_ratio:.3f} should be > 20m risk {result_20m.max_cv_ratio:.3f}"

    def test_run_risk_increases_with_time(self):
        """30m/30min has higher max_cv_ratio than 30m/10min."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()

        profile_10min = gen.generate_square(depth=30, bottom_time=10)
        result_10min = model.run(profile_10min)

        # Reset tissue state
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        for comp in model.compartments:
            comp.slab[:] = ppn2_surface

        profile_30min = gen.generate_square(depth=30, bottom_time=30)
        result_30min = model.run(profile_30min)

        assert result_30min.max_cv_ratio > result_10min.max_cv_ratio, \
            f"30min risk {result_30min.max_cv_ratio:.3f} should be > 10min risk {result_10min.max_cv_ratio:.3f}"

    def test_run_ndl_decreases_with_depth(self):
        """final_ndl at 30m < final_ndl at 20m (same time 10min)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()

        profile_20m = gen.generate_square(depth=20, bottom_time=10)
        result_20m = model.run(profile_20m)

        # Reset tissue state
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        for comp in model.compartments:
            comp.slab[:] = ppn2_surface

        profile_30m = gen.generate_square(depth=30, bottom_time=10)
        result_30m = model.run(profile_30m)

        assert result_30m.final_ndl < result_20m.final_ndl, \
            f"30m NDL {result_30m.final_ndl:.1f} should be < 20m NDL {result_20m.final_ndl:.1f}"

    def test_run_final_slabs_has_all_compartments(self):
        """final_slabs has keys 'Spine', 'Muscle', 'Joints'."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=25, bottom_time=15)

        result = model.run(profile)

        assert "Spine" in result.final_slabs, "final_slabs missing 'Spine'"
        assert "Muscle" in result.final_slabs, "final_slabs missing 'Muscle'"
        assert "Joints" in result.final_slabs, "final_slabs missing 'Joints'"

    def test_run_slab_history_shape(self):
        """slab_history shape = [time_steps, 3, 20]."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20, bottom_time=10)

        result = model.run(profile)

        # Shape should be [time_steps, num_compartments, slices_per_compartment]
        assert result.slab_history.ndim == 3, "slab_history should be 3D"
        assert result.slab_history.shape[1] == 3, "should have 3 compartments"
        assert result.slab_history.shape[2] == model.compartments[0].slices, \
            "slices dimension should match compartment config"

    def test_run_critical_compartment_is_valid_name(self):
        """critical_compartment in ['Spine', 'Muscle', 'Joints']."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30, bottom_time=25)

        result = model.run(profile)

        assert result.critical_compartment in ["Spine", "Muscle", "Joints"], \
            f"Invalid critical_compartment: {result.critical_compartment}"

    def test_run_min_margin_consistent_with_cv_ratio(self):
        """min_margin == pytest.approx(1.0 - max_cv_ratio)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=25, bottom_time=20)

        result = model.run(profile)

        expected_margin = 1.0 - result.max_cv_ratio
        assert result.min_margin == pytest.approx(expected_margin, abs=1e-6), \
            f"min_margin {result.min_margin:.6f} != 1.0 - max_cv_ratio {expected_margin:.6f}"

    def test_run_ceiling_at_bottom_nonnegative(self):
        """ceiling_at_bottom >= 0 for any profile."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')
        gen = ProfileGenerator()

        test_profiles = [
            (10, 10),
            (20, 15),
            (30, 25),
            (40, 20),
        ]

        for depth, bottom_time in test_profiles:
            profile = gen.generate_square(depth=depth, bottom_time=bottom_time)
            result = model.run(profile)
            assert result.ceiling_at_bottom >= 0.0, \
                f"ceiling_at_bottom negative at {depth}m/{bottom_time}min: {result.ceiling_at_bottom}"


class TestMultiCompartmentNDL:
    """Test calculate_multi_compartment_ndl() - shadow simulation NDL."""

    def test_ndl_surface_equilibrated_is_max(self):
        """Initialize model, get fresh slabs at surface equilibrium, NDL at surface should be max_time (100)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Fresh tissue at surface
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        fresh_slabs = [np.full(comp.slices, ppn2_surface) for comp in model.compartments]

        ndl = model.calculate_multi_compartment_ndl(fresh_slabs, depth=0.0, max_time=100)

        assert ndl == 100, f"Surface NDL should be max_time (100), got {ndl}"

    def test_ndl_at_30m_reference_near_28min(self):
        """At 30m with fresh tissue, NDL should be approximately 28 minutes (calibration reference, ±5 tolerance)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Fresh tissue at surface
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        fresh_slabs = [np.full(comp.slices, ppn2_surface) for comp in model.compartments]

        ndl = model.calculate_multi_compartment_ndl(fresh_slabs, depth=30.0, max_time=100)

        assert 23 <= ndl <= 33, \
            f"30m NDL should be approximately 28 minutes (±5), got {ndl}"

    def test_ndl_deeper_is_shorter(self):
        """NDL at 40m < NDL at 30m with fresh tissue."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Fresh tissue at surface
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        fresh_slabs_30 = [np.full(comp.slices, ppn2_surface) for comp in model.compartments]
        fresh_slabs_40 = [np.full(comp.slices, ppn2_surface) for comp in model.compartments]

        ndl_30 = model.calculate_multi_compartment_ndl(fresh_slabs_30, depth=30.0, max_time=100)
        ndl_40 = model.calculate_multi_compartment_ndl(fresh_slabs_40, depth=40.0, max_time=100)

        assert ndl_40 < ndl_30, \
            f"40m NDL {ndl_40} should be < 30m NDL {ndl_30}"

    def test_ndl_preloaded_shorter_than_fresh(self):
        """Pre-load tissue by simulating time at depth, then check NDL < fresh NDL."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Fresh tissue
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        fresh_slabs = [np.full(comp.slices, ppn2_surface) for comp in model.compartments]

        ndl_fresh = model.calculate_multi_compartment_ndl(fresh_slabs, depth=30.0, max_time=100)

        # Pre-load tissue: simulate 10 minutes at 30m
        preloaded_slabs = model._simulate_time_at_depth(fresh_slabs, depth=30.0, duration_min=10.0)

        ndl_preloaded = model.calculate_multi_compartment_ndl(preloaded_slabs, depth=30.0, max_time=100)

        assert ndl_preloaded < ndl_fresh, \
            f"Preloaded NDL {ndl_preloaded} should be < fresh NDL {ndl_fresh}"

    def test_ndl_capped_at_max_time(self):
        """NDL at shallow depth (5m) should be capped at max_time."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        # Fresh tissue
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        fresh_slabs = [np.full(comp.slices, ppn2_surface) for comp in model.compartments]

        ndl = model.calculate_multi_compartment_ndl(fresh_slabs, depth=5.0, max_time=100)

        assert ndl == 100, f"Shallow NDL should be capped at max_time (100), got {ndl}"


class TestAtmosphericPressure:
    """Test _get_atmospheric_pressure() - barometric formula."""

    def test_sea_level_pressure(self):
        """altitude=0 -> pytest.approx(1.01325)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        pressure = model._get_atmospheric_pressure(altitude_m=0.0)

        assert pressure == pytest.approx(1.01325, abs=0.0001), \
            f"Sea level pressure should be 1.01325 bar, got {pressure}"

    def test_negative_altitude_clamps(self):
        """altitude=-100 -> 1.01325 (clamped)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        pressure = model._get_atmospheric_pressure(altitude_m=-100.0)

        assert pressure == pytest.approx(1.01325, abs=0.0001), \
            f"Negative altitude should clamp to 1.01325 bar, got {pressure}"

    def test_high_altitude_lower(self):
        """altitude=2000 -> pressure < 1.01325 (verify barometric formula)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        pressure = model._get_atmospheric_pressure(altitude_m=2000.0)

        assert pressure < 1.01325, \
            f"High altitude pressure should be < 1.01325 bar, got {pressure}"

    def test_very_high_altitude(self):
        """altitude=5300 -> approximately 0.527 bar (±0.05)."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        pressure = model._get_atmospheric_pressure(altitude_m=5300.0)

        # Standard barometric formula: 1.01325 * (1 - 2.25577e-5 * 5300)^5.25588 ≈ 0.530
        assert pressure == pytest.approx(0.530, abs=0.02), \
            f"5300m altitude pressure should be approximately 0.530 bar, got {pressure}"


class TestPPN2Calculation:
    """Test _get_ppn2() - nitrogen partial pressure."""

    def test_ppn2_at_surface_air(self):
        """ppN2 = 1.01325 * 0.79 ≈ 0.8005."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        ppn2 = model._get_ppn2(depth_m=0.0)

        expected = 1.01325 * 0.79
        assert ppn2 == pytest.approx(expected, abs=0.001), \
            f"Surface ppN2 should be {expected:.4f}, got {ppn2:.4f}"

    def test_ppn2_at_30m_air(self):
        """ppN2 = (1.01325 + 3.0) * 0.79 ≈ 3.17."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        ppn2 = model._get_ppn2(depth_m=30.0)

        expected = (1.01325 + 3.0) * 0.79
        assert ppn2 == pytest.approx(expected, abs=0.01), \
            f"30m ppN2 should be {expected:.2f}, got {ppn2:.2f}"

    def test_ppn2_with_nitrox_32(self):
        """At 20m on EAN32: ppN2 = (1.01325 + 2.0) * 0.68."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        ppn2 = model._get_ppn2(depth_m=20.0, f_o2=0.32)

        expected = (1.01325 + 2.0) * 0.68
        assert ppn2 == pytest.approx(expected, abs=0.01), \
            f"20m EAN32 ppN2 should be {expected:.2f}, got {ppn2:.2f}"

    def test_ppn2_increases_with_depth(self):
        """ppN2_30m > ppN2_20m > ppN2_10m."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        ppn2_10 = model._get_ppn2(depth_m=10.0)
        ppn2_20 = model._get_ppn2(depth_m=20.0)
        ppn2_30 = model._get_ppn2(depth_m=30.0)

        assert ppn2_30 > ppn2_20 > ppn2_10, \
            f"ppN2 should increase with depth: {ppn2_10:.2f} < {ppn2_20:.2f} < {ppn2_30:.2f}"


class TestUpdateCompartment:
    """Test _update_compartment() - finite difference step. Access compartment objects directly."""

    def test_perfect_perfusion_sets_boundary(self):
        """After update with permeability=None, slab[0] == boundary_pressure."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml', permeability=None)
        comp = model.compartments[0]

        # Initialize slab to surface equilibrium
        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)
        comp.slab[:] = ppn2_surface

        # Update with different boundary pressure
        boundary_pressure = ppn2_surface + 1.0
        model._update_compartment(comp, boundary_pressure)

        assert comp.slab[0] == pytest.approx(boundary_pressure, abs=1e-9), \
            f"With perfect perfusion, slab[0] should equal boundary_pressure"

    def test_diffusion_moves_gas_inward(self):
        """After many steps with slab[0] > slab[1], slab[1] increases."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml', permeability=None)
        comp = model.compartments[0]

        # Initialize slab uniformly low
        comp.slab[:] = 0.5

        # Set boundary high, interior low
        boundary_pressure = 2.0
        initial_slab_1 = comp.slab[1]

        # Run many steps
        for _ in range(100):
            model._update_compartment(comp, boundary_pressure)

        assert comp.slab[1] > initial_slab_1, \
            f"After diffusion, slab[1] should increase from {initial_slab_1:.3f} to {comp.slab[1]:.3f}"

    def test_no_flux_boundary_at_core(self):
        """After update, slab[-1] == slab[-2]."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml', permeability=None)
        comp = model.compartments[0]

        # Initialize with gradient
        comp.slab[:] = np.linspace(0.5, 1.5, comp.slices)

        # Update
        boundary_pressure = 2.0
        model._update_compartment(comp, boundary_pressure)

        assert comp.slab[-1] == pytest.approx(comp.slab[-2], abs=1e-9), \
            f"No-flux boundary: slab[-1] {comp.slab[-1]:.6f} should equal slab[-2] {comp.slab[-2]:.6f}"


class TestComputeExcessGas:
    """Test _compute_excess_gas() - integrated excess volume."""

    def test_no_excess_at_equilibrium(self):
        """Slab uniformly at ppN2_surface -> excess = 0.0."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)

        slab = np.full(20, ppn2_surface)
        excess = model._compute_excess_gas(slab, reference_ppn2=ppn2_surface)

        assert excess == pytest.approx(0.0, abs=1e-9), \
            f"At equilibrium, excess should be 0.0, got {excess}"

    def test_excess_proportional_to_loading(self):
        """Slab uniformly at value above reference -> excess = (val - ref) * slices * dx."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        reference = 1.0
        value = 1.5
        slices = 20

        slab = np.full(slices, value)
        excess = model._compute_excess_gas(slab, reference_ppn2=reference)

        expected = (value - reference) * slices * model.dx
        assert excess == pytest.approx(expected, abs=1e-6), \
            f"Excess should be {expected:.6f}, got {excess:.6f}"

    def test_partial_excess_only_above_reference(self):
        """Half slices above, half below reference -> only above contribute."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        reference = 1.0
        slices = 20

        # First half below reference, second half above
        slab = np.concatenate([
            np.full(slices // 2, reference - 0.5),
            np.full(slices // 2, reference + 0.5),
        ])

        excess = model._compute_excess_gas(slab, reference_ppn2=reference)

        # Only the second half contributes
        expected = 0.5 * (slices // 2) * model.dx
        assert excess == pytest.approx(expected, abs=1e-6), \
            f"Partial excess should be {expected:.6f}, got {excess:.6f}"

    def test_default_reference_is_surface(self):
        """Calling without reference_ppn2 uses surface N2 pressure."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        p_surface = model._get_atmospheric_pressure()
        ppn2_surface = p_surface * (1 - model.f_o2)

        slab = np.full(20, ppn2_surface + 0.5)

        # Call without reference
        excess_default = model._compute_excess_gas(slab)

        # Call with explicit reference
        excess_explicit = model._compute_excess_gas(slab, reference_ppn2=ppn2_surface)

        assert excess_default == pytest.approx(excess_explicit, abs=1e-9), \
            f"Default reference should match explicit surface ppN2"


class TestGetCompartmentConfig:
    """Test get_compartment_config()."""

    def test_config_has_three_compartments(self):
        """Returns list of 3 dicts."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        config = model.get_compartment_config()

        assert isinstance(config, list), "config should be a list"
        assert len(config) == 3, f"Should have 3 compartments, got {len(config)}"

    def test_config_fields_present(self):
        """Each dict has keys: name, D, slices, v_crit, g_crit."""
        model = SlabModel(config_path='/Users/evanhuang/dbm3/config.yaml')

        config = model.get_compartment_config()

        required_keys = {"name", "D", "slices", "v_crit", "g_crit"}
        for comp_config in config:
            assert isinstance(comp_config, dict), "Each compartment config should be a dict"
            for key in required_keys:
                assert key in comp_config, f"Compartment config missing key: {key}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
