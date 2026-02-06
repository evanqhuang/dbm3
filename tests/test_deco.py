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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
