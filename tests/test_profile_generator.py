"""
Comprehensive unit tests for backtest/profile_generator.py.

Tests all methods of DiveProfile and ProfileGenerator, validating:
- Data integrity and edge cases
- Physical dive simulation (descent/ascent rates, depth bounds)
- Gas fraction handling
- Profile type variations
- Batch generation
"""

import pytest
from backtest.profile_generator import DiveProfile, ProfileGenerator


class TestDiveProfileDataclass:
    """Tests for DiveProfile dataclass and basic operations."""

    def test_empty_profile_defaults(self):
        """Fresh DiveProfile has points=[], max_depth=0.0, name='unnamed'."""
        profile = DiveProfile()
        assert profile.points == []
        assert profile.max_depth == 0.0
        assert profile.name == "unnamed"
        assert profile.bottom_time == 0.0

    def test_add_point_updates_max_depth(self):
        """Adding deeper point updates max_depth, shallower doesn't reduce it."""
        profile = DiveProfile()

        # Add point at 20m
        profile.add_point(1.0, 20.0)
        assert profile.max_depth == pytest.approx(20.0)

        # Add deeper point at 30m
        profile.add_point(2.0, 30.0)
        assert profile.max_depth == pytest.approx(30.0)

        # Add shallower point at 15m (should not reduce max_depth)
        profile.add_point(3.0, 15.0)
        assert profile.max_depth == pytest.approx(30.0)

    def test_add_point_stores_gas_fractions(self):
        """fO2=0.32, fHe=0.10 stored correctly."""
        profile = DiveProfile()
        profile.add_point(1.0, 20.0, fO2=0.32, fHe=0.10)

        assert len(profile.points) == 1
        time, depth, fO2, fHe = profile.points[0]
        assert time == pytest.approx(1.0)
        assert depth == pytest.approx(20.0)
        assert fO2 == pytest.approx(0.32)
        assert fHe == pytest.approx(0.10)

    def test_add_point_default_gas_fractions(self):
        """Default fO2=0.21, fHe=0.0."""
        profile = DiveProfile()
        profile.add_point(1.0, 20.0)

        time, depth, fO2, fHe = profile.points[0]
        assert fO2 == pytest.approx(0.21)
        assert fHe == pytest.approx(0.0)


class TestToBuhlmannFormat:
    """Tests for Buhlmann format conversion."""

    def test_pressure_formula_at_30m(self):
        """30m depth -> pressure 4.0 bar (depth/10 + 1)."""
        profile = DiveProfile()
        profile.add_point(0.0, 30.0)

        output = profile.to_buhlmann_format()
        lines = output.strip().split("\n")

        # Parse first line: time pressure O2 He
        parts = lines[0].split()
        pressure = float(parts[1])
        assert pressure == pytest.approx(4.0)  # 30/10 + 1 = 4.0

    def test_surface_pressure_is_1_bar(self):
        """0m depth -> 1.0 bar."""
        profile = DiveProfile()
        profile.add_point(0.0, 0.0)

        output = profile.to_buhlmann_format()
        lines = output.strip().split("\n")

        parts = lines[0].split()
        pressure = float(parts[1])
        assert pressure == pytest.approx(1.0)

    def test_format_multiple_points_structure(self):
        """Each line has 4 space-separated values."""
        profile = DiveProfile()
        profile.add_point(0.0, 0.0, 0.21, 0.0)
        profile.add_point(1.0, 10.0, 0.21, 0.0)
        profile.add_point(2.0, 20.0, 0.32, 0.0)

        output = profile.to_buhlmann_format()
        lines = output.strip().split("\n")

        assert len(lines) == 3
        for line in lines:
            parts = line.split()
            assert len(parts) == 4  # time pressure O2 He

    def test_empty_profile_produces_empty_string(self):
        """No points -> empty string."""
        profile = DiveProfile()
        output = profile.to_buhlmann_format()
        assert output == ""

    def test_nitrox_gas_fractions_preserved(self):
        """fO2=0.32 appears correctly in output."""
        profile = DiveProfile()
        profile.add_point(0.0, 20.0, fO2=0.32, fHe=0.0)

        output = profile.to_buhlmann_format()
        lines = output.strip().split("\n")

        parts = lines[0].split()
        fO2 = float(parts[2])
        fHe = float(parts[3])
        assert fO2 == pytest.approx(0.32)
        assert fHe == pytest.approx(0.0)


class TestGetDepthAtTime:
    """Tests for depth interpolation."""

    def test_empty_profile_returns_zero(self):
        """No points -> 0.0."""
        profile = DiveProfile()
        depth = profile.get_depth_at_time(5.0)
        assert depth == pytest.approx(0.0)

    def test_time_before_first_point(self):
        """Returns first depth."""
        profile = DiveProfile()
        profile.add_point(5.0, 20.0)
        profile.add_point(10.0, 30.0)

        depth = profile.get_depth_at_time(2.0)
        assert depth == pytest.approx(20.0)

    def test_time_after_last_point(self):
        """Returns last depth."""
        profile = DiveProfile()
        profile.add_point(5.0, 20.0)
        profile.add_point(10.0, 30.0)

        depth = profile.get_depth_at_time(15.0)
        assert depth == pytest.approx(30.0)

    def test_exact_point_time(self):
        """Returns exact depth at that time."""
        profile = DiveProfile()
        profile.add_point(5.0, 20.0)
        profile.add_point(10.0, 30.0)

        depth = profile.get_depth_at_time(5.0)
        assert depth == pytest.approx(20.0)

        depth = profile.get_depth_at_time(10.0)
        assert depth == pytest.approx(30.0)

    def test_linear_interpolation_midpoint(self):
        """(0,0)→(10,30) at t=5 -> 15.0."""
        profile = DiveProfile()
        profile.add_point(0.0, 0.0)
        profile.add_point(10.0, 30.0)

        depth = profile.get_depth_at_time(5.0)
        assert depth == pytest.approx(15.0)

    def test_linear_interpolation_quarter(self):
        """(0,0)→(10,40) at t=2.5 -> 10.0."""
        profile = DiveProfile()
        profile.add_point(0.0, 0.0)
        profile.add_point(10.0, 40.0)

        depth = profile.get_depth_at_time(2.5)
        assert depth == pytest.approx(10.0)

    def test_descending_profile_interpolation(self):
        """(0,30)→(3,0) at t=1 -> 20.0."""
        profile = DiveProfile()
        profile.add_point(0.0, 30.0)
        profile.add_point(3.0, 0.0)

        depth = profile.get_depth_at_time(1.0)
        assert depth == pytest.approx(20.0)


class TestGenerateSquare:
    """Tests for square profile generation."""

    def test_square_max_depth_matches_request(self):
        """profile.max_depth == 30.0."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30.0, bottom_time=10.0)
        assert profile.max_depth == pytest.approx(30.0)

    def test_square_starts_at_surface(self):
        """First point depth is 0.0."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20.0, bottom_time=5.0)

        first_depth = profile.points[0][1]
        assert first_depth == pytest.approx(0.0)

    def test_square_ends_at_surface(self):
        """Last point depth is 0.0."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20.0, bottom_time=5.0)

        last_depth = profile.points[-1][1]
        assert last_depth == pytest.approx(0.0)

    def test_square_reaches_target_depth(self):
        """At least one point at target depth."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=25.0, bottom_time=10.0)

        depths = [point[1] for point in profile.points]
        assert max(depths) == pytest.approx(25.0)

    def test_square_descent_time_physics(self):
        """Descent phase ≈ depth/descent_rate minutes."""
        gen = ProfileGenerator(descent_rate=20.0)
        profile = gen.generate_square(depth=40.0, bottom_time=10.0)

        # Expected descent time = 40m / 20m/min = 2 minutes
        # Find when max depth is first reached
        for i, (time, depth, _, _) in enumerate(profile.points):
            if depth >= 39.9:  # Within tolerance
                assert time <= 2.1  # Should reach within ~2 minutes
                break

    def test_square_ascent_time_physics(self):
        """Ascent phase ≈ depth/ascent_rate minutes."""
        gen = ProfileGenerator(ascent_rate=10.0)
        profile = gen.generate_square(depth=30.0, bottom_time=5.0)

        # Expected ascent time = 30m / 10m/min = 3 minutes
        # Find last point at max depth and first point back at surface
        last_deep_time = None
        surface_time = None

        for time, depth, _, _ in profile.points:
            if depth >= 29.9:
                last_deep_time = time
            if depth <= 0.1 and last_deep_time is not None and surface_time is None:
                surface_time = time
                break

        assert last_deep_time is not None, "Profile never reached target depth"
        assert surface_time is not None, "Profile never returned to surface"
        ascent_duration = surface_time - last_deep_time
        expected_ascent = 30.0 / 10.0  # 3 minutes
        assert ascent_duration == pytest.approx(expected_ascent, abs=0.5), \
            f"Ascent should take ~{expected_ascent} min, took {ascent_duration:.2f} min"

    def test_square_custom_gas_fractions(self):
        """fO2=0.32 in all points."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=20.0, bottom_time=5.0, fO2=0.32, fHe=0.0)

        # Check that all points have correct gas mix
        for _, _, fO2, fHe in profile.points:
            assert fO2 == pytest.approx(0.32)
            assert fHe == pytest.approx(0.0)

    def test_square_bottom_time_attribute(self):
        """profile.bottom_time matches request."""
        gen = ProfileGenerator()
        profile = gen.generate_square(depth=30.0, bottom_time=15.0)
        assert profile.bottom_time == pytest.approx(15.0)


class TestGenerateSawtooth:
    """Tests for sawtooth profile generation."""

    def test_sawtooth_reaches_max_depth(self):
        """max_depth reached (within 1m tolerance)."""
        gen = ProfileGenerator()
        profile = gen.generate_sawtooth(
            max_depth=40.0, min_depth=20.0, total_time=10.0, oscillations=3
        )

        assert profile.max_depth >= 39.0  # Within 1m tolerance

    def test_sawtooth_oscillates(self):
        """At least 2 direction changes during oscillation."""
        gen = ProfileGenerator()
        profile = gen.generate_sawtooth(
            max_depth=30.0, min_depth=15.0, total_time=8.0, oscillations=2
        )

        # Count direction changes by tracking when depth derivative changes sign
        depths = [point[1] for point in profile.points]
        direction_changes = 0
        going_down = None

        for i in range(1, len(depths)):
            if depths[i] > depths[i-1]:
                current_direction = "down"
            elif depths[i] < depths[i-1]:
                current_direction = "up"
            else:
                continue

            if going_down is not None and going_down != current_direction:
                direction_changes += 1
            going_down = current_direction

        assert direction_changes >= 2  # At least 2 oscillations

    def test_sawtooth_ends_at_surface(self):
        """Last point at 0.0."""
        gen = ProfileGenerator()
        profile = gen.generate_sawtooth(
            max_depth=25.0, min_depth=10.0, total_time=5.0
        )

        last_depth = profile.points[-1][1]
        assert last_depth == pytest.approx(0.0)

    def test_sawtooth_name_format(self):
        """Name contains 'sawtooth'."""
        gen = ProfileGenerator()
        profile = gen.generate_sawtooth(
            max_depth=30.0, min_depth=15.0, total_time=10.0, oscillations=4
        )

        assert "sawtooth" in profile.name


class TestGenerateMultilevel:
    """Tests for multi-level profile generation."""

    def test_multilevel_visits_all_levels(self):
        """At least one point near each level depth."""
        gen = ProfileGenerator()
        levels = [(40.0, 5.0), (30.0, 5.0), (20.0, 5.0)]
        profile = gen.generate_multilevel(levels)

        depths = [point[1] for point in profile.points]

        # Check that we get close to each target depth (within 2m)
        for target_depth, _ in levels:
            closest = min(depths, key=lambda d: abs(d - target_depth))
            assert abs(closest - target_depth) < 2.0

    def test_multilevel_deepest_first(self):
        """Deeper phases chronologically precede shallower ones."""
        gen = ProfileGenerator()
        levels = [(40.0, 3.0), (30.0, 3.0), (20.0, 3.0)]
        profile = gen.generate_multilevel(levels)

        # Find when diver stays at each depth (looking for sustained time at depth)
        # Count consecutive points at each depth to find the "stay" phase
        depth_stay_times = {}

        for target_depth, _ in levels:
            consecutive_count = 0
            for time, depth, _, _ in profile.points:
                if abs(depth - target_depth) < 0.5:
                    consecutive_count += 1
                    if consecutive_count == 3 and target_depth not in depth_stay_times:
                        # Found sustained stay at this depth
                        depth_stay_times[target_depth] = time
                        break
                else:
                    consecutive_count = 0

        # Verify chronological order (deepest level visited first)
        assert depth_stay_times[40.0] < depth_stay_times[30.0]
        assert depth_stay_times[30.0] < depth_stay_times[20.0]

    def test_multilevel_ends_at_surface(self):
        """Last point at 0.0."""
        gen = ProfileGenerator()
        levels = [(30.0, 5.0), (15.0, 5.0)]
        profile = gen.generate_multilevel(levels)

        last_depth = profile.points[-1][1]
        assert last_depth == pytest.approx(0.0)


class TestGenerateRandomWalk:
    """Tests for random walk profile generation."""

    def test_random_walk_deterministic_with_seed(self):
        """Same seed -> identical profiles."""
        gen = ProfileGenerator()

        profile1 = gen.generate_random_walk(
            max_depth=30.0, total_time=10.0, seed=42
        )
        profile2 = gen.generate_random_walk(
            max_depth=30.0, total_time=10.0, seed=42
        )

        # Should have same number of points
        assert len(profile1.points) == len(profile2.points)

        # Depths should match exactly (same random sequence)
        for p1, p2 in zip(profile1.points, profile2.points):
            assert p1[1] == pytest.approx(p2[1])

    def test_random_walk_depth_bounded(self):
        """Depth never exceeds max_depth."""
        gen = ProfileGenerator()
        profile = gen.generate_random_walk(
            max_depth=40.0, total_time=10.0, seed=123
        )

        depths = [point[1] for point in profile.points]
        assert max(depths) <= 40.0

    def test_random_walk_different_seeds_differ(self):
        """seed=42 vs seed=99 differ."""
        gen = ProfileGenerator()

        profile1 = gen.generate_random_walk(
            max_depth=30.0, total_time=10.0, seed=42
        )
        profile2 = gen.generate_random_walk(
            max_depth=30.0, total_time=10.0, seed=99
        )

        # Extract depths from both profiles
        depths1 = [point[1] for point in profile1.points]
        depths2 = [point[1] for point in profile2.points]

        # At least some depths should differ
        differences = sum(1 for d1, d2 in zip(depths1, depths2) if abs(d1 - d2) > 0.1)
        assert differences > 0

    def test_random_walk_ends_at_surface(self):
        """Last point at 0.0."""
        gen = ProfileGenerator()
        profile = gen.generate_random_walk(
            max_depth=30.0, total_time=10.0, seed=42
        )

        last_depth = profile.points[-1][1]
        assert last_depth == pytest.approx(0.0)


class TestGenerateDecoSquare:
    """Tests for deco square profile generation."""

    def test_deco_square_includes_stops(self):
        """Stop depths appear in profile."""
        gen = ProfileGenerator()
        deco_stops = [(9.0, 2.0), (6.0, 3.0), (3.0, 5.0)]
        profile = gen.generate_deco_square(
            depth=40.0, bottom_time=20.0, deco_stops=deco_stops
        )

        depths = [point[1] for point in profile.points]

        # Check that each stop depth appears in the profile (within 1m)
        for stop_depth, _ in deco_stops:
            closest = min(depths, key=lambda d: abs(d - stop_depth))
            assert abs(closest - stop_depth) < 1.0

    def test_deco_square_stop_order_deepest_first(self):
        """Deeper stops come before shallower (chronologically)."""
        gen = ProfileGenerator()
        deco_stops = [(9.0, 2.0), (6.0, 2.0), (3.0, 2.0)]
        profile = gen.generate_deco_square(
            depth=40.0, bottom_time=15.0, deco_stops=deco_stops
        )

        # Find when diver stays at each stop (sustained time at depth, not just passing through)
        stop_times = {}
        for stop_depth, _ in deco_stops:
            consecutive_count = 0
            for time, depth, _, _ in profile.points:
                if abs(depth - stop_depth) < 0.5:
                    consecutive_count += 1
                    if consecutive_count == 5 and stop_depth not in stop_times:
                        # Found sustained stop at this depth
                        stop_times[stop_depth] = time
                        break
                else:
                    consecutive_count = 0

        # Verify chronological order (deeper stops before shallower)
        assert stop_times[9.0] < stop_times[6.0]
        assert stop_times[6.0] < stop_times[3.0]

    def test_deco_square_name_includes_stops(self):
        """Name contains 'deco_'."""
        gen = ProfileGenerator()
        deco_stops = [(6.0, 3.0), (3.0, 5.0)]
        profile = gen.generate_deco_square(
            depth=35.0, bottom_time=18.0, deco_stops=deco_stops
        )

        assert "deco_" in profile.name


class TestGenerateBatch:
    """Tests for batch profile generation."""

    def test_batch_cartesian_product_count(self):
        """3 depths x 4 times = 12 profiles."""
        gen = ProfileGenerator()
        depths = [20.0, 30.0, 40.0]
        times = [5.0, 10.0, 15.0, 20.0]

        profiles = gen.generate_batch(depths, times, profile_type="square")

        assert len(profiles) == 12

    def test_batch_square_type_works(self):
        """All names contain 'square'."""
        gen = ProfileGenerator()
        depths = [20.0, 30.0]
        times = [10.0, 15.0]

        profiles = gen.generate_batch(depths, times, profile_type="square")

        for profile in profiles:
            assert "square" in profile.name

    def test_batch_sawtooth_type_works(self):
        """All names contain 'sawtooth'."""
        gen = ProfileGenerator()
        depths = [25.0, 35.0]
        times = [8.0, 12.0]

        profiles = gen.generate_batch(depths, times, profile_type="sawtooth")

        for profile in profiles:
            assert "sawtooth" in profile.name

    def test_batch_unknown_type_raises(self):
        """ValueError for unknown type."""
        gen = ProfileGenerator()
        depths = [20.0]
        times = [10.0]

        with pytest.raises(ValueError, match="Unknown profile type"):
            gen.generate_batch(depths, times, profile_type="invalid_type")
