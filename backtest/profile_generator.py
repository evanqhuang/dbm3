"""
Dive profile generator for backtesting decompression models.

Generates various dive profiles:
- Square profiles (constant depth)
- Sawtooth profiles (oscillating depth)
- Multi-level profiles (stepped depths)
- Random walk profiles (realistic diving patterns)
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from io import StringIO


@dataclass
class DiveProfile:
    """Represents a dive profile as a sequence of (time, depth, fO2, fHe) points."""

    points: List[Tuple[float, float, float, float]] = field(default_factory=list)
    name: str = "unnamed"
    max_depth: float = 0.0
    bottom_time: float = 0.0

    def add_point(self, time: float, depth: float, fO2: float = 0.21, fHe: float = 0.0):
        """Add a point to the profile. Depth in meters, time in minutes."""
        self.points.append((time, depth, fO2, fHe))
        if depth > self.max_depth:
            self.max_depth = depth

    def to_buhlmann_format(self) -> str:
        """
        Convert to libbuhlmann input format.
        Format: time pressure O2 He
        Where pressure = depth/10 + 1 (bar)
        """
        output = StringIO()
        for time, depth, fO2, fHe in self.points:
            pressure = depth / 10.0 + 1.0
            output.write(f"{time:.2f} {pressure:.2f} {fO2:.4f} {fHe:.4f}\n")
        return output.getvalue()

    def get_depth_at_time(self, t: float) -> float:
        """Interpolate depth at a given time."""
        if not self.points:
            return 0.0
        if t <= self.points[0][0]:
            return self.points[0][1]
        if t >= self.points[-1][0]:
            return self.points[-1][1]

        for i in range(len(self.points) - 1):
            t1, d1, _, _ = self.points[i]
            t2, d2, _, _ = self.points[i + 1]
            if t1 <= t <= t2:
                # Linear interpolation
                if t2 == t1:
                    return d1
                ratio = (t - t1) / (t2 - t1)
                return d1 + ratio * (d2 - d1)
        return self.points[-1][1]


class ProfileGenerator:
    """Generate various dive profiles for backtesting."""

    def __init__(
        self,
        descent_rate: float = 20.0,  # m/min
        ascent_rate: float = 10.0,  # m/min (conservative)
        sampling_interval: float = 0.1,
    ):  # minutes
        self.descent_rate = descent_rate
        self.ascent_rate = ascent_rate
        self.sampling_interval = sampling_interval

    def generate_square(
        self, depth: float, bottom_time: float, fO2: float = 0.21, fHe: float = 0.0
    ) -> DiveProfile:
        """
        Generate a square profile (simple recreational dive).

        Args:
            depth: Maximum depth in meters
            bottom_time: Time at depth in minutes
            fO2: Oxygen fraction
            fHe: Helium fraction
        """
        profile = DiveProfile(name=f"square_{depth}m_{bottom_time}min")
        profile.bottom_time = bottom_time

        time = 0.0
        current_depth = 0.0

        # Descent phase
        descent_time = depth / self.descent_rate
        while current_depth < depth:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = min(
                current_depth + self.descent_rate * self.sampling_interval, depth
            )
            time += self.sampling_interval

        # Bottom phase
        end_bottom = time + bottom_time
        while time < end_bottom:
            profile.add_point(time, depth, fO2, fHe)
            time += self.sampling_interval

        # Ascent phase
        while current_depth > 0:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = max(
                current_depth - self.ascent_rate * self.sampling_interval, 0
            )
            time += self.sampling_interval

        # Surface interval (1 minute)
        end_surface = time + 1.0
        while time < end_surface:
            profile.add_point(time, 0.0, fO2, fHe)
            time += self.sampling_interval

        return profile

    def generate_sawtooth(
        self,
        max_depth: float,
        min_depth: float,
        total_time: float,
        oscillations: int = 3,
        fO2: float = 0.21,
        fHe: float = 0.0,
    ) -> DiveProfile:
        """
        Generate a sawtooth profile (yo-yo diving pattern).

        Args:
            max_depth: Maximum depth in meters
            min_depth: Minimum depth during oscillations
            total_time: Total bottom time in minutes
            oscillations: Number of depth oscillations
            fO2: Oxygen fraction
            fHe: Helium fraction
        """
        profile = DiveProfile(name=f"sawtooth_{max_depth}m_{oscillations}osc")
        profile.bottom_time = total_time

        time = 0.0
        current_depth = 0.0

        # Initial descent to max depth
        while current_depth < max_depth:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = min(
                current_depth + self.descent_rate * self.sampling_interval, max_depth
            )
            time += self.sampling_interval

        # Oscillation phase
        time_per_oscillation = total_time / oscillations
        going_up = True

        oscillation_start = time
        while time < oscillation_start + total_time:
            profile.add_point(time, current_depth, fO2, fHe)

            if going_up:
                current_depth -= self.ascent_rate * self.sampling_interval
                if current_depth <= min_depth:
                    current_depth = min_depth
                    going_up = False
            else:
                current_depth += self.descent_rate * self.sampling_interval
                if current_depth >= max_depth:
                    current_depth = max_depth
                    going_up = True

            time += self.sampling_interval

        # Final ascent
        while current_depth > 0:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = max(
                current_depth - self.ascent_rate * self.sampling_interval, 0
            )
            time += self.sampling_interval

        # Surface interval
        end_surface = time + 1.0
        while time < end_surface:
            profile.add_point(time, 0.0, fO2, fHe)
            time += self.sampling_interval

        return profile

    def generate_multilevel(
        self, levels: List[Tuple[float, float]], fO2: float = 0.21, fHe: float = 0.0
    ) -> DiveProfile:
        """
        Generate a multi-level profile.

        Args:
            levels: List of (depth, duration) tuples, deepest first
            fO2: Oxygen fraction
            fHe: Helium fraction
        """
        profile = DiveProfile(name=f"multilevel_{len(levels)}levels")
        profile.bottom_time = sum(d[1] for d in levels)

        time = 0.0
        current_depth = 0.0

        for target_depth, duration in levels:
            # Transition to target depth
            if target_depth > current_depth:
                # Descend
                while current_depth < target_depth:
                    profile.add_point(time, current_depth, fO2, fHe)
                    current_depth = min(
                        current_depth + self.descent_rate * self.sampling_interval,
                        target_depth,
                    )
                    time += self.sampling_interval
            else:
                # Ascend
                while current_depth > target_depth:
                    profile.add_point(time, current_depth, fO2, fHe)
                    current_depth = max(
                        current_depth - self.ascent_rate * self.sampling_interval,
                        target_depth,
                    )
                    time += self.sampling_interval

            # Stay at level
            end_level = time + duration
            while time < end_level:
                profile.add_point(time, target_depth, fO2, fHe)
                time += self.sampling_interval

        # Final ascent
        while current_depth > 0:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = max(
                current_depth - self.ascent_rate * self.sampling_interval, 0
            )
            time += self.sampling_interval

        # Surface interval
        end_surface = time + 1.0
        while time < end_surface:
            profile.add_point(time, 0.0, fO2, fHe)
            time += self.sampling_interval

        return profile

    def generate_random_walk(
        self,
        max_depth: float,
        total_time: float,
        volatility: float = 0.3,
        fO2: float = 0.21,
        fHe: float = 0.0,
        seed: Optional[int] = None,
    ) -> DiveProfile:
        """
        Generate a random walk profile (realistic diving pattern).

        Args:
            max_depth: Maximum allowed depth
            total_time: Total dive time in minutes
            volatility: Depth change volatility (0-1)
            fO2: Oxygen fraction
            fHe: Helium fraction
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        profile = DiveProfile(name=f"random_{max_depth}m_{total_time}min")
        profile.bottom_time = total_time

        time = 0.0
        current_depth = 0.0
        target_depth = max_depth * random.uniform(0.6, 1.0)

        # Initial descent
        while current_depth < target_depth:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = min(
                current_depth + self.descent_rate * self.sampling_interval, target_depth
            )
            time += self.sampling_interval

        # Random walk phase
        dive_end = time + total_time
        while time < dive_end:
            profile.add_point(time, current_depth, fO2, fHe)

            # Random depth change
            change = random.gauss(0, volatility * max_depth * self.sampling_interval)
            current_depth = max(5.0, min(max_depth, current_depth + change))

            time += self.sampling_interval

        # Final ascent
        while current_depth > 0:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = max(
                current_depth - self.ascent_rate * self.sampling_interval, 0
            )
            time += self.sampling_interval

        # Surface interval
        end_surface = time + 1.0
        while time < end_surface:
            profile.add_point(time, 0.0, fO2, fHe)
            time += self.sampling_interval

        return profile

    def generate_deco_square(
        self,
        depth: float,
        bottom_time: float,
        deco_stops: List[Tuple[float, float]],
        fO2: float = 0.21,
        fHe: float = 0.0,
    ) -> DiveProfile:
        """Generate a square profile with explicit decompression stops.

        Args:
            depth: Bottom depth in meters
            bottom_time: Time at depth in minutes
            deco_stops: List of (stop_depth_m, stop_duration_min) tuples, deepest first
            fO2: Oxygen fraction
            fHe: Helium fraction
        """
        stop_desc = "+".join(f"{d:.0f}m/{t:.0f}min" for d, t in deco_stops) if deco_stops else "nodeco"
        profile = DiveProfile(name=f"deco_{depth}m_{bottom_time}min_{stop_desc}")
        profile.bottom_time = bottom_time

        time = 0.0
        current_depth = 0.0

        # Descent phase
        while current_depth < depth:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = min(
                current_depth + self.descent_rate * self.sampling_interval, depth
            )
            time += self.sampling_interval

        # Bottom phase
        end_bottom = time + bottom_time
        while time < end_bottom:
            profile.add_point(time, depth, fO2, fHe)
            time += self.sampling_interval

        # Deco stops (ascend to each stop, hold, repeat)
        for stop_depth, stop_duration in deco_stops:
            # Ascend to stop depth
            while current_depth > stop_depth:
                profile.add_point(time, current_depth, fO2, fHe)
                current_depth = max(
                    current_depth - self.ascent_rate * self.sampling_interval,
                    stop_depth,
                )
                time += self.sampling_interval

            # Hold at stop depth
            end_stop = time + stop_duration
            while time < end_stop:
                profile.add_point(time, stop_depth, fO2, fHe)
                time += self.sampling_interval

        # Final ascent to surface
        while current_depth > 0:
            profile.add_point(time, current_depth, fO2, fHe)
            current_depth = max(
                current_depth - self.ascent_rate * self.sampling_interval, 0
            )
            time += self.sampling_interval

        # Surface interval (1 minute)
        end_surface = time + 1.0
        while time < end_surface:
            profile.add_point(time, 0.0, fO2, fHe)
            time += self.sampling_interval

        return profile

    def generate_batch(
        self,
        depths: List[float],
        times: List[float],
        profile_type: str = "square",
        **kwargs,
    ) -> List[DiveProfile]:
        """
        Generate a batch of profiles for systematic testing.

        Args:
            depths: List of depths to test
            times: List of bottom times to test
            profile_type: Type of profile ("square", "sawtooth", "random")
            **kwargs: Additional arguments for profile generator

        Returns:
            List of DiveProfile objects
        """
        profiles = []

        for depth in depths:
            for time in times:
                if profile_type == "square":
                    profile = self.generate_square(depth, time, **kwargs)
                elif profile_type == "sawtooth":
                    profile = self.generate_sawtooth(depth, depth * 0.3, time, **kwargs)
                elif profile_type == "random":
                    profile = self.generate_random_walk(depth, time, **kwargs)
                else:
                    raise ValueError(f"Unknown profile type: {profile_type}")

                profiles.append(profile)

        return profiles


if __name__ == "__main__":
    # Demo: Generate and print a square profile
    gen = ProfileGenerator()
    profile = gen.generate_square(depth=20, bottom_time=5)
    print(f"Generated profile: {profile.name}")
    print(f"Max depth: {profile.max_depth}m")
    print(f"Points: {len(profile.points)}")
    print("\nFirst 10 lines of Bühlmann format:")
    lines = profile.to_buhlmann_format().split("\n")[:10]
    for line in lines:
        print(line)
