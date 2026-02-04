"""
Finite Difference Slab Model for tissue gas diffusion.

Implements a 1D slab diffusion model using the finite difference method
to simulate nitrogen uptake and offgassing in tissue.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .profile_generator import DiveProfile


@dataclass
class SlabResult:
    """Results from running a dive profile through the Slab model."""

    # Time series data
    times: List[float]
    depths: List[float]

    # Slab state over time: [time_idx][slice_idx]
    slab_history: np.ndarray

    # Per-timestep metrics
    max_loads: List[float]  # Max tissue load at each timestep
    margins: List[float]  # Minimum margin to M-value at each timestep

    # Summary metrics
    max_tissue_load: float
    min_margin: float
    critical_slice: int
    max_supersaturation: float  # Tissue load / M-value ratio
    final_ndl: float  # NDL remaining at end of dive (minutes)
    final_slab: np.ndarray  # Final tissue state

    @property
    def exceeded_limit(self) -> bool:
        """True if any slice exceeded its M-value."""
        return self.min_margin < 0

    @property
    def risk_score(self) -> float:
        """
        Normalized risk score (0-1+).
        Based on maximum supersaturation relative to M-value.
        >1.0 indicates M-value exceeded.
        """
        return self.max_supersaturation


class SlabModel:
    """
    Finite Difference Slab Model for decompression calculations.

    Models tissue as a 1D slab with:
    - Blood-tissue interface at slice 0 (boundary condition)
    - Core tissue at slice N-1 (no-flux boundary)
    - Diffusion governed by Fick's law
    """

    def __init__(
        self,
        slices: int = 50,
        diffusion_coefficient: float = 0.1,
        dt: float = 0.5,  # seconds
        dx: float = 1.0,
        permeability: Optional[float] = 0.0003,
        m_surface: float = 3.0,
        m_core: float = 1.5,
        f_o2: float = 0.21,
        surface_altitude_m: float = 0.0,
    ):
        """
        Initialize the Slab model.

        Args:
            slices: Number of tissue layers (resolution)
            diffusion_coefficient: D in Fick's law (speed of gas moving)
            dt: Time step in seconds (smaller = more precise)
            dx: Distance between slices (arbitrary units)
            permeability: Blood-tissue barrier permeability (None = perfect perfusion)
                          Controls delay at blood-tissue interface. Higher = faster equilibration.
                          Typical range: 0.001 (slow barrier) to 0.1 (fast barrier)
            m_surface: M-value at surface slice (fast tissue) - HIGH tolerance
            m_core: M-value at core slice (slow tissue) - LOW tolerance
            f_o2: Breathing gas O2 fraction (Air = 0.21)
            surface_altitude_m: Altitude of the water surface (0m = Sea Level)
        """
        self.slices = slices
        self.D = diffusion_coefficient
        self.dt = dt
        self.dx = dx
        self.permeability = permeability
        self.f_o2 = f_o2
        self.surface_altitude_m = surface_altitude_m

        # Stability Check: The finite difference method explodes if k > 0.5
        self.k = (self.D * self.dt) / (self.dx**2)
        if self.k > 0.5:
            raise ValueError(
                f"Stability warning! k={self.k:.4f} is > 0.5. Reduce dt or increase dx."
            )

        # Create M-Value Array (the "Death Line")
        # Smooth curve from high tolerance (surface) to low tolerance (core)
        self.m_values = np.linspace(m_surface, m_core, slices)

    def _get_atmospheric_pressure(self, altitude_m: float = None) -> float:
        """
        Calculate atmospheric pressure at altitude (bar).
        Uses Standard Barometric Formula: P = P0 * (1 - L*h/T0)^(gM/RL)
        """
        if altitude_m is None:
            altitude_m = self.surface_altitude_m
        if altitude_m < 0:
            return 1.01325  # Clamp to sea level if negative
        return 1.01325 * (1 - 2.25577e-5 * altitude_m) ** 5.25588

    def _get_hydrostatic_pressure(self, depth_m: float) -> float:
        """Calculate hydrostatic pressure at depth. Standard rule: 1 bar per 10m."""
        return depth_m / 10.0

    def _get_ppn2(self, depth_m: float, f_o2: float = None) -> float:
        """Calculate nitrogen partial pressure at depth."""
        if f_o2 is None:
            f_o2 = self.f_o2
        p_total = self._get_atmospheric_pressure() + self._get_hydrostatic_pressure(
            depth_m
        )
        return p_total * (1 - f_o2)

    def run(self, profile: DiveProfile) -> SlabResult:
        """
        Run a dive profile through the Slab model.

        Args:
            profile: DiveProfile object with dive data

        Returns:
            SlabResult with tissue states and risk metrics
        """
        # Calculate Surface Pressure (Start)
        p_surface_bar = self._get_atmospheric_pressure()
        ppn2_surface = p_surface_bar * (1 - self.f_o2)

        # Initialize slab at surface equilibrium
        # The tissue starts fully saturated at surface pressure
        slab = np.full(self.slices, ppn2_surface)

        # Storage for results
        times = []
        depths = []
        slab_history = []
        max_loads = []
        margins = []

        # Convert profile to time-depth pairs
        if not profile.points:
            raise ValueError("Empty dive profile")

        # Process profile
        current_time = 0.0
        sample_interval = 1.0  # Record every 1 second for history

        for i in range(len(profile.points) - 1):
            t1, d1, o2_1, _ = profile.points[i]
            t2, d2, o2_2, _ = profile.points[i + 1]

            # Convert times from minutes to seconds
            t1_sec = t1 * 60
            t2_sec = t2 * 60

            # Simulate this segment
            while current_time < t2_sec:
                # Interpolate depth and O2 at current time
                if t2_sec > t1_sec:
                    ratio = (current_time - t1_sec) / (t2_sec - t1_sec)
                else:
                    ratio = 0

                current_depth = d1 + ratio * (d2 - d1)
                current_o2 = o2_1 + ratio * (o2_2 - o2_1)

                # Calculate ppN2 at current depth
                ppn2_current = self._get_ppn2(current_depth, current_o2)

                # 1. Update Boundary Condition (Blood-Tissue Interface)
                if self.permeability is None:
                    # Perfect Perfusion: surface instantly matches blood
                    slab[0] = ppn2_current
                else:
                    # Permeability Barrier: adds delay at interface
                    # Flux = Permeability * (Blood_Pressure - Surface_Pressure)
                    flux = self.permeability * (ppn2_current - slab[0])
                    slab[0] += flux * self.dt

                # 2. Diffusion (Finite Difference Method)
                # The math: New = Old + k * (Left_Neighbor - 2*Me + Right_Neighbor)
                # We use vectorization (slab[1:-1]) to do all slices at once for speed
                slab[1:-1] += self.k * (slab[:-2] - 2 * slab[1:-1] + slab[2:])

                # 3. No-Flux Boundary at the deep end
                # (Gas hits the back of the cartilage and stops, or bounces back)
                slab[-1] = slab[-2]

                # Record state periodically
                if current_time % sample_interval < self.dt:
                    times.append(current_time / 60)  # Back to minutes
                    depths.append(current_depth)
                    slab_history.append(slab.copy())
                    max_loads.append(np.max(slab))
                    margins.append(np.min(self.m_values - slab))

                current_time += self.dt

        # Process final point
        if profile.points:
            t_final, d_final, _, _ = profile.points[-1]
            times.append(t_final)
            depths.append(d_final)
            slab_history.append(slab.copy())
            max_loads.append(np.max(slab))
            margins.append(np.min(self.m_values - slab))

        # Calculate summary metrics
        slab_history = np.array(slab_history)
        max_tissue_load = np.max(slab_history)
        min_margin = np.min(margins)

        # Check which slice is closest to its limit (critical slice)
        margin = self.m_values - slab  # How much headroom each slice has
        critical_slice = int(np.argmin(margin))

        # Calculate max supersaturation
        # In the slab model, each slice has its own M-value limit.
        # NDL is exceeded when ANY slice exceeds its corresponding M-value.
        max_supersaturation = np.max(slab_history / self.m_values)

        # Calculate NDL from final state at the last depth
        final_depth = depths[-1] if depths else 0.0
        final_ndl = self.calculate_ndl(slab, final_depth)

        return SlabResult(
            times=times,
            depths=depths,
            slab_history=slab_history,
            max_loads=max_loads,
            margins=margins,
            max_tissue_load=max_tissue_load,
            min_margin=min_margin,
            critical_slice=critical_slice,
            max_supersaturation=max_supersaturation,
            final_ndl=float(final_ndl),
            final_slab=slab.copy(),
        )

    def calculate_ndl(
        self, current_slab: np.ndarray, depth: float, max_time: int = 99
    ) -> int:
        """
        Simulates the future to find how many minutes until we hit the limit.
        Uses binary search for efficiency (O(log n) vs O(n)).

        Args:
            current_slab: Current tissue state
            depth: Current depth in meters
            max_time: Maximum time to simulate (minutes)

        Returns:
            NDL in minutes (max_time if > max_time, essentially unlimited)
        """
        # Calculate pressure at current depth (we are staying here)
        p_bottom = self._get_atmospheric_pressure() + self._get_hydrostatic_pressure(
            depth
        )
        ppn2_bottom = p_bottom * (1 - self.f_o2)

        # Binary search for NDL
        lo, hi = 0, max_time

        # First check if we're already over the limit
        if np.any(current_slab > self.m_values):
            return 0

        # Check if max_time is safe (early exit)
        test_slab = self._simulate_minutes(current_slab, ppn2_bottom, max_time)
        if not np.any(test_slab > self.m_values):
            return max_time

        # Binary search for the exact minute
        while lo < hi:
            mid = (lo + hi) // 2
            test_slab = self._simulate_minutes(current_slab, ppn2_bottom, mid)
            if np.any(test_slab > self.m_values):
                hi = mid
            else:
                lo = mid + 1

        return max(0, lo - 1)

    def _simulate_minutes(
        self, initial_slab: np.ndarray, ppn2_bottom: float, minutes: int
    ) -> np.ndarray:
        """
        Simulate tissue state after given minutes at constant depth.
        Optimized for NDL calculation.
        """
        if minutes <= 0:
            return initial_slab.copy()

        shadow_slab = initial_slab.copy()
        steps_per_min = int(60 / self.dt)
        total_steps = minutes * steps_per_min

        for _ in range(total_steps):
            # Update Boundary
            if self.permeability is None:
                shadow_slab[0] = ppn2_bottom
            else:
                flux = self.permeability * (ppn2_bottom - shadow_slab[0])
                shadow_slab[0] += flux * self.dt

            # Diffuse (vectorized)
            shadow_slab[1:-1] += self.k * (
                shadow_slab[:-2] - 2 * shadow_slab[1:-1] + shadow_slab[2:]
            )
            # No-flux deep end
            shadow_slab[-1] = shadow_slab[-2]

        return shadow_slab

    def run_batch(self, profiles: List[DiveProfile]) -> List[SlabResult]:
        """
        Run multiple profiles through the model.

        Args:
            profiles: List of DiveProfile objects

        Returns:
            List of SlabResult objects
        """
        results = []
        for profile in profiles:
            try:
                result = self.run(profile)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to process {profile.name}: {e}")
                results.append(None)
        return results


if __name__ == "__main__":
    # Demo: Run a simple profile
    from .profile_generator import ProfileGenerator

    gen = ProfileGenerator()
    profile = gen.generate_square(depth=30, bottom_time=20)

    model = SlabModel()
    result = model.run(profile)

    print(f"Profile: {profile.name}")
    print(f"Max tissue load: {result.max_tissue_load:.3f} bar")
    print(f"Min margin: {result.min_margin:.3f} bar")
    print(f"Critical slice: {result.critical_slice}")
    print(f"Max supersaturation: {result.max_supersaturation:.2%}")
    print(f"Exceeded limit: {result.exceeded_limit}")
