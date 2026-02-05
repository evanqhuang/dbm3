"""
Finite Difference Slab Model for tissue gas diffusion.

Implements a 1D slab diffusion model using the finite difference method
to simulate nitrogen uptake and offgassing in tissue.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import yaml
import os

from backtest.profile_generator import DiveProfile


@dataclass
class TissueCompartment:
    """Represents a single tissue compartment with its own properties."""
    name: str
    D: float  # Diffusion coefficient
    slices: int  # Number of slices
    m_value_max: float  # M-value at surface
    m_value_min: float  # M-value at core
    slab: np.ndarray  # Current tissue state
    m_values: np.ndarray  # M-value array for this compartment
    k: float  # Stability constant for this compartment


@dataclass
class SlabResult:
    """Results from running a dive profile through the Slab model."""

    # Time series data
    times: List[float]
    depths: List[float]

    # Slab state over time: [time_idx][compartment_idx][slice_idx]
    slab_history: np.ndarray

    # Per-timestep metrics
    max_loads: List[float]  # Max tissue load at each timestep
    margins: List[float]  # Minimum margin to M-value at each timestep

    # Summary metrics
    max_tissue_load: float
    min_margin: float
    critical_compartment: str  # Name of the compartment with highest risk
    critical_slice: int
    max_supersaturation: float  # Tissue load / M-value ratio
    final_ndl: float  # NDL remaining at end of dive (minutes)
    final_slabs: dict  # Final tissue state for each compartment

    @property
    def exceeded_limit(self) -> bool:
        """True if any slice in any compartment exceeded its M-value."""
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
    Multi-Compartment Finite Difference Slab Model for decompression calculations.

    Models tissue as multiple compartments, each with:
    - Blood-tissue interface at slice 0 (boundary condition)
    - Core tissue at slice N-1 (no-flux boundary)
    - Diffusion governed by Fick's law
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        compartments_config=None,
        dt: float = None,  # seconds
        dx: float = None,
        permeability: Optional[float] = None,
        f_o2: float = None,
        surface_altitude_m: float = None,
    ):
        """
        Initialize the Multi-Compartment Slab model.

        Args:
            config_path: Path to YAML config file. If provided, loads parameters from file.
            compartments_config: List of dicts with compartment params (overrides config file)
                                Each dict has: name, D, slices, m_value_max, m_value_min
                                If None, uses default 3-compartment model or loads from config file
            dt: Time step in seconds (smaller = more precise)
            dx: Distance between slices (arbitrary units)
            permeability: Blood-tissue barrier permeability (None = perfect perfusion)
                          Controls delay at blood-tissue interface. Higher = faster equilibration.
                          Typical range: 0.001 (slow barrier) to 0.1 (fast barrier)
            f_o2: Breathing gas O2 fraction (Air = 0.21)
            surface_altitude_m: Altitude of the water surface (0m = Sea Level)
        """

        # Load configuration from file if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

        # Use config values if not explicitly provided
        self.dt = dt if dt is not None else config.get('dt', 0.5)
        self.dx = dx if dx is not None else config.get('dx', 1.0)
        self.permeability = permeability if permeability is not None else config.get('permeability', 0.0003)
        self.f_o2 = f_o2 if f_o2 is not None else config.get('f_o2', 0.21)
        self.surface_altitude_m = surface_altitude_m if surface_altitude_m is not None else config.get('surface_altitude_m', 0.0)

        # Load compartments config
        if compartments_config is not None:
            # Use explicitly provided config
            final_compartments_config = compartments_config
        elif config_path and 'compartments' in config:
            # Load from config file
            final_compartments_config = config['compartments']
        else:
            # Use default compartments based on example.md
            final_compartments_config = [
                # Fast, Sensitive (Spine)
                {"name": "Spine", "D": 0.002, "slices": 20, "m_value_max": 3.0, "m_value_min": 1.2},
                # Medium (Muscle)
                {"name": "Muscle", "D": 0.0005, "slices": 20, "m_value_max": 3.5, "m_value_min": 1.5},
                # Slow, Robust but traps gas (Joints)
                {"name": "Joints", "D": 0.0001, "slices": 20, "m_value_max": 4.0, "m_value_min": 2.0}
            ]

        # Create compartments
        self.compartments = []
        for config_item in final_compartments_config:
            name = config_item["name"]
            D = config_item["D"]
            slices = config_item["slices"]
            m_value_max = config_item["m_value_max"]
            m_value_min = config_item["m_value_min"]

            # Create the tissue slab
            slab = np.zeros(slices)

            # Create M-Value Array (the "Death Line")
            m_values = np.linspace(m_value_max, m_value_min, slices)

            # Stability Constant (k)
            k = (D * self.dt) / (self.dx**2)
            if k > 0.5:
                raise ValueError(f"Stability Error in {name}")

            compartment = TissueCompartment(
                name=name,
                D=D,
                slices=slices,
                m_value_max=m_value_max,
                m_value_min=m_value_min,
                slab=slab,
                m_values=m_values,
                k=k
            )
            self.compartments.append(compartment)

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

    def _update_compartment(self, compartment: TissueCompartment, boundary_pressure: float):
        """
        Update a single compartment with the given boundary pressure.

        Args:
            compartment: Tissue compartment to update
            boundary_pressure: Pressure at the blood-tissue interface
        """
        # 1. Update Boundary Condition (Blood-Tissue Interface)
        if self.permeability is None:
            # Perfect Perfusion: surface instantly matches blood
            compartment.slab[0] = boundary_pressure
        else:
            # Permeability Barrier: adds delay at interface
            # Flux = Permeability * (Blood_Pressure - Surface_Pressure)
            flux = self.permeability * (boundary_pressure - compartment.slab[0])
            compartment.slab[0] += flux * self.dt

        # 2. Diffusion (Finite Difference Method)
        # The math: New = Old + k * (Left_Neighbor - 2*Me + Right_Neighbor)
        # We use vectorization (slab[1:-1]) to do all slices at once for speed
        compartment.slab[1:-1] += compartment.k * (compartment.slab[:-2] - 2 * compartment.slab[1:-1] + compartment.slab[2:])

        # 3. No-Flux Boundary at the deep end
        # (Gas hits the back of the cartilage and stops, or bounces back)
        compartment.slab[-1] = compartment.slab[-2]

    def run(self, profile: DiveProfile) -> SlabResult:
        """
        Run a dive profile through the Multi-Compartment Slab model.

        Args:
            profile: DiveProfile object with dive data

        Returns:
            SlabResult with tissue states and risk metrics
        """
        # Calculate Surface Pressure (Start)
        p_surface_bar = self._get_atmospheric_pressure()
        ppn2_surface = p_surface_bar * (1 - self.f_o2)

        # Initialize all compartments at surface equilibrium
        # Each compartment starts fully saturated at surface pressure
        for compartment in self.compartments:
            compartment.slab[:] = ppn2_surface

        # Storage for results
        times = []
        depths = []
        slab_history = []  # Will store state of all compartments
        max_loads = []
        margins = []

        # Track tissue state at max depth for NDL calculation
        max_depth_seen = 0.0
        max_depth_slabs = [compartment.slab.copy() for compartment in self.compartments]

        # Track max supersaturation across ALL timesteps (not just final)
        max_supersaturation = 0.0

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

                # Update all compartments
                for compartment in self.compartments:
                    self._update_compartment(compartment, ppn2_current)

                # Track supersaturation at every timestep
                for compartment in self.compartments:
                    step_supersaturation = np.max(compartment.slab / compartment.m_values)
                    if step_supersaturation > max_supersaturation:
                        max_supersaturation = step_supersaturation

                # Capture tissue state at deepest point for NDL
                if current_depth >= max_depth_seen:
                    max_depth_seen = current_depth
                    max_depth_slabs = [compartment.slab.copy() for compartment in self.compartments]

                # Record state periodically
                if current_time % sample_interval < self.dt:
                    times.append(current_time / 60)  # Back to minutes
                    depths.append(current_depth)

                    # Store state of all compartments
                    current_state = [compartment.slab.copy() for compartment in self.compartments]
                    slab_history.append(current_state)

                    # Calculate max loads across all compartments
                    all_max_loads = [np.max(compartment.slab) for compartment in self.compartments]
                    max_loads.append(max(all_max_loads))

                    # Calculate margins across all compartments
                    all_margins = [np.min(compartment.m_values - compartment.slab) for compartment in self.compartments]
                    margins.append(min(all_margins))

                current_time += self.dt

        # Process final point
        if profile.points:
            t_final, d_final, _, _ = profile.points[-1]
            times.append(t_final)
            depths.append(d_final)

            # Store final state of all compartments
            current_state = [compartment.slab.copy() for compartment in self.compartments]
            slab_history.append(current_state)

            # Calculate max loads across all compartments
            all_max_loads = [np.max(compartment.slab) for compartment in self.compartments]
            max_loads.append(max(all_max_loads))

            # Calculate margins across all compartments
            all_margins = [np.min(compartment.m_values - compartment.slab) for compartment in self.compartments]
            margins.append(min(all_margins))

        # Calculate summary metrics
        slab_history = np.array(slab_history)  # Shape: [time_steps, num_compartments, slices_per_compartment]
        max_tissue_load = np.max(slab_history)
        min_margin = np.min(margins)

        # Find which compartment and slice is closest to its limit (critical compartment/slice)
        final_slabs = {compartment.name: compartment.slab.copy() for compartment in self.compartments}

        # Determine which compartment has the highest risk at the end
        critical_compartment = ""
        overall_critical_slice = 0
        max_risk = 0

        for compartment in self.compartments:
            risk_array = compartment.slab / compartment.m_values
            max_risk_in_compartment = np.max(risk_array)
            max_slice_in_compartment = int(np.argmax(risk_array))

            if max_risk_in_compartment > max_risk:
                max_risk = max_risk_in_compartment
                critical_compartment = compartment.name
                overall_critical_slice = max_slice_in_compartment

        # Also check final state supersaturation (covers the final profile point)
        for compartment in self.compartments:
            final_supersaturation = np.max(compartment.slab / compartment.m_values)
            if final_supersaturation > max_supersaturation:
                max_supersaturation = final_supersaturation

        # Calculate NDL from tissue state at max depth (not surface)
        final_ndl = self.calculate_multi_compartment_ndl(max_depth_slabs, max_depth_seen)

        return SlabResult(
            times=times,
            depths=depths,
            slab_history=slab_history,
            max_loads=max_loads,
            margins=margins,
            max_tissue_load=max_tissue_load,
            min_margin=min_margin,
            critical_compartment=critical_compartment,
            critical_slice=overall_critical_slice,
            max_supersaturation=max_supersaturation,
            final_ndl=float(final_ndl),
            final_slabs=final_slabs,
        )

    def calculate_multi_compartment_ndl(
        self, current_slabs: List[np.ndarray], depth: float, max_time: int = 100
    ) -> int:
        """
        Simulates the future to find how many minutes until any compartment hits its limit.
        Uses a linear search approach - clones state and runs forward in time.

        Args:
            current_slabs: Current tissue states for each compartment
            depth: Current depth in meters
            max_time: Maximum time to simulate (minutes)

        Returns:
            NDL in minutes (max_time if > max_time, essentially unlimited)
        """
        # First check if we're already over the limit in any compartment
        for i, compartment in enumerate(self.compartments):
            if np.any(current_slabs[i] > compartment.m_values):
                return 0

        # Calculate pressure at current depth (we are staying here)
        p_bottom = self._get_atmospheric_pressure() + self._get_hydrostatic_pressure(
            depth
        )
        ppn2_bottom = p_bottom * (1 - self.f_o2)

        # Create shadow slabs so we don't mess up the real dive
        shadow_slabs = [slab.copy() for slab in current_slabs]

        # Look ahead up to max_time minutes
        for minute in range(max_time + 1):
            # Run simulation for 60 seconds (one minute block)
            steps_per_min = int(60 / self.dt)

            for _ in range(steps_per_min):
                # Update each compartment
                for i, compartment in enumerate(self.compartments):
                    # Update Boundary
                    if self.permeability is None:
                        shadow_slabs[i][0] = ppn2_bottom
                    else:
                        flux = self.permeability * (ppn2_bottom - shadow_slabs[i][0])
                        shadow_slabs[i][0] += flux * self.dt

                    # Diffuse (vectorized)
                    shadow_slabs[i][1:-1] += compartment.k * (
                        shadow_slabs[i][:-2] - 2 * shadow_slabs[i][1:-1] + shadow_slabs[i][2:]
                    )
                    # No-flux deep end
                    shadow_slabs[i][-1] = shadow_slabs[i][-2]

            # The NDL Check:
            # If ANY slice in ANY compartment exceeds its corresponding M-Value
            for i, compartment in enumerate(self.compartments):
                if np.any(shadow_slabs[i] > compartment.m_values):
                    return minute  # We found the limit!

        return max_time  # More than max_time mins (essentially unlimited)

    def calculate_ndl(
        self, current_slab: np.ndarray, depth: float, max_time: int = 100
    ) -> int:
        """
        Legacy method for single compartment NDL calculation.
        For backward compatibility.
        """
        # This is kept for backward compatibility, but the multi-compartment version should be used
        # First check if we're already over the limit
        if np.any(current_slab > self.m_values):
            return 0

        p_bottom = self._get_atmospheric_pressure() + self._get_hydrostatic_pressure(
            depth
        )
        ppn2_bottom = p_bottom * (1 - self.f_o2)

        # Create shadow slab so we don't mess up the real dive
        shadow_slab = current_slab.copy()

        # Look ahead up to max_time minutes
        for minute in range(max_time + 1):
            # Run simulation for 60 seconds (one minute block)
            steps_per_min = int(60 / self.dt)

            for _ in range(steps_per_min):
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

            # The NDL Check:
            # If ANY slice in the shadow slab is higher than its matching M-Value
            if np.any(shadow_slab > self.m_values):
                return minute  # We found the limit!

        return max_time  # More than max_time mins (essentially unlimited)

    def get_compartment_config(self):
        """Return the compartment configuration for serialization/pickling."""
        config = []
        for compartment in self.compartments:
            config.append({
                "name": compartment.name,
                "D": compartment.D,
                "slices": compartment.slices,
                "m_value_max": compartment.m_value_max,
                "m_value_min": compartment.m_value_min
            })
        return config

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
    from backtest.profile_generator import ProfileGenerator

    gen = ProfileGenerator()
    profile = gen.generate_square(depth=30, bottom_time=20)

    # Use the new multi-compartment model
    model = SlabModel()
    result = model.run(profile)

    print(f"Profile: {profile.name}")
    print(f"Max tissue load: {result.max_tissue_load:.3f} bar")
    print(f"Min margin: {result.min_margin:.3f} bar")
    print(f"Critical compartment: {result.critical_compartment}")
    print(f"Critical slice: {result.critical_slice}")
    print(f"Max supersaturation: {result.max_supersaturation:.2%}")
    print(f"Exceeded limit: {result.exceeded_limit}")
    print(f"Final NDL: {result.final_ndl:.2f} minutes")
