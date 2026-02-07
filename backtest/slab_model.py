"""
Finite Difference Slab Model for tissue gas diffusion.

Implements a 1D slab diffusion model using the finite difference method
to simulate nitrogen uptake and offgassing in tissue.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import yaml
import os

from backtest.profile_generator import DiveProfile


def _compute_v_crit(D: float, slices: int, dx: float, permeability: Optional[float], k: float) -> float:
    """Compute critical volume threshold for a compartment from global k.
    V_crit = k / (L²/D + L/permeability)
    With perfect perfusion (permeability=None), the permeability term is 0.
    Used as fallback when per-compartment v_crit is not specified.
    """
    L = slices * dx
    tau_diffusion = (L * L) / D
    tau_permeability = L / permeability if permeability is not None else 0.0
    return k / (tau_diffusion + tau_permeability)


@dataclass
class TissueCompartment:
    """Represents a single tissue compartment with its own properties."""
    name: str
    D: float  # Diffusion coefficient
    slices: int  # Number of slices
    slab: np.ndarray  # Current tissue state
    v_crit: float  # Critical volume threshold (for NDL & risk scoring)
    k: float  # Stability constant for this compartment
    g_crit: float = 0.0  # Critical gradient at surface (for ceiling/deco checks)


@dataclass
class DecoStop:
    """A single decompression stop."""
    depth: float         # Stop depth in meters
    duration_min: float  # Duration at this stop in minutes


@dataclass
class DecoSchedule:
    """Complete decompression schedule from the Slab model."""
    stops: List[DecoStop]               # Ordered deepest-first
    tts: float                          # Total Time to Surface (minutes)
    total_deco_time: float              # Sum of stop durations (minutes)
    final_slabs: dict                   # Tissue state at surface after deco
    requires_deco: bool                 # True if any stops were needed
    controlling_compartment: str        # Compartment driving the longest stop
    ceiling_at_start: float             # Ceiling depth when planning began (meters)


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

    # Summary metrics
    max_tissue_load: float
    min_margin: float  # 1.0 - max_cv_ratio (positive = safe)
    critical_compartment: str  # Name of controlling compartment
    max_cv_ratio: float  # Peak critical volume ratio (>1.0 = exceeded)
    final_ndl: float  # NDL remaining at end of dive (minutes)
    final_slabs: dict  # Final tissue state for each compartment

    # Deco fields (populated when model computes ceiling)
    ceiling_at_bottom: float = 0.0  # Ceiling depth at end of bottom time (meters)
    deco_schedule: Optional[DecoSchedule] = None  # Populated by generate_deco_profile

    @property
    def exceeded_limit(self) -> bool:
        return self.max_cv_ratio > 1.0

    @property
    def risk_score(self) -> float:
        return self.max_cv_ratio


_UNSET = object()  # Sentinel to distinguish "not provided" from None


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
        dt=_UNSET,
        dx=_UNSET,
        permeability=_UNSET,
        f_o2=_UNSET,
        surface_altitude_m=_UNSET,
        critical_volume_k=_UNSET,
        conservatism=_UNSET,
    ):
        """
        Initialize the Multi-Compartment Slab model.

        Args:
            config_path: Path to YAML config file. If provided, loads parameters from file.
            compartments_config: List of dicts with compartment params (overrides config file)
                                Each dict has: name, D, slices, and optionally v_crit
                                If None, uses default 3-compartment model or loads from config file
            dt: Time step in seconds (smaller = more precise)
            dx: Distance between slices (arbitrary units)
            permeability: Blood-tissue barrier permeability (None = perfect perfusion)
                          Controls delay at blood-tissue interface. Higher = faster equilibration.
                          Typical range: 0.001 (slow barrier) to 0.1 (fast barrier)
            f_o2: Breathing gas O2 fraction (Air = 0.21)
            surface_altitude_m: Altitude of the water surface (0m = Sea Level)
            critical_volume_k: Calibration constant for critical volume threshold (default 1.0)
            conservatism: Safety factor for v_crit scaling (default 1.0).
                          Values < 1.0 are more conservative (e.g., 0.85 = 15% safety margin).
                          Effective v_crit = v_crit * conservatism.
        """

        # Load configuration from file if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

        # Use explicit values if provided, otherwise fall back to config, then defaults
        self.dt = dt if dt is not _UNSET else config.get('dt', 0.5)
        self.dx = dx if dx is not _UNSET else config.get('dx', 1.0)
        self.permeability = permeability if permeability is not _UNSET else config.get('permeability', None)
        self.f_o2 = f_o2 if f_o2 is not _UNSET else config.get('f_o2', 0.21)
        self.surface_altitude_m = surface_altitude_m if surface_altitude_m is not _UNSET else config.get('surface_altitude_m', 0.0)
        self.critical_volume_k = critical_volume_k if critical_volume_k is not _UNSET else config.get('critical_volume_k', 1.0)
        self.conservatism = conservatism if conservatism is not _UNSET else config.get('conservatism', 1.0)

        # Load deco planning config
        deco_config = config.get('deco', {})
        self.deco_stop_increment = deco_config.get('stop_increment', 3.0)
        self.deco_last_stop_depth = deco_config.get('last_stop_depth', 3.0)
        self.deco_ascent_rate = deco_config.get('ascent_rate', 10.0)
        self.deco_max_stop_time = deco_config.get('max_stop_time', 120)
        self.boyle_exponent = deco_config.get('boyle_exponent', 0.5)

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
                {"name": "Spine", "D": 0.002, "slices": 20},
                # Medium (Muscle)
                {"name": "Muscle", "D": 0.0005, "slices": 20},
                # Slow, Robust but traps gas (Joints)
                {"name": "Joints", "D": 0.0001, "slices": 20},
            ]

        # Create compartments
        self.compartments = []
        for config_item in final_compartments_config:
            name = config_item["name"]
            D = config_item["D"]
            slices = config_item["slices"]

            slab = np.zeros(slices)

            k = (D * self.dt) / (self.dx ** 2)
            if k > 0.5:
                raise ValueError(f"Stability Error in {name}")

            # Use per-compartment v_crit if provided, otherwise derive from global k
            if "v_crit" in config_item:
                v_crit = config_item["v_crit"]
            else:
                v_crit = _compute_v_crit(D, slices, self.dx, self.permeability, self.critical_volume_k)

            # g_crit: critical gradient at surface for ceiling/deco checks.
            # Calibrated as slab[1] - ppN2_surface at the NDL boundary (30m reference).
            g_crit = config_item.get("g_crit", 0.0)

            compartment = TissueCompartment(
                name=name, D=D, slices=slices, slab=slab,
                v_crit=v_crit, k=k, g_crit=g_crit,
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

    def _compute_excess_gas(self, slab: np.ndarray, reference_ppn2: float = None) -> float:
        """Compute total excess dissolved gas above a reference pressure.

        Args:
            slab: Tissue state array
            reference_ppn2: Reference N2 pressure. If None, uses surface equilibrium.
        """
        if reference_ppn2 is None:
            reference_ppn2 = self._get_atmospheric_pressure() * (1 - self.f_o2)
        excess = np.maximum(0, slab - reference_ppn2)
        return np.sum(excess) * self.dx

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

        # Track tissue state at max depth for NDL calculation
        max_depth_seen = 0.0
        max_depth_slabs = [compartment.slab.copy() for compartment in self.compartments]

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

        # Calculate summary metrics
        slab_history = np.array(slab_history)  # Shape: [time_steps, num_compartments, slices_per_compartment]
        max_tissue_load = np.max(slab_history)

        # Store final tissue state for each compartment
        final_slabs = {compartment.name: compartment.slab.copy() for compartment in self.compartments}

        # Evaluate critical volume ratio at end of profile (final tissue state).
        # This measures the decompression stress after the dive is complete,
        # avoiding transient ascent supersaturation artifacts.
        max_cv_ratio = 0.0
        critical_compartment_name = ""
        for compartment in self.compartments:
            excess_gas = self._compute_excess_gas(compartment.slab)
            effective_v_crit = compartment.v_crit * self.conservatism
            cv_ratio = excess_gas / effective_v_crit if effective_v_crit > 0 else 0.0
            if cv_ratio > max_cv_ratio:
                max_cv_ratio = cv_ratio
                critical_compartment_name = compartment.name

        min_margin = 1.0 - max_cv_ratio

        # Calculate ceiling at max depth (how deep must we stop?)
        ceiling_at_bottom = self.calculate_ceiling(max_depth_slabs, max_depth_seen)

        # Calculate NDL from tissue state at max depth (not surface)
        final_ndl = self.calculate_multi_compartment_ndl(max_depth_slabs, max_depth_seen)

        return SlabResult(
            times=times,
            depths=depths,
            slab_history=slab_history,
            max_loads=max_loads,
            max_tissue_load=max_tissue_load,
            min_margin=min_margin,
            critical_compartment=critical_compartment_name,
            max_cv_ratio=max_cv_ratio,
            final_ndl=float(final_ndl),
            final_slabs=final_slabs,
            ceiling_at_bottom=ceiling_at_bottom,
        )

    def _simulate_ascent_to_depth(
        self,
        slabs: List[np.ndarray],
        from_depth: float,
        to_depth: float,
        ascent_rate: float = 10.0,
    ) -> List[np.ndarray]:
        """Simulate ascent from one depth to another (shallower) depth.

        Args:
            slabs: Current tissue states (will NOT be modified — copies are used)
            from_depth: Starting depth in meters
            to_depth: Target depth in meters (must be <= from_depth)
            ascent_rate: Ascent rate in m/min (default 10)

        Returns:
            List of slab arrays representing tissue state at to_depth after ascent
        """
        ascent_slabs = [s.copy() for s in slabs]
        depth_delta = from_depth - to_depth
        if depth_delta <= 0:
            return ascent_slabs

        ascent_time_sec = (depth_delta / ascent_rate) * 60
        steps = int(ascent_time_sec / self.dt)

        for step in range(steps):
            ratio = step / max(steps, 1)
            current_depth = from_depth - ratio * depth_delta
            ppn2 = self._get_ppn2(current_depth)

            for i, compartment in enumerate(self.compartments):
                if self.permeability is None:
                    ascent_slabs[i][0] = ppn2
                else:
                    flux = self.permeability * (ppn2 - ascent_slabs[i][0])
                    ascent_slabs[i][0] += flux * self.dt
                ascent_slabs[i][1:-1] += compartment.k * (
                    ascent_slabs[i][:-2] - 2 * ascent_slabs[i][1:-1] + ascent_slabs[i][2:]
                )
                ascent_slabs[i][-1] = ascent_slabs[i][-2]

        return ascent_slabs

    def _simulate_ascent(self, slabs: List[np.ndarray], depth: float, ascent_rate: float = 10.0) -> List[np.ndarray]:
        """Simulate direct ascent to surface and return tissue state at surface."""
        return self._simulate_ascent_to_depth(slabs, depth, 0.0, ascent_rate)

    def _check_safe_at_depth(self, slabs: List[np.ndarray], depth: float) -> bool:
        """Check if current tissue state allows being at a given depth.

        Uses the boundary gradient approach (Hennessy critical gradient model):
        With perfect perfusion, slab[0] = ambient ppN2 (no supersaturation at
        boundary). The first interior slice slab[1] holds the highest tension
        and drives outward diffusion — this is where bubble formation risk is
        highest. The critical gradient g_crit scales with ambient pressure
        via Boyle's Law (higher pressure compresses bubbles).

        Args:
            slabs: Current tissue states
            depth: Candidate depth to check safety at (meters)

        Returns:
            True if safe at this depth
        """
        ppn2_at_depth = self._get_ppn2(depth)
        p_ambient = self._get_atmospheric_pressure() + self._get_hydrostatic_pressure(depth)
        p_surface = self._get_atmospheric_pressure()
        pressure_ratio = p_ambient / p_surface

        for i, compartment in enumerate(self.compartments):
            gradient = slabs[i][1] - ppn2_at_depth
            effective_g_crit = compartment.g_crit * self.conservatism * (pressure_ratio ** self.boyle_exponent)
            if effective_g_crit > 0 and gradient > effective_g_crit:
                return False
        return True

    def _simulate_time_at_depth(
        self,
        slabs: List[np.ndarray],
        depth: float,
        duration_min: float,
    ) -> List[np.ndarray]:
        """Simulate holding at a constant depth for a given duration.

        Args:
            slabs: Current tissue states (will NOT be modified — copies are used)
            depth: Depth in meters
            duration_min: Duration in minutes

        Returns:
            List of slab arrays representing tissue state after holding
        """
        result_slabs = [s.copy() for s in slabs]
        ppn2 = self._get_ppn2(depth)
        steps = int(duration_min * 60 / self.dt)

        for _ in range(steps):
            for i, compartment in enumerate(self.compartments):
                if self.permeability is None:
                    result_slabs[i][0] = ppn2
                else:
                    flux = self.permeability * (ppn2 - result_slabs[i][0])
                    result_slabs[i][0] += flux * self.dt
                result_slabs[i][1:-1] += compartment.k * (
                    result_slabs[i][:-2] - 2 * result_slabs[i][1:-1] + result_slabs[i][2:]
                )
                result_slabs[i][-1] = result_slabs[i][-2]

        return result_slabs

    def calculate_ceiling(
        self,
        slabs: List[np.ndarray],
        current_depth: float,
        stop_increment: float = None,
    ) -> float:
        """Find the shallowest safe depth (ceiling) given current tissue state.

        Uses binary search over stop-increment depths. The ceiling is the
        shallowest depth where excess_gas / (v_crit * conservatism) <= 1.0
        for all compartments, evaluated at that depth's ambient pressure.

        Args:
            slabs: Current tissue states for each compartment
            current_depth: Current depth in meters (upper bound for search)
            stop_increment: Depth increment for stops (default from config)

        Returns:
            Ceiling depth in meters, rounded up to next stop_increment.
            0.0 if safe to surface directly.
        """
        if stop_increment is None:
            stop_increment = self.deco_stop_increment

        # Quick check: safe at surface?
        if self._check_safe_at_depth(slabs, 0.0):
            return 0.0

        # Build candidate depths: [stop_increment, 2*stop_increment, ..., max]
        import math
        max_stop = math.ceil(current_depth / stop_increment) * stop_increment
        # candidates are ordered shallow-to-deep
        candidates = [i * stop_increment for i in range(1, int(max_stop / stop_increment) + 1)]

        if not candidates:
            return 0.0

        # Binary search: find shallowest depth where safe
        # Property: if safe at depth D, also safe at any deeper depth
        lo, hi = 0, len(candidates) - 1

        # Verify safe at deepest candidate (current depth area)
        if not self._check_safe_at_depth(slabs, candidates[hi]):
            return max_stop  # Tissue overloaded even at current depth

        while lo < hi:
            mid = (lo + hi) // 2
            if self._check_safe_at_depth(slabs, candidates[mid]):
                hi = mid  # Try shallower
            else:
                lo = mid + 1  # Need deeper

        return candidates[lo]

    def plan_deco(
        self,
        slabs: List[np.ndarray],
        depth: float,
        ascent_rate: float = None,
        last_stop_depth: float = None,
        stop_increment: float = None,
        max_stop_time: int = None,
        gas_switches: Optional[List[dict]] = None,
    ) -> DecoSchedule:
        """Generate a complete decompression schedule from current tissue state.

        Starting from current depth and tissue state, iteratively:
        1. Calculate ceiling
        2. Ascend to ceiling (simulating tissue changes during ascent)
        3. Hold at ceiling until ceiling clears to next shallower stop
        4. Repeat until ceiling is 0 (safe to surface)

        Args:
            slabs: Current tissue states for each compartment
            depth: Current depth in meters
            ascent_rate: Ascent rate in m/min (default from config)
            last_stop_depth: Shallowest possible stop (default from config)
            stop_increment: Depth increment between stops (default from config)
            max_stop_time: Max time at any single stop in minutes (default from config)
            gas_switches: Future placeholder for gas switch depths (not yet implemented)

        Returns:
            DecoSchedule with stops, TTS, and final tissue state
        """
        if ascent_rate is None:
            ascent_rate = self.deco_ascent_rate
        if last_stop_depth is None:
            last_stop_depth = self.deco_last_stop_depth
        if stop_increment is None:
            stop_increment = self.deco_stop_increment
        if max_stop_time is None:
            max_stop_time = self.deco_max_stop_time

        working_slabs = [s.copy() for s in slabs]
        current_depth = depth
        stops = []
        total_ascent_time = 0.0
        controlling_compartment = ""
        max_stop_duration = 0

        # Record initial ceiling
        ceiling_at_start = self.calculate_ceiling(working_slabs, current_depth, stop_increment)

        while True:
            ceiling = self.calculate_ceiling(working_slabs, current_depth, stop_increment)

            if ceiling <= 0.0:
                # Safe to ascend to surface
                working_slabs = self._simulate_ascent_to_depth(
                    working_slabs, current_depth, 0.0, ascent_rate
                )
                total_ascent_time += current_depth / ascent_rate
                break

            # Ascend to ceiling
            stop_depth = ceiling
            working_slabs = self._simulate_ascent_to_depth(
                working_slabs, current_depth, stop_depth, ascent_rate
            )
            total_ascent_time += (current_depth - stop_depth) / ascent_rate
            current_depth = stop_depth

            # Determine next target: one increment shallower, or surface
            next_stop = max(0.0, stop_depth - stop_increment)

            # Hold at stop until ceiling clears to next_stop
            stop_time = 0
            for minute in range(1, max_stop_time + 1):
                working_slabs = self._simulate_time_at_depth(
                    working_slabs, stop_depth, 1.0
                )
                stop_time = minute

                new_ceiling = self.calculate_ceiling(working_slabs, current_depth, stop_increment)
                if new_ceiling <= next_stop:
                    break

            stops.append(DecoStop(depth=stop_depth, duration_min=stop_time))

            # Track which compartment controls the longest stop
            if stop_time > max_stop_duration:
                max_stop_duration = stop_time
                # Find controlling compartment using boundary gradient
                ppn2_at_stop = self._get_ppn2(stop_depth)
                p_amb = self._get_atmospheric_pressure() + self._get_hydrostatic_pressure(stop_depth)
                p_surf = self._get_atmospheric_pressure()
                p_ratio = p_amb / p_surf
                max_ratio = 0.0
                for i, comp in enumerate(self.compartments):
                    gradient = working_slabs[i][1] - ppn2_at_stop
                    effective_g_crit = comp.g_crit * self.conservatism * (p_ratio ** self.boyle_exponent)
                    ratio = gradient / effective_g_crit if effective_g_crit > 0 else 0.0
                    if ratio > max_ratio:
                        max_ratio = ratio
                        controlling_compartment = comp.name

            if stop_time >= max_stop_time:
                break  # Safety limit reached

        total_deco_time = sum(s.duration_min for s in stops)
        tts = total_deco_time + total_ascent_time
        final_slabs = {
            comp.name: working_slabs[i].copy()
            for i, comp in enumerate(self.compartments)
        }

        return DecoSchedule(
            stops=stops,
            tts=tts,
            total_deco_time=total_deco_time,
            final_slabs=final_slabs,
            requires_deco=len(stops) > 0,
            controlling_compartment=controlling_compartment,
            ceiling_at_start=ceiling_at_start,
        )

    def generate_deco_profile(
        self,
        depth: float,
        bottom_time: float,
        ascent_rate: float = None,
        descent_rate: float = 20.0,
        fO2: float = 0.21,
        fHe: float = 0.0,
        last_stop_depth: float = None,
    ) -> Tuple[DiveProfile, DecoSchedule]:
        """Generate a complete dive profile including computed deco stops.

        Simulates descent + bottom time to get tissue state, then plans
        decompression and builds a DiveProfile with stops included.

        Args:
            depth: Bottom depth in meters
            bottom_time: Time at depth in minutes
            ascent_rate: Ascent rate in m/min (default from config)
            descent_rate: Descent rate in m/min
            fO2: Oxygen fraction
            fHe: Helium fraction
            last_stop_depth: Shallowest deco stop depth (default from config)

        Returns:
            Tuple of (DiveProfile with deco stops, DecoSchedule)
        """
        from backtest.profile_generator import ProfileGenerator

        if ascent_rate is None:
            ascent_rate = self.deco_ascent_rate
        if last_stop_depth is None:
            last_stop_depth = self.deco_last_stop_depth

        # Initialize tissue at surface equilibrium
        p_surface_bar = self._get_atmospheric_pressure()
        ppn2_surface = p_surface_bar * (1 - fO2)
        for compartment in self.compartments:
            compartment.slab[:] = ppn2_surface

        # Simulate descent
        descent_time_sec = (depth / descent_rate) * 60
        descent_steps = int(descent_time_sec / self.dt)
        for step in range(descent_steps):
            ratio = step / max(descent_steps, 1)
            current_depth = ratio * depth
            ppn2 = self._get_ppn2(current_depth, fO2)
            for compartment in self.compartments:
                self._update_compartment(compartment, ppn2)

        # Simulate bottom time
        ppn2_bottom = self._get_ppn2(depth, fO2)
        bottom_steps = int(bottom_time * 60 / self.dt)
        for _ in range(bottom_steps):
            for compartment in self.compartments:
                self._update_compartment(compartment, ppn2_bottom)

        # Extract tissue state for deco planning
        tissue_slabs = [comp.slab.copy() for comp in self.compartments]

        # Plan decompression
        schedule = self.plan_deco(
            tissue_slabs, depth,
            ascent_rate=ascent_rate,
            last_stop_depth=last_stop_depth,
        )

        # Build the DiveProfile using ProfileGenerator
        gen = ProfileGenerator(
            descent_rate=descent_rate,
            ascent_rate=ascent_rate,
        )
        deco_stops = [(stop.depth, stop.duration_min) for stop in schedule.stops]
        profile = gen.generate_deco_square(
            depth, bottom_time, deco_stops, fO2=fO2, fHe=fHe,
        )

        return profile, schedule

    def calculate_multi_compartment_ndl(
        self, current_slabs: List[np.ndarray], depth: float, max_time: int = 100
    ) -> int:
        """
        Simulates the future to find how many minutes until any compartment hits its limit.

        NDL answers: "how long before I can't ascend directly to surface?"
        For each minute at depth, simulates a direct ascent and checks if
        the post-ascent tissue state would exceed V_crit.

        Args:
            current_slabs: Current tissue states for each compartment
            depth: Current depth in meters
            max_time: Maximum time to simulate (minutes)

        Returns:
            NDL in minutes (max_time if > max_time, essentially unlimited)
        """
        p_surface_equil = self._get_atmospheric_pressure() * (1 - self.f_o2)

        # Check if already over limit (simulate ascent from current state)
        surface_slabs = self._simulate_ascent(current_slabs, depth)
        for i, comp in enumerate(self.compartments):
            excess = np.sum(np.maximum(0, surface_slabs[i] - p_surface_equil)) * self.dx
            if excess > comp.v_crit * self.conservatism:
                return 0

        ppn2_bottom = self._get_ppn2(depth)
        shadow_slabs = [s.copy() for s in current_slabs]

        steps_per_min = int(60 / self.dt)
        for minute in range(1, max_time + 1):
            # Simulate 1 minute at depth
            for _ in range(steps_per_min):
                for i, compartment in enumerate(self.compartments):
                    if self.permeability is None:
                        shadow_slabs[i][0] = ppn2_bottom
                    else:
                        flux = self.permeability * (ppn2_bottom - shadow_slabs[i][0])
                        shadow_slabs[i][0] += flux * self.dt
                    shadow_slabs[i][1:-1] += compartment.k * (
                        shadow_slabs[i][:-2] - 2 * shadow_slabs[i][1:-1] + shadow_slabs[i][2:]
                    )
                    shadow_slabs[i][-1] = shadow_slabs[i][-2]

            # Simulate ascent and check at surface
            surface_slabs = self._simulate_ascent(shadow_slabs, depth)
            for i, comp in enumerate(self.compartments):
                excess = np.sum(np.maximum(0, surface_slabs[i] - p_surface_equil)) * self.dx
                if excess > comp.v_crit * self.conservatism:
                    return minute

        return max_time

    def calculate_ndl(
        self, current_slab: np.ndarray, depth: float, max_time: int = 100
    ) -> int:
        """Legacy single-compartment NDL. Deprecated."""
        raise NotImplementedError(
            "Legacy single-compartment NDL not supported with depth-dependent M-values. "
            "Use calculate_multi_compartment_ndl instead."
        )

    def get_compartment_config(self):
        """Return the compartment configuration for serialization/pickling."""
        config = []
        for compartment in self.compartments:
            config.append({
                "name": compartment.name,
                "D": compartment.D,
                "slices": compartment.slices,
                "v_crit": compartment.v_crit,
                "g_crit": compartment.g_crit,
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
    print(f"Min margin: {result.min_margin:.3f}")
    print(f"Critical compartment: {result.critical_compartment}")
    print(f"Max CV ratio: {result.max_cv_ratio:.4f}")
    print(f"Exceeded limit: {result.exceeded_limit}")
    print(f"Final NDL: {result.final_ndl:.2f} minutes")
