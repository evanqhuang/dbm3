"""
Interface to libbuhlmann binary for running Bühlmann decompression calculations.

Treats the src/dive binary as a black-box function f(profile) -> risk_metrics.
"""

import subprocess
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from .profile_generator import DiveProfile


@dataclass
class BuhlmannResult:
    """Results from running a dive profile through the Bühlmann model."""

    # Time series data
    times: List[float]
    pressures: List[float]

    # Per-compartment data (16 compartments)
    compartment_n2: List[List[float]]  # [time_idx][compartment_idx]
    compartment_he: List[List[float]]  # [time_idx][compartment_idx]

    # Derived metrics
    ceilings: List[float]  # Decompression ceiling at each time step
    ndl_times: List[float]  # No-decompression limit at each time step

    # Summary metrics
    max_ceiling: float  # Maximum ceiling reached (bar)
    min_ndl: float  # Minimum NDL during dive (minutes)
    max_supersaturation: float  # Maximum M-value percentage reached

    @property
    def requires_deco(self) -> bool:
        """True if any ceiling exceeded surface pressure (1 bar)."""
        return self.max_ceiling > 1.0

    @property
    def risk_score(self) -> float:
        """
        Normalized risk score (0-1+).
        Based on maximum supersaturation relative to M-value.
        >1.0 indicates M-value exceeded.
        """
        return self.max_supersaturation


class BuhlmannRunner:
    """
    Interface to run dive profiles through libbuhlmann.

    Uses subprocess to pipe dive profile data to the compiled binary
    and parse the output for compartment states and safety metrics.
    """

    def __init__(self, binary_path: Optional[str] = None):
        """
        Initialize the runner.

        Args:
            binary_path: Path to the dive binary. If None, auto-detect.
        """
        if binary_path is None:
            # Auto-detect binary location
            possible_paths = [
                Path(__file__).parent.parent / "libbuhlmann" / "src" / "dive",
                Path("libbuhlmann/src/dive"),
                Path("./src/dive"),
            ]
            for path in possible_paths:
                if path.exists():
                    binary_path = str(path.resolve())
                    break

            if binary_path is None:
                raise FileNotFoundError(
                    "Could not find libbuhlmann dive binary. "
                    "Please compile it or specify the path."
                )

        self.binary_path = binary_path

        # Verify binary exists and is executable
        if not os.path.isfile(self.binary_path):
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")
        if not os.access(self.binary_path, os.X_OK):
            raise PermissionError(f"Binary not executable: {self.binary_path}")

    def run(self, profile: DiveProfile) -> BuhlmannResult:
        """
        Run a dive profile through the Bühlmann model.

        Args:
            profile: DiveProfile object with dive data

        Returns:
            BuhlmannResult with compartment states and risk metrics
        """
        # Convert profile to libbuhlmann input format
        input_data = profile.to_buhlmann_format()

        # Run the binary
        process = subprocess.Popen(
            [self.binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate(input=input_data)

        if process.returncode != 0:
            raise RuntimeError(f"libbuhlmann failed: {stderr}")

        # Parse output
        return self._parse_output(stdout)

    def _parse_output(self, output: str) -> BuhlmannResult:
        """
        Parse libbuhlmann output.

        Output format per line:
        time pressure [n2_p he_p]*16 ceiling nodectime

        That's: 2 + 16*2 + 2 = 36 values per line
        """
        times = []
        pressures = []
        compartment_n2 = []
        compartment_he = []
        ceilings = []
        ndl_times = []

        num_compartments = 16

        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            values = line.split()
            if len(values) < 36:
                continue  # Skip malformed lines

            try:
                time = float(values[0])
                pressure = float(values[1])

                # Parse compartment data (16 compartments, n2 and he each)
                n2_values = []
                he_values = []
                for i in range(num_compartments):
                    n2_idx = 2 + i * 2
                    he_idx = 3 + i * 2
                    n2_values.append(float(values[n2_idx]))
                    he_values.append(float(values[he_idx]))

                # Last two values are ceiling and nodectime
                ceiling = float(values[-2])
                ndl = float(values[-1])

                times.append(time)
                pressures.append(pressure)
                compartment_n2.append(n2_values)
                compartment_he.append(he_values)
                ceilings.append(ceiling)
                ndl_times.append(ndl)

            except (ValueError, IndexError) as e:
                continue  # Skip parse errors

        if not times:
            raise ValueError("No valid output from libbuhlmann")

        # Calculate summary metrics
        max_ceiling = max(ceilings)
        min_ndl = min(ndl_times)

        # Calculate max supersaturation
        # Supersaturation = tissue_pressure / M_value
        # For simplicity, we use ceiling as a proxy: if ceiling > surface, we're supersaturated
        # A more accurate calculation would require the M-value constants
        max_supersaturation = self._calculate_max_supersaturation(
            compartment_n2, compartment_he, pressures
        )

        return BuhlmannResult(
            times=times,
            pressures=pressures,
            compartment_n2=compartment_n2,
            compartment_he=compartment_he,
            ceilings=ceilings,
            ndl_times=ndl_times,
            max_ceiling=max_ceiling,
            min_ndl=min_ndl,
            max_supersaturation=max_supersaturation,
        )

    def _calculate_max_supersaturation(
        self,
        compartment_n2: List[List[float]],
        compartment_he: List[List[float]],
        pressures: List[float],
    ) -> float:
        """
        Calculate maximum supersaturation across all compartments and time steps.

        Uses ZH-L16A M-values for the calculation.
        Returns value as fraction of M-value (1.0 = at limit, >1.0 = exceeded).
        """
        # ZH-L16 a and b values for N2 (simplified, first 16 compartments)
        # These are the surface M-value parameters
        zh_l16_n2_a = [
            1.2599,
            1.0000,
            0.8618,
            0.7562,
            0.6667,
            0.5933,
            0.5282,
            0.4701,
            0.4187,
            0.3798,
            0.3497,
            0.3223,
            0.2971,
            0.2737,
            0.2523,
            0.2327,
        ]
        zh_l16_n2_b = [
            0.5050,
            0.6514,
            0.7222,
            0.7825,
            0.8126,
            0.8434,
            0.8693,
            0.8910,
            0.9092,
            0.9222,
            0.9319,
            0.9403,
            0.9477,
            0.9544,
            0.9602,
            0.9653,
        ]

        max_ratio = 0.0

        for t_idx, pressure in enumerate(pressures):
            for c_idx in range(min(16, len(compartment_n2[t_idx]))):
                # Total inert gas pressure
                p_inert = compartment_n2[t_idx][c_idx]
                if compartment_he and len(compartment_he[t_idx]) > c_idx:
                    p_inert += compartment_he[t_idx][c_idx]

                # M-value at current ambient pressure
                # M = a + P_ambient / b
                a = zh_l16_n2_a[c_idx]
                b = zh_l16_n2_b[c_idx]
                m_value = a + pressure / b

                # Supersaturation ratio
                ratio = p_inert / m_value if m_value > 0 else 0
                max_ratio = max(max_ratio, ratio)

        return max_ratio

    def run_batch(self, profiles: List[DiveProfile]) -> List[BuhlmannResult]:
        """
        Run multiple profiles through the model.

        Args:
            profiles: List of DiveProfile objects

        Returns:
            List of BuhlmannResult objects
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
    profile = gen.generate_square(depth=20, bottom_time=5)

    runner = BuhlmannRunner()
    result = runner.run(profile)

    print(f"Profile: {profile.name}")
    print(f"Max ceiling: {result.max_ceiling:.2f} bar")
    print(f"Min NDL: {result.min_ndl:.1f} min")
    print(f"Max supersaturation: {result.max_supersaturation:.2%}")
    print(f"Requires deco: {result.requires_deco}")
