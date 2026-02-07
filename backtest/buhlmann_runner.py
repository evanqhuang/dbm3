"""
Interface to libbuhlmann binary for running Bühlmann decompression calculations.

Treats the src/dive binary as a black-box function f(profile) -> risk_metrics.
"""

import subprocess
import os
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from .profile_generator import DiveProfile
from .buhlmann_constants import (
    GradientFactors,
    GF_DEFAULT,
    compute_ceilings_gf,
    compute_ndl_gf,
    compute_max_supersaturation_gf,
)
from .buhlmann_engine import BuhlmannEngine


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

    # Gradient factors used (1.0/1.0 = standard Bühlmann)
    gf_low: float = 1.0
    gf_high: float = 1.0

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

    def __init__(
        self,
        binary_path: Optional[str] = None,
        gf: Optional[GradientFactors] = None,
        use_python_engine: bool = True,
    ):
        """
        Initialize the runner.

        Args:
            binary_path: Path to the dive binary. If None, auto-detect.
            gf: Gradient factors for M-value adjustments. Defaults to GF 100/100.
            use_python_engine: If True, use Python engine; if False, use C binary.
        """
        self.gf = gf or GF_DEFAULT
        self.use_python_engine = use_python_engine

        if use_python_engine:
            self.engine = BuhlmannEngine()
            self.binary_path = None
        else:
            self.engine = None
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
        if self.use_python_engine:
            return self._run_python(profile)
        else:
            return self._run_binary(profile)

    def _run_binary(self, profile: DiveProfile) -> BuhlmannResult:
        """
        Run profile through C binary subprocess.

        Args:
            profile: DiveProfile object with dive data

        Returns:
            BuhlmannResult with compartment states and risk metrics
        """
        assert self.binary_path is not None
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

    def _run_python(self, profile: DiveProfile) -> BuhlmannResult:
        """
        Run profile through Python engine.

        Args:
            profile: DiveProfile object with dive data

        Returns:
            BuhlmannResult with compartment states and risk metrics
        """
        assert self.engine is not None
        raw = self.engine.simulate(profile)
        return self._build_result(raw)

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

        raw = {
            'times': times,
            'pressures': pressures,
            'compartment_n2': compartment_n2,
            'compartment_he': compartment_he,
            'ceilings': ceilings,
            'ndl_times': ndl_times,
        }
        return self._build_result(raw)

    def _build_result(self, raw: dict) -> BuhlmannResult:
        """
        Build BuhlmannResult from raw data, applying GF adjustments if needed.

        Args:
            raw: Dictionary with keys: times, pressures, compartment_n2,
                 compartment_he, ceilings, ndl_times

        Returns:
            BuhlmannResult with GF-adjusted metrics
        """
        times = raw['times']
        pressures = raw['pressures']
        compartment_n2 = raw['compartment_n2']
        compartment_he = raw['compartment_he']
        ceilings = raw['ceilings']
        ndl_times = raw['ndl_times']

        if self.gf.is_standard:
            # GF 100/100: use raw values
            max_ceiling = max(ceilings)
            min_ndl = min(ndl_times)
            max_supersaturation = self._calculate_max_supersaturation(
                compartment_n2, compartment_he, pressures
            )
        else:
            # Recalculate with GF-adjusted M-values
            gf_ceilings = []
            gf_ndls = []
            for t_idx in range(len(times)):
                ceil, _ = compute_ceilings_gf(
                    compartment_n2[t_idx],
                    compartment_he[t_idx],
                    self.gf.gf_low,
                )
                gf_ceilings.append(ceil)

                ndl = compute_ndl_gf(
                    compartment_n2[t_idx],
                    compartment_he[t_idx],
                    pressures[t_idx],
                    0.79,  # f_inert for air
                    self.gf.gf_high,
                )
                gf_ndls.append(ndl)

            ceilings = gf_ceilings
            ndl_times = gf_ndls
            max_ceiling = max(ceilings)
            min_ndl = min(ndl_times)
            max_supersaturation = compute_max_supersaturation_gf(
                compartment_n2, compartment_he, pressures,
                self.gf.gf_low, self.gf.gf_high,
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
            gf_low=self.gf.gf_low,
            gf_high=self.gf.gf_high,
        )

    @staticmethod
    def _calculate_max_supersaturation(
        compartment_n2: List[List[float]],
        compartment_he: List[List[float]],
        pressures: List[float],
    ) -> float:
        """
        Calculate maximum supersaturation across all compartments and time steps.

        Uses standard (GF 100/100) M-values. For GF-adjusted supersaturation,
        use compute_max_supersaturation_gf() instead.
        """
        return compute_max_supersaturation_gf(
            compartment_n2, compartment_he, pressures, 1.0, 1.0
        )

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

    runner = BuhlmannRunner(use_python_engine=False)
    result = runner.run(profile)

    print(f"Profile: {profile.name}")
    print(f"GF: {result.gf_low*100:.0f}/{result.gf_high*100:.0f}")
    print(f"Max ceiling: {result.max_ceiling:.2f} bar")
    print(f"Min NDL: {result.min_ndl:.1f} min")
    print(f"Max supersaturation: {result.max_supersaturation:.2%}")
    print(f"Requires deco: {result.requires_deco}")
