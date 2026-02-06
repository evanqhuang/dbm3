"""
Model Comparator for backtesting Bühlmann vs Slab decompression models.

Runs batch comparisons and generates divergence matrices to identify
where the models differ in their risk predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time as time_module

from .profile_generator import ProfileGenerator, DiveProfile
from .buhlmann_runner import BuhlmannRunner, BuhlmannResult
from .slab_model import SlabModel, SlabResult


def _run_buhlmann_single(
    binary_path: str, profile: DiveProfile
) -> Optional[BuhlmannResult]:
    """Helper function for parallel Bühlmann execution."""
    import subprocess

    try:
        input_data = profile.to_buhlmann_format()
        process = subprocess.Popen(
            [binary_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=input_data)
        if process.returncode != 0:
            return None
        # Parse output inline to avoid circular imports
        return _parse_buhlmann_output(stdout)
    except Exception:
        return None


def _parse_buhlmann_output(output: str) -> Optional[BuhlmannResult]:
    """Parse libbuhlmann output for parallel execution."""
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
            continue
        try:
            time = float(values[0])
            pressure = float(values[1])
            n2_values = []
            he_values = []
            for i in range(num_compartments):
                n2_idx = 2 + i * 2
                he_idx = 3 + i * 2
                n2_values.append(float(values[n2_idx]))
                he_values.append(float(values[he_idx]))
            ceiling = float(values[-2])
            ndl = float(values[-1])
            times.append(time)
            pressures.append(pressure)
            compartment_n2.append(n2_values)
            compartment_he.append(he_values)
            ceilings.append(ceiling)
            ndl_times.append(ndl)
        except (ValueError, IndexError):
            continue

    if not times:
        return None

    max_ceiling = max(ceilings)
    min_ndl = min(ndl_times)

    # Calculate max supersaturation (simplified from BuhlmannRunner)
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
            p_inert = compartment_n2[t_idx][c_idx]
            if compartment_he and len(compartment_he[t_idx]) > c_idx:
                p_inert += compartment_he[t_idx][c_idx]
            a = zh_l16_n2_a[c_idx]
            b = zh_l16_n2_b[c_idx]
            m_value = a + pressure / b
            ratio = p_inert / m_value if m_value > 0 else 0
            max_ratio = max(max_ratio, ratio)

    return BuhlmannResult(
        times=times,
        pressures=pressures,
        compartment_n2=compartment_n2,
        compartment_he=compartment_he,
        ceilings=ceilings,
        ndl_times=ndl_times,
        max_ceiling=max_ceiling,
        min_ndl=min_ndl,
        max_supersaturation=max_ratio,
    )


def _run_slab_single(slab_params: dict, profile: DiveProfile) -> Optional[SlabResult]:
    """Helper function for parallel Slab execution."""
    try:
        # Use the new multi-compartment model
        model = SlabModel(
            compartments_config=slab_params.get("compartments_config"),
            dt=slab_params["dt"],
            dx=slab_params["dx"],
            permeability=slab_params["permeability"],
            f_o2=slab_params["f_o2"],
            surface_altitude_m=slab_params["surface_altitude_m"],
            critical_volume_k=slab_params.get("critical_volume_k", 1.0),
        )
        return model.run(profile)
    except Exception:
        return None


@dataclass
class ComparisonResult:
    """Result of comparing two models on a single dive profile."""

    profile: DiveProfile
    buhlmann_result: Optional[BuhlmannResult]
    slab_result: Optional[SlabResult]

    @property
    def buhlmann_risk(self) -> float:
        """Bühlmann risk score (M-value fraction)."""
        if self.buhlmann_result is None:
            return float("nan")
        return self.buhlmann_result.risk_score

    @property
    def slab_risk(self) -> float:
        """Slab model risk score (M-value fraction)."""
        if self.slab_result is None:
            return float("nan")
        return self.slab_result.risk_score

    @property
    def delta_risk(self) -> float:
        """
        Risk divergence: Slab - Bühlmann.
        Positive = Slab more conservative (predicts higher risk)
        Negative = Bühlmann more conservative
        """
        if self.buhlmann_result is None or self.slab_result is None:
            return float("nan")
        return self.slab_risk - self.buhlmann_risk

    @property
    def buhlmann_ndl(self) -> float:
        """Bühlmann NDL (no-decompression limit) in minutes."""
        if self.buhlmann_result is None:
            return float("nan")
        return self.buhlmann_result.min_ndl

    @property
    def slab_ndl(self) -> float:
        """Slab model NDL in minutes."""
        if self.slab_result is None:
            return float("nan")
        return self.slab_result.final_ndl

    @property
    def delta_ndl(self) -> float:
        """
        NDL divergence: Slab - Bühlmann (minutes).
        Positive = Slab allows more time (less conservative)
        Negative = Slab allows less time (more conservative)
        """
        if self.buhlmann_result is None or self.slab_result is None:
            return float("nan")
        return self.slab_ndl - self.buhlmann_ndl

    @property
    def buhlmann_ceiling(self) -> float:
        """
        Buhlmann ceiling depth in meters.
        Converts max ceiling from bar to depth: max(0, (ceiling_bar - P_surface) * 10)
        """
        if self.buhlmann_result is None:
            return float("nan")
        P_surface = 1.01325  # bar at sea level
        ceiling_depth = max(0, (self.buhlmann_result.max_ceiling - P_surface) * 10)
        return ceiling_depth

    @property
    def slab_ceiling(self) -> float:
        """Slab model ceiling depth in meters."""
        if self.slab_result is None:
            return float("nan")
        return self.slab_result.ceiling_at_bottom

    @property
    def delta_ceiling(self) -> float:
        """
        Ceiling divergence: Slab - Buhlmann (meters).
        Positive = Slab requires deeper stops (more conservative)
        Negative = Buhlmann requires deeper stops (more conservative)
        """
        if self.buhlmann_result is None or self.slab_result is None:
            return float("nan")
        return self.slab_ceiling - self.buhlmann_ceiling

    @property
    def slab_requires_deco(self) -> bool:
        """True if slab model ceiling exceeds surface (requires deco stops)."""
        if self.slab_result is None:
            return False
        return self.slab_ceiling > 0

    @property
    def buhlmann_requires_deco(self) -> bool:
        """True if Buhlmann ceiling exceeds surface (requires deco stops)."""
        if self.buhlmann_result is None:
            return False
        return self.buhlmann_ceiling > 0

    @property
    def is_valid(self) -> bool:
        """True if both models produced valid results."""
        return self.buhlmann_result is not None and self.slab_result is not None


class ModelComparator:
    """
    Compare Bühlmann and Slab decompression models across dive profiles.

    Generates divergence matrices showing where the models agree/disagree
    on risk assessments.
    """

    _DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

    def __init__(
        self,
        buhlmann_runner: Optional[BuhlmannRunner] = None,
        slab_model: Optional[SlabModel] = None,
        config_path: Optional[str] = None,
    ):
        """
        Initialize the comparator.

        Args:
            buhlmann_runner: BuhlmannRunner instance (auto-created if None)
            slab_model: SlabModel instance (auto-created if None)
            config_path: Path to config.yaml for SlabModel (auto-detected if None)
        """
        self.buhlmann = buhlmann_runner or BuhlmannRunner()
        if slab_model is not None:
            self.slab = slab_model
        else:
            cfg = config_path or self._DEFAULT_CONFIG
            self.slab = SlabModel(config_path=cfg) if os.path.exists(cfg) else SlabModel()
        self.generator = ProfileGenerator()

    def compare_profile(self, profile: DiveProfile) -> ComparisonResult:
        """
        Compare both models on a single dive profile.

        Args:
            profile: DiveProfile to analyze

        Returns:
            ComparisonResult with both model outputs
        """
        buhlmann_result = None
        slab_result = None

        try:
            buhlmann_result = self.buhlmann.run(profile)
        except Exception as e:
            print(f"Bühlmann failed for {profile.name}: {e}")

        try:
            slab_result = self.slab.run(profile)
        except Exception as e:
            print(f"Slab failed for {profile.name}: {e}")

        return ComparisonResult(
            profile=profile, buhlmann_result=buhlmann_result, slab_result=slab_result
        )

    def compare_batch(
        self,
        profiles: List[DiveProfile],
        verbose: bool = True,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[ComparisonResult]:
        """
        Compare both models on multiple profiles.

        Args:
            profiles: List of profiles to analyze
            verbose: Print progress updates
            parallel: Use parallel processing (default True)
            max_workers: Number of worker processes (default: CPU count)

        Returns:
            List of ComparisonResult objects
        """
        total = len(profiles)
        start_time = time_module.time()

        if parallel and total > 1:
            return self._compare_batch_parallel(profiles, verbose, max_workers)

        # Sequential fallback
        results = []
        for i, profile in enumerate(profiles):
            result = self.compare_profile(profile)
            results.append(result)

            if verbose and (i + 1) % 100 == 0:
                elapsed = time_module.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate if rate > 0 else 0
                print(
                    f"Processed {i + 1}/{total} profiles "
                    f"({rate:.1f}/s, ~{remaining:.0f}s remaining)"
                )

        if verbose:
            elapsed = time_module.time() - start_time
            print(f"Completed {total} profiles in {elapsed:.1f}s")

        return results

    def _compare_batch_parallel(
        self,
        profiles: List[DiveProfile],
        verbose: bool = True,
        max_workers: Optional[int] = None,
    ) -> List[ComparisonResult]:
        """
        Run batch comparison using parallel processing.

        Uses ProcessPoolExecutor for CPU-bound Slab model calculations
        and ThreadPoolExecutor for I/O-bound Bühlmann subprocess calls.
        """
        total = len(profiles)
        start_time = time_module.time()
        n_workers = max_workers or min(cpu_count(), 8)  # Cap at 8 workers

        if verbose:
            print(f"Running parallel backtest with {n_workers} workers...")

        # Run models in parallel using helper functions
        buhlmann_results = [None] * total
        slab_results = [None] * total

        # Bühlmann: Use ThreadPoolExecutor (I/O bound - subprocess calls)
        binary_path = self.buhlmann.binary_path
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_buhlmann_single, binary_path, p): i
                for i, p in enumerate(profiles)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    buhlmann_results[idx] = future.result()
                except Exception as e:
                    if verbose:
                        print(f"Bühlmann failed for profile {idx}: {e}")
                completed += 1
                if verbose and completed % 100 == 0:
                    elapsed = time_module.time() - start_time
                    print(
                        f"Bühlmann: {completed}/{total} ({completed / elapsed:.1f}/s)"
                    )

        buhlmann_time = time_module.time() - start_time
        if verbose:
            print(f"Bühlmann completed in {buhlmann_time:.1f}s")

        # Slab: Use ProcessPoolExecutor (CPU bound)
        slab_start = time_module.time()
        slab_params = {
            "compartments_config": self.slab.get_compartment_config(),
            "dt": self.slab.dt,
            "dx": self.slab.dx,
            "permeability": self.slab.permeability,
            "f_o2": self.slab.f_o2,
            "surface_altitude_m": self.slab.surface_altitude_m,
            "critical_volume_k": self.slab.critical_volume_k,
        }
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_slab_single, slab_params, p): i
                for i, p in enumerate(profiles)
            }
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    slab_results[idx] = future.result()
                except Exception as e:
                    if verbose:
                        print(f"Slab failed for profile {idx}: {e}")
                completed += 1
                if verbose and completed % 100 == 0:
                    elapsed = time_module.time() - slab_start
                    print(f"Slab: {completed}/{total} ({completed / elapsed:.1f}/s)")

        if verbose:
            slab_time = time_module.time() - slab_start
            print(f"Slab completed in {slab_time:.1f}s")

        # Combine results
        results = [
            ComparisonResult(
                profile=profiles[i],
                buhlmann_result=buhlmann_results[i],
                slab_result=slab_results[i],
            )
            for i in range(total)
        ]

        if verbose:
            elapsed = time_module.time() - start_time
            print(
                f"Total: {total} profiles in {elapsed:.1f}s ({total / elapsed:.1f}/s)"
            )

        return results

    def generate_divergence_matrix(
        self,
        depths: List[float],
        times: List[float],
        profile_type: str = "square",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate divergence matrices across depth/time combinations.

        Args:
            depths: List of depths to test (meters)
            times: List of bottom times to test (minutes)
            profile_type: Type of profile to generate
            **kwargs: Additional profile generator arguments

        Returns:
            Dict with matrices: delta_risk, buhlmann_risk, slab_risk,
                               delta_ndl, buhlmann_ndl, slab_ndl
            Each matrix is [len(times), len(depths)]
        """
        # Generate all profiles
        profiles = self.generator.generate_batch(depths, times, profile_type, **kwargs)

        # Run comparisons
        results = self.compare_batch(profiles, verbose=True)

        # Build matrices
        n_times = len(times)
        n_depths = len(depths)

        matrices = {
            "delta_risk": np.full((n_times, n_depths), np.nan),
            "buhlmann_risk": np.full((n_times, n_depths), np.nan),
            "slab_risk": np.full((n_times, n_depths), np.nan),
            "delta_ndl": np.full((n_times, n_depths), np.nan),
            "buhlmann_ndl": np.full((n_times, n_depths), np.nan),
            "slab_ndl": np.full((n_times, n_depths), np.nan),
            "delta_ceiling": np.full((n_times, n_depths), np.nan),
            "buhlmann_ceiling": np.full((n_times, n_depths), np.nan),
            "slab_ceiling": np.full((n_times, n_depths), np.nan),
        }

        for idx, result in enumerate(results):
            depth_idx = idx % n_depths
            time_idx = idx // n_depths

            if result.is_valid:
                matrices["delta_risk"][time_idx, depth_idx] = result.delta_risk
                matrices["buhlmann_risk"][time_idx, depth_idx] = result.buhlmann_risk
                matrices["slab_risk"][time_idx, depth_idx] = result.slab_risk
                matrices["delta_ndl"][time_idx, depth_idx] = result.delta_ndl
                matrices["buhlmann_ndl"][time_idx, depth_idx] = result.buhlmann_ndl
                matrices["slab_ndl"][time_idx, depth_idx] = result.slab_ndl
                matrices["delta_ceiling"][time_idx, depth_idx] = result.delta_ceiling
                matrices["buhlmann_ceiling"][time_idx, depth_idx] = result.buhlmann_ceiling
                matrices["slab_ceiling"][time_idx, depth_idx] = result.slab_ceiling

        return matrices

    def plot_divergence_matrix(
        self,
        depths: List[float],
        times: List[float],
        delta_matrix: np.ndarray,
        title: str = "Model Divergence Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the divergence matrix as a heatmap.

        Args:
            depths: Depth values for x-axis
            times: Time values for y-axis
            delta_matrix: Divergence values [times, depths]
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Use diverging colormap centered at 0
        vmax = max(abs(np.nanmin(delta_matrix)), abs(np.nanmax(delta_matrix)))
        vmax = max(vmax, 0.1)  # Ensure some range

        im = ax.imshow(
            delta_matrix,
            aspect="auto",
            cmap="RdBu_r",  # Red = Slab more conservative, Blue = Bühlmann more conservative
            origin="lower",
            extent=[min(depths), max(depths), min(times), max(times)],
            vmin=-vmax,
            vmax=vmax,
        )

        cbar = plt.colorbar(im, ax=ax, label="ΔRisk (Slab - Bühlmann)")

        # Add contour lines at key thresholds
        contour_levels = [-0.2, -0.1, 0, 0.1, 0.2]
        valid_levels = [l for l in contour_levels if -vmax <= l <= vmax]
        if valid_levels:
            ax.contour(
                delta_matrix,
                levels=valid_levels,
                colors="black",
                alpha=0.5,
                origin="lower",
                extent=[min(depths), max(depths), min(times), max(times)],
            )

        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Bottom Time (min)")
        ax.set_title(title)

        # Add interpretation legend
        ax.text(
            0.02,
            0.98,
            "Red: Slab more conservative\nBlue: Bühlmann more conservative",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_comparison_summary(
        self,
        depths: List[float],
        times: List[float],
        buhlmann_matrix: np.ndarray,
        slab_matrix: np.ndarray,
        delta_matrix: np.ndarray,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a 3-panel comparison showing both models and their divergence.

        Args:
            depths: Depth values for x-axis
            times: Time values for y-axis
            buhlmann_matrix: Bühlmann risk values
            slab_matrix: Slab risk values
            delta_matrix: Divergence values
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        extent = [min(depths), max(depths), min(times), max(times)]

        # Panel 1: Bühlmann risk
        vmax_risk = max(np.nanmax(buhlmann_matrix), np.nanmax(slab_matrix), 1.0)
        im1 = axes[0].imshow(
            buhlmann_matrix,
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=vmax_risk,
        )
        axes[0].set_xlabel("Depth (m)")
        axes[0].set_ylabel("Bottom Time (min)")
        axes[0].set_title("Bühlmann Risk (M-value %)")
        plt.colorbar(im1, ax=axes[0])

        # Add M-value = 1.0 contour (limit)
        if np.any(buhlmann_matrix >= 1.0):
            axes[0].contour(
                buhlmann_matrix,
                levels=[1.0],
                colors="red",
                linewidths=2,
                origin="lower",
                extent=extent,
            )

        # Panel 2: Slab risk
        im2 = axes[1].imshow(
            slab_matrix,
            aspect="auto",
            cmap="YlOrRd",
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=vmax_risk,
        )
        axes[1].set_xlabel("Depth (m)")
        axes[1].set_ylabel("Bottom Time (min)")
        axes[1].set_title("Slab Risk (M-value %)")
        plt.colorbar(im2, ax=axes[1])

        if np.any(slab_matrix >= 1.0):
            axes[1].contour(
                slab_matrix,
                levels=[1.0],
                colors="red",
                linewidths=2,
                origin="lower",
                extent=extent,
            )

        # Panel 3: Divergence
        vmax_delta = max(
            abs(np.nanmin(delta_matrix)), abs(np.nanmax(delta_matrix)), 0.1
        )
        im3 = axes[2].imshow(
            delta_matrix,
            aspect="auto",
            cmap="RdBu_r",
            origin="lower",
            extent=extent,
            vmin=-vmax_delta,
            vmax=vmax_delta,
        )
        axes[2].set_xlabel("Depth (m)")
        axes[2].set_ylabel("Bottom Time (min)")
        axes[2].set_title("ΔRisk (Slab - Bühlmann)")
        plt.colorbar(im3, ax=axes[2])

        # Add zero contour
        axes[2].contour(
            delta_matrix,
            levels=[0],
            colors="black",
            linewidths=1,
            linestyles="--",
            origin="lower",
            extent=extent,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(self, results: List[ComparisonResult]) -> Dict:
        """
        Generate summary statistics from comparison results.

        Args:
            results: List of ComparisonResult objects

        Returns:
            Dictionary with summary statistics
        """
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            return {"error": "No valid results"}

        # Risk metrics
        deltas = [r.delta_risk for r in valid_results]
        buhlmann_risks = [r.buhlmann_risk for r in valid_results]
        slab_risks = [r.slab_risk for r in valid_results]

        # NDL metrics
        delta_ndls = [r.delta_ndl for r in valid_results]
        buhlmann_ndls = [r.buhlmann_ndl for r in valid_results]
        slab_ndls = [r.slab_ndl for r in valid_results]

        # Find profiles where models disagree most on risk
        sorted_by_delta = sorted(
            valid_results, key=lambda r: abs(r.delta_risk), reverse=True
        )

        # Find profiles where models disagree most on NDL
        sorted_by_ndl_delta = sorted(
            valid_results, key=lambda r: abs(r.delta_ndl), reverse=True
        )

        # Profiles where Slab is more conservative (higher risk)
        slab_conservative = [r for r in valid_results if r.delta_risk > 0.1]

        # Profiles where Bühlmann is more conservative
        buhlmann_conservative = [r for r in valid_results if r.delta_risk < -0.1]

        # Ceiling metrics
        delta_ceilings = [r.delta_ceiling for r in valid_results]
        buhlmann_ceilings = [r.buhlmann_ceiling for r in valid_results]
        slab_ceilings = [r.slab_ceiling for r in valid_results]

        # Deco agreement analysis
        deco_agreements = [
            r for r in valid_results
            if r.buhlmann_requires_deco == r.slab_requires_deco
        ]
        deco_agreement_pct = (
            len(deco_agreements) / len(valid_results) * 100 if valid_results else 0
        )

        # Count profiles requiring deco
        buhlmann_deco_count = sum(1 for r in valid_results if r.buhlmann_requires_deco)
        slab_deco_count = sum(1 for r in valid_results if r.slab_requires_deco)

        # Determine which model is more conservative for ceilings
        slab_deeper_ceiling = [r for r in valid_results if r.delta_ceiling > 0.5]
        buhlmann_deeper_ceiling = [r for r in valid_results if r.delta_ceiling < -0.5]

        # Find profiles with largest ceiling divergence
        sorted_by_ceiling_delta = sorted(
            valid_results, key=lambda r: abs(r.delta_ceiling), reverse=True
        )

        return {
            "total_profiles": len(results),
            "valid_profiles": len(valid_results),
            # Risk divergence stats
            "mean_delta_risk": np.mean(deltas),
            "std_delta_risk": np.std(deltas),
            "max_delta_risk": np.max(deltas),
            "min_delta_risk": np.min(deltas),
            "mean_buhlmann_risk": np.mean(buhlmann_risks),
            "mean_slab_risk": np.mean(slab_risks),
            "risk_correlation": np.corrcoef(buhlmann_risks, slab_risks)[0, 1],
            # NDL divergence stats
            "mean_delta_ndl": np.mean(delta_ndls),
            "std_delta_ndl": np.std(delta_ndls),
            "max_delta_ndl": np.max(delta_ndls),
            "min_delta_ndl": np.min(delta_ndls),
            "mean_buhlmann_ndl": np.mean(buhlmann_ndls),
            "mean_slab_ndl": np.mean(slab_ndls),
            "ndl_correlation": np.corrcoef(buhlmann_ndls, slab_ndls)[0, 1],
            # Ceiling divergence stats
            "mean_delta_ceiling": np.mean(delta_ceilings),
            "std_delta_ceiling": np.std(delta_ceilings),
            "max_delta_ceiling": np.max(delta_ceilings),
            "min_delta_ceiling": np.min(delta_ceilings),
            "mean_buhlmann_ceiling": np.mean(buhlmann_ceilings),
            "mean_slab_ceiling": np.mean(slab_ceilings),
            "ceiling_correlation": np.corrcoef(buhlmann_ceilings, slab_ceilings)[0, 1],
            # Deco agreement analysis
            "deco_agreement_pct": deco_agreement_pct,
            "buhlmann_deco_count": buhlmann_deco_count,
            "slab_deco_count": slab_deco_count,
            "both_require_deco": sum(
                1 for r in valid_results
                if r.buhlmann_requires_deco and r.slab_requires_deco
            ),
            "neither_require_deco": sum(
                1 for r in valid_results
                if not r.buhlmann_requires_deco and not r.slab_requires_deco
            ),
            "slab_only_deco": sum(
                1 for r in valid_results
                if r.slab_requires_deco and not r.buhlmann_requires_deco
            ),
            "buhlmann_only_deco": sum(
                1 for r in valid_results
                if r.buhlmann_requires_deco and not r.slab_requires_deco
            ),
            # Ceiling conservatism
            "slab_deeper_ceiling_count": len(slab_deeper_ceiling),
            "buhlmann_deeper_ceiling_count": len(buhlmann_deeper_ceiling),
            # Conservative counts
            "slab_conservative_count": len(slab_conservative),
            "buhlmann_conservative_count": len(buhlmann_conservative),
            # Top divergent profiles (by risk)
            "top_divergent_risk": [
                {
                    "name": r.profile.name,
                    "depth": r.profile.max_depth,
                    "delta_risk": r.delta_risk,
                    "buhlmann_risk": r.buhlmann_risk,
                    "slab_risk": r.slab_risk,
                }
                for r in sorted_by_delta[:5]
            ],
            # Top divergent profiles (by NDL)
            "top_divergent_ndl": [
                {
                    "name": r.profile.name,
                    "depth": r.profile.max_depth,
                    "delta_ndl": r.delta_ndl,
                    "buhlmann_ndl": r.buhlmann_ndl,
                    "slab_ndl": r.slab_ndl,
                }
                for r in sorted_by_ndl_delta[:5]
            ],
            # Top divergent profiles (by ceiling)
            "top_divergent_ceiling": [
                {
                    "name": r.profile.name,
                    "depth": r.profile.max_depth,
                    "delta_ceiling": r.delta_ceiling,
                    "buhlmann_ceiling": r.buhlmann_ceiling,
                    "slab_ceiling": r.slab_ceiling,
                }
                for r in sorted_by_ceiling_delta[:5]
            ],
        }


def run_full_backtest(
    depths: Optional[List[float]] = None,
    times: Optional[List[float]] = None,
    save_plots: bool = True,
    save_data: bool = True,
    output_dir: str = ".",
) -> Dict:
    """
    Run a complete backtest comparison between Bühlmann and Slab models.

    Args:
        depths: List of depths to test (default: 10-60m in 5m steps)
        times: List of times to test (default: 5-60min in 5min steps)
        save_plots: Whether to save plots to disk
        save_data: Whether to save raw data (CSV/JSON) to disk
        output_dir: Directory for output files

    Returns:
        Dictionary with results and statistics
    """
    import os
    import json
    import csv

    if depths is None:
        depths = list(range(10, 65, 5))  # 10, 15, 20, ..., 60m

    if times is None:
        times = list(range(5, 65, 5))  # 5, 10, 15, ..., 60min

    print(
        f"Running backtest: {len(depths)} depths x {len(times)} times = {len(depths) * len(times)} profiles"
    )

    comparator = ModelComparator()

    # Generate divergence matrices (risk and NDL)
    matrices = comparator.generate_divergence_matrix(
        depths, times, profile_type="square"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    if save_plots:
        # Risk divergence plot
        comparator.plot_divergence_matrix(
            depths,
            times,
            matrices["delta_risk"],
            title="Bühlmann vs Slab Risk Divergence (Square Profiles)",
            save_path=os.path.join(output_dir, "divergence_risk.png"),
        )

        # NDL divergence plot
        comparator.plot_divergence_matrix(
            depths,
            times,
            matrices["delta_ndl"],
            title="Bühlmann vs Slab NDL Divergence (Square Profiles)",
            save_path=os.path.join(output_dir, "divergence_ndl.png"),
        )

        # Risk comparison summary
        comparator.plot_comparison_summary(
            depths,
            times,
            matrices["buhlmann_risk"],
            matrices["slab_risk"],
            matrices["delta_risk"],
            save_path=os.path.join(output_dir, "comparison_risk.png"),
        )

        # NDL comparison summary
        comparator.plot_comparison_summary(
            depths,
            times,
            matrices["buhlmann_ndl"],
            matrices["slab_ndl"],
            matrices["delta_ndl"],
            save_path=os.path.join(output_dir, "comparison_ndl.png"),
        )

    # Generate all profiles for report
    profiles = comparator.generator.generate_batch(depths, times, "square")
    results = comparator.compare_batch(profiles, verbose=False)
    report = comparator.generate_report(results)

    # Save raw data
    if save_data:
        # Save matrices as CSV files
        for matrix_name, matrix_data in matrices.items():
            csv_path = os.path.join(output_dir, f"matrix_{matrix_name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Header row: depth values
                writer.writerow(["time\\depth"] + [str(d) for d in depths])
                # Data rows: time value + matrix row
                for i, t in enumerate(times):
                    row = [str(t)] + [f"{v:.6f}" for v in matrix_data[i, :]]
                    writer.writerow(row)

        # Save detailed per-profile results as CSV
        results_csv_path = os.path.join(output_dir, "results_detailed.csv")
        with open(results_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "profile_name",
                    "depth_m",
                    "bottom_time_min",
                    "buhlmann_risk",
                    "slab_risk",
                    "delta_risk",
                    "buhlmann_ndl",
                    "slab_ndl",
                    "delta_ndl",
                    "buhlmann_ceiling_bar",
                    "buhlmann_ceiling_m",
                    "slab_ceiling_m",
                    "delta_ceiling_m",
                    "buhlmann_requires_deco",
                    "slab_requires_deco",
                    "slab_exceeded_limit",
                    "slab_critical_compartment",
                ]
            )
            for result in results:
                if result.is_valid:
                    writer.writerow(
                        [
                            result.profile.name,
                            result.profile.max_depth,
                            result.profile.bottom_time,
                            f"{result.buhlmann_risk:.6f}",
                            f"{result.slab_risk:.6f}",
                            f"{result.delta_risk:.6f}",
                            f"{result.buhlmann_ndl:.2f}",
                            f"{result.slab_ndl:.2f}",
                            f"{result.delta_ndl:.2f}",
                            f"{result.buhlmann_result.max_ceiling:.4f}",
                            f"{result.buhlmann_ceiling:.2f}",
                            f"{result.slab_ceiling:.2f}",
                            f"{result.delta_ceiling:.2f}",
                            result.buhlmann_requires_deco,
                            result.slab_requires_deco,
                            result.slab_result.exceeded_limit,
                            result.slab_result.critical_compartment,
                        ]
                    )

        # Save report as JSON
        report_json_path = os.path.join(output_dir, "report.json")
        with open(report_json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save summary statistics as JSON
        summary = {
            "depths": depths,
            "times": times,
            "n_profiles": len(results),
            "n_valid": sum(1 for r in results if r.is_valid),
            "report": report,
        }
        summary_json_path = os.path.join(output_dir, "summary.json")
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"Raw data saved to {output_dir}/")

    return {
        "depths": depths,
        "times": times,
        "matrices": matrices,
        "results": results,
        "report": report,
    }


if __name__ == "__main__":
    # Demo: Run a small backtest
    results = run_full_backtest(
        depths=[10, 20, 30, 40],
        times=[10, 20, 30, 40],
        save_plots=True,
        output_dir="backtest_output",
    )

    print("\n=== BACKTEST REPORT ===")
    for key, value in results["report"].items():
        if key != "top_divergent_profiles":
            print(f"{key}: {value}")

    print("\nTop divergent profiles:")
    for p in results["report"]["top_divergent_profiles"]:
        print(
            f"  {p['name']}: Δ={p['delta']:.3f} (B={p['buhlmann']:.3f}, S={p['slab']:.3f})"
        )
