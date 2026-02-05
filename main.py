"""
SlabDive - Interactive Dive Profile Simulator

Uses the multi-compartment Slab Diffusion model to simulate nitrogen uptake
and offgassing for user-defined dive profiles. Supports square, multilevel,
sawtooth, and custom profiles.

Usage:
    python main.py                          # Run with default settings below
    python main.py --depth 30 --time 20     # Quick square profile override
    python main.py --profile multilevel     # Use a multilevel profile
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from backtest.slab_model import SlabModel
from backtest.profile_generator import ProfileGenerator, DiveProfile


# --- USER CONFIGURATION ---
# Edit these values to plan your dive, or override via CLI arguments.

DIVE_CONFIG = {
    "profile_type": "square",       # "square", "multilevel", "sawtooth", or "custom"
    "depth_m": 30,                  # Depth for square/sawtooth profiles (meters)
    "bottom_time_min": 30,          # Bottom time (minutes)
    "fO2": 0.21,                    # Breathing gas O2 fraction (Air = 0.21)

    # Multilevel profile: list of (depth_m, duration_min), deepest first
    "multilevel_levels": [
        (30, 10),
        (20, 10),
        (10, 10),
    ],

    # Sawtooth profile settings
    "sawtooth_min_depth_m": 10,     # Minimum depth during oscillations
    "sawtooth_oscillations": 3,     # Number of depth oscillations

    # Descent/ascent rates
    "descent_rate": 20.0,           # m/min
    "ascent_rate": 10.0,            # m/min (conservative)
}


def build_profile(config: dict) -> DiveProfile:
    """Build a DiveProfile from user configuration."""
    gen = ProfileGenerator(
        descent_rate=config["descent_rate"],
        ascent_rate=config["ascent_rate"],
    )

    profile_type = config["profile_type"]

    if profile_type == "square":
        return gen.generate_square(
            depth=config["depth_m"],
            bottom_time=config["bottom_time_min"],
            fO2=config["fO2"],
        )
    elif profile_type == "multilevel":
        return gen.generate_multilevel(
            levels=config["multilevel_levels"],
            fO2=config["fO2"],
        )
    elif profile_type == "sawtooth":
        return gen.generate_sawtooth(
            max_depth=config["depth_m"],
            min_depth=config["sawtooth_min_depth_m"],
            total_time=config["bottom_time_min"],
            oscillations=config["sawtooth_oscillations"],
            fO2=config["fO2"],
        )
    else:
        raise ValueError(
            f"Unknown profile type: {profile_type}. "
            "Use 'square', 'multilevel', or 'sawtooth'."
        )


def print_dive_plan(profile: DiveProfile, model: SlabModel) -> None:
    """Print dive plan summary before simulation."""
    p_surface = model._get_atmospheric_pressure()
    p_bottom = p_surface + profile.max_depth / 10.0
    ppn2_surface = p_surface * (1 - model.f_o2)
    ppn2_bottom = p_bottom * (1 - model.f_o2)

    print("--- DIVE PLAN ---")
    print(f"Profile: {profile.name}")
    print(f"Max depth: {profile.max_depth:.0f}m")
    print(f"Bottom time: {profile.bottom_time:.0f} min")
    print(f"Gas mix: {model.f_o2 * 100:.0f}% O2 (fO2={model.f_o2})")
    print(f"Altitude: {model.surface_altitude_m:.0f}m (Atm: {p_surface:.3f} bar)")
    print(f"Bottom pressure: {p_bottom:.2f} bar")
    print(f"ppN2: surface={ppn2_surface:.3f}, bottom={ppn2_bottom:.3f} bar")
    print(f"Compartments: {', '.join(c.name for c in model.compartments)}")
    print(f"Sat limits: bottom={model.sat_limit_bottom}, surface={model.sat_limit_surface}")


def print_results(result) -> None:
    """Print simulation results."""
    print("\n--- SIMULATION RESULTS ---")
    print(f"Max tissue load: {result.max_tissue_load:.3f} bar")
    print(f"Max supersaturation: {result.max_supersaturation:.2%}")
    print(f"Critical compartment: {result.critical_compartment} (slice {result.critical_slice})")
    print(f"Min margin to M-value: {result.min_margin:.3f} bar")

    if result.exceeded_limit:
        print("\nWARNING: M-value limit EXCEEDED!")
        print("Mandatory decompression stops required before surfacing.")
    else:
        print(f"\nAll compartments within limits. NDL remaining: {result.final_ndl:.0f} min")


def plot_results(profile: DiveProfile, result, model: SlabModel) -> None:
    """Visualize dive profile and tissue loading."""
    compartments = model.compartments
    num_compartments = len(compartments)

    # Layout: depth profile + one heatmap per compartment + gas load vs M-values
    num_rows = 2 + num_compartments
    _fig, axes = plt.subplots(
        num_rows, 1, figsize=(12, 4 * num_rows),
        gridspec_kw={"height_ratios": [1] + [1.5] * num_compartments + [1.5]},
    )

    # --- Row 0: Depth Profile ---
    ax_depth = axes[0]
    ax_depth.plot(result.times, result.depths, "b-", linewidth=2)
    ax_depth.set_ylabel("Depth (m)")
    ax_depth.set_xlabel("Time (min)")
    ax_depth.set_title(f"Dive Profile: {profile.name}")
    ax_depth.invert_yaxis()
    ax_depth.grid(True, alpha=0.3)
    ax_depth.fill_between(result.times, result.depths, alpha=0.15, color="blue")

    # --- Rows 1..N: Heatmaps per compartment ---
    for c_idx, compartment in enumerate(compartments):
        ax = axes[1 + c_idx]

        # Extract this compartment's history: slab_history shape is [time, compartment, slices]
        # But slab_history stores lists of arrays, so index accordingly
        compartment_history = result.slab_history[:, c_idx, :]

        im = ax.imshow(
            compartment_history.T,
            aspect="auto",
            cmap="hot",
            origin="lower",
            extent=[result.times[0], result.times[-1], 0, compartment.slices],
        )
        plt.colorbar(im, ax=ax, label="ppN2 (bar)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Slice (0=surface)")
        ax.set_title(f"N2 Diffusion: {compartment.name} (D={compartment.D})")
        ax.contour(
            compartment_history.T,
            levels=8,
            colors="black",
            alpha=0.2,
            origin="lower",
            extent=[result.times[0], result.times[-1], 0, compartment.slices],
        )

    # --- Bottom Row: Gas Load vs M-Values for all compartments ---
    ax_mv = axes[-1]
    p_surface = model._get_atmospheric_pressure()

    cmap = colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, num_compartments))

    for c_idx, compartment in enumerate(compartments):
        color = colors[c_idx]
        slice_indices = np.arange(compartment.slices)
        m_vals = compartment.get_limited_m_values(p_surface, model.sat_limit_surface)

        ax_mv.plot(
            slice_indices + c_idx * compartment.slices,
            compartment.slab,
            color=color, linewidth=2,
            label=f"{compartment.name} load",
        )
        ax_mv.plot(
            slice_indices + c_idx * compartment.slices,
            m_vals,
            color=color, linewidth=2, linestyle="--",
            label=f"{compartment.name} M-limit",
        )

        # Mark exceeded slices
        exceeded = np.where(compartment.slab > m_vals)[0]
        if len(exceeded) > 0:
            ax_mv.scatter(
                exceeded + c_idx * compartment.slices,
                compartment.slab[exceeded],
                color="red", s=50, zorder=5,
            )

        # Separator line between compartments
        if c_idx < num_compartments - 1:
            sep_x = (c_idx + 1) * compartment.slices
            ax_mv.axvline(x=sep_x, color="gray", linestyle=":", alpha=0.5)

    # Reference lines
    ppn2_surface = p_surface * (1 - model.f_o2)
    total_slices = sum(c.slices for c in compartments)
    ax_mv.axhline(
        y=ppn2_surface, color="green", linestyle="-", linewidth=1, alpha=0.5,
        label=f"Surface ppN2 = {ppn2_surface:.2f}",
    )

    ax_mv.set_xlabel("Slice Index (grouped by compartment)")
    ax_mv.set_ylabel("ppN2 (bar)")
    ax_mv.set_title("Final Gas Load vs. Surface M-Value Limits")
    ax_mv.legend(loc="upper right", fontsize=8)
    ax_mv.set_xlim(0, total_slices)
    ax_mv.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for quick overrides."""
    parser = argparse.ArgumentParser(
        description="SlabDive - Slab Diffusion Dive Simulator",
    )
    parser.add_argument("--depth", type=float, help="Dive depth in meters")
    parser.add_argument("--time", type=float, help="Bottom time in minutes")
    parser.add_argument("--fO2", type=float, help="O2 fraction (e.g. 0.32 for EAN32)")
    parser.add_argument(
        "--profile", choices=["square", "multilevel", "sawtooth"],
        help="Profile type",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to model config YAML (default: config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    config = DIVE_CONFIG.copy()
    if args.depth is not None:
        config["depth_m"] = args.depth
    if args.time is not None:
        config["bottom_time_min"] = args.time
    if args.fO2 is not None:
        config["fO2"] = args.fO2
    if args.profile is not None:
        config["profile_type"] = args.profile

    # Build profile
    profile = build_profile(config)

    # Initialize model from config
    model = SlabModel(config_path=args.config, f_o2=config["fO2"])

    # Print plan
    print_dive_plan(profile, model)

    # Run simulation
    result = model.run(profile)

    # Print results
    print_results(result)

    # Visualize
    plot_results(profile, result, model)


if __name__ == "__main__":
    main()
