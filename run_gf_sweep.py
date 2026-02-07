#!/usr/bin/env python3
"""
Gradient Factor sweep orchestration script.

Runs backtests across multiple GF settings to evaluate how Gradient Factors
affect the divergence between Buhlmann and Slab models.

Usage:
    python run_gf_sweep.py              # Run both synthetic and real data tests
    python run_gf_sweep.py --mode full  # Only synthetic grid (7,920 profiles)
    python run_gf_sweep.py --mode real  # Only real scraped profiles
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_backtest import run_full_test, run_real_data_test
from backtest.output_naming import load_effective_config, build_settings_dirname


# GF pairs to test: (gf_low, gf_high, label)
GF_SETS = [
    (100, 100, "Standard Buhlmann"),
    (85, 85, "Recreational Flat"),
    (70, 85, "Moderate Conservative"),
    (50, 80, "Tech Conservative"),
    (30, 85, "Deep Stops Moderate"),
    (30, 70, "Deep Tech"),
]


def print_header():
    """Print header showing all GF pairs that will be tested."""
    print("\n" + "=" * 70)
    print("GRADIENT FACTOR SWEEP")
    print("=" * 70)
    print(f"\nTesting {len(GF_SETS)} GF configurations:\n")
    for i, (low, high, label) in enumerate(GF_SETS, 1):
        print(f"  [{i}/{len(GF_SETS)}] GF {low}/{high} — {label}")
    print()


def check_libbuhlmann():
    """Check if libbuhlmann binary exists."""
    binary_path = os.path.join(
        os.path.dirname(__file__), "libbuhlmann", "src", "dive"
    )
    if not os.path.exists(binary_path):
        print(f"WARNING: libbuhlmann binary not found at {binary_path}")
        print("Please compile it first:")
        print("  cd libbuhlmann && ./bootstrap.sh && ./configure && make")
        print("\nContinuing anyway (Buhlmann tests may fail)...\n")
        return False
    return True


def run_sweep(mode: str = "both", results: list | None = None) -> list:
    """Run the GF sweep.

    Args:
        mode: 'full' (synthetic only), 'real' (real data only), or 'both' (default)
        results: Mutable list to accumulate results into (for partial results on interrupt).
                 If None, creates a new list.

    Returns:
        List of dicts with keys: gf_low, gf_high, label, output_dir, elapsed_time
    """
    if results is None:
        results = []

    for i, (gf_low, gf_high, label) in enumerate(GF_SETS, 1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(GF_SETS)}] GF {gf_low}/{gf_high} — {label}")
        print("=" * 70)

        effective_config = load_effective_config(gf_override=(gf_low, gf_high))
        gf = effective_config["gf"]

        settings_dir = build_settings_dirname(
            gf,
            effective_config["conservatism"],
            effective_config["boyle_exponent"],
            effective_config["f_o2"],
        )

        start_time = time.time()

        try:
            if mode in ("full", "both"):
                print(f"\nRunning synthetic grid backtest (GF {gf_low}/{gf_high})...")
                run_full_test(effective_config)

            if mode in ("real", "both"):
                print(f"\nRunning real data backtest (GF {gf_low}/{gf_high})...")
                run_real_data_test(effective_config)

            elapsed = time.time() - start_time

            results.append({
                "gf_low": gf_low,
                "gf_high": gf_high,
                "label": label,
                "output_dir": settings_dir,
                "elapsed_time": elapsed,
            })

            print(f"\nCompleted GF {gf_low}/{gf_high} in {elapsed:.1f}s")

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            results.append({
                "gf_low": gf_low,
                "gf_high": gf_high,
                "label": label,
                "output_dir": settings_dir,
                "elapsed_time": elapsed,
                "interrupted": True,
            })
            raise

        except Exception as e:
            print(f"\nERROR running GF {gf_low}/{gf_high}: {e}")
            elapsed = time.time() - start_time
            results.append({
                "gf_low": gf_low,
                "gf_high": gf_high,
                "label": label,
                "output_dir": settings_dir,
                "elapsed_time": elapsed,
                "error": str(e),
            })

    return results


def print_summary(results: list, total_time: float):
    """Print summary table of all runs."""
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print()

    # Table header
    print(f"{'GF':<12} {'Label':<25} {'Time (s)':<10} {'Output Dir'}")
    print("-" * 70)

    # Table rows
    for r in results:
        gf_str = f"{r['gf_low']}/{r['gf_high']}"
        time_str = f"{r['elapsed_time']:.1f}"

        status = ""
        if r.get("interrupted"):
            status = " (interrupted)"
        elif r.get("error"):
            status = " (error)"

        print(
            f"{gf_str:<12} {r['label']:<25} {time_str:<10} {r['output_dir']}{status}"
        )

    print()
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run backtests across multiple Gradient Factor settings"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "real", "both"],
        default="both",
        help="Test mode: 'full' (synthetic grid), 'real' (scraped profiles), "
             "'both' (default)",
    )
    args = parser.parse_args()

    print_header()
    check_libbuhlmann()

    total_start = time.time()
    results = []

    try:
        run_sweep(mode=args.mode, results=results)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Showing partial results...")

    total_time = time.time() - total_start

    if results:
        print_summary(results, total_time)
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()
