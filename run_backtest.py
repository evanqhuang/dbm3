#!/usr/bin/env python3
"""
Backtesting runner script for comparing Bühlmann and Slab decompression models.

Usage:
    python run_backtest.py                    # Run default backtest
    python run_backtest.py --quick            # Quick test with fewer profiles
    python run_backtest.py --full             # Full test with 50k+ profiles
    python run_backtest.py --profile 30 20    # Test single profile (30m, 20min)
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import ProfileGenerator, BuhlmannRunner, SlabModel, ModelComparator
from backtest.comparator import run_full_backtest


def test_single_profile(depth: float, time: float):
    """Test a single dive profile through both models."""
    print(f"\n{'=' * 60}")
    print(f"Single Profile Test: {depth}m for {time} min")
    print(f"{'=' * 60}")

    # Generate profile
    gen = ProfileGenerator()
    profile = gen.generate_square(depth=depth, bottom_time=time)

    print(f"\nProfile: {profile.name}")
    print(f"Points: {len(profile.points)}")
    print(f"Max depth: {profile.max_depth}m")

    # Test Bühlmann
    print("\n--- Bühlmann Model ---")
    try:
        runner = BuhlmannRunner()
        buhl_result = runner.run(profile)
        print(f"Max ceiling: {buhl_result.max_ceiling:.3f} bar")
        print(f"Min NDL: {buhl_result.min_ndl:.1f} min")
        print(f"Max supersaturation: {buhl_result.max_supersaturation:.2%}")
        print(f"Requires deco: {buhl_result.requires_deco}")
    except Exception as e:
        print(f"ERROR: {e}")
        buhl_result = None

    # Test Slab
    print("\n--- Slab Model ---")
    try:
        slab = SlabModel()
        slab_result = slab.run(profile)
        print(f"Max tissue load: {slab_result.max_tissue_load:.3f} bar")
        print(f"Min margin: {slab_result.min_margin:.3f} bar")
        print(f"Critical slice: {slab_result.critical_slice}")
        print(f"Max supersaturation: {slab_result.max_supersaturation:.2%}")
        print(f"Final NDL: {slab_result.final_ndl:.1f} min")
        print(f"Exceeded limit: {slab_result.exceeded_limit}")
    except Exception as e:
        print(f"ERROR: {e}")
        slab_result = None

    # Compare
    if buhl_result and slab_result:
        delta_risk = slab_result.risk_score - buhl_result.risk_score
        delta_ndl = slab_result.final_ndl - buhl_result.min_ndl
        print("\n--- Risk Comparison ---")
        print(f"Bühlmann risk: {buhl_result.risk_score:.3f}")
        print(f"Slab risk: {slab_result.risk_score:.3f}")
        print(f"Delta Risk (Slab - Bühlmann): {delta_risk:.3f}")
        if delta_risk > 0:
            print("→ Slab predicts HIGHER risk")
        elif delta_risk < 0:
            print("→ Bühlmann predicts HIGHER risk")
        else:
            print("→ Models agree on risk")

        print("\n--- NDL Comparison ---")
        print(f"Bühlmann NDL: {buhl_result.min_ndl:.1f} min")
        print(f"Slab NDL: {slab_result.final_ndl:.1f} min")
        print(f"Delta NDL (Slab - Bühlmann): {delta_ndl:.1f} min")
        if delta_ndl > 0:
            print("→ Slab allows MORE bottom time")
        elif delta_ndl < 0:
            print("→ Bühlmann allows MORE bottom time")
        else:
            print("→ Models agree on NDL")


def run_quick_test():
    """Run a quick test with few profiles."""
    print("\n" + "=" * 60)
    print("QUICK BACKTEST")
    print("=" * 60)

    results = run_full_backtest(
        depths=[10.0, 20.0, 30.0, 40.0],
        times=[10, 20, 30],
        save_plots=True,
        output_dir="backtest_output",
    )

    print_report(results["report"])
    print(f"\nPlots saved to: backtest_output/")


def run_default_test():
    """Run default backtest."""
    print("\n" + "=" * 60)
    print("DEFAULT BACKTEST")
    print("=" * 60)

    results = run_full_backtest(
        depths=[float(x) for x in range(10, 55, 5)],  # 10-50m in 5m steps
        times=list(range(5, 45, 5)),  # 5-40min in 5min steps
        save_plots=True,
        output_dir="backtest_output",
    )

    print_report(results["report"])
    print(f"\nPlots saved to: backtest_output/")


def run_full_test():
    """Run full backtest with many profiles."""
    print("\n" + "=" * 60)
    print("FULL BACKTEST (This may take a while...)")
    print("=" * 60)

    # Generate depths: 5-70m in 1m steps
    depths = [float(x) for x in range(5, 71, 1)]
    # Generate times: 1-120min in 1min steps
    times = [float(x) for x in range(1, 121, 1)]

    total = len(depths) * len(times)
    print(f"Generating {total} profiles...")

    results = run_full_backtest(
        depths=depths, times=times, save_plots=True, output_dir="backtest_output_full"
    )

    print_report(results["report"])
    print(f"\nPlots saved to: backtest_output_full/")


def print_report(report: dict):
    """Print formatted report."""
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)

    print(
        f"\nProfiles analyzed: {report.get('valid_profiles', 0)}/{report.get('total_profiles', 0)}"
    )

    print("\n--- Risk Divergence Statistics ---")
    print(f"Mean ΔRisk: {report.get('mean_delta_risk', 0):.4f}")
    print(f"Std ΔRisk: {report.get('std_delta_risk', 0):.4f}")
    print(f"Max ΔRisk: {report.get('max_delta_risk', 0):.4f}")
    print(f"Min ΔRisk: {report.get('min_delta_risk', 0):.4f}")
    print(f"Risk Correlation: {report.get('risk_correlation', 0):.4f}")

    print("\n--- NDL Divergence Statistics ---")
    print(f"Mean ΔNDL: {report.get('mean_delta_ndl', 0):.1f} min")
    print(f"Std ΔNDL: {report.get('std_delta_ndl', 0):.1f} min")
    print(f"Max ΔNDL: {report.get('max_delta_ndl', 0):.1f} min")
    print(f"Min ΔNDL: {report.get('min_delta_ndl', 0):.1f} min")
    print(f"NDL Correlation: {report.get('ndl_correlation', 0):.4f}")

    print("\n--- Average Values ---")
    print(f"Mean Bühlmann risk: {report.get('mean_buhlmann_risk', 0):.4f}")
    print(f"Mean Slab risk: {report.get('mean_slab_risk', 0):.4f}")
    print(f"Mean Bühlmann NDL: {report.get('mean_buhlmann_ndl', 0):.1f} min")
    print(f"Mean Slab NDL: {report.get('mean_slab_ndl', 0):.1f} min")

    print("\n--- Conservative Counts ---")
    print(
        f"Slab more conservative (risk): {report.get('slab_conservative_count', 0)} profiles"
    )
    print(
        f"Bühlmann more conservative (risk): {report.get('buhlmann_conservative_count', 0)} profiles"
    )

    if "top_divergent_risk" in report:
        print("\n--- Top 5 Most Divergent (Risk) ---")
        for p in report["top_divergent_risk"]:
            print(
                f"  {p['name']}: ΔRisk={p['delta_risk']:.3f} "
                f"(B={p['buhlmann_risk']:.3f}, S={p['slab_risk']:.3f})"
            )

    if "top_divergent_ndl" in report:
        print("\n--- Top 5 Most Divergent (NDL) ---")
        for p in report["top_divergent_ndl"]:
            print(
                f"  {p['name']}: ΔNDL={p['delta_ndl']:.1f}min "
                f"(B={p['buhlmann_ndl']:.1f}, S={p['slab_ndl']:.1f})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Bühlmann vs Slab decompression models"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with fewer profiles"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full test with many profiles (slow)"
    )
    parser.add_argument(
        "--profile",
        nargs=2,
        type=float,
        metavar=("DEPTH", "TIME"),
        help="Test single profile at DEPTH (m) for TIME (min)",
    )

    args = parser.parse_args()

    # Check if libbuhlmann binary exists
    binary_path = os.path.join(os.path.dirname(__file__), "libbuhlmann", "src", "dive")
    if not os.path.exists(binary_path):
        print(f"WARNING: libbuhlmann binary not found at {binary_path}")
        print("Please compile it first: cd libbuhlmann/src && make")
        print("Continuing anyway (Bühlmann tests may fail)...\n")

    if args.profile:
        test_single_profile(args.profile[0], args.profile[1])
    elif args.quick:
        run_quick_test()
    elif args.full:
        run_full_test()
    else:
        run_default_test()


if __name__ == "__main__":
    main()
