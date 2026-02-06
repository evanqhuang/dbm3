#!/usr/bin/env python3
"""
Analyze backtest divergence data from results_detailed.csv.

This script loads CSV output from full backtest runs and performs statistical
analysis of the divergence between Bühlmann and Slab decompression models across
depth and time regimes.
"""

import matplotlib
matplotlib.use('Agg')  # Headless rendering

import csv
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), "backtest_output_full", "results_detailed.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "backtest_output_full")


def load_csv_data():
    """Load and parse CSV into numpy arrays and lists."""
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    profile_names = []
    depths = []
    times = []
    buhlmann_risks = []
    slab_risks = []
    delta_risks = []
    buhlmann_ndls = []
    slab_ndls = []
    delta_ndls = []
    slab_critical_slices = []

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            profile_names.append(row['profile_name'])
            depths.append(float(row['depth_m']))
            times.append(float(row['bottom_time_min']))
            buhlmann_risks.append(float(row['buhlmann_risk']))
            slab_risks.append(float(row['slab_risk']))
            delta_risks.append(float(row['delta_risk']))
            buhlmann_ndls.append(float(row['buhlmann_ndl']))
            slab_ndls.append(float(row['slab_ndl']))
            delta_ndls.append(float(row['delta_ndl']))
            slab_critical_slices.append(int(row['slab_critical_slice']))

    return {
        'profile_name': profile_names,
        'depth': np.array(depths),
        'time': np.array(times),
        'buhlmann_risk': np.array(buhlmann_risks),
        'slab_risk': np.array(slab_risks),
        'delta_risk': np.array(delta_risks),
        'buhlmann_ndl': np.array(buhlmann_ndls),
        'slab_ndl': np.array(slab_ndls),
        'delta_ndl': np.array(delta_ndls),
        'slab_critical_slice': np.array(slab_critical_slices),
    }


def print_overall_statistics(data):
    """Print overall statistical summary."""
    print("=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    total = len(data['depth'])
    print(f"\nTotal profiles: {total}")

    print("\n--- Risk Divergence ---")
    print(f"Mean ΔRisk: {np.mean(data['delta_risk']):.6f}")
    print(f"Std ΔRisk: {np.std(data['delta_risk']):.6f}")
    print(f"Min ΔRisk: {np.min(data['delta_risk']):.6f}")
    print(f"Max ΔRisk: {np.max(data['delta_risk']):.6f}")

    risk_corr = np.corrcoef(data['buhlmann_risk'], data['slab_risk'])[0, 1]
    print(f"Risk correlation (Bühlmann vs Slab): {risk_corr:.4f}")

    print("\n--- NDL Divergence ---")
    print(f"Mean ΔNDL: {np.mean(data['delta_ndl']):.2f} min")
    print(f"Std ΔNDL: {np.std(data['delta_ndl']):.2f} min")
    print(f"Min ΔNDL: {np.min(data['delta_ndl']):.2f} min")
    print(f"Max ΔNDL: {np.max(data['delta_ndl']):.2f} min")

    ndl_corr = np.corrcoef(data['buhlmann_ndl'], data['slab_ndl'])[0, 1]
    print(f"NDL correlation (Bühlmann vs Slab): {ndl_corr:.4f}")

    print("\n--- Conservative Model Counts ---")
    buhlmann_conservative = np.sum(data['delta_risk'] < 0)
    slab_conservative = np.sum(data['delta_risk'] > 0)
    equal = np.sum(data['delta_risk'] == 0)

    print(f"Bühlmann more conservative (ΔRisk < 0): {buhlmann_conservative} ({100*buhlmann_conservative/total:.1f}%)")
    print(f"Slab more conservative (ΔRisk > 0): {slab_conservative} ({100*slab_conservative/total:.1f}%)")
    print(f"Equal risk: {equal} ({100*equal/total:.1f}%)")


def print_depth_binned_analysis(data):
    """Analyze data by depth bins."""
    print("\n" + "=" * 60)
    print("DEPTH-BINNED ANALYSIS")
    print("=" * 60)

    bins = [
        ('Shallow (5-15m)', 5, 15),
        ('Medium (16-35m)', 16, 35),
        ('Deep (36-55m)', 36, 55),
        ('Very Deep (56-70m)', 56, 70),
    ]

    print(f"\n{'Bin':<25} {'Count':>8} {'Mean ΔRisk':>12} {'Mean ΔNDL':>12} {'Max |ΔRisk|':>13} {'Max |ΔNDL|':>13}")
    print("-" * 90)

    for name, dmin, dmax in bins:
        mask = (data['depth'] >= dmin) & (data['depth'] <= dmax)
        count = np.sum(mask)

        if count == 0:
            print(f"{name:<25} {count:>8} {'N/A':>12} {'N/A':>12} {'N/A':>13} {'N/A':>13}")
            continue

        mean_delta_risk = np.mean(data['delta_risk'][mask])
        mean_delta_ndl = np.mean(data['delta_ndl'][mask])
        max_abs_risk = np.max(np.abs(data['delta_risk'][mask]))
        max_abs_ndl = np.max(np.abs(data['delta_ndl'][mask]))

        print(f"{name:<25} {count:>8} {mean_delta_risk:>12.6f} {mean_delta_ndl:>12.2f} {max_abs_risk:>13.6f} {max_abs_ndl:>13.2f}")


def print_time_binned_analysis(data):
    """Analyze data by time bins."""
    print("\n" + "=" * 60)
    print("TIME-BINNED ANALYSIS")
    print("=" * 60)

    bins = [
        ('Short (1-20min)', 1, 20),
        ('Medium (21-50min)', 21, 50),
        ('Long (51-90min)', 51, 90),
        ('Very Long (91-120min)', 91, 120),
    ]

    print(f"\n{'Bin':<25} {'Count':>8} {'Mean ΔRisk':>12} {'Mean ΔNDL':>12} {'Max |ΔRisk|':>13} {'Max |ΔNDL|':>13}")
    print("-" * 90)

    for name, tmin, tmax in bins:
        mask = (data['time'] >= tmin) & (data['time'] <= tmax)
        count = np.sum(mask)

        if count == 0:
            print(f"{name:<25} {count:>8} {'N/A':>12} {'N/A':>12} {'N/A':>13} {'N/A':>13}")
            continue

        mean_delta_risk = np.mean(data['delta_risk'][mask])
        mean_delta_ndl = np.mean(data['delta_ndl'][mask])
        max_abs_risk = np.max(np.abs(data['delta_risk'][mask]))
        max_abs_ndl = np.max(np.abs(data['delta_ndl'][mask]))

        print(f"{name:<25} {count:>8} {mean_delta_risk:>12.6f} {mean_delta_ndl:>12.2f} {max_abs_risk:>13.6f} {max_abs_ndl:>13.2f}")


def print_divergence_boundaries(data):
    """Find minimum time where divergence exceeds thresholds for each depth."""
    print("\n" + "=" * 60)
    print("DIVERGENCE BOUNDARY DETECTION")
    print("=" * 60)
    print("\nMinimum bottom time where |ΔRisk| > 0.1 or |ΔNDL| > 10")

    unique_depths = np.unique(data['depth'])

    print(f"\n{'Depth (m)':>10} {'Risk Threshold Time (min)':>28} {'NDL Threshold Time (min)':>27}")
    print("-" * 70)

    risk_threshold_depths = []
    risk_threshold_times = []
    ndl_threshold_depths = []
    ndl_threshold_times = []

    for depth in unique_depths:
        mask = data['depth'] == depth
        times_at_depth = data['time'][mask]
        delta_risks = data['delta_risk'][mask]
        delta_ndls = data['delta_ndl'][mask]

        # Sort by time
        sort_idx = np.argsort(times_at_depth)
        times_sorted = times_at_depth[sort_idx]
        delta_risks_sorted = delta_risks[sort_idx]
        delta_ndls_sorted = delta_ndls[sort_idx]

        # Find first time where |delta_risk| > 0.1
        risk_idx = np.where(np.abs(delta_risks_sorted) > 0.1)[0]
        risk_time = times_sorted[risk_idx[0]] if len(risk_idx) > 0 else None

        # Find first time where |delta_ndl| > 10
        ndl_idx = np.where(np.abs(delta_ndls_sorted) > 10)[0]
        ndl_time = times_sorted[ndl_idx[0]] if len(ndl_idx) > 0 else None

        risk_str = f"{risk_time:.0f}" if risk_time is not None else "Never"
        ndl_str = f"{ndl_time:.0f}" if ndl_time is not None else "Never"

        print(f"{depth:>10.1f} {risk_str:>28} {ndl_str:>27}")

        if risk_time is not None:
            risk_threshold_depths.append(depth)
            risk_threshold_times.append(risk_time)

        if ndl_time is not None:
            ndl_threshold_depths.append(depth)
            ndl_threshold_times.append(ndl_time)

    return {
        'risk_depths': risk_threshold_depths,
        'risk_times': risk_threshold_times,
        'ndl_depths': ndl_threshold_depths,
        'ndl_times': ndl_threshold_times,
    }


def print_top_divergent_profiles(data):
    """Show top 10 most divergent profiles by risk and NDL."""
    print("\n" + "=" * 60)
    print("TOP 10 DIVERGENT PROFILES")
    print("=" * 60)

    # Top 10 by |delta_risk|
    print("\n--- Top 10 by |ΔRisk| ---")
    risk_idx = np.argsort(np.abs(data['delta_risk']))[::-1][:10]

    print(f"\n{'Profile':<30} {'Depth':>7} {'Time':>7} {'B.Risk':>8} {'S.Risk':>8} {'ΔRisk':>10}")
    print("-" * 80)
    for idx in risk_idx:
        print(f"{data['profile_name'][idx]:<30} {data['depth'][idx]:>7.1f} "
              f"{data['time'][idx]:>7.1f} {data['buhlmann_risk'][idx]:>8.4f} "
              f"{data['slab_risk'][idx]:>8.4f} {data['delta_risk'][idx]:>10.6f}")

    # Top 10 by |delta_ndl|
    print("\n--- Top 10 by |ΔNDL| ---")
    ndl_idx = np.argsort(np.abs(data['delta_ndl']))[::-1][:10]

    print(f"\n{'Profile':<30} {'Depth':>7} {'Time':>7} {'B.NDL':>8} {'S.NDL':>8} {'ΔNDL':>10}")
    print("-" * 80)
    for idx in ndl_idx:
        print(f"{data['profile_name'][idx]:<30} {data['depth'][idx]:>7.1f} "
              f"{data['time'][idx]:>7.1f} {data['buhlmann_ndl'][idx]:>8.2f} "
              f"{data['slab_ndl'][idx]:>8.2f} {data['delta_ndl'][idx]:>10.2f}")


def print_critical_slice_analysis(data):
    """Analyze critical slice distribution across depth bins."""
    print("\n" + "=" * 60)
    print("CRITICAL SLICE ANALYSIS")
    print("=" * 60)

    bins = [
        ('Shallow (5-15m)', 5, 15),
        ('Medium (16-35m)', 16, 35),
        ('Deep (36-55m)', 36, 55),
        ('Very Deep (56-70m)', 56, 70),
    ]

    for name, dmin, dmax in bins:
        mask = (data['depth'] >= dmin) & (data['depth'] <= dmax)
        slices = data['slab_critical_slice'][mask]

        if len(slices) == 0:
            print(f"\n{name}: No data")
            continue

        unique_slices, counts = np.unique(slices, return_counts=True)
        total = len(slices)

        print(f"\n{name}:")
        print(f"{'Slice':>8} {'Count':>10} {'Percentage':>12}")
        print("-" * 35)

        for slice_val, count in zip(unique_slices, counts):
            pct = 100 * count / total
            print(f"{slice_val:>8} {count:>10} {pct:>11.1f}%")


def print_regime_classification(data):
    """Classify profiles into agreement/divergence regimes."""
    print("\n" + "=" * 60)
    print("REGIME CLASSIFICATION")
    print("=" * 60)

    abs_delta_risk = np.abs(data['delta_risk'])
    abs_delta_ndl = np.abs(data['delta_ndl'])
    total = len(data['depth'])

    # High divergence: |delta_risk| > 0.2 OR |delta_ndl| > 20
    high = (abs_delta_risk > 0.2) | (abs_delta_ndl > 20)
    high_count = np.sum(high)

    # Agreement zone: |delta_risk| < 0.05 AND |delta_ndl| < 5
    agreement = (abs_delta_risk < 0.05) & (abs_delta_ndl < 5)
    agreement_count = np.sum(agreement)

    # Moderate divergence: everything else
    moderate = ~agreement & ~high
    moderate_count = np.sum(moderate)

    print(f"\n{'Regime':<25} {'Count':>10} {'Percentage':>12}")
    print("-" * 50)
    print(f"{'Agreement':<25} {agreement_count:>10} {100*agreement_count/total:>11.1f}%")
    print(f"  (|ΔRisk| < 0.05 AND |ΔNDL| < 5)")
    print(f"{'Moderate Divergence':<25} {moderate_count:>10} {100*moderate_count/total:>11.1f}%")
    print(f"  (|ΔRisk| 0.05-0.2 OR |ΔNDL| 5-20)")
    print(f"{'High Divergence':<25} {high_count:>10} {100*high_count/total:>11.1f}%")
    print(f"  (|ΔRisk| > 0.2 OR |ΔNDL| > 20)")


def plot_risk_divergence_vs_depth(data):
    """Scatter plot of delta_risk vs depth, colored by time."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(data['depth'], data['delta_risk'], c=data['time'],
                        cmap='viridis', alpha=0.6, s=10)

    ax.set_xlabel('Depth (m)', fontsize=12)
    ax.set_ylabel('ΔRisk (Slab - Bühlmann)', fontsize=12)
    ax.set_title('Risk Divergence vs Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Bottom Time (min)', fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'divergence_risk_vs_depth.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_ndl_divergence_vs_depth(data):
    """Scatter plot of delta_ndl vs depth, colored by time."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(data['depth'], data['delta_ndl'], c=data['time'],
                        cmap='viridis', alpha=0.6, s=10)

    ax.set_xlabel('Depth (m)', fontsize=12)
    ax.set_ylabel('ΔNDL (Slab - Bühlmann) [min]', fontsize=12)
    ax.set_title('NDL Divergence vs Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Bottom Time (min)', fontsize=11)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'divergence_ndl_vs_depth.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_divergence_boundaries(boundaries):
    """Plot divergence boundary curves."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    if boundaries['risk_depths']:
        ax.plot(boundaries['risk_depths'], boundaries['risk_times'],
               marker='o', markersize=4, label='Risk threshold (|ΔRisk| > 0.1)',
               linewidth=2, color='red')

    if boundaries['ndl_depths']:
        ax.plot(boundaries['ndl_depths'], boundaries['ndl_times'],
               marker='s', markersize=4, label='NDL threshold (|ΔNDL| > 10)',
               linewidth=2, color='blue')

    ax.set_xlabel('Depth (m)', fontsize=12)
    ax.set_ylabel('Minimum Bottom Time (min)', fontsize=12)
    ax.set_title('Divergence Boundary Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'divergence_boundary.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_divergence_histograms(data):
    """Plot histograms of delta_risk and delta_ndl."""
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Delta risk histogram
    ax1.hist(data['delta_risk'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('ΔRisk (Slab - Bühlmann)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Risk Divergence Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')

    # Delta NDL histogram
    ax2.hist(data['delta_ndl'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('ΔNDL (Slab - Bühlmann) [min]', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('NDL Divergence Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'divergence_histograms.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main analysis entry point."""
    print("\nLoading CSV data from:", CSV_PATH)
    data = load_csv_data()
    print(f"Loaded {len(data['depth'])} data points")

    # Terminal analysis
    print_overall_statistics(data)
    print_depth_binned_analysis(data)
    print_time_binned_analysis(data)
    boundaries = print_divergence_boundaries(data)
    print_top_divergent_profiles(data)
    print_critical_slice_analysis(data)
    print_regime_classification(data)

    # Plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    print("")

    plot_risk_divergence_vs_depth(data)
    plot_ndl_divergence_vs_depth(data)
    plot_divergence_boundaries(boundaries)
    plot_divergence_histograms(data)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
