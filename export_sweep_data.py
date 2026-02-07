#!/usr/bin/env python3
"""
Consolidate all GF sweep output into a single JSON file.

Reads from both backtest_output_real/ (real scraped profiles) and
backtest_output_full/ (synthetic grid profiles), merging report.json,
results_detailed.csv, and matrix CSVs into one structured JSON.

Usage:
    python export_sweep_data.py                    # Export to data/gf_sweep.json
    python export_sweep_data.py -o path/out.json   # Custom output path
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_gf_sweep import GF_SETS
from backtest.output_naming import load_effective_config, build_settings_dirname


# Map source label to base output directory
DATASETS = [
    ("real", "backtest_output_real"),
    ("synthetic", "backtest_output_full"),
]

MATRIX_NAMES = [
    "delta_risk", "delta_ndl", "delta_ceiling",
    "buhlmann_risk", "buhlmann_ndl", "buhlmann_ceiling",
    "slab_risk", "slab_ndl", "slab_ceiling",
]


def _coerce_value(value: str):
    """Convert a CSV string value to the appropriate Python type."""
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def _load_profiles(csv_path: str) -> list[dict]:
    """Load per-profile results from a results_detailed.csv file."""
    profiles = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles.append({k: _coerce_value(v) for k, v in row.items()})
    return profiles


def _load_matrix(csv_path: str) -> dict:
    """Load a matrix CSV into a dict with depths, times, and 2D data array.

    Matrix CSVs have the format:
        time\\depth, 5.0, 6.0, ...
        1.0, val, val, ...
        2.0, val, val, ...
    """
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        depths = [float(d) for d in header[1:]]
        times = []
        data = []
        for row in reader:
            times.append(float(row[0]))
            data.append([float(v) for v in row[1:]])
    return {"depths": depths, "times": times, "data": data}


def _resolve_gf_dir(gf_low: int, gf_high: int) -> str:
    """Build the settings directory name for a GF pair."""
    effective_config = load_effective_config(gf_override=(gf_low, gf_high))
    return build_settings_dirname(
        effective_config["gf"],
        effective_config["conservatism"],
        effective_config["boyle_exponent"],
        effective_config["f_o2"],
    )


def export(output_path: str):
    """Build consolidated JSON from all sweep output directories."""
    consolidated = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [],
    }

    for source, base_dir in DATASETS:
        include_matrices = source == "synthetic"
        dataset = {
            "source": source,
            "base_dir": base_dir,
            "gf_settings": [],
        }

        total_profiles = 0

        for gf_low, gf_high, label in GF_SETS:
            settings_dir = _resolve_gf_dir(gf_low, gf_high)
            run_dir = os.path.join(base_dir, settings_dir)

            report_path = os.path.join(run_dir, "report.json")
            csv_path = os.path.join(run_dir, "results_detailed.csv")

            if not os.path.exists(report_path):
                print(f"  SKIP [{source}] GF {gf_low}/{gf_high}: {report_path} not found")
                continue

            with open(report_path) as f:
                summary = json.load(f)

            profiles = []
            if os.path.exists(csv_path):
                profiles = _load_profiles(csv_path)

            entry = {
                "gf_low": gf_low,
                "gf_high": gf_high,
                "label": label,
                "summary": summary,
                "profiles": profiles,
            }

            if include_matrices:
                matrices = {}
                for name in MATRIX_NAMES:
                    matrix_path = os.path.join(run_dir, f"matrix_{name}.csv")
                    if os.path.exists(matrix_path):
                        matrices[name] = _load_matrix(matrix_path)
                    else:
                        print(f"  WARN [{source}] GF {gf_low}/{gf_high}: matrix_{name}.csv not found")
                if matrices:
                    entry["matrices"] = matrices

            dataset["gf_settings"].append(entry)
            total_profiles += len(profiles)
            print(f"  [{source}] GF {gf_low}/{gf_high}: {len(profiles)} profiles loaded")

        dataset["total_profiles"] = total_profiles
        consolidated["datasets"].append(dataset)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(consolidated, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported to {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate GF sweep outputs into a single JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/gf_sweep.json",
        help="Output JSON path (default: data/gf_sweep.json)",
    )
    args = parser.parse_args()
    export(args.output)


if __name__ == "__main__":
    main()
