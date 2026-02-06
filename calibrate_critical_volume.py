#!/usr/bin/env python3
"""
Calibrate per-compartment critical volume thresholds (V_crit).

For each reference depth:
  1. Get the Buhlmann NDL
  2. Run slab model for that NDL (full profile with ascent)
  3. At end of profile (surface), measure excess_gas for each compartment

V_crit for each compartment = median excess_gas across calibration depths.
This ensures the controlling compartment has risk ≈ 1.0 at the NDL boundary.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.profile_generator import ProfileGenerator
from backtest.buhlmann_runner import BuhlmannRunner
from backtest.slab_model import SlabModel


def main():
    print("=" * 70)
    print("PER-COMPARTMENT CRITICAL VOLUME CALIBRATION")
    print("=" * 70)

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    atmospheric_pressure = 1.01325
    # Read f_o2 from config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    f_o2 = config.get('f_o2', 0.21)
    p_surface_equil = atmospheric_pressure * (1 - f_o2)

    depths = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70]

    gen = ProfileGenerator()
    buhl_runner = BuhlmannRunner()

    # {compartment_name: [excess_gas_at_ndl_for_each_depth]}
    compartment_excess = {}

    print(f"\nSurface equilibrium ppN2: {p_surface_equil:.4f} bar")
    print(f"O2 fraction: {f_o2}\n")

    for depth in depths:
        print(f"Processing {depth}m...", end=" ", flush=True)

        test_profile = gen.generate_square(depth=depth, bottom_time=1)
        try:
            buhl_result = buhl_runner.run(test_profile)
            ndl = buhl_result.min_ndl
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        if ndl >= 100:
            print(f"NDL={ndl:.0f}min (unlimited, skipping)")
            continue

        ndl_int = int(np.floor(ndl))
        if ndl_int < 1:
            ndl_int = 1

        # Run full profile (descent + bottom + ascent) at NDL
        ndl_profile = gen.generate_square(depth=depth, bottom_time=ndl_int)
        model = SlabModel(config_path=config_path)
        slab_result = model.run(ndl_profile)

        # Measure excess gas at end of profile (surface) for each compartment
        excess_per_comp = {}
        for compartment in model.compartments:
            final_slab = slab_result.final_slabs[compartment.name]
            excess_gas = np.sum(np.maximum(0, final_slab - p_surface_equil)) * model.dx
            excess_per_comp[compartment.name] = excess_gas

            if compartment.name not in compartment_excess:
                compartment_excess[compartment.name] = []
            compartment_excess[compartment.name].append(excess_gas)

        # Find controlling compartment (highest excess / current v_crit)
        ctrl = max(excess_per_comp, key=lambda c: excess_per_comp[c] / next(
            comp.v_crit for comp in model.compartments if comp.name == c
        ))

        excess_str = ", ".join(f"{name}={val:.4f}" for name, val in excess_per_comp.items())
        print(f"NDL={ndl_int}min, ctrl={ctrl}, excess=[{excess_str}]")

    if not compartment_excess:
        print("\nERROR: No valid calibration data collected")
        return

    # Compute V_crit per compartment
    print("\n" + "=" * 70)
    print("PER-COMPARTMENT EXCESS GAS AT NDL BOUNDARIES")
    print("=" * 70)

    compartment_names = list(compartment_excess.keys())
    v_crit_values = {}

    for name in compartment_names:
        values = compartment_excess[name]
        median_val = float(np.median(values))
        max_val = float(np.max(values))
        min_val = float(np.min(values))
        std_val = float(np.std(values))

        # Use median as V_crit (robust to outliers)
        v_crit_values[name] = median_val

        print(f"\n{name}:")
        print(f"  Values: {[f'{v:.4f}' for v in values]}")
        print(f"  Min={min_val:.4f}, Median={median_val:.4f}, Max={max_val:.4f}, Std={std_val:.4f}")
        print(f"  -> V_crit = {median_val:.6f}")

    # Write to config.yaml
    print(f"\n{'=' * 70}")
    print("UPDATING config.yaml")
    print("=" * 70)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update each compartment with v_crit
    for comp_config in config['compartments']:
        name = comp_config['name']
        if name in v_crit_values:
            comp_config['v_crit'] = round(v_crit_values[name], 6)
            print(f"  {name}: v_crit = {v_crit_values[name]:.6f}")

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated config.yaml with per-compartment v_crit values")

    # Verify: re-run a test to check risk values
    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print("=" * 70)

    for depth in [20, 30, 40]:
        test_profile = gen.generate_square(depth=depth, bottom_time=1)
        try:
            buhl_result = buhl_runner.run(test_profile)
            ndl = int(np.floor(buhl_result.min_ndl))
            if ndl >= 100 or ndl < 1:
                continue
        except Exception:
            continue

        # At NDL
        ndl_profile = gen.generate_square(depth=depth, bottom_time=ndl)
        model = SlabModel(config_path=config_path)
        result = model.run(ndl_profile)
        print(f"  {depth}m/{ndl}min (NDL): cv_ratio={result.max_cv_ratio:.4f}, ctrl={result.critical_compartment}")

        # 5min before NDL
        if ndl > 5:
            before_profile = gen.generate_square(depth=depth, bottom_time=ndl - 5)
            model2 = SlabModel(config_path=config_path)
            result2 = model2.run(before_profile)
            print(f"  {depth}m/{ndl-5}min (NDL-5): cv_ratio={result2.max_cv_ratio:.4f}, ctrl={result2.critical_compartment}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
