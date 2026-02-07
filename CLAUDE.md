# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

SlabDive is a computational framework for backtesting a Slab Diffusion decompression model (Hempleman's linear bulk diffusion theory) against the established Buhlmann ZH-L16C algorithm. It generates dive profiles, runs them through both models, and produces divergence matrices comparing risk predictions.

## Build & Run Commands

### Prerequisites

```bash
# Install Python dependencies (requires pipenv)
pipenv install

# Compile the libbuhlmann C library (required for Buhlmann model)
cd libbuhlmann && ./bootstrap.sh && ./configure && make && cd ..
```

The compiled `libbuhlmann/src/dive` binary is auto-detected by `BuhlmannRunner`.

### Running

```bash
# Activate virtualenv first
pipenv shell

# Standalone slab model with visualization
python main.py

# Backtesting
python run_backtest.py                    # Default: 10-50m depths, 5-40min times
python run_backtest.py --quick            # Quick: 4 depths x 3 times
python run_backtest.py --full             # Full: 5-70m x 1-120min (~7800 profiles)
python run_backtest.py --profile 30 20    # Single profile: 30m depth, 20min
python run_backtest.py --profile 30 20 --gf 70 85   # Override GF from command line

# Run individual modules
python -m backtest.slab_model
python -m backtest.profile_generator
python -m backtest.comparator
```

### Tests

```bash
# Run deco tests
python -m pytest tests/test_deco.py -v

# Run all tests
python -m pytest tests/ -v
```

The libbuhlmann submodule has tests at `libbuhlmann/test/test_all_of_the_units.py`.

## Architecture

### Data Flow

```
ProfileGenerator -> DiveProfile -> BuhlmannRunner (subprocess to libbuhlmann/src/dive)
                                -> SlabModel (numpy finite difference solver)
                                        |
                                   ModelComparator -> divergence matrices, plots, CSV/JSON reports
```

### Key Types

- **`DiveProfile`** (`backtest/profile_generator.py`): Sequence of `(time_min, depth_m, fO2, fHe)` tuples. Converts to libbuhlmann text-stream format via `to_buhlmann_format()`.
- **`BuhlmannResult`** (`backtest/buhlmann_runner.py`): 16-compartment tissue tensions, ceilings, NDL times, max supersaturation, `gf_low` and `gf_high` values. Output parsed from the `src/dive` binary's stdout (36 values per line).
- **`GradientFactors`** (`backtest/buhlmann_constants.py`): Frozen dataclass with `gf_low` and `gf_high` (0-1.0 fractions). GF 100/100 = standard Buhlmann. GF 70/85 = common conservative setting. `gf_low` controls first stop depth, `gf_high` controls NDL and surface ascent.
- **`SlabResult`** (`backtest/slab_model.py`): Multi-compartment slab history, critical compartment name, final NDL, max critical volume ratio, ceiling at bottom, optional deco schedule. Risk scores use the critical volume approach: `risk = excess_gas / V_crit`.
- **`TissueCompartment`** (`backtest/slab_model.py`): Per-compartment state with diffusion coefficient D, slice count, `v_crit` (critical volume threshold for NDL/risk), and `g_crit` (critical gradient at surface for ceiling/deco).
- **`DecoStop`** (`backtest/slab_model.py`): Single decompression stop with depth (meters) and duration (minutes).
- **`DecoSchedule`** (`backtest/slab_model.py`): Complete deco schedule with stops (deepest-first), TTS, total deco time, final tissue state, requires_deco flag, controlling compartment name, and initial ceiling.
- **`ComparisonResult`** (`backtest/comparator.py`): Wraps both results with computed `delta_risk` and `delta_ndl` properties.

### Model Details

**Buhlmann** (`BuhlmannRunner`): Wraps the C binary via subprocess. Profile data is piped to stdin as `time pressure O2 He` lines. The binary outputs 36 values per line: time, pressure, 16x(N2, He) compartment tensions, ceiling, nodectime. The C binary computes raw tissue tensions; Gradient Factor (GF) adjustments are applied in Python post-processing via `buhlmann_constants.py`. GF modifies M-value limits: at first stop depth, M-values are scaled by `gf_low`; at surface, by `gf_high`. Intermediate depths use linear interpolation. This affects ceiling calculations, NDL, and supersaturation risk scoring. At GF 100/100, all calculations reduce to standard Buhlmann ZH-L16C (backward compatible).

**Slab Model** (`SlabModel`): Multi-compartment finite difference solver (Fick's Second Law). Default 3 compartments configured in `config.yaml`:
- Spine (fast, D=0.002) - sensitive, fastest gas uptake
- Muscle (medium, D=0.0005)
- Joints (slow, D=0.0001) - traps gas, slowest uptake

Each compartment is a 1D slab with: perfect perfusion at slice[0] (blood-tissue interface instantly matches ambient ppN2), diffusion through interior slices, no-flux boundary at the core. Stability requires `k = (D * dt) / dx^2 <= 0.5`.

**Dual Metric Approach** — The model uses two complementary risk metrics from Hennessy-Hempleman theory:

1. **Critical Volume (for NDL & Risk)**: Total integrated excess dissolved gas across the slab vs threshold:
   - `excess_gas = sum(max(0, slab[i] - P_surface_equil) * dx)` — gas above surface equilibrium
   - `risk = excess_gas / V_crit` — 1.0 = at limit, >1.0 = exceeded
   - `V_crit` calibrated at 30m reference depth against Buhlmann ZH-L16C NDL (28 minutes)
   - Evaluated at end of profile (post-ascent), not per-timestep, to avoid ascent transient artifacts
   - NDL calculation simulates ascent to surface before checking excess gas (ascent-aware)

2. **Boundary Gradient (for Ceiling & Deco)**: Gradient at blood-tissue interface vs critical gradient:
   - With perfect perfusion, `slab[0] = ambient ppN2` (no supersaturation at boundary)
   - Uses `slab[1]` (first interior slice) as tissue tension for ceiling checks
   - `gradient = slab[1] - ppN2_ambient`
   - At surface: `safe if gradient <= g_crit * conservatism`
   - At depth: `g_crit` scales with sublinear Boyle's Law: `effective_g_crit = g_crit × (P_ambient / P_surface) ^ boyle_exponent`
   - Higher ambient pressure compresses bubbles, allowing higher gradients
   - `g_crit` calibrated as `slab[1] - ppN2_surface` at 30m NDL

**Known Limitation**: Slow compartments (Joints) may show risk > 1.0 on deco dives because the boundary gradient clears before the integrated volume metric. This divergence is expected — the gradient controls ceiling (local bubble risk), while volume controls overall tissue loading.

### Deco Planning

**Methods** (`SlabModel`):
- `calculate_ceiling(slabs, current_depth)`: Binary search over stop-increment depths to find shallowest safe depth using boundary gradient checks. Returns ceiling in meters (0.0 if safe to surface).
- `plan_deco(slabs, depth, ...)`: Iterative deco schedule generation. Calculates ceiling, ascends, holds at stop until ceiling clears to next shallower depth, repeats until safe to surface. Returns `DecoSchedule` with stops (deepest-first), TTS, controlling compartment, and final tissue state.
- `generate_deco_profile(depth, bottom_time, ...)`: End-to-end profile generation. Simulates descent + bottom time to get tissue state, plans deco, builds `DiveProfile` with stops via `ProfileGenerator.generate_deco_square()`.

**Stop Logic**: At each stop depth, simulates forward 1 minute at a time, checking ceiling after each minute. Exits stop when `ceiling <= next_stop_depth`. Max stop time (default 120 min) prevents infinite loops.

**Controlling Compartment**: Identified as compartment with highest `gradient / effective_g_crit` ratio at the longest stop. Typically Spine (fast) for short dives, Joints (slow) for longer exposures.

**Conservatism**: Config parameter scales both `v_crit` and `g_crit`. Values < 1.0 add safety margin (e.g., 0.85 = 15% more conservative stops).

### Parallel Processing

`ModelComparator.compare_batch()` uses `ThreadPoolExecutor` for Buhlmann (I/O-bound subprocess calls) and `ProcessPoolExecutor` for Slab (CPU-bound numpy). Helper functions `_run_buhlmann_single` and `_run_slab_single` are module-level for pickling compatibility.

### Configuration

`config.yaml` holds slab model parameters:
- **Simulation**: `dt` (time step, seconds), `dx` (slice spacing), `f_o2`, `surface_altitude_m`, `permeability` (null = perfect perfusion)
- **Safety**: `conservatism` (1.0 = match Buhlmann, < 1.0 = more conservative, scales both v_crit and g_crit)
- **Deco**: `stop_increment` (3m), `last_stop_depth` (3m), `ascent_rate` (10 m/min), `max_stop_time` (120 min), `boyle_exponent` (0.5, sublinear pressure scaling: 1.0 = linear Boyle's, 0.5 = sqrt)
- **Compartments**: Each has `name`, `D` (diffusion coeff), `slices`, `v_crit` (critical volume for NDL/risk), `g_crit` (critical gradient at surface for ceiling/deco)
- **Buhlmann**: Optional `buhlmann:` block with `gf_low` (gradient factor at first stop, 0-1.0) and `gf_high` (gradient factor at surface, 0-1.0). Defaults to 1.0/1.0 (standard Buhlmann). Example conservative setting: `gf_low: 0.70`, `gf_high: 0.85`

Both `v_crit` and `g_crit` are calibrated at 30m reference depth against Buhlmann ZH-L16C NDL (28 minutes). `SlabModel.__init__` accepts either a config file path or explicit kwargs, with explicit values taking priority. `calibrate_critical_volume.py` recalibrates v_crit values; `g_crit` calibration is manual (derived from slab[1] - ppN2_surface at NDL).

### Buhlmann Constants Module

**`buhlmann_constants.py`** (`backtest/buhlmann_constants.py`): Single source of truth for Buhlmann ZH-L16 tissue parameters and Gradient Factor calculations. Contains:
- `ZH_L16_N2_HALFTIMES`, `ZH_L16_N2_A`, `ZH_L16_N2_B`: Standard ZH-L16 N2 coefficients (16 compartments)
- `GradientFactors(gf_low, gf_high)`: Frozen dataclass with validation (0 < gf <= 1.0, gf_low <= gf_high)
- `GF_DEFAULT`: Standard GF 100/100 (no adjustment)
- `m_value(compartment_idx, ambient_pressure)`: Standard M-value: `a + P/b`
- `m_value_gf(compartment_idx, ambient_pressure, gf)`: GF-adjusted: `P + gf * (a + P/b - P)`
- `ceiling_pressure_gf(compartment_idx, tissue_pressure, gf)`: GF-adjusted ceiling per compartment
- `compute_ceilings_gf(compartment_n2, compartment_he, gf)`: Max ceiling across all compartments
- `compute_ndl_gf(compartment_n2, compartment_he, ambient_pressure, f_inert, gf_high)`: GF-adjusted NDL
- `compute_max_supersaturation_gf(n2_series, he_series, pressures, gf_low, gf_high)`: Max supersaturation with interpolated GF

All functions are pure (no side effects) for safe use in parallel execution paths. GF is interpolated linearly between `gf_low` (at first stop depth) and `gf_high` (at surface).

### Output

Backtest runs save to config-based subfolders under `backtest_output/`, `backtest_output_full/`, or `backtest_output_real/`:
- Subfolder naming: `gf{low}-{high}_cons{conservatism}_boyle{exponent}_o2-{fraction}` (e.g., `gf70-85_cons100_boyle050_o2-21`)
- Example full path: `backtest_output/gf70-85_cons100_boyle050_o2-21/divergence_risk.png`
- A copy of `config.yaml` is saved into each output folder for reproducibility
- PNG plots: divergence heatmaps, comparison summaries
- CSV: divergence matrices (`matrix_*.csv`), per-profile results (`results_detailed.csv`)
- JSON: report summary, full statistics

Output directory naming is handled by `backtest/output_naming.py` (`build_settings_dirname`, `load_effective_config`, `save_config_snapshot`).

## Important Notes

- `libbuhlmann/` is a git submodule pointing to `https://github.com/AquaBSD/libbuhlmann`
- `main.py` is an older standalone single-compartment slab implementation; the `backtest/slab_model.py` multi-compartment version supersedes it
- `con.py` is a standalone lzma+base64 file decompression utility, not part of the core pipeline
- Risk scores from both models are normalized: 1.0 = at limit, >1.0 = exceeded. Buhlmann uses M-value fractions (with optional GF scaling); Slab uses critical volume ratios (excess_gas / V_crit).
- At GF 100/100, all Buhlmann calculations reduce to standard ZH-L16C (backward compatible). Lower GF values add conservatism: shallower first stops (gf_low), reduced NDL and surface supersaturation tolerance (gf_high).
- NDL calculation uses a "shadow simulation" approach: clones tissue state, simulates forward at depth, then simulates ascent to surface before checking if excess gas exceeds V_crit. Capped at 100 minutes to match libbuhlmann.
- Deco planning uses the boundary gradient metric (slab[1] vs g_crit), NOT the integrated volume metric, because it reflects local bubble formation risk. The volume metric is consulted for risk scoring only.
- Deco stops are typically more conservative than Buhlmann for slow compartments (Joints), as the model requires full gradient clearance at each stop rather than M-value tolerance.
- For deco dives, slow compartments may show final risk > 1.0 even after completing all stops — this is expected, as the boundary gradient clears first (controls ceiling) while integrated excess lags behind (controls risk score).
