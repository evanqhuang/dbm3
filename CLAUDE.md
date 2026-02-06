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

# Run individual modules
python -m backtest.slab_model
python -m backtest.profile_generator
python -m backtest.comparator
```

### Tests

The libbuhlmann submodule has tests at `libbuhlmann/test/test_all_of_the_units.py`. There is no test suite for the Python `backtest/` module yet.

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
- **`BuhlmannResult`** (`backtest/buhlmann_runner.py`): 16-compartment tissue tensions, ceilings, NDL times, max supersaturation. Output parsed from the `src/dive` binary's stdout (36 values per line).
- **`SlabResult`** (`backtest/slab_model.py`): Multi-compartment slab history, critical compartment name, final NDL, max critical volume ratio. Risk scores use the critical volume approach: `risk = excess_gas / V_crit`.
- **`TissueCompartment`** (`backtest/slab_model.py`): Per-compartment state with diffusion coefficient D, slice count, and `v_crit` (critical volume threshold calibrated against Buhlmann NDLs at 30m reference depth).
- **`ComparisonResult`** (`backtest/comparator.py`): Wraps both results with computed `delta_risk` and `delta_ndl` properties.

### Model Details

**Buhlmann** (`BuhlmannRunner`): Wraps the C binary via subprocess. Profile data is piped to stdin as `time pressure O2 He` lines. The binary outputs 36 values per line: time, pressure, 16x(N2, He) compartment tensions, ceiling, nodectime.

**Slab Model** (`SlabModel`): Multi-compartment finite difference solver (Fick's Second Law). Default 3 compartments configured in `config.yaml`:
- Spine (fast, D=0.002) - sensitive, fastest gas uptake
- Muscle (medium, D=0.0005)
- Joints (slow, D=0.0001) - traps gas, slowest uptake

Each compartment is a 1D slab with: perfect perfusion at slice[0] (blood-tissue interface instantly matches ambient ppN2), diffusion through interior slices, no-flux boundary at the core. Stability requires `k = (D * dt) / dx^2 <= 0.5`.

**Critical Volume Risk** (Hennessy & Hempleman): Risk is measured by total excess dissolved gas across the slab, compared to a per-compartment threshold:
- `excess_gas = sum(max(0, slab[i] - P_surface_equil) * dx)` — total gas above surface equilibrium
- `risk = excess_gas / V_crit` — 1.0 = at limit, >1.0 = exceeded
- `V_crit` per compartment is calibrated at 30m reference depth against Buhlmann ZH-L16C NDLs
- Evaluated at end of profile (post-ascent), not per-timestep, to avoid ascent transient artifacts
- NDL calculation simulates ascent to surface before checking excess gas (ascent-aware)

### Parallel Processing

`ModelComparator.compare_batch()` uses `ThreadPoolExecutor` for Buhlmann (I/O-bound subprocess calls) and `ProcessPoolExecutor` for Slab (CPU-bound numpy). Helper functions `_run_buhlmann_single` and `_run_slab_single` are module-level for pickling compatibility.

### Configuration

`config.yaml` holds slab model parameters (dt, dx, permeability, fO2, compartment definitions with D, slices, and v_crit). `permeability: null` enables perfect perfusion (classic Hempleman slab). `SlabModel.__init__` accepts either a config file path or explicit kwargs, with explicit values taking priority. `calibrate_critical_volume.py` recalibrates v_crit values against Buhlmann NDLs.

### Output

Backtest runs save to `backtest_output/` or `backtest_output_full/`:
- PNG plots: divergence heatmaps, comparison summaries
- CSV: divergence matrices (`matrix_*.csv`), per-profile results (`results_detailed.csv`)
- JSON: report summary, full statistics

## Important Notes

- `libbuhlmann/` is a git submodule pointing to `https://github.com/AquaBSD/libbuhlmann`
- `main.py` is an older standalone single-compartment slab implementation; the `backtest/slab_model.py` multi-compartment version supersedes it
- `con.py` is a standalone lzma+base64 file decompression utility, not part of the core pipeline
- Risk scores from both models are normalized: 1.0 = at limit, >1.0 = exceeded. Buhlmann uses M-value fractions; Slab uses critical volume ratios (excess_gas / V_crit).
- NDL calculation uses a "shadow simulation" approach: clones tissue state, simulates forward at depth, then simulates ascent to surface before checking if excess gas exceeds V_crit. Capped at 100 minutes to match libbuhlmann.
