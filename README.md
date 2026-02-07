# DBM-3 — Diffusion Barrier Matrix

A computational framework for modelling decompression risk using **slab diffusion physics** (Hempleman's linear bulk diffusion theory) as an alternative to the classical Buhlmann ZH-L16C compartment model. DBM-3 generates dive profiles, runs them through both models, and produces divergence matrices comparing their risk predictions across the depth-time envelope.

## Why Slab Diffusion?

The Buhlmann ZH-L16C model treats each tissue compartment as a perfectly-stirred tank with exponential gas kinetics. This is convenient but biologically unrealistic -- real tissues have spatial structure, and gas must physically diffuse through them.

The slab diffusion model solves **Fick's Second Law** on a 1D tissue slab using finite differences. Gas enters through a permeability barrier at the blood-tissue interface and diffuses inward slice by slice. This captures phenomena that exponential models miss:

- **Gradient formation** -- outer tissue layers saturate before inner layers, creating a gas concentration gradient across the slab
- **Diffusion lag** -- deep tissue layers respond more slowly to pressure changes, with delay proportional to distance squared
- **Asymmetric on/off-gassing** -- a fully saturated slab off-gasses more slowly than a partially saturated one, since the interior has no nearby boundary to diffuse toward

## The Three-Compartment Model

DBM-3 models the body as three tissue compartments with distinct diffusion properties, each representing a class of tissue with different gas transport characteristics:

### Spine (Fast)
- **D = 0.002** -- Highest diffusion coefficient
- Well-vascularized neural tissue with rapid gas exchange
- First to approach saturation, first to off-gas
- Primary risk driver for short/deep dives

### Muscle (Medium)
- **D = 0.0005** -- Moderate diffusion coefficient
- Represents the large mass of moderately perfused tissue
- Dominates risk for medium-duration dives
- Significant gas storage capacity

### Joints (Slow)
- **D = 0.0001** -- Lowest diffusion coefficient
- Poorly perfused connective tissue (cartilage, tendons, ligaments)
- Traps gas due to slow transport
- Primary risk driver for long/repetitive dives
- Slowest to off-gas after surfacing

Each compartment is discretized into **20 slices** forming a 1D slab:

```
Blood ─── [Permeability Barrier] ─── Slice 0 ─── Slice 1 ─── ... ─── Slice 19 (Core)
  │              │                       │            │                      │
  │         k=permeability          Fick's 2nd Law diffusion           No-flux boundary
  │                                                                    (sealed core)
  Ambient pN2
```

## Depth-Dependent M-Values

Rather than using fixed tolerated supersaturation limits, DBM-3 derives M-values from the diffusion physics of each slice. Every slice has an **effective half-time** based on its position in the slab:

```
t_half(i) = ln(2)/permeability + (i * dx)^2 / (C * D)
```

- Slice 0 is **barrier-limited** (fast, controlled by permeability)
- Deeper slices are **diffusion-limited** (slow, controlled by distance squared)

These half-times map to Buhlmann-compatible M-value parameters using empirical fits from ZH-L16C:

```
a = 2.0 * t_half^(-1/3)        (intercept: higher for fast tissues)
b = 1.005 - t_half^(-1/2)      (slope: approaches 1.0 for slow tissues)

M(depth) = a + P_ambient / b   (tolerated tissue pressure at depth)
```

This means each slice in each compartment has its own M-value line, and the critical constraint comes from whichever slice is closest to its limit -- providing spatial resolution that classical compartment models lack.

## Risk Scoring

Both models produce a **risk score** normalized as an M-value fraction:

| Score | Meaning |
|-------|---------|
| 0.0 | No gas loading |
| 0.5 | 50% of tolerated supersaturation |
| 1.0 | At the decompression limit |
| >1.0 | Exceeded -- mandatory deco required |

The **No-Decompression Limit (NDL)** is calculated using a shadow simulation: the model clones the current tissue state, simulates continued exposure at depth, and checks when tissue tension would exceed **surface** M-values (ascent-aware). Capped at 100 minutes.

## Getting Started

### Prerequisites

- Python 3.14+
- pipenv
- C compiler (for libbuhlmann)

### Installation

```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>
cd dbm3

# Install Python dependencies
pipenv install

# Build the Buhlmann C library
cd libbuhlmann && ./bootstrap.sh && ./configure && make && cd ..
```

### Running

```bash
# Activate the virtual environment
pipenv shell

# Interactive single-dive simulation with visualization
python main.py

# Backtesting (compare Slab vs Buhlmann across depth-time matrix)
python run_backtest.py                    # Default: 10-50m, 5-40min (72 profiles)
python run_backtest.py --quick            # Quick: 4 depths x 3 times
python run_backtest.py --full             # Full: 5-70m x 1-120min (~7800 profiles)
python run_backtest.py --profile 30 20    # Single profile: 30m depth, 20min
```

Output is saved to `backtest_output/` (or `backtest_output_full/` for `--full` runs) and includes PNG heatmaps, CSV divergence matrices, and JSON reports.

## Configuration

All slab model parameters are in `config.yaml`:

```yaml
dt: 0.5                    # Time step (seconds)
dx: 1.0                    # Slice spacing (arbitrary units)
permeability: 0.01         # Blood-tissue barrier permeability
f_o2: 0.21                 # Breathing gas O2 fraction
sat_limit_bottom: 1.0      # Bottom M-value scale factor (1.0 = no conservatism)
sat_limit_surface: 1.0     # Surface M-value scale factor

compartments:
  - name: "Spine"
    D: 0.002               # Fast diffusion
    slices: 20
  - name: "Muscle"
    D: 0.0005              # Medium diffusion
    slices: 20
  - name: "Joints"
    D: 0.0001              # Slow diffusion
    slices: 20
```

Lowering `sat_limit_bottom` / `sat_limit_surface` below 1.0 adds conservatism by scaling down the tolerated M-values.

## Architecture

```
ProfileGenerator ──> DiveProfile ──> BuhlmannRunner (subprocess: libbuhlmann/src/dive)
                                 ──> SlabModel (numpy finite difference solver)
                                           │
                                     ModelComparator ──> heatmaps, CSV, JSON reports
```

| Module | Description |
|--------|-------------|
| `backtest/profile_generator.py` | Generates square, multilevel, sawtooth, and random-walk dive profiles |
| `backtest/slab_model.py` | Multi-compartment finite difference solver (Fick's 2nd Law) |
| `backtest/buhlmann_runner.py` | Subprocess wrapper for the libbuhlmann ZH-L16C binary |
| `backtest/comparator.py` | Batch comparison with parallel execution, divergence matrices, and plotting |
| `config.yaml` | Slab model parameters and compartment definitions |
| `libbuhlmann/` | Git submodule -- C implementation of Buhlmann ZH-L16C |

### Parallel Processing

Batch comparisons use `ThreadPoolExecutor` for Buhlmann (I/O-bound subprocess calls) and `ProcessPoolExecutor` for Slab (CPU-bound numpy), running both models concurrently.

## Numerical Stability

The finite difference scheme requires the stability condition:

```
k = (D * dt) / dx^2 <= 0.5
```

With the default parameters (`D_max=0.002`, `dt=0.5`, `dx=1.0`), the fastest compartment yields `k=0.001`, well within the stable regime. The model validates this constraint at initialization.

## References

- Hempleman, H.V. (1952). A new theoretical basis for the calculation of decompression tables. *Report UPS 131*, Royal Naval Physiological Laboratory
- Buhlmann, A.A. (1984). *Decompression-Decompression Sickness*. Springer-Verlag
- Fick, A. (1855). Ueber Diffusion. *Annalen der Physik*, 170(1), 59-86

## License

MIT
