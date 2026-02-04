# SlabDive - Computational Framework for Decompression Model Backtesting

## Project Overview

SlabDive is a computational framework for comparing decompression models, specifically the Bühlmann ZH-L16C algorithm (a perfusion-limited neo-Haldanian model) against a Slab Diffusion Model (based on Hempleman's linear bulk diffusion theory). The project implements a finite difference method to simulate nitrogen diffusion in tissue slabs and compares the results against the established Bühlmann model.

The project leverages the open-source `libbuhlmann` library as a reference implementation for Bühlmann calculations and implements a custom Slab model using finite difference methods to solve Fick's Second Law of Diffusion in a one-dimensional medium.

### Key Components

1. **Slab Model Implementation** (`backtest/slab_model.py`): Implements a 1D slab diffusion model using finite difference methods to simulate nitrogen uptake and offgassing in tissue.

2. **Bühlmann Interface** (`backtest/buhlmann_runner.py`): Interfaces with the `libbuhlmann` C library to run Bühlmann ZH-L16C calculations.

3. **Profile Generator** (`backtest/profile_generator.py`): Generates various dive profiles including square, sawtooth, multi-level, and random walk profiles for systematic testing.

4. **Model Comparator** (`backtest/comparator.py`): Compares both models across dive profiles and generates divergence matrices showing where models agree/disagree on risk assessments.

5. **Main Simulation** (`main.py`): Standalone implementation of the slab model with visualization capabilities.

## Building and Running

### Prerequisites

- Python 3.14+
- NumPy
- Matplotlib
- libbuhlmann C library (compiled)

### Dependencies

Install Python dependencies using pipenv:
```bash
pipenv install
```

Or directly with pip:
```bash
pip install numpy matplotlib
```

### Compiling libbuhlmann

The project requires the `libbuhlmann` library to be compiled:

```bash
cd libbuhlmann
./bootstrap.sh
./configure
make
```

This creates the `src/dive` binary used by the Bühlmann runner.

### Running the Project

1. **Simple Slab Model Simulation**:
   ```bash
   python main.py
   ```
   This runs the standalone slab model with configurable parameters and generates visualizations.

2. **Backtesting**:
   ```bash
   python run_backtest.py                    # Run default backtest
   python run_backtest.py --quick            # Quick test with fewer profiles
   python run_backtest.py --full             # Full test with 50k+ profiles
   python run_backtest.py --profile 30 20    # Test single profile (30m, 20min)
   ```

3. **Direct Module Usage**:
   ```bash
   python -m backtest.profile_generator      # Generate profiles
   python -m backtest.slab_model             # Run slab model
   python -m backtest.buhlmann_runner        # Run Bühlmann model
   python -m backtest.comparator             # Run comparisons
   ```

## Development Conventions

### Code Structure

- `backtest/` - Main module containing all backtesting functionality
  - `profile_generator.py` - Dive profile generation
  - `buhlmann_runner.py` - Interface to libbuhlmann
  - `slab_model.py` - Slab model implementation
  - `comparator.py` - Model comparison logic
  - `__init__.py` - Package initialization

- `main.py` - Standalone slab model implementation with visualization
- `run_backtest.py` - Command-line interface for running backtests
- `example.py` - Example usage of the tissue slab class
- `con.py` - Compression/decompression utilities
- `libbuhlmann/` - External dependency (Bühlmann library)

### Mathematical Approach

The project implements two different approaches to decompression modeling:

1. **Bühlmann ZH-L16C (Perfusion Limited)**: Uses 16 parallel tissue compartments with exponential uptake/elimination kinetics based on perfusion rates.

2. **Slab Diffusion Model (Diffusion Limited)**: Models tissue as a 1D slab where gas transport follows Fick's Second Law of Diffusion, resulting in square-root-of-time kinetics initially.

### Testing and Validation

The project includes comprehensive backtesting capabilities that:
- Generate systematic dive profiles across depth and time ranges
- Compare risk predictions between models
- Generate divergence matrices showing where models disagree
- Create visualizations of model behavior
- Support parallel processing for large-scale testing

### Configuration Parameters

The slab model can be configured with parameters such as:
- Number of tissue slices (resolution)
- Diffusion coefficient
- Time step size
- Permeability barrier at blood-tissue interface
- M-value limits for different tissue slices

## Project Architecture

The project follows a modular design with clear separation of concerns:
- Profile generation is separated from model execution
- Model implementations are independent
- Comparison logic is abstracted from individual models
- Visualization is handled separately from computation

The codebase emphasizes numerical stability (checking finite difference stability conditions) and provides detailed visualization of the diffusion process through heatmaps and profile plots.