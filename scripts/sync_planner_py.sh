#!/bin/bash
# Sync Python model files to docs/py/backtest/ for Pyodide serving.
# Run locally for development, or automatically by CI before deployment.
#
# These files are NOT checked into git (docs/py/backtest/ is in .gitignore).
# The source of truth is always backtest/*.py in the repo root.

set -euo pipefail
SCRIPT_DIR="$(cd "$(/usr/bin/dirname "$0")" && pwd)"
PROJECT_ROOT="$(/usr/bin/dirname "$SCRIPT_DIR")"

DEST="$PROJECT_ROOT/docs/py/backtest"
/bin/mkdir -p "$DEST"

# Copy the 4 core model files
for f in profile_generator.py buhlmann_constants.py buhlmann_engine.py slab_model.py; do
    /bin/cp "$PROJECT_ROOT/backtest/$f" "$DEST/$f"
    echo "  Synced backtest/$f -> docs/py/backtest/$f"
done

# Generate simplified __init__.py (avoids importing BuhlmannRunner/subprocess,
# ModelComparator/concurrent.futures, output_naming)
/bin/cat > "$DEST/__init__.py" << 'PYEOF'
"""Minimal backtest package for Pyodide (browser) use."""
from .profile_generator import DiveProfile, ProfileGenerator
from .buhlmann_engine import BuhlmannEngine
from .buhlmann_constants import GradientFactors, GF_DEFAULT
from .slab_model import SlabModel
PYEOF

echo "Planner Python files synced successfully."
