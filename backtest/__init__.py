"""
Backtesting utilities for comparing Bühlmann and Slab decompression models.

Modules:
    - profile_generator: Generate various dive profiles (square, sawtooth, multi-level)
    - buhlmann_runner: Interface to libbuhlmann binary
    - buhlmann_constants: ZH-L16 constants and gradient factor calculations
    - slab_model: Finite difference Slab model implementation
    - comparator: Run batch comparisons and generate divergence matrices
"""

from .profile_generator import DiveProfile, ProfileGenerator
from .buhlmann_runner import BuhlmannRunner
from .buhlmann_engine import BuhlmannEngine
from .buhlmann_constants import GradientFactors, GF_DEFAULT
from .slab_model import SlabModel
from .comparator import ModelComparator
from .output_naming import build_settings_dirname, load_effective_config

__all__ = [
    "DiveProfile",
    "ProfileGenerator",
    "BuhlmannRunner",
    "BuhlmannEngine",
    "GradientFactors",
    "GF_DEFAULT",
    "SlabModel",
    "ModelComparator",
    "build_settings_dirname",
    "load_effective_config",
]
