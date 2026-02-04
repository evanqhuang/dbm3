"""
Backtesting utilities for comparing Bühlmann and Slab decompression models.

Modules:
    - profile_generator: Generate various dive profiles (square, sawtooth, multi-level)
    - buhlmann_runner: Interface to libbuhlmann binary
    - slab_model: Finite difference Slab model implementation
    - comparator: Run batch comparisons and generate divergence matrices
"""

from .profile_generator import DiveProfile, ProfileGenerator
from .buhlmann_runner import BuhlmannRunner
from .slab_model import SlabModel
from .comparator import ModelComparator

__all__ = [
    "DiveProfile",
    "ProfileGenerator",
    "BuhlmannRunner",
    "SlabModel",
    "ModelComparator",
]
