"""
Bühlmann ZH-L16 constants and gradient factor calculations.

Single source of truth for M-value parameters and GF-adjusted decompression math.
All functions are pure (no side effects) for safe use in parallel execution paths.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ZH-L16 N2 compartment parameters (16 compartments)
# Half-times in minutes
ZH_L16_N2_HALFTIMES: Tuple[float, ...] = (
    4.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0,
    109.0, 146.0, 187.0, 239.0, 305.0, 390.0, 498.0, 635.0,
)

# M-value coefficients: M(P) = a + P/b
ZH_L16_N2_A: Tuple[float, ...] = (
    1.2599, 1.0000, 0.8618, 0.7562, 0.6667, 0.5933, 0.5282, 0.4701,
    0.4187, 0.3798, 0.3497, 0.3223, 0.2971, 0.2737, 0.2523, 0.2327,
)

ZH_L16_N2_B: Tuple[float, ...] = (
    0.5050, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910,
    0.9092, 0.9222, 0.9319, 0.9403, 0.9477, 0.9544, 0.9602, 0.9653,
)

NUM_COMPARTMENTS = 16


@dataclass(frozen=True)
class GradientFactors:
    """Gradient factor pair for Bühlmann decompression adjustments.

    gf_low:  applied at the deepest ceiling (first stop) — controls first stop depth
    gf_high: applied at the surface — controls final ascent and NDL
    Values are fractions (0.0–1.0), where 1.0 = use full M-value (standard Bühlmann).
    """
    gf_low: float
    gf_high: float

    def __post_init__(self):
        if not (0.0 < self.gf_low <= 1.0):
            raise ValueError(f"gf_low must be in (0, 1.0], got {self.gf_low}")
        if not (0.0 < self.gf_high <= 1.0):
            raise ValueError(f"gf_high must be in (0, 1.0], got {self.gf_high}")
        if self.gf_low > self.gf_high:
            raise ValueError(
                f"gf_low ({self.gf_low}) must be <= gf_high ({self.gf_high})"
            )

    @property
    def is_standard(self) -> bool:
        """True if GF 100/100 (no adjustment)."""
        return self.gf_low == 1.0 and self.gf_high == 1.0


GF_DEFAULT = GradientFactors(gf_low=1.0, gf_high=1.0)


def m_value(compartment_idx: int, ambient_pressure: float) -> float:
    """Standard M-value at a given ambient pressure.

    M(P) = a + P/b
    """
    a = ZH_L16_N2_A[compartment_idx]
    b = ZH_L16_N2_B[compartment_idx]
    return a + ambient_pressure / b


def m_value_gf(
    compartment_idx: int, ambient_pressure: float, gf: float
) -> float:
    """GF-adjusted M-value at a given ambient pressure.

    M_gf(P) = P + gf * (M(P) - P) = P + gf * (a + P/b - P)
    At gf=1.0, reduces to standard M(P).
    """
    m = m_value(compartment_idx, ambient_pressure)
    return ambient_pressure + gf * (m - ambient_pressure)


def ceiling_pressure_gf(
    compartment_idx: int, tissue_pressure: float, gf: float
) -> float:
    """GF-adjusted ceiling pressure (bar) for a single compartment.

    Solves for ambient pressure P where tissue_pressure = M_gf(P):
        P_ceil = (tissue_pressure - gf * a) / (1 - gf + gf / b)

    Returns 0.0 if tissue is not supersaturated (ceiling below surface vacuum).
    """
    a = ZH_L16_N2_A[compartment_idx]
    b = ZH_L16_N2_B[compartment_idx]
    denominator = 1.0 - gf + gf / b
    if denominator <= 0:
        return 0.0
    ceil_p = (tissue_pressure - gf * a) / denominator
    return max(0.0, ceil_p)


def compute_ceilings_gf(
    compartment_n2: List[float],
    compartment_he: List[float],
    gf: float,
) -> Tuple[float, List[float]]:
    """Compute GF-adjusted ceiling for all compartments at a single timestep.

    Args:
        compartment_n2: N2 tensions per compartment (bar)
        compartment_he: He tensions per compartment (bar)
        gf: gradient factor to apply (typically gf_low for ceiling)

    Returns:
        (max_ceiling_bar, per_compartment_ceilings)
    """
    per_compartment = []
    max_ceil = 0.0
    n = min(NUM_COMPARTMENTS, len(compartment_n2))
    for c in range(n):
        p_inert = compartment_n2[c]
        if compartment_he and len(compartment_he) > c:
            p_inert += compartment_he[c]
        ceil = ceiling_pressure_gf(c, p_inert, gf)
        per_compartment.append(ceil)
        if ceil > max_ceil:
            max_ceil = ceil
    return max_ceil, per_compartment


def compute_ndl_gf(
    compartment_n2: List[float],
    compartment_he: List[float],
    ambient_pressure: float,
    f_inert: float,
    gf_high: float,
) -> float:
    """Compute GF-adjusted NDL using exponential Bühlmann gas loading.

    NDL is the time at current depth until any compartment reaches
    the GF_high-adjusted surface M-value.

    Uses analytic exponential: P(t) = P_ambient_inert + (P0 - P_ambient_inert) * exp(-kt)
    Solves for t when P(t) = M_gf_high(P_surface=1.01325).

    Args:
        compartment_n2: current N2 tensions per compartment
        compartment_he: current He tensions per compartment
        ambient_pressure: current ambient pressure (bar)
        f_inert: inert gas fraction (1 - fO2)
        gf_high: gradient factor at surface

    Returns:
        NDL in minutes. Capped at 100.0 to match libbuhlmann behavior.
    """
    NDL_CAP = 100.0  # Match C binary's cap
    p_inspired_inert = ambient_pressure * f_inert
    p_surface = 1.01325  # sea level pressure
    min_ndl = NDL_CAP

    n = min(NUM_COMPARTMENTS, len(compartment_n2))
    for c in range(n):
        p_tissue = compartment_n2[c]
        if compartment_he and len(compartment_he) > c:
            p_tissue += compartment_he[c]

        # Target: GF_high-adjusted M-value at surface
        m_target = m_value_gf(c, p_surface, gf_high)

        if p_tissue >= m_target:
            # Already exceeded — NDL is 0
            return 0.0

        if p_inspired_inert <= m_target:
            # This compartment will never exceed at this depth — skip
            continue

        # Exponential gas loading: P(t) = p_inspired + (p_tissue - p_inspired) * exp(-k*t)
        # Solve for t when P(t) = m_target
        k = math.log(2) / ZH_L16_N2_HALFTIMES[c]
        ratio = (m_target - p_inspired_inert) / (p_tissue - p_inspired_inert)
        if ratio <= 0:
            return 0.0
        t = -math.log(ratio) / k  # minutes
        if t < min_ndl:
            min_ndl = t

    return min(min_ndl, NDL_CAP)


def compute_max_supersaturation_gf(
    compartment_n2_series: List[List[float]],
    compartment_he_series: List[List[float]],
    pressures: List[float],
    gf_low: float,
    gf_high: float,
) -> float:
    """Compute max supersaturation ratio across all timesteps using interpolated GF.

    The GF is interpolated between gf_low (at the first stop / deepest ceiling)
    and gf_high (at the surface). For simplicity, we use the ambient pressure
    to linearly interpolate between gf_low and gf_high.

    supersaturation_ratio = p_tissue / M_gf(P_ambient, gf_interpolated)

    Returns the maximum ratio (1.0 = at M-value limit, >1.0 = exceeded).
    """
    p_surface = 1.01325
    max_ratio = 0.0

    # Find deepest ceiling for GF interpolation reference
    # Use the timestep with maximum tissue loading to estimate first stop depth
    max_ceil_bar = p_surface
    for t_idx in range(len(pressures)):
        n = min(NUM_COMPARTMENTS, len(compartment_n2_series[t_idx]))
        for c in range(n):
            p_inert = compartment_n2_series[t_idx][c]
            if compartment_he_series and len(compartment_he_series[t_idx]) > c:
                p_inert += compartment_he_series[t_idx][c]
            ceil = ceiling_pressure_gf(c, p_inert, gf_low)
            if ceil > max_ceil_bar:
                max_ceil_bar = ceil

    for t_idx, pressure in enumerate(pressures):
        # Interpolate GF based on ambient pressure between surface and first stop
        if max_ceil_bar > p_surface:
            gf_interp = gf_high + (gf_low - gf_high) * (
                (pressure - p_surface) / (max_ceil_bar - p_surface)
            )
            gf_interp = max(gf_high, min(gf_low, gf_interp))
        else:
            gf_interp = gf_high

        n = min(NUM_COMPARTMENTS, len(compartment_n2_series[t_idx]))
        for c in range(n):
            p_inert = compartment_n2_series[t_idx][c]
            if compartment_he_series and len(compartment_he_series[t_idx]) > c:
                p_inert += compartment_he_series[t_idx][c]

            m_gf = m_value_gf(c, pressure, gf_interp)
            if m_gf > 0:
                ratio = p_inert / m_gf
                if ratio > max_ratio:
                    max_ratio = ratio

    return max_ratio


# ---------------------------------------------------------------------------
# Physiological constants (matching libbuhlmann C binary)
# ---------------------------------------------------------------------------
WATER_VAPOR_PRESSURE = 0.0627   # bar
CO2_PRESSURE = 0.0534           # bar
BUHLMANN_RQ = 1.0               # Respiratory quotient
SURFACE_N2_FRACTION = 0.78084   # Atmospheric N2 fraction
P_SURFACE = 1.01325             # Sea level pressure (bar)

# ---------------------------------------------------------------------------
# ZH-L16 He compartment parameters (16 compartments, from ZH-L16C table)
# Compartment 1b (halftime 1.88 min) is excluded to match 16-compartment layout.
# ---------------------------------------------------------------------------
ZH_L16_HE_HALFTIMES: Tuple[float, ...] = (
    1.51, 3.02, 4.72, 6.99, 10.21, 14.48, 20.53, 29.11,
    41.20, 55.19, 70.69, 90.34, 115.29, 147.42, 188.24, 240.03,
)

ZH_L16_HE_A: Tuple[float, ...] = (
    1.7424, 1.3830, 1.1919, 1.0458, 0.9220, 0.8205, 0.7305, 0.6502,
    0.5950, 0.5545, 0.5333, 0.5189, 0.5181, 0.5176, 0.5172, 0.5119,
)

ZH_L16_HE_B: Tuple[float, ...] = (
    0.4245, 0.5747, 0.6527, 0.7223, 0.7582, 0.7957, 0.8279, 0.8553,
    0.8757, 0.8903, 0.8997, 0.9073, 0.9122, 0.9171, 0.9217, 0.9267,
)


# ---------------------------------------------------------------------------
# Tissue loading equations (vectorized for numpy)
# ---------------------------------------------------------------------------

def alveolar_pressure(
    ambient_pressure: float, gas_fraction: float, rq: float = BUHLMANN_RQ
) -> float:
    """Compute alveolar partial pressure of an inert gas.

    Matches libbuhlmann ventilation():
        palv = (pamb - WVP + (1-RQ)/RQ * CO2) * gas_fraction
    With RQ=1.0 this simplifies to: (pamb - 0.0627) * gas_fraction
    """
    return (
        ambient_pressure - WATER_VAPOR_PRESSURE
        + (1.0 - rq) / rq * CO2_PRESSURE
    ) * gas_fraction


def schreiner_vec(
    pt0: np.ndarray,
    palv0: float,
    r: float,
    t: float,
    k: np.ndarray,
) -> np.ndarray:
    """Vectorized Schreiner equation across compartments.

    Computes new tissue pressure after time t with linearly changing
    ambient pressure (descent/ascent).

    Args:
        pt0:   current tissue pressures, shape (N,)
        palv0: alveolar pressure at START of interval (scalar, broadcasts)
        r:     rate of change of alveolar pressure (bar/min, scalar)
        t:     time interval (minutes)
        k:     pre-computed ln(2)/halftime per compartment, shape (N,)

    Returns:
        New tissue pressures, shape (N,)
    """
    return palv0 + r * (t - 1.0 / k) - (palv0 - pt0 - r / k) * np.exp(-k * t)


def haldane_vec(
    pt0: np.ndarray,
    palv0: float,
    t: float,
    k: np.ndarray,
) -> np.ndarray:
    """Vectorized Haldane equation across compartments.

    Computes new tissue pressure after time t at constant ambient pressure.

    Args:
        pt0:   current tissue pressures, shape (N,)
        palv0: alveolar pressure (constant, scalar)
        t:     time interval (minutes)
        k:     pre-computed ln(2)/halftime per compartment, shape (N,)

    Returns:
        New tissue pressures, shape (N,)
    """
    return pt0 + (palv0 - pt0) * (1.0 - np.exp(-k * t))
