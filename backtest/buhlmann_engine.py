"""
Pure Python/numpy Bühlmann ZH-L16 tissue simulation engine.

Replaces the libbuhlmann C binary subprocess with equivalent vectorized numpy
computation. Processes DiveProfile objects and returns tissue state histories
compatible with the existing BuhlmannResult format.
"""

import math
import numpy as np

from .buhlmann_constants import (
    NUM_COMPARTMENTS,
    SURFACE_N2_FRACTION,
    ZH_L16_N2_HALFTIMES,
    ZH_L16_N2_A,
    ZH_L16_N2_B,
    ZH_L16_HE_HALFTIMES,
    ZH_L16_HE_A,
    ZH_L16_HE_B,
    alveolar_pressure,
    schreiner_vec,
)
from .profile_generator import DiveProfile


class BuhlmannEngine:
    """Pure Python Bühlmann tissue simulation using ZH-L16 constants.

    All tissue math is vectorized across 16 compartments using numpy.
    Produces output compatible with BuhlmannResult construction.
    """

    def __init__(self):
        # Pre-convert to numpy arrays (shape (16,))
        self.n2_a = np.array(ZH_L16_N2_A)
        self.n2_b = np.array(ZH_L16_N2_B)
        self.he_a = np.array(ZH_L16_HE_A)
        self.he_b = np.array(ZH_L16_HE_B)

        # Pre-compute decay constants k = ln(2) / halftime
        self.n2_k = np.log(2) / np.array(ZH_L16_N2_HALFTIMES)
        self.he_k = np.log(2) / np.array(ZH_L16_HE_HALFTIMES)

        # Surface M-values for NDL calculation (standard, no GF)
        self._surface_pressure = 1.01325
        self._surface_m = self.n2_a + self._surface_pressure / self.n2_b

    def simulate(self, profile: DiveProfile) -> dict:
        """Run a full dive profile simulation.

        Args:
            profile: DiveProfile with (time_min, depth_m, fO2, fHe) points

        Returns:
            dict with keys: times, pressures, compartment_n2, compartment_he,
            ceilings, ndl_times — matching BuhlmannRunner._parse_output format.
        """
        if not profile.points:
            raise ValueError("Empty profile")

        n = NUM_COMPARTMENTS

        # Initialize tissue state at surface equilibrium
        n2_p = np.full(n, alveolar_pressure(1.0, SURFACE_N2_FRACTION))
        he_p = np.zeros(n)

        # Output accumulators
        times = []
        pressures = []
        compartment_n2 = []
        compartment_he = []
        ceilings = []
        ndl_times = []

        last_t = 0.0
        last_p = 1.0

        for time_min, depth_m, f_o2, f_he in profile.points:
            p = depth_m / 10.0 + 1.0
            dt = time_min - last_t
            dp = p - last_p

            if dt < 0:
                continue

            if dt > 0:
                rate = dp / dt
                n2_ratio = 1.0 - f_o2 - f_he
                he_ratio = f_he

                # Alveolar pressures at START of interval (matches C binary)
                palv_n2 = alveolar_pressure(last_p, n2_ratio)
                palv_he = alveolar_pressure(last_p, he_ratio)

                # Schreiner equation for both gases
                n2_p = schreiner_vec(n2_p, palv_n2, rate * n2_ratio, dt, self.n2_k)
                he_p = schreiner_vec(he_p, palv_he, rate * he_ratio, dt, self.he_k)

            # Record state
            times.append(time_min)
            pressures.append(p)
            compartment_n2.append(n2_p.tolist())
            compartment_he.append(he_p.tolist())

            # Raw ceiling (matches C getCeiling: separate N2/He, take max)
            ceil = self._compute_ceiling(n2_p, he_p)
            ceilings.append(ceil)

            # NDL (analytic solve, faster than C's iterative search)
            n2_ratio = 1.0 - f_o2 - f_he
            ndl = self._compute_ndl(n2_p, he_p, last_p, n2_ratio)
            ndl_times.append(ndl)

            last_t = time_min
            last_p = p

        return {
            "times": times,
            "pressures": pressures,
            "compartment_n2": compartment_n2,
            "compartment_he": compartment_he,
            "ceilings": ceilings,
            "ndl_times": ndl_times,
        }

    def _compute_ceiling(self, n2_p: np.ndarray, he_p: np.ndarray) -> float:
        """Raw ceiling matching C getCeiling().

        C computes N2 and He ceilings separately per compartment using
        their respective a/b coefficients, then takes the max.
        """
        p_stop_n2 = (n2_p - self.n2_a) * self.n2_b
        p_stop_he = (he_p - self.he_a) * self.he_b
        return float(np.max(np.maximum(p_stop_n2, p_stop_he)))

    def _compute_ndl(
        self,
        n2_p: np.ndarray,
        he_p: np.ndarray,
        p_ambient: float,
        n2_ratio: float,
    ) -> float:
        """Analytic NDL computation.

        For each compartment, solves for time until total tissue pressure
        (N2 + He) reaches the surface M-value using N2 halftimes.
        Returns min across compartments, capped at 100.0 minutes.
        """
        NDL_CAP = 100.0
        p_inspired = alveolar_pressure(p_ambient, n2_ratio)
        p_tissue = n2_p + he_p
        min_ndl = NDL_CAP

        for c in range(NUM_COMPARTMENTS):
            m_target = self._surface_m[c]

            if p_tissue[c] >= m_target:
                return 0.0

            if p_inspired <= m_target:
                continue

            ratio = (m_target - p_inspired) / (p_tissue[c] - p_inspired)
            if ratio <= 0:
                return 0.0

            t = -math.log(ratio) / self.n2_k[c]
            if t < min_ndl:
                min_ndl = t

        return min(min_ndl, NDL_CAP)
