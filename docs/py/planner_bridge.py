"""
Dive Planner Bridge for Pyodide

Provides a JSON-in/JSON-out API for the dive planner web interface.
Runs in WebAssembly via Pyodide - no YAML, no file I/O.

Input: JSON string with levels, gas mix, GF, conservatism
Output: JSON string with Buhlmann and Slab model results
"""

import json
from typing import Dict, List, Any, Tuple

from backtest.profile_generator import ProfileGenerator
from backtest.buhlmann_engine import BuhlmannEngine
from backtest.buhlmann_constants import (
    GradientFactors,
    P_SURFACE,
    NUM_COMPARTMENTS,
    ZH_L16_N2_HALFTIMES,
    ZH_L16_HE_HALFTIMES,
    compute_ceilings_gf,
    compute_ndl_gf,
    compute_max_supersaturation_gf,
    m_value_gf,
    alveolar_pressure,
    haldane_vec,
)
from backtest.slab_model import SlabModel

# Default compartment configuration (hardcoded, no YAML in Pyodide)
DEFAULT_COMPARTMENTS = [
    {"name": "Spine", "D": 0.002, "slices": 20, "v_crit": 3.6001, "g_crit": 1.3406},
    {"name": "Muscle", "D": 0.0005, "slices": 20, "v_crit": 1.4735, "g_crit": 0.9827},
    {"name": "Joints", "D": 0.0001, "slices": 20, "v_crit": 0.3902, "g_crit": 0.3440},
]

STOP_INCREMENT = 3.0  # meters
ASCENT_RATE = 10.0  # m/min
MAX_STOP_TIME = 120  # minutes per stop
MAX_DECIMATION_POINTS = 200  # max points for chart data


def _to_python(value: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [_to_python(v) for v in value]
    elif isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    return value


def _decimate_series(times: List[float], values: List[float], max_points: int) -> Tuple[List[float], List[float]]:
    """Decimate time series to max_points for efficient charting."""
    if len(times) <= max_points:
        return times, values

    step = len(times) // max_points
    return times[::step], values[::step]


def _plan_buhlmann_deco(
    final_n2: List[float],
    final_he: List[float],
    bottom_depth: float,
    gf_low: float,
    gf_high: float,
    f_o2: float,
    f_he: float,
) -> List[Dict[str, float]]:
    """
    Plan Buhlmann decompression schedule using iterative ceiling checks.

    Args:
        final_n2: Final N2 tissue tensions (16 compartments)
        final_he: Final He tissue tensions (16 compartments)
        bottom_depth: Depth at end of bottom time (meters)
        gf_low: Gradient factor at first stop
        gf_high: Gradient factor at surface
        f_o2: Oxygen fraction
        f_he: Helium fraction

    Returns:
        List of deco stops [{depth, duration}], deepest first
    """
    import numpy as np

    current_n2 = np.array(final_n2, dtype=float)
    current_he = np.array(final_he, dtype=float)
    f_n2 = 1.0 - f_o2 - f_he

    # Pre-compute decay constants k = ln(2) / halftime
    n2_k = np.log(2) / np.array(ZH_L16_N2_HALFTIMES)
    he_k = np.log(2) / np.array(ZH_L16_HE_HALFTIMES)

    stops = []
    current_depth = bottom_depth

    # Find initial ceiling with gf_low
    ceiling_bar, _ = compute_ceilings_gf(current_n2.tolist(), current_he.tolist(), gf_low)
    ceiling_depth = (ceiling_bar - 1.0) * 10.0

    if ceiling_depth <= 0.0:
        return []  # No deco required

    # Round ceiling up to next stop increment
    first_stop = (int(ceiling_depth / STOP_INCREMENT) + 1) * STOP_INCREMENT
    deepest_stop = first_stop

    # Ascend to first stop
    current_depth = first_stop

    while current_depth > 0.0:
        stop_duration = 0
        ambient_pressure = current_depth / 10.0 + 1.0

        # Compute interpolated GF for current depth
        if deepest_stop > 0:
            depth_fraction = current_depth / deepest_stop
            gf_current = gf_high + (gf_low - gf_high) * depth_fraction
        else:
            gf_current = gf_high

        # Alveolar partial pressures at this depth
        ppn2 = alveolar_pressure(ambient_pressure, f_n2)
        pphe = alveolar_pressure(ambient_pressure, f_he)

        # Hold at stop until ceiling clears
        for _ in range(MAX_STOP_TIME):
            stop_duration += 1

            # Offgas for 1 minute using Haldane equation
            current_n2 = haldane_vec(current_n2, ppn2, 1.0, n2_k)
            current_he = haldane_vec(current_he, pphe, 1.0, he_k)

            # Check ceiling
            ceiling_bar, _ = compute_ceilings_gf(current_n2.tolist(), current_he.tolist(), gf_current)
            ceiling_depth = (ceiling_bar - 1.0) * 10.0

            # Can we ascend to next stop?
            next_stop = max(0.0, current_depth - STOP_INCREMENT)
            if ceiling_depth <= next_stop:
                break

        if stop_duration > 0:
            stops.append({"depth": float(current_depth), "duration": float(stop_duration)})

        # Ascend to next stop
        if current_depth <= STOP_INCREMENT:
            break
        current_depth = max(0.0, current_depth - STOP_INCREMENT)

    return stops


def _plan_dive_impl(params: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation of dive planning logic."""
    # Parse input
    levels = params["levels"]  # [{"depth": 30, "duration": 20}, ...]
    f_o2 = params["fO2"]
    f_he = params["fHe"]
    gf_low = params["gfLow"]
    gf_high = params["gfHigh"]
    conservatism = params["conservatism"]

    # Generate profile
    profile_generator = ProfileGenerator()
    if len(levels) == 1:
        profile = profile_generator.generate_square(
            depth=levels[0]["depth"],
            bottom_time=levels[0]["duration"],
            fO2=f_o2,
            fHe=f_he,
        )
    else:
        level_tuples = [(level["depth"], level["duration"]) for level in levels]
        profile = profile_generator.generate_multilevel(
            levels=level_tuples,
            fO2=f_o2,
            fHe=f_he,
        )

    # Extract profile data for chart (decimated)
    profile_times = [p[0] for p in profile.points]
    profile_depths = [p[1] for p in profile.points]
    dec_times, dec_depths = _decimate_series(profile_times, profile_depths, MAX_DECIMATION_POINTS)

    # Find max depth for calculations
    max_depth = max(profile_depths)

    # Run Buhlmann simulation
    buhlmann_engine = BuhlmannEngine()
    buhlmann_result = buhlmann_engine.simulate(profile)

    # Validate GF pair
    GradientFactors(gf_low=gf_low, gf_high=gf_high)

    # Buhlmann analysis
    times = buhlmann_result["times"]
    pressures = buhlmann_result["pressures"]
    compartment_n2_series = buhlmann_result["compartment_n2"]
    compartment_he_series = buhlmann_result["compartment_he"]

    # Final tissue state
    final_n2 = compartment_n2_series[-1]
    final_he = compartment_he_series[-1]

    # Risk (max supersaturation with GF)
    buhlmann_risk = compute_max_supersaturation_gf(
        compartment_n2_series,
        compartment_he_series,
        pressures,
        gf_low,
        gf_high,
    )

    # NDL at bottom time (use last point before ascent)
    bottom_idx = len(times) - 1
    for i in range(len(times)):
        depth_idx = min(i, len(profile_depths) - 1)
        if profile_depths[depth_idx] < max_depth * 0.99:
            bottom_idx = max(0, i - 1)
            break

    bottom_pressure = pressures[bottom_idx]
    f_inert = 1.0 - f_o2
    buhlmann_ndl = compute_ndl_gf(
        compartment_n2_series[bottom_idx],
        compartment_he_series[bottom_idx],
        bottom_pressure,
        f_inert,
        gf_high,
    )

    # Ceiling over time (sample every ~1 minute or every 10th point)
    sample_step = max(1, len(times) // min(len(times), 60))
    buhlmann_ceilings = []
    for i in range(0, len(times), sample_step):
        ceiling_bar, _ = compute_ceilings_gf(
            compartment_n2_series[i],
            compartment_he_series[i],
            gf_low,
        )
        ceiling_depth = (ceiling_bar - 1.0) * 10.0
        buhlmann_ceilings.append(float(max(0.0, ceiling_depth)))

    # Max ceiling
    max_ceiling = max(buhlmann_ceilings)

    # M-values at surface for each compartment (with GF)
    m_values = [m_value_gf(i, P_SURFACE, gf_high) for i in range(NUM_COMPARTMENTS)]

    # Deco planning
    buhlmann_deco_stops = []
    if max_ceiling > 0.0:
        buhlmann_deco_stops = _plan_buhlmann_deco(
            final_n2,
            final_he,
            max_depth,
            gf_low,
            gf_high,
            f_o2,
            f_he,
        )

    buhlmann_data = {
        "risk": float(buhlmann_risk),
        "ndl": float(buhlmann_ndl),
        "maxCeiling": float(max_ceiling),
        "requiresDeco": max_ceiling > 0.0,
        "ceilingsOverTime": _to_python(buhlmann_ceilings),
        "tissueN2": _to_python(final_n2),
        "mValues": _to_python(m_values),
        "decoStops": _to_python(buhlmann_deco_stops),
    }

    # Run Slab simulation
    slab_model = SlabModel(
        compartments_config=DEFAULT_COMPARTMENTS,
        conservatism=conservatism,
    )
    slab_result = slab_model.run(profile)

    # Slab analysis
    slab_risk = slab_result.max_cv_ratio
    slab_ndl = slab_result.final_ndl
    ceiling_at_bottom = slab_result.ceiling_at_bottom

    # Ceiling over time (sample from slab_history)
    slab_times = slab_result.times
    slab_depths = slab_result.depths
    num_compartments = len(slab_model.compartments)
    sample_step = max(1, len(slab_times) // min(len(slab_times), 60))
    slab_ceilings = []

    for i in range(0, len(slab_times), sample_step):
        if i < len(slab_result.slab_history):
            slabs_at_t = [slab_result.slab_history[i][c] for c in range(num_compartments)]
            ceiling = slab_model.calculate_ceiling(slabs_at_t, slab_depths[i])
        else:
            # Use final slabs for any remaining points
            slabs_at_t = [slab_result.final_slabs[comp.name] for comp in slab_model.compartments]
            ceiling = slab_model.calculate_ceiling(slabs_at_t, slab_depths[i])
        slab_ceilings.append(float(ceiling))

    # Deco planning — plan_deco() must be called explicitly (run() doesn't populate deco_schedule)
    slab_deco_stops = []
    if ceiling_at_bottom > 0.0:
        final_slabs_list = [slab_result.final_slabs[comp.name] for comp in slab_model.compartments]
        deco_schedule = slab_model.plan_deco(final_slabs_list, max_depth)
        if deco_schedule and deco_schedule.stops:
            slab_deco_stops = [
                {"depth": float(stop.depth), "duration": float(stop.duration_min)}
                for stop in deco_schedule.stops
            ]

    # Compartment details
    compartment_data = []
    ppn2_surface = alveolar_pressure(P_SURFACE, 1.0 - f_o2 - f_he)

    for comp_config in DEFAULT_COMPARTMENTS:
        name = comp_config["name"]
        slab = slab_result.final_slabs[name]

        # Gradient at boundary (slab[1] - surface ppN2)
        gradient = float(slab[1] - ppn2_surface)

        # Excess gas (integrated above surface equilibrium)
        # Must use slab_model.dx (default 1.0), NOT 1/slices — v_crit was calibrated with dx=1.0
        dx = slab_model.dx
        excess_gas = float(sum(max(0.0, s - ppn2_surface) * dx for s in slab))

        compartment_data.append({
            "name": name,
            "gradient": gradient,
            "gCrit": float(comp_config["g_crit"]),
            "excessGas": excess_gas,
            "vCrit": float(comp_config["v_crit"]),
        })

    slab_data = {
        "risk": float(slab_risk),
        "ndl": float(slab_ndl),
        "ceiling": float(ceiling_at_bottom),
        "requiresDeco": ceiling_at_bottom > 0.0,
        "criticalCompartment": slab_result.critical_compartment,
        "ceilingsOverTime": _to_python(slab_ceilings),
        "compartments": compartment_data,
        "decoStops": _to_python(slab_deco_stops),
    }

    return {
        "profile": {
            "times": _to_python(dec_times),
            "depths": _to_python(dec_depths),
        },
        "buhlmann": buhlmann_data,
        "slab": slab_data,
    }


def plan_dive(params_json: str) -> str:
    """
    Main entry point for dive planning.

    Args:
        params_json: JSON string with dive parameters

    Returns:
        JSON string with Buhlmann and Slab model results
    """
    try:
        params = json.loads(params_json)
        result = _plan_dive_impl(params)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
