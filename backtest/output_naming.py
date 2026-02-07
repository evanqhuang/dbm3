"""
Output directory naming based on active config settings.

Generates deterministic subfolder names so multiple backtest runs with different
configurations coexist without overwriting each other.
"""

import os
import shutil

import yaml

from .buhlmann_constants import GradientFactors, GF_DEFAULT


def build_settings_dirname(
    gf: GradientFactors,
    conservatism: float,
    boyle_exponent: float,
    f_o2: float,
) -> str:
    """Build a deterministic directory name encoding the active settings.

    Format: gf{low}-{high}_cons{conservatism}_boyle{exponent}_o2-{fraction}

    All values are expressed as integers (fractions multiplied to remove dots).

    Examples:
        GF 70/85, cons 1.0, boyle 0.5, fO2 0.21  -> gf70-85_cons100_boyle050_o2-21
        GF 100/100, cons 0.85, boyle 1.0, fO2 0.32 -> gf100-100_cons085_boyle100_o2-32
    """
    gf_low_pct = int(round(gf.gf_low * 100))
    gf_high_pct = int(round(gf.gf_high * 100))
    cons_int = int(round(conservatism * 100))
    boyle_int = int(round(boyle_exponent * 100))
    # fO2 as two-digit integer (0.21 -> 21, 0.32 -> 32)
    o2_int = int(round(f_o2 * 100))

    return (
        f"gf{gf_low_pct}-{gf_high_pct}"
        f"_cons{cons_int:03d}"
        f"_boyle{boyle_int:03d}"
        f"_o2-{o2_int}"
    )


def load_effective_config(
    gf_override: tuple = None,
    config_path: str = None,
) -> dict:
    """Load configuration from config.yaml with optional CLI GF override.

    Returns a dict with resolved settings:
        gf:              GradientFactors instance
        conservatism:    float
        boyle_exponent:  float
        f_o2:            float
        config_path:     str (resolved path)
        gf_source:       'cli' | 'config' | 'default'
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config.yaml"
        )

    # Defaults
    conservatism = 1.0
    boyle_exponent = 0.5
    f_o2 = 0.21
    gf = GF_DEFAULT
    gf_source = "default"

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        conservatism = float(config.get("conservatism", conservatism))
        f_o2 = float(config.get("f_o2", f_o2))

        deco_cfg = config.get("deco", {})
        boyle_exponent = float(deco_cfg.get("boyle_exponent", boyle_exponent))

        buhlmann_cfg = config.get("buhlmann", {})
        if buhlmann_cfg:
            gf = GradientFactors(
                gf_low=float(buhlmann_cfg.get("gf_low", 1.0)),
                gf_high=float(buhlmann_cfg.get("gf_high", 1.0)),
            )
            gf_source = "config"

    if gf_override:
        gf = GradientFactors(
            gf_low=gf_override[0] / 100.0,
            gf_high=gf_override[1] / 100.0,
        )
        gf_source = "cli"

    return {
        "gf": gf,
        "conservatism": conservatism,
        "boyle_exponent": boyle_exponent,
        "f_o2": f_o2,
        "config_path": config_path,
        "gf_source": gf_source,
    }


def save_config_snapshot(config_path: str, output_dir: str) -> None:
    """Copy config.yaml into the output directory for reproducibility."""
    if os.path.exists(config_path):
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(config_path, os.path.join(output_dir, "config.yaml"))
