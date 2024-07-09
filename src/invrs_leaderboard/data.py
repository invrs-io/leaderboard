"""Utilities for loading leaderboard data.

Copyright (c) 2024 The INVRS-IO authors.
"""

import glob
from typing import Any, Dict

import numpy as onp
import pandas as pd

PyTree = Any

SEP = ", "
PATH = "path"


def leaderboard_dataframe(base_path: str = "") -> pd.DataFrame:
    """Return leaderboard data as a dataframe."""
    leaderboard = load_leaderboard(base_path=base_path)
    df = pd.DataFrame.from_dict(leaderboard.values())
    df["challenge"] = df["path"].str.split("/").str[1]
    df["file"] = df["path"].str.split("/").str[-1]
    df["file_prefix"] = df["file"].str.split("_").str[:2].str.join("_")
    df["minimum_length_scale"] = onp.minimum(df["minimum_width"], df["minimum_spacing"])
    return df


def load_leaderboard(base_path: str = "") -> Dict[str, Any]:
    """Load leaderboard data for the given `base_path`."""
    base_path = fix_base_path(base_path)
    leaderboard_paths = glob.glob(f"{base_path}challenges/*/leaderboard.txt")
    leaderboard = {}

    for path in leaderboard_paths:
        with open(path) as f:
            leaderboard_data = f.read().strip()
        if not leaderboard_data:
            continue
        lines = leaderboard_data.split("\n")
        for line in lines:
            entry_dict = {}
            for d in line.split(SEP):
                key, value = d.split("=")
                entry_dict[key] = try_float(value)
            solution_path = entry_dict[PATH]
            if solution_path in leaderboard:
                raise ValueError(
                    f"Found duplicate entry in leaderboard: {solution_path}"
                )
            leaderboard[solution_path] = entry_dict

    return leaderboard


def try_float(value: Any) -> Any:
    """Convert `value` to float if possible."""
    try:
        return float(value)
    except ValueError:
        return value


def fix_base_path(base_path: str) -> str:
    """Prepend the base path, if it is not empty."""
    if not base_path or base_path.endswith("/"):
        return base_path
    return f"{base_path}/"
