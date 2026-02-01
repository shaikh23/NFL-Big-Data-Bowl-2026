"""
Data loading and cleaning for NFL Big Data Bowl 2026.

Main cleaning step: normalize play_direction so all plays have the offense
moving right (increasing x). When play_direction is "left", we flip:
    x  -> 120 - x
    dir -> (360 - dir) % 360
    o   -> (360 - o) % 360
    ball_land_x -> 120 - ball_land_x

This is done at load time so everything downstream works in a consistent
coordinate system.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR


def _normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates for plays where offense moves left."""
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = 120 - df.loc[left, "x"]
    df.loc[left, "dir"] = (360 - df.loc[left, "dir"]) % 360
    df.loc[left, "o"] = (360 - df.loc[left, "o"]) % 360
    df.loc[left, "ball_land_x"] = 120 - df.loc[left, "ball_land_x"]
    return df


def load_input(week: int) -> pd.DataFrame:
    """Load and clean a single week's input file."""
    path = TRAIN_DIR / f"input_2023_w{week:02d}.csv"
    df = pd.read_csv(path)
    return _normalize_direction(df)


def load_output(week: int) -> pd.DataFrame:
    """Load a single week's output file. No direction normalization needed
    here — apply after merging with input if you need it, since output lacks
    play_direction."""
    path = TRAIN_DIR / f"output_2023_w{week:02d}.csv"
    return pd.read_csv(path)


def load_all_input(weeks: list[int] | None = None) -> pd.DataFrame:
    """Load and clean input files for given weeks. Defaults to all 18 weeks."""
    if weeks is None:
        weeks = list(range(1, 19))
    frames = [load_input(w) for w in weeks]
    return pd.concat(frames, ignore_index=True)


def load_all_output(weeks: list[int] | None = None) -> pd.DataFrame:
    """Load output files for given weeks. Defaults to all 18 weeks."""
    if weeks is None:
        weeks = list(range(1, 19))
    frames = [load_output(w) for w in weeks]
    return pd.concat(frames, ignore_index=True)


def load_paired(weeks: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load input and output together, normalizing output x coordinates
    to match the input's play_direction normalization."""
    if weeks is None:
        weeks = list(range(1, 19))

    inputs = []
    outputs = []
    for w in weeks:
        inp = load_input(w)
        out = load_output(w)

        # Propagate play_direction into output so we can flip output x too
        direction_map = (
            inp[["game_id", "play_id", "play_direction"]]
            .drop_duplicates()
        )
        out = out.merge(direction_map, on=["game_id", "play_id"])
        left = out["play_direction"] == "left"
        out.loc[left, "x"] = 120 - out.loc[left, "x"]
        out.drop(columns=["play_direction"], inplace=True)

        inputs.append(inp)
        outputs.append(out)

    return pd.concat(inputs, ignore_index=True), pd.concat(outputs, ignore_index=True)
