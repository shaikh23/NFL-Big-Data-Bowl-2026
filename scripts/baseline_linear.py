"""
Approach 1: Linear extrapolation baseline.

For each player, take the last input frame (position at throw time) and project
forward in a straight line using speed and direction:

    vx = s * sin(dir)
    vy = s * cos(dir)
    x(t) = x_throw + vx * t
    y(t) = y_throw + vy * t

where t = frame_id * 0.1 seconds (10 fps).

No play_direction normalization needed — the projection math is coordinate-
system agnostic.

Usage:
    uv run scripts/baseline_linear.py            # evaluate on train + generate test preds
    uv run scripts/baseline_linear.py --train-only  # train eval only
    uv run scripts/baseline_linear.py --test-only   # test preds only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR, TEST_INPUT, TEST_CSV


def _last_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the last input frame per (game_id, play_id, nfl_id) and
    compute velocity components from speed and direction."""
    last = (
        input_df.sort_values("frame_id")
        .groupby(["game_id", "play_id", "nfl_id"])
        .last()
        .reset_index()
    )
    dir_rad = np.radians(last["dir"])
    last["vx"] = last["s"] * np.sin(dir_rad)
    last["vy"] = last["s"] * np.cos(dir_rad)
    return last


def predict(test_input_batch: pd.DataFrame, test_batch: pd.DataFrame) -> pd.DataFrame:
    """Core prediction function — matches the gateway batch interface.

    Args:
        test_input_batch: input tracking data for one play (all players, all frames)
        test_batch: target rows for this play (game_id, play_id, nfl_id, frame_id)

    Returns:
        DataFrame with columns ['x', 'y'], same length as test_batch, same order.
    """
    last = _last_frame(test_input_batch)

    # Merge last-frame info onto each target row
    merged = test_batch.merge(
        last[["game_id", "play_id", "nfl_id", "x", "y", "vx", "vy"]],
        on=["game_id", "play_id", "nfl_id"],
        how="left",
    )

    t = merged["frame_id"] * 0.1
    predictions = pd.DataFrame(
        {
            "x": (merged["x"] + merged["vx"] * t).clip(0, 120),
            "y": (merged["y"] + merged["vy"] * t).clip(0, 53.33),
        }
    )
    return predictions


def evaluate_train(weeks: list[int] | None = None) -> dict:
    """Evaluate linear baseline on training data. Returns per-role and overall RMSE."""
    if weeks is None:
        weeks = list(range(1, 19))

    all_errors = []

    for w in weeks:
        inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w:02d}.csv")
        out = pd.read_csv(TRAIN_DIR / f"output_2023_w{w:02d}.csv")

        last = _last_frame(inp)

        # Merge velocity onto output
        merged = out.merge(
            last[["game_id", "play_id", "nfl_id", "x", "y", "vx", "vy", "player_role"]],
            on=["game_id", "play_id", "nfl_id"],
            suffixes=("_actual", "_throw"),
        )

        t = merged["frame_id"] * 0.1
        merged["x_pred"] = merged["x_throw"] + merged["vx"] * t
        merged["y_pred"] = merged["y_throw"] + merged["vy"] * t
        merged["err_sq"] = (merged["x_pred"] - merged["x_actual"]) ** 2 + (
            merged["y_pred"] - merged["y_actual"]
        ) ** 2

        all_errors.append(merged[["player_role", "err_sq", "frame_id"]])
        print(f"  Week {w:2d}: {len(merged):>7,} output frames processed")

    errors = pd.concat(all_errors, ignore_index=True)

    overall_rmse = np.sqrt(errors["err_sq"].mean())

    role_rmse = (
        errors.groupby("player_role")["err_sq"]
        .mean()
        .apply(np.sqrt)
        .sort_values(ascending=False)
    )

    # RMSE by forecast horizon (frame_id), averaged across all plays
    horizon_rmse = (
        errors.groupby("frame_id")["err_sq"]
        .mean()
        .apply(np.sqrt)
    )

    return {
        "overall_rmse": overall_rmse,
        "role_rmse": role_rmse,
        "horizon_rmse": horizon_rmse,
        "total_frames": len(errors),
    }


def generate_test_predictions() -> pd.DataFrame:
    """Generate predictions for the test set and return as a DataFrame
    with columns ['id', 'x', 'y'] matching test.csv row order."""
    test = pd.read_csv(TEST_CSV)
    test_input = pd.read_csv(TEST_INPUT)

    # Process play by play to match gateway behavior
    predictions = []
    plays = test[["game_id", "play_id"]].drop_duplicates()

    for _, play in plays.iterrows():
        gid, pid = play["game_id"], play["play_id"]
        test_batch = test[(test["game_id"] == gid) & (test["play_id"] == pid)]
        input_batch = test_input[
            (test_input["game_id"] == gid) & (test_input["play_id"] == pid)
        ]
        pred = predict(input_batch, test_batch)
        pred["id"] = test_batch["id"].values
        predictions.append(pred)

    result = pd.concat(predictions, ignore_index=True)
    # Return in test.csv row order
    result = result.set_index("id").loc[test["id"]].reset_index()
    return result


def main():
    parser = argparse.ArgumentParser(description="Linear extrapolation baseline")
    parser.add_argument(
        "--train-only", action="store_true", help="Only evaluate on training data"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only generate test predictions"
    )
    parser.add_argument(
        "--weeks", type=int, nargs="+", default=None,
        help="Which training weeks to evaluate (default: all 18)"
    )
    args = parser.parse_args()

    if not args.test_only:
        print("Evaluating on training data...")
        results = evaluate_train(args.weeks)
        print(f"\nOverall RMSE: {results['overall_rmse']:.4f} yards")
        print(f"Total output frames evaluated: {results['total_frames']:,}")
        print(f"\nRMSE by player role:")
        for role, rmse in results["role_rmse"].items():
            print(f"  {role:25s}: {rmse:.4f}")
        print(f"\nRMSE by forecast horizon (first 20 frames):")
        for fid, rmse in results["horizon_rmse"].head(20).items():
            print(f"  frame {fid:2d} ({fid*0.1:.1f}s): {rmse:.4f}")

    if not args.train_only:
        print("\nGenerating test predictions...")
        preds = generate_test_predictions()
        out_path = Path(__file__).resolve().parent.parent / "test_predictions_baseline.csv"
        preds.to_csv(out_path, index=False)
        print(f"Saved {len(preds)} predictions to {out_path.name}")
        print(preds.describe())


if __name__ == "__main__":
    main()
