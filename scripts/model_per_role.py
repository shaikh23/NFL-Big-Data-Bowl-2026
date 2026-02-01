"""
Approach 2: Per-role gradient boosting models.

Two LightGBM models (one per predicted role: Targeted Receiver, Defensive
Coverage), each predicting x and y independently. Features include the
last-frame kinematics, physics-informed quantities (distance/angle to ball
landing spot, speed needed to reach ball), and the linear baseline prediction.

All coordinates are normalized so the offense moves right. Test predictions
are un-normalized before output.

Train/val split is temporal: weeks 1-14 train, weeks 15-18 validation.
For test predictions we retrain on all 18 weeks.

Usage:
    uv run scripts/model_per_role.py              # train + val eval + test preds
    uv run scripts/model_per_role.py --train-only # train + val eval only
    uv run scripts/model_per_role.py --test-only  # retrain on all weeks + test preds
"""

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR, TEST_INPUT, TEST_CSV

TRAIN_WEEKS = list(range(1, 15))
VAL_WEEKS = list(range(15, 19))

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

FEATURES = [
    # Position and motion at throw time
    "x", "y", "s", "a", "dir", "o", "vx", "vy",
    # Ball and play context
    "ball_land_x", "ball_land_y", "absolute_yardline_number",
    # Time / horizon
    "frame_id", "t", "frac_time", "num_frames_output", "time_in_air",
    # Physics features
    "dist_to_ball", "angle_to_ball", "angle_diff", "speed_needed",
    # Linear baseline as feature
    "x_linear", "y_linear", "linear_dist_to_ball",
    # Player info
    "player_position",
]

CATEGORICAL_FEATURES = ["player_position"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates for left-direction plays so offense always moves right."""
    df = df.copy()
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = 120 - df.loc[left, "x"]
    df.loc[left, "dir"] = (360 - df.loc[left, "dir"]) % 360
    df.loc[left, "o"] = (360 - df.loc[left, "o"]) % 360
    df.loc[left, "ball_land_x"] = 120 - df.loc[left, "ball_land_x"]
    return df


def _last_frame(input_df: pd.DataFrame) -> pd.DataFrame:
    """Last input frame per player per play, plus velocity components."""
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


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features. Expects df to already have last-frame
    columns merged in (x, y, s, a, dir, o, vx, vy, ball_land_x, ball_land_y,
    num_frames_output) plus frame_id."""
    df = df.copy()

    # Time
    df["t"] = df["frame_id"] * 0.1
    df["frac_time"] = df["frame_id"] / df["num_frames_output"]
    df["time_in_air"] = df["num_frames_output"] * 0.1

    # Physics: relationship between player and ball landing spot
    dx = df["ball_land_x"] - df["x"]
    dy = df["ball_land_y"] - df["y"]
    df["dist_to_ball"] = np.sqrt(dx**2 + dy**2)
    df["angle_to_ball"] = np.degrees(np.arctan2(dx, dy)) % 360
    # Signed angular difference: how far off is the player's heading from the ball?
    df["angle_diff"] = (df["dir"] - df["angle_to_ball"] + 180) % 360 - 180
    df["speed_needed"] = df["dist_to_ball"] / df["time_in_air"].clip(lower=0.1)

    # Linear baseline prediction as feature
    df["x_linear"] = (df["x"] + df["vx"] * df["t"]).clip(0, 120)
    df["y_linear"] = (df["y"] + df["vy"] * df["t"]).clip(0, 53.33)
    df["linear_dist_to_ball"] = np.sqrt(
        (df["x_linear"] - df["ball_land_x"]) ** 2
        + (df["y_linear"] - df["ball_land_y"]) ** 2
    )

    return df


def _load_week(week: int) -> pd.DataFrame:
    """Load one week's input+output, normalize, merge, engineer features.
    Returns a single DataFrame with features + x_actual, y_actual + player_role."""
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{week:02d}.csv")
    out = pd.read_csv(TRAIN_DIR / f"output_2023_w{week:02d}.csv")

    inp = _normalize(inp)

    # Propagate play_direction and player_role into output
    meta = inp[["game_id", "play_id", "nfl_id", "play_direction", "player_role"]].drop_duplicates()
    out = out.merge(meta, on=["game_id", "play_id", "nfl_id"])

    # Normalize output x the same way
    left = out["play_direction"] == "left"
    out.loc[left, "x"] = 120 - out.loc[left, "x"]

    # Rename targets so they don't collide with throw-time x, y after merge
    out = out.rename(columns={"x": "x_actual", "y": "y_actual"})

    # Merge last-frame info
    last = _last_frame(inp)
    last_cols = [
        "game_id", "play_id", "nfl_id",
        "x", "y", "s", "a", "dir", "o", "vx", "vy",
        "ball_land_x", "ball_land_y", "num_frames_output",
        "absolute_yardline_number", "player_position",
    ]
    df = out.merge(last[last_cols], on=["game_id", "play_id", "nfl_id"])

    df = _engineer_features(df)
    return df


def load_weeks(weeks: list[int]) -> pd.DataFrame:
    """Load and prepare multiple weeks."""
    frames = []
    for w in weeks:
        frames.append(_load_week(w))
        print(f"  Loaded week {w:2d}")
    return pd.concat(frames, ignore_index=True)


def train_models(df: pd.DataFrame) -> dict:
    """Train LightGBM models. Returns {(role, target): model}."""
    models = {}
    roles = ["Targeted Receiver", "Defensive Coverage"]

    for role in roles:
        role_df = df[df["player_role"] == role].copy()
        for col in CATEGORICAL_FEATURES:
            role_df[col] = role_df[col].astype("category")

        X = role_df[FEATURES]
        for target in ["x_actual", "y_actual"]:
            y = role_df[target]
            dataset = lgb.Dataset(X, label=y, categorical_feature=CATEGORICAL_FEATURES)
            model = lgb.train(LGBM_PARAMS, dataset, num_boost_round=500)
            models[(role, target)] = model
            print(f"  {role} -> {target}: trained ({len(role_df):,} rows)")

    return models


def predict_with_models(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Add x_pred, y_pred columns using trained models."""
    df = df.copy()
    df["x_pred"] = np.nan
    df["y_pred"] = np.nan

    for role in ["Targeted Receiver", "Defensive Coverage"]:
        mask = df["player_role"] == role
        if mask.sum() == 0:
            continue
        role_df = df.loc[mask].copy()
        for col in CATEGORICAL_FEATURES:
            role_df[col] = role_df[col].astype("category")

        X = role_df[FEATURES]
        df.loc[mask, "x_pred"] = models[(role, "x_actual")].predict(X)
        df.loc[mask, "y_pred"] = models[(role, "y_actual")].predict(X)

    df["x_pred"] = df["x_pred"].clip(0, 120)
    df["y_pred"] = df["y_pred"].clip(0, 53.33)
    return df


def evaluate(df: pd.DataFrame) -> None:
    """Print RMSE overall, by role, and by horizon. Also compare to linear baseline."""
    # Model RMSE
    df["err_sq"] = (df["x_pred"] - df["x_actual"]) ** 2 + (df["y_pred"] - df["y_actual"]) ** 2
    overall = np.sqrt(df["err_sq"].mean())

    # Linear baseline RMSE on same rows
    df["err_sq_linear"] = (df["x_linear"] - df["x_actual"]) ** 2 + (df["y_linear"] - df["y_actual"]) ** 2
    linear_rmse = np.sqrt(df["err_sq_linear"].mean())

    print(f"\n{'':=<50}")
    print(f"  Model RMSE:  {overall:.4f} yards")
    print(f"  Linear RMSE: {linear_rmse:.4f} yards")
    print(f"  Improvement: {linear_rmse - overall:.4f} yards ({(1 - overall / linear_rmse) * 100:.1f}%)")
    print(f"{'':=<50}")

    print(f"\nBy role:")
    print(f"  {'Role':<28} {'Model':>8} {'Linear':>8} {'Delta':>8}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8}")
    for role in ["Targeted Receiver", "Defensive Coverage"]:
        mask = df["player_role"] == role
        m_rmse = np.sqrt(df.loc[mask, "err_sq"].mean())
        l_rmse = np.sqrt(df.loc[mask, "err_sq_linear"].mean())
        print(f"  {role:<28} {m_rmse:>8.4f} {l_rmse:>8.4f} {m_rmse - l_rmse:>+8.4f}")

    print(f"\nBy horizon (first 20 frames):")
    print(f"  {'Frame':<10} {'Model':>8} {'Linear':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8}")
    for fid in range(1, 21):
        mask = df["frame_id"] == fid
        if mask.sum() == 0:
            continue
        m = np.sqrt(df.loc[mask, "err_sq"].mean())
        l = np.sqrt(df.loc[mask, "err_sq_linear"].mean())
        print(f"  {fid:<3} ({fid*0.1:.1f}s)  {m:>8.4f} {l:>8.4f}")

    # Feature importance (average across all 4 models)
    print(f"\nTop 10 features by importance (avg across models):")
    imp = pd.Series(0.0, index=FEATURES)
    for model in models.values():
        imp += pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES)
    imp /= len(models)
    for feat, val in imp.sort_values(ascending=False).head(10).items():
        print(f"  {feat:<30} {val:>10.1f}")


def generate_test_predictions(models: dict) -> pd.DataFrame:
    """Predict on test set, un-normalize, return DataFrame with id, x, y."""
    test = pd.read_csv(TEST_CSV)
    test_input = pd.read_csv(TEST_INPUT)

    # Save original play_direction for un-normalization
    dir_map = test_input[["game_id", "play_id", "play_direction"]].drop_duplicates()

    # Normalize
    test_input = _normalize(test_input)

    # Last frame + velocity
    last = _last_frame(test_input)

    # Propagate player_role to test target rows
    role_map = test_input[["game_id", "play_id", "nfl_id", "player_role"]].drop_duplicates()
    df = test.merge(role_map, on=["game_id", "play_id", "nfl_id"])

    # Merge last-frame info
    last_cols = [
        "game_id", "play_id", "nfl_id",
        "x", "y", "s", "a", "dir", "o", "vx", "vy",
        "ball_land_x", "ball_land_y", "num_frames_output",
        "absolute_yardline_number", "player_position",
    ]
    df = df.merge(last[last_cols], on=["game_id", "play_id", "nfl_id"])
    df = _engineer_features(df)

    # Predict (in normalized coordinates)
    df = predict_with_models(models, df)

    # Un-normalize x predictions for left-direction plays
    df = df.merge(dir_map, on=["game_id", "play_id"])
    left = df["play_direction"] == "left"
    df.loc[left, "x_pred"] = 120 - df.loc[left, "x_pred"]

    # Output in test.csv row order
    result = df.set_index("id")[["x_pred", "y_pred"]].loc[test["id"]].reset_index()
    result = result.rename(columns={"x_pred": "x", "y_pred": "y"})
    return result


def main():
    parser = argparse.ArgumentParser(description="Per-role LightGBM models")
    parser.add_argument("--train-only", action="store_true", help="Train + val eval only")
    parser.add_argument("--test-only", action="store_true", help="Train on all weeks + test preds only")
    args = parser.parse_args()

    global models

    if not args.test_only:
        print("Loading training data (weeks 1-14)...")
        train_df = load_weeks(TRAIN_WEEKS)
        print(f"  Total: {len(train_df):,} rows\n")

        print("Loading validation data (weeks 15-18)...")
        val_df = load_weeks(VAL_WEEKS)
        print(f"  Total: {len(val_df):,} rows\n")

        print("Training models...")
        models = train_models(train_df)

        print("\nEvaluating on validation set...")
        val_df = predict_with_models(models, val_df)
        evaluate(val_df)

    if not args.train_only:
        print("\nTraining on all 18 weeks for test predictions...")
        all_df = load_weeks(list(range(1, 19)))
        print(f"  Total: {len(all_df):,} rows\n")

        print("Training models (full data)...")
        models = train_models(all_df)

        print("\nGenerating test predictions...")
        preds = generate_test_predictions(models)
        out_path = Path(__file__).resolve().parent.parent / "test_predictions_per_role.csv"
        preds.to_csv(out_path, index=False)
        print(f"Saved {len(preds)} predictions to {out_path.name}")


if __name__ == "__main__":
    main()
