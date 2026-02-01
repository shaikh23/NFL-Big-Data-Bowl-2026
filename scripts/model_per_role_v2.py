"""
Approach 3: Per-role gradient boosting with trajectory and contextual features.

Builds on Approach 2 (model_per_role.py). Adds two feature categories:

1. Trajectory summary (from last 10 input frames per player):
   - Speed trend: is the player accelerating or decelerating into the throw?
   - Direction stability: mean resultant length (1 = straight line, 0 = random)
   - Smoothed velocity: linear regression slope over x/y positions
   - Mean speed over the window

2. Contextual / relative features (spatial relationships between players):
   - Distance, offset, and relative velocity to the targeted receiver
   - Geometry relative to the TR-to-ball line: projection fraction (how far
     along the line from TR to ball) and perpendicular distance (how far off it).
     High proj_frac + low perp_dist = defender is in position to intercept.
   - For the targeted receiver: nearest defender distance and defender count
     within 5 yards (defensive pressure)

Same model structure as v1: LightGBM, one model per (role, target),
play_direction normalized, temporal train/val split (weeks 1-14 / 15-18).

Usage:
    uv run scripts/model_per_role_v2.py              # train + val eval + test preds
    uv run scripts/model_per_role_v2.py --train-only
    uv run scripts/model_per_role_v2.py --test-only
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

LGBM_BASE_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

# Grid search space — evaluated on train/val split, best params reused for full retrain
LR_GRID = [0.02, 0.05, 0.1]
LEAVES_GRID = [31, 63, 127]

# Short-horizon blend: linear wins at very early frames; blend smoothly
# frame_id 1-2: pure linear, frame 3: 50% blend, frame 4+: pure model
BLEND_WEIGHTS = {1: 0.0, 2: 0.0, 3: 0.5}  # weight on MODEL prediction

FEATURES = [
    # --- v1 features ---
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
    # --- NEW: Trajectory summary ---
    "s_trend",          # slope of speed over last 10 frames (+ = accelerating)
    "dir_stability",    # mean resultant length of direction (1 = straight, 0 = random)
    "s_mean_10",        # mean speed over last 10 frames
    "x_vel_smooth",     # regression slope of x over last 10 frames (yards/frame)
    "y_vel_smooth",     # regression slope of y over last 10 frames
    # --- NEW: Contextual / relative ---
    "dist_to_tr",           # distance to targeted receiver at throw time
    "dx_to_tr",             # x offset to TR
    "dy_to_tr",             # y offset to TR
    "rel_vx_to_tr",         # vx - vx_TR (relative velocity)
    "rel_vy_to_tr",         # vy - vy_TR
    "rel_speed_to_tr",      # magnitude of relative velocity to TR
    "proj_frac_tr_ball",    # projection along TR-to-ball line (0=at TR, 1=at ball)
    "perp_dist_tr_ball",    # perpendicular distance to TR-to-ball line
    "min_defender_dist",    # closest defender to TR (meaningful for TR model)
    "n_defenders_close",    # defenders within 5 yards of TR
]

CATEGORICAL_FEATURES = ["player_position"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flip coordinates for left-direction plays so offense always moves right."""
    df = df.copy()
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = 120 - df.loc[left, "x"]
    df.loc[left, "dir"] = (360 - df.loc[left, "dir"]) % 360
    df.loc[left, "o"] = (360 - df.loc[left, "o"]) % 360
    df.loc[left, "ball_land_x"] = 120 - df.loc[left, "ball_land_x"]
    return df


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

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


def _trajectory_summary(input_df: pd.DataFrame, n_frames: int = 10) -> pd.DataFrame:
    """Summarize the pre-throw trajectory for each player.

    Only needs to be called on predicted players (those with output rows).
    Uses the last n_frames of each player's input sequence.
    """
    results = []
    for (gid, pid, nid), group in input_df.groupby(["game_id", "play_id", "nfl_id"]):
        group = group.sort_values("frame_id").tail(n_frames)
        n = len(group)
        s_mean = group["s"].mean()

        if n < 3:
            results.append({
                "game_id": gid, "play_id": pid, "nfl_id": nid,
                "s_trend": 0.0, "dir_stability": 1.0, "s_mean_10": s_mean,
                "x_vel_smooth": 0.0, "y_vel_smooth": 0.0,
            })
            continue

        t = np.arange(n, dtype=float)

        # Speed trend and smoothed position velocity via linear regression
        s_trend = np.polyfit(t, group["s"].values, 1)[0]
        x_vel_smooth = np.polyfit(t, group["x"].values, 1)[0]
        y_vel_smooth = np.polyfit(t, group["y"].values, 1)[0]

        # Direction stability: mean resultant length of circular data
        # 1.0 = perfectly consistent direction, 0.0 = uniformly random
        dir_rad = np.radians(group["dir"].values)
        R = np.sqrt(np.sin(dir_rad).mean() ** 2 + np.cos(dir_rad).mean() ** 2)

        results.append({
            "game_id": gid, "play_id": pid, "nfl_id": nid,
            "s_trend": s_trend,
            "dir_stability": R,
            "s_mean_10": s_mean,
            "x_vel_smooth": x_vel_smooth,
            "y_vel_smooth": y_vel_smooth,
        })

    return pd.DataFrame(results)


def _play_context(last_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute contextual features encoding relationships between players.

    Uses ALL players' last-frame positions (not just predicted ones) so that
    defender counts/distances are accurate even for non-predicted defenders.
    Returns one row per player in last_frame with contextual feature columns.
    """
    # --- Targeted receiver position per play ---
    tr = (
        last_frame[last_frame["player_role"] == "Targeted Receiver"][
            ["game_id", "play_id", "x", "y", "vx", "vy"]
        ]
        .rename(columns={"x": "x_tr", "y": "y_tr", "vx": "vx_tr", "vy": "vy_tr"})
    )

    # --- Defender summary per play (for TR model: pressure metrics) ---
    defenders = last_frame[last_frame["player_role"] == "Defensive Coverage"][
        ["game_id", "play_id", "x", "y"]
    ]
    def_with_tr = defenders.merge(tr, on=["game_id", "play_id"])
    def_with_tr["_dist"] = np.sqrt(
        (def_with_tr["x"] - def_with_tr["x_tr"]) ** 2
        + (def_with_tr["y"] - def_with_tr["y_tr"]) ** 2
    )
    def_summary = (
        def_with_tr.groupby(["game_id", "play_id"])
        .agg(
            min_defender_dist=("_dist", "min"),
            n_defenders_close=("_dist", lambda x: (x < 5).sum()),
        )
        .reset_index()
    )

    # --- Per-player contextual features ---
    ctx = last_frame[
        ["game_id", "play_id", "nfl_id", "x", "y", "vx", "vy",
         "ball_land_x", "ball_land_y"]
    ].copy()

    # Merge TR position into every player's row
    ctx = ctx.merge(tr, on=["game_id", "play_id"], how="left")

    # Distance and offset to TR
    ctx["dist_to_tr"] = np.sqrt(
        (ctx["x"] - ctx["x_tr"]) ** 2 + (ctx["y"] - ctx["y_tr"]) ** 2
    )
    ctx["dx_to_tr"] = ctx["x_tr"] - ctx["x"]
    ctx["dy_to_tr"] = ctx["y_tr"] - ctx["y"]

    # Relative velocity to TR
    ctx["rel_vx_to_tr"] = ctx["vx"] - ctx["vx_tr"]
    ctx["rel_vy_to_tr"] = ctx["vy"] - ctx["vy_tr"]
    ctx["rel_speed_to_tr"] = np.sqrt(
        ctx["rel_vx_to_tr"] ** 2 + ctx["rel_vy_to_tr"] ** 2
    )

    # Geometry: where does this player sit relative to the TR-to-ball line?
    # proj_frac: 0 = at TR, 1 = at ball landing spot
    # perp_dist: perpendicular distance off that line
    dx_tr_ball = ctx["ball_land_x"] - ctx["x_tr"]
    dy_tr_ball = ctx["ball_land_y"] - ctx["y_tr"]
    line_len_sq = (dx_tr_ball ** 2 + dy_tr_ball ** 2).clip(lower=1e-6)

    dx_player = ctx["x"] - ctx["x_tr"]
    dy_player = ctx["y"] - ctx["y_tr"]

    proj = (dx_player * dx_tr_ball + dy_player * dy_tr_ball) / line_len_sq
    ctx["proj_frac_tr_ball"] = proj.clip(0, 1)

    closest_x = ctx["x_tr"] + ctx["proj_frac_tr_ball"] * dx_tr_ball
    closest_y = ctx["y_tr"] + ctx["proj_frac_tr_ball"] * dy_tr_ball
    ctx["perp_dist_tr_ball"] = np.sqrt(
        (ctx["x"] - closest_x) ** 2 + (ctx["y"] - closest_y) ** 2
    )

    # Merge defender pressure summary (meaningful for TR; filled with defaults for others)
    ctx = ctx.merge(def_summary, on=["game_id", "play_id"], how="left")
    ctx["min_defender_dist"] = ctx["min_defender_dist"].fillna(999.0)
    ctx["n_defenders_close"] = ctx["n_defenders_close"].fillna(0).astype(int)

    context_cols = [
        "game_id", "play_id", "nfl_id",
        "dist_to_tr", "dx_to_tr", "dy_to_tr",
        "rel_vx_to_tr", "rel_vy_to_tr", "rel_speed_to_tr",
        "proj_frac_tr_ball", "perp_dist_tr_ball",
        "min_defender_dist", "n_defenders_close",
    ]
    return ctx[context_cols]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time, physics, and linear baseline features (same as v1)."""
    df = df.copy()

    df["t"] = df["frame_id"] * 0.1
    df["frac_time"] = df["frame_id"] / df["num_frames_output"]
    df["time_in_air"] = df["num_frames_output"] * 0.1

    dx = df["ball_land_x"] - df["x"]
    dy = df["ball_land_y"] - df["y"]
    df["dist_to_ball"] = np.sqrt(dx ** 2 + dy ** 2)
    df["angle_to_ball"] = np.degrees(np.arctan2(dx, dy)) % 360
    df["angle_diff"] = (df["dir"] - df["angle_to_ball"] + 180) % 360 - 180
    df["speed_needed"] = df["dist_to_ball"] / df["time_in_air"].clip(lower=0.1)

    df["x_linear"] = (df["x"] + df["vx"] * df["t"]).clip(0, 120)
    df["y_linear"] = (df["y"] + df["vy"] * df["t"]).clip(0, 53.33)
    df["linear_dist_to_ball"] = np.sqrt(
        (df["x_linear"] - df["ball_land_x"]) ** 2
        + (df["y_linear"] - df["ball_land_y"]) ** 2
    )

    return df


# ---------------------------------------------------------------------------
# Data loading pipeline
# ---------------------------------------------------------------------------

def _load_week(week: int) -> pd.DataFrame:
    """Full pipeline for one week: load -> normalize -> extract all features -> merge."""
    inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{week:02d}.csv")
    out = pd.read_csv(TRAIN_DIR / f"output_2023_w{week:02d}.csv")

    inp = _normalize(inp)

    # Propagate metadata into output
    meta = inp[["game_id", "play_id", "nfl_id", "play_direction", "player_role"]].drop_duplicates()
    out = out.merge(meta, on=["game_id", "play_id", "nfl_id"])
    left = out["play_direction"] == "left"
    out.loc[left, "x"] = 120 - out.loc[left, "x"]
    out = out.rename(columns={"x": "x_actual", "y": "y_actual"})

    # Last frame for all players (needed for context)
    last = _last_frame(inp)

    # Trajectory summary — only for predicted players (those with output rows)
    traj = _trajectory_summary(inp[inp["player_to_predict"]])

    # Play context — uses all players' positions
    context = _play_context(last)

    # Merge last-frame info onto output rows
    last_cols = [
        "game_id", "play_id", "nfl_id",
        "x", "y", "s", "a", "dir", "o", "vx", "vy",
        "ball_land_x", "ball_land_y", "num_frames_output",
        "absolute_yardline_number", "player_position",
    ]
    df = out.merge(last[last_cols], on=["game_id", "play_id", "nfl_id"])
    df = df.merge(traj, on=["game_id", "play_id", "nfl_id"])
    df = df.merge(context, on=["game_id", "play_id", "nfl_id"])
    df = _engineer_features(df)

    return df


def load_weeks(weeks: list[int]) -> pd.DataFrame:
    frames = []
    for w in weeks:
        frames.append(_load_week(w))
        print(f"  Loaded week {w:2d}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Model training, prediction, evaluation
# ---------------------------------------------------------------------------

def _train_single(X_train, y_train, X_val, y_val, params: dict, max_rounds: int = 5000) -> lgb.Booster:
    """Train one LightGBM model with early stopping on a validation set."""
    train_ds = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    val_ds = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_ds)
    model = lgb.train(
        params, train_ds,
        num_boost_round=max_rounds,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def tune_hyperparams(df: pd.DataFrame) -> dict[str, dict]:
    """Grid search over learning_rate x num_leaves, tuned independently per role.

    Within df (already the train weeks 1-14), split the last 20% of game_ids
    as an early-stopping validation set. Returns {role: best_params}.
    """
    game_ids = sorted(df["game_id"].unique())
    n_tune_val = max(1, int(len(game_ids) * 0.2))
    tune_val_games = set(game_ids[-n_tune_val:])

    tune_train = df[~df["game_id"].isin(tune_val_games)]
    tune_val = df[df["game_id"].isin(tune_val_games)]

    print(f"  Tuning grid: {len(LR_GRID)} LRs x {len(LEAVES_GRID)} num_leaves "
          f"| tune-train {len(tune_train):,} rows, tune-val {len(tune_val):,} rows")

    best_per_role = {}
    for role in ["Targeted Receiver", "Defensive Coverage"]:
        best_rmse = float("inf")
        best_params = {}
        for lr in LR_GRID:
            for leaves in LEAVES_GRID:
                params = {**LGBM_BASE_PARAMS, "learning_rate": lr, "num_leaves": leaves}
                role_rmses = []

                for target in ["x_actual", "y_actual"]:
                    tr = tune_train[tune_train["player_role"] == role].copy()
                    va = tune_val[tune_val["player_role"] == role].copy()
                    for col in CATEGORICAL_FEATURES:
                        tr[col] = tr[col].astype("category")
                        va[col] = va[col].astype("category")

                    model = _train_single(
                        tr[FEATURES], tr[target],
                        va[FEATURES], va[target],
                        params,
                    )
                    preds = model.predict(va[FEATURES])
                    rmse = np.sqrt(((preds - va[target]) ** 2).mean())
                    role_rmses.append(rmse)

                avg_rmse = np.mean(role_rmses)
                flag = " <-- best" if avg_rmse < best_rmse else ""
                print(f"    {role[:2]} lr={lr:<5} leaves={leaves:<4} rmse={avg_rmse:.4f}{flag}")
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = params

        best_per_role[role] = best_params
        print(f"  {role}: lr={best_params['learning_rate']}, num_leaves={best_params['num_leaves']}, rmse={best_rmse:.4f}\n")

    return best_per_role


def train_models(df: pd.DataFrame, params_per_role: dict[str, dict] | None = None) -> dict:
    """Train 4 LightGBM models with early stopping, using per-role hyperparameters.

    params_per_role: {role: params_dict} from tune_hyperparams(). If None, uses
    defaults (lr=0.05, leaves=63) for all roles.
    """
    defaults = {**LGBM_BASE_PARAMS, "learning_rate": 0.05, "num_leaves": 63}

    # Use last 10% of game_ids as early-stopping val within the training set
    game_ids = sorted(df["game_id"].unique())
    n_es_val = max(1, int(len(game_ids) * 0.1))
    es_val_games = set(game_ids[-n_es_val:])

    models = {}
    for role in ["Targeted Receiver", "Defensive Coverage"]:
        params = (params_per_role or {}).get(role, defaults)
        role_df = df[df["player_role"] == role].copy()
        for col in CATEGORICAL_FEATURES:
            role_df[col] = role_df[col].astype("category")

        tr = role_df[~role_df["game_id"].isin(es_val_games)]
        va = role_df[role_df["game_id"].isin(es_val_games)]

        for target in ["x_actual", "y_actual"]:
            model = _train_single(
                tr[FEATURES], tr[target],
                va[FEATURES], va[target],
                params,
            )
            models[(role, target)] = model
            print(f"  {role} -> {target}: {model.best_iteration} rounds "
                  f"(train {len(tr):,} / es-val {len(va):,})")
    return models


def predict_with_models(models: dict, df: pd.DataFrame) -> pd.DataFrame:
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


def _apply_blend(df: pd.DataFrame) -> pd.DataFrame:
    """Post-processing: blend linear baseline into model predictions at short horizons.

    The model optimizes for the full sequence and tends to overshoot at frames 1-3.
    Linear extrapolation is more accurate there because the player hasn't had time
    to change course yet. BLEND_WEIGHTS controls the interpolation per frame_id.
    """
    df = df.copy()
    for fid, model_weight in BLEND_WEIGHTS.items():
        mask = df["frame_id"] == fid
        if mask.sum() == 0:
            continue
        df.loc[mask, "x_pred"] = (
            model_weight * df.loc[mask, "x_pred"]
            + (1 - model_weight) * df.loc[mask, "x_linear"]
        )
        df.loc[mask, "y_pred"] = (
            model_weight * df.loc[mask, "y_pred"]
            + (1 - model_weight) * df.loc[mask, "y_linear"]
        )
    return df


def evaluate(df: pd.DataFrame, models: dict) -> None:
    """Print RMSE comparisons: v2 model vs linear baseline, by role and horizon."""
    df["err_sq"] = (df["x_pred"] - df["x_actual"]) ** 2 + (df["y_pred"] - df["y_actual"]) ** 2
    df["err_sq_linear"] = (
        (df["x_linear"] - df["x_actual"]) ** 2
        + (df["y_linear"] - df["y_actual"]) ** 2
    )

    overall = np.sqrt(df["err_sq"].mean())
    linear_rmse = np.sqrt(df["err_sq_linear"].mean())

    print(f"\n{'':=<50}")
    print(f"  Model RMSE:  {overall:.4f} yards")
    print(f"  Linear RMSE: {linear_rmse:.4f} yards")
    print(f"  Improvement: {linear_rmse - overall:.4f} yards ({(1 - overall / linear_rmse) * 100:.1f}%)")
    print(f"{'':=<50}")

    print(f"\nBy role:")
    print(f"  {'Role':<28} {'Model':>8} {'Linear':>8} {'Delta':>8}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}")
    for role in ["Targeted Receiver", "Defensive Coverage"]:
        mask = df["player_role"] == role
        m = np.sqrt(df.loc[mask, "err_sq"].mean())
        l = np.sqrt(df.loc[mask, "err_sq_linear"].mean())
        print(f"  {role:<28} {m:>8.4f} {l:>8.4f} {m - l:>+8.4f}")

    print(f"\nBy horizon (first 20 frames):")
    print(f"  {'Frame':<10} {'Model':>8} {'Linear':>8}")
    print(f"  {'-' * 10} {'-' * 8} {'-' * 8}")
    for fid in range(1, 21):
        mask = df["frame_id"] == fid
        if mask.sum() == 0:
            continue
        m = np.sqrt(df.loc[mask, "err_sq"].mean())
        l = np.sqrt(df.loc[mask, "err_sq_linear"].mean())
        print(f"  {fid:<3} ({fid * 0.1:.1f}s)  {m:>8.4f} {l:>8.4f}")

    # Feature importance (averaged across all 4 models)
    print(f"\nTop 15 features by importance (avg gain across models):")
    imp = pd.Series(0.0, index=FEATURES)
    for model in models.values():
        imp += pd.Series(model.feature_importance(importance_type="gain"), index=FEATURES)
    imp /= len(models)
    for feat, val in imp.sort_values(ascending=False).head(15).items():
        print(f"  {feat:<30} {val:>12.1f}")


# ---------------------------------------------------------------------------
# Test prediction
# ---------------------------------------------------------------------------

def _prepare_test_features(test_input: pd.DataFrame) -> tuple:
    """Normalize test input and extract last frame, trajectory, and context."""
    test_input = _normalize(test_input)
    last = _last_frame(test_input)
    traj = _trajectory_summary(test_input[test_input["player_to_predict"]])
    context = _play_context(last)
    return last, traj, context


def generate_test_predictions(models: dict) -> pd.DataFrame:
    test = pd.read_csv(TEST_CSV)
    test_input = pd.read_csv(TEST_INPUT)

    # Save original direction for un-normalization
    dir_map = test_input[["game_id", "play_id", "play_direction"]].drop_duplicates()

    # Extract features from normalized input
    last, traj, context = _prepare_test_features(test_input)

    # Start from test target rows, merge in everything
    role_map = test_input[["game_id", "play_id", "nfl_id", "player_role"]].drop_duplicates()
    df = test.merge(role_map, on=["game_id", "play_id", "nfl_id"])

    last_cols = [
        "game_id", "play_id", "nfl_id",
        "x", "y", "s", "a", "dir", "o", "vx", "vy",
        "ball_land_x", "ball_land_y", "num_frames_output",
        "absolute_yardline_number", "player_position",
    ]
    df = df.merge(last[last_cols], on=["game_id", "play_id", "nfl_id"])
    df = df.merge(traj, on=["game_id", "play_id", "nfl_id"])
    df = df.merge(context, on=["game_id", "play_id", "nfl_id"])
    df = _engineer_features(df)

    # Predict in normalized coordinates, then blend short-horizon frames
    df = predict_with_models(models, df)
    df = _apply_blend(df)

    # Un-normalize x for left-direction plays
    df = df.merge(dir_map, on=["game_id", "play_id"])
    left = df["play_direction"] == "left"
    df.loc[left, "x_pred"] = 120 - df.loc[left, "x_pred"]

    # Output in test.csv row order
    result = df.set_index("id")[["x_pred", "y_pred"]].loc[test["id"]].reset_index()
    result = result.rename(columns={"x_pred": "x", "y_pred": "y"})
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-role LightGBM v2 (trajectory + context)")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    models = None
    tuned_params = None

    if not args.test_only:
        print("Loading training data (weeks 1-14)...")
        train_df = load_weeks(TRAIN_WEEKS)
        print(f"  Total: {len(train_df):,} rows\n")

        print("Loading validation data (weeks 15-18)...")
        val_df = load_weeks(VAL_WEEKS)
        print(f"  Total: {len(val_df):,} rows\n")

        print("Tuning hyperparameters (per role)...")
        tuned_params = tune_hyperparams(train_df)

        print("Training models with tuned params...")
        models = train_models(train_df, tuned_params)

        print("\nEvaluating on validation set...")
        val_df = predict_with_models(models, val_df)
        val_df = _apply_blend(val_df)
        evaluate(val_df, models)

    if not args.train_only:
        print("\nTraining on all 18 weeks for test predictions...")
        all_df = load_weeks(list(range(1, 19)))
        print(f"  Total: {len(all_df):,} rows\n")

        print("Training models (full data, tuned params)...")
        models = train_models(all_df, tuned_params)

        print("\nGenerating test predictions...")
        preds = generate_test_predictions(models)
        out_path = Path(__file__).resolve().parent.parent / "test_predictions_per_role_v2.csv"
        preds.to_csv(out_path, index=False)
        print(f"Saved {len(preds)} predictions to {out_path.name}")


if __name__ == "__main__":
    main()
