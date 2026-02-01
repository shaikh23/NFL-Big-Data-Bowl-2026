"""
Approach 5: LSTM encoder + per-frame MLP decoder.

Encodes the last 10 input frames per player with a 2-layer LSTM, then predicts
each output frame's (x, y) by decoding the encoder's hidden state together with
per-frame physics features (linear baseline, dist-to-ball, etc.).

Trained separately per role (TR, DC).  All coordinates normalized (offense moves
right); test predictions are un-normalized before output.

Compare to gradient boosting (model_per_role_v2.py) to see whether sequence
modelling adds value beyond the hand-crafted trajectory summaries in v2.

Usage:
    uv run scripts/model_lstm.py              # train + val eval
    uv run scripts/model_lstm.py --train-only # same (no test preds yet)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_DIR
from model_per_role_v2 import _normalize, load_weeks, TRAIN_WEEKS, VAL_WEEKS

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEQ_LEN = 10       # input frames fed to encoder
HIDDEN_DIM = 64    # LSTM hidden size
NUM_LAYERS = 2     # LSTM depth
NUM_EPOCHS = 50
BATCH_SIZE = 4096
LR = 1e-3
PATIENCE = 10      # epochs without val improvement before stopping

# Features extracted per timestep from the input sequence
SEQ_FEATURES = ["x", "y", "s", "a", "vx", "vy"]

# Per-output-frame features concatenated with the encoder hidden state
FRAME_FEATURES = [
    "frame_id", "t", "frac_time", "time_in_air",
    "x_linear", "y_linear",
    "dist_to_ball", "angle_to_ball", "angle_diff", "speed_needed",
    "ball_land_x", "ball_land_y",
]


# ---------------------------------------------------------------------------
# Dataset & model
# ---------------------------------------------------------------------------

class FrameDataset(Dataset):
    """Each item is one output frame.  The input sequence is shared across all
    output frames of the same player and is looked up via player_indices."""

    def __init__(self, input_seqs, player_indices, frame_features, targets):
        self.input_seqs = input_seqs          # (num_players, seq_len, seq_dim)
        self.player_indices = player_indices  # (num_frames,)
        self.frame_features = frame_features  # (num_frames, frame_dim)
        self.targets = targets                # (num_frames, 2)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        pidx = self.player_indices[idx]
        return self.input_seqs[pidx], self.frame_features[idx], self.targets[idx]


class PlayerTracker(nn.Module):
    """LSTM encoder over the input sequence -> MLP decoder per output frame."""

    def __init__(self, seq_dim: int, frame_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.LSTM(
            seq_dim, hidden_dim,
            num_layers=NUM_LAYERS, batch_first=True, dropout=0.1,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + frame_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, input_seq: torch.Tensor, frame_features: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(input_seq)   # h: (num_layers, batch, hidden)
        h = h[-1]                              # take last layer
        return self.decoder(torch.cat([h, frame_features], dim=-1))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_input_sequences(weeks: list[int]) -> dict:
    """Load and normalize raw input CSVs; extract last SEQ_LEN frames per player.

    Returns: {(game_id, play_id, nfl_id): float32 array of shape (SEQ_LEN, 6)}
    Sequences shorter than SEQ_LEN are left-padded by repeating the first frame.
    """
    sequences = {}
    for w in weeks:
        inp = pd.read_csv(TRAIN_DIR / f"input_2023_w{w:02d}.csv")
        inp = _normalize(inp)
        dir_rad = np.radians(inp["dir"])
        inp["vx"] = inp["s"] * np.sin(dir_rad)
        inp["vy"] = inp["s"] * np.cos(dir_rad)
        inp = inp.sort_values("frame_id")

        for (gid, pid, nid), grp in inp.groupby(["game_id", "play_id", "nfl_id"]):
            arr = grp.tail(SEQ_LEN)[SEQ_FEATURES].values.astype(np.float32)
            if len(arr) < SEQ_LEN:
                arr = np.vstack([np.tile(arr[0:1], (SEQ_LEN - len(arr), 1)), arr])
            sequences[(gid, pid, nid)] = arr
        print(f"  Sequences: week {w:2d}")
    return sequences


def prepare_role_data(df: pd.DataFrame, sequences: dict, role: str) -> tuple:
    """Build numpy arrays for one role ready for the dataset.

    Returns: (input_seqs, player_indices, frame_features, targets)
        input_seqs:       (num_players, SEQ_LEN, 6)
        player_indices:   (num_frames,) int — maps each frame to its player
        frame_features:   (num_frames, len(FRAME_FEATURES))
        targets:          (num_frames, 2) — x_actual, y_actual
    """
    role_df = df[df["player_role"] == role].copy()

    # Unique players + integer index
    players = role_df[["game_id", "play_id", "nfl_id"]].drop_duplicates().reset_index(drop=True)
    players["_pidx"] = np.arange(len(players))
    player_keys = list(zip(players["game_id"], players["play_id"], players["nfl_id"]))

    # Stack sequences in player order
    input_seqs = np.stack([sequences[k] for k in player_keys])  # (P, SEQ_LEN, 6)

    # Map each output frame to its player index via merge (vectorised)
    role_df = role_df.merge(
        players[["game_id", "play_id", "nfl_id", "_pidx"]],
        on=["game_id", "play_id", "nfl_id"],
    )
    pidx = role_df["_pidx"].values.astype(np.int64)
    frame_feats = role_df[FRAME_FEATURES].values.astype(np.float32)
    targets = role_df[["x_actual", "y_actual"]].values.astype(np.float32)

    return input_seqs, pidx, frame_feats, targets


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def _standardize(train_seqs, train_feats, val_seqs=None, val_feats=None):
    """Fit on training arrays, optionally transform val arrays too.

    Returns (transformed arrays..., stats_dict).  stats_dict can be passed to
    predict_role() later.
    """
    seq_mean = train_seqs.reshape(-1, train_seqs.shape[-1]).mean(0)
    seq_std = train_seqs.reshape(-1, train_seqs.shape[-1]).std(0) + 1e-8
    feat_mean = train_feats.mean(0)
    feat_std = train_feats.std(0) + 1e-8

    stats = {"seq_mean": seq_mean, "seq_std": seq_std,
             "feat_mean": feat_mean, "feat_std": feat_std}

    train_seqs = (train_seqs - seq_mean) / seq_std
    train_feats = (train_feats - feat_mean) / feat_std

    if val_seqs is not None:
        val_seqs = (val_seqs - seq_mean) / seq_std
        val_feats = (val_feats - feat_mean) / feat_std
        return train_seqs, train_feats, val_seqs, val_feats, stats

    return train_seqs, train_feats, stats


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_role_model(train_data: tuple, val_data: tuple, role: str) -> tuple:
    """Train one LSTM model for a role with early stopping.

    Returns: (model, best_val_rmse, scaler_stats)
    """
    train_seqs, train_pidx, train_feats, train_targets = train_data
    val_seqs, val_pidx, val_feats, val_targets = val_data

    train_seqs, train_feats, val_seqs, val_feats, stats = _standardize(
        train_seqs, train_feats, val_seqs, val_feats
    )

    train_ds = FrameDataset(
        torch.from_numpy(train_seqs), torch.from_numpy(train_pidx),
        torch.from_numpy(train_feats), torch.from_numpy(train_targets),
    )
    val_ds = FrameDataset(
        torch.from_numpy(val_seqs), torch.from_numpy(val_pidx),
        torch.from_numpy(val_feats), torch.from_numpy(val_targets),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = PlayerTracker(len(SEQ_FEATURES), len(FRAME_FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        for seq, feat, tgt in train_loader:
            loss = criterion(model(seq, feat), tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Val ---
        model.eval()
        preds, tgts_list = [], []
        with torch.no_grad():
            for seq, feat, tgt in val_loader:
                preds.append(model(seq, feat).numpy())
                tgts_list.append(tgt.numpy())
        preds_np = np.vstack(preds)
        tgts_np = np.vstack(tgts_list)
        rmse = np.sqrt(((preds_np - tgts_np) ** 2).sum(axis=1).mean())

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            flag = " *"
        else:
            no_improve += 1
            flag = ""

        if epoch % 5 == 0 or flag:
            print(f"    epoch {epoch:3d}: val_rmse={rmse:.4f} (best={best_rmse:.4f}){flag}")

        if no_improve >= PATIENCE:
            print(f"    early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_rmse, stats


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_role(model: PlayerTracker, data: tuple, stats: dict) -> np.ndarray:
    """Run inference for one role.  Returns (n_frames, 2) array of predictions."""
    seqs, pidx, feats, _ = data
    seqs = (seqs - stats["seq_mean"]) / stats["seq_std"]
    feats = (feats - stats["feat_mean"]) / stats["feat_std"]

    ds = FrameDataset(
        torch.from_numpy(seqs.astype(np.float32)),
        torch.from_numpy(pidx),
        torch.from_numpy(feats.astype(np.float32)),
        torch.zeros(len(pidx), 2),  # dummy targets
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for seq, feat, _ in loader:
            preds.append(model(seq, feat).numpy())
    return np.vstack(preds)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(val_df: pd.DataFrame, predictions: dict) -> None:
    """Print RMSE by role and horizon, compared to linear baseline."""
    df = val_df.copy()

    # Stitch predictions back into df in role order
    for role in ["Targeted Receiver", "Defensive Coverage"]:
        mask = df["player_role"] == role
        df.loc[mask, "x_pred"] = predictions[role][:, 0]
        df.loc[mask, "y_pred"] = predictions[role][:, 1]

    df["x_pred"] = df["x_pred"].clip(0, 120)
    df["y_pred"] = df["y_pred"].clip(0, 53.33)

    df["err_sq"] = (df["x_pred"] - df["x_actual"]) ** 2 + (df["y_pred"] - df["y_actual"]) ** 2
    df["err_sq_linear"] = (
        (df["x_linear"] - df["x_actual"]) ** 2
        + (df["y_linear"] - df["y_actual"]) ** 2
    )

    overall = np.sqrt(df["err_sq"].mean())
    linear = np.sqrt(df["err_sq_linear"].mean())

    print(f"\n{'':=<50}")
    print(f"  LSTM RMSE:   {overall:.4f} yards")
    print(f"  Linear RMSE: {linear:.4f} yards")
    print(f"  Improvement: {linear - overall:.4f} yards ({(1 - overall / linear) * 100:.1f}%)")
    print(f"{'':=<50}")

    print(f"\nBy role:")
    print(f"  {'Role':<28} {'LSTM':>8} {'Linear':>8}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8}")
    for role in ["Targeted Receiver", "Defensive Coverage"]:
        mask = df["player_role"] == role
        m = np.sqrt(df.loc[mask, "err_sq"].mean())
        l = np.sqrt(df.loc[mask, "err_sq_linear"].mean())
        print(f"  {role:<28} {m:>8.4f} {l:>8.4f}")

    print(f"\nBy horizon (first 20 frames):")
    print(f"  {'Frame':<10} {'LSTM':>8} {'Linear':>8}")
    print(f"  {'-' * 10} {'-' * 8} {'-' * 8}")
    for fid in range(1, 21):
        mask = df["frame_id"] == fid
        if mask.sum() == 0:
            continue
        m = np.sqrt(df.loc[mask, "err_sq"].mean())
        l = np.sqrt(df.loc[mask, "err_sq_linear"].mean())
        print(f"  {fid:<3} ({fid * 0.1:.1f}s)  {m:>8.4f} {l:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LSTM per-role model")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    print("Loading output data (v2 pipeline: weeks 1-14 train, 15-18 val)...")
    train_df = load_weeks(TRAIN_WEEKS)
    val_df = load_weeks(VAL_WEEKS)
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows\n")

    print("Loading input sequences...")
    train_seqs = load_input_sequences(TRAIN_WEEKS)
    val_seqs = load_input_sequences(VAL_WEEKS)
    print()

    predictions = {}

    for role in ["Targeted Receiver", "Defensive Coverage"]:
        print(f"Training {role}...")
        train_data = prepare_role_data(train_df, train_seqs, role)
        val_data = prepare_role_data(val_df, val_seqs, role)

        model, best_rmse, stats = train_role_model(train_data, val_data, role)
        print(f"  {role}: best val RMSE = {best_rmse:.4f}\n")

        predictions[role] = predict_role(model, val_data, stats)

    print("\nOverall evaluation (val set):")
    evaluate(val_df, predictions)


if __name__ == "__main__":
    main()
