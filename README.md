# NFL Big Data Bowl 2026 — Prediction Competition

Predict player movement during NFL pass plays while the ball is in the air.

## What we're predicting

Given pre-snap and pre-throw tracking data (positions, speed, acceleration, orientation) plus the ball landing location, predict the (x, y) position of each player at every frame after the ball is released until it arrives.

Evaluated on RMSE between predicted and actual positions.

## Project layout

```
config.py              — central data path config (change this if data moves)
notebooks/             — exploration and analysis
scripts/               — training and prediction scripts
kaggle_evaluation/     — Kaggle-provided evaluation framework (do not modify)
data/                  — training and test CSVs (gitignored, lives on external drive)
```

## Data setup

Data is large (~1 GB). It is not tracked in git. If your data is on an external drive, edit `config.py` and point `DATA_DIR` at the right path. Everything imports paths from there.

## Environment

Uses `uv` for dependency management.

```bash
uv sync          # install dependencies
uv run jupyter   # launch notebooks with the right venv
```

## Approach

Approaches are listed roughly in order of implementation. Status updates added as we go.

### 1. Linear extrapolation baseline
Project each player forward using their pre-throw position, speed, direction, and acceleration. Straight-line kinematics, no ML. Purpose: establish a floor for RMSE and see how much player behavior actually deviates from simple physics after the ball is thrown.

**Status:** done — `scripts/baseline_linear.py`

Results (all 18 training weeks, 562,936 output frames):
- Overall RMSE: **2.41 yards**
- Targeted Receiver: 1.99 yards
- Defensive Coverage: 2.56 yards
- Error at 1.0s: 2.07 yards | at 2.0s: 5.30 yards (grows roughly linearly with horizon)

Note: Defenders are harder to predict with straight lines than receivers. Receivers run relatively direct routes toward the ball; defenders react and change direction more.

### 2. Per-player-role models
Players have fundamentally different jobs on each play (Targeted Receiver, Defensive Coverage, Other Route Runner, Passer). Their post-throw movement patterns differ accordingly. Split the prediction problem by `player_role` and build separate models per role. The targeted receiver is running toward the ball; defenders are reacting; route runners are finishing their routes.

**Status:** done — `scripts/model_per_role.py`

Results (LightGBM, weeks 1–14 train / 15–18 val):
- Overall RMSE: **1.45 yards** vs linear baseline 2.42 (40% improvement)
- Targeted Receiver: 0.90 vs 1.90 (53% improvement)
- Defensive Coverage: 1.62 vs 2.60 (38% improvement)

Key findings:
- The linear baseline prediction (`x_linear`, `y_linear`) is by far the most important feature — the model is learning to correct the straight-line projection.
- `ball_land_y` is the 4th most important feature, confirming the ball landing spot is doing real work (especially for receivers).
- The model is worse than linear at very short horizons (frames 1–3, < 0.3s). At those timescales the player has barely moved and straight-line projection is nearly exact. Not worth special-casing for now.
- Error still grows with horizon but much more slowly than linear: 1.21 yards at 1s vs 2.06 for linear; 3.09 at 2s vs 5.32.

### 3. Gradient boosting with trajectory and contextual features
Builds on Approach 2. Adds two feature categories: (a) trajectory summaries from the last 10 input frames (speed trend, direction stability, smoothed velocity), and (b) contextual features encoding spatial relationships between players on the same play (distance/relative velocity to targeted receiver, geometry relative to the TR-to-ball line, defender pressure on the receiver).

**Status:** done — `scripts/model_per_role_v2.py`

Results (LightGBM, weeks 1–14 train / 15–18 val, with per-role tuning + short-horizon blend):
- Overall RMSE: **1.199 yards** vs v1's 1.45 (17% improvement over v1, 50.5% over linear)
- Targeted Receiver: 0.769 vs v1's 0.90
- Defensive Coverage: 1.332 vs v1's 1.62

Tuning details:
- Per-role grid search over learning_rate ∈ {0.02, 0.05, 0.1} and num_leaves ∈ {31, 63, 127}. Both roles converged on lr=0.02, num_leaves=31 (simpler, slower-learning trees).
- Early stopping with patience=100, max 5000 rounds. TR models stop at ~3500-4100 rounds; DC at ~4700-5000.
- Short-horizon blend: frames 1–2 use pure linear prediction, frame 3 is 50/50 linear+model, frame 4+ is pure model. Fixes the model's tendency to overshoot at very short horizons.

Key findings:
- Defenders benefited most from the new features. Knowing where the TR is and the geometry of the TR-to-ball line helps predict how defenders react.
- `y_vel_smooth` (trajectory) enters the top-15 feature importance — the pre-throw trajectory is contributing, particularly for y-direction prediction.
- The contextual features don't dominate individual importance rankings but are clearly driving the defender improvement.
- Tuning + blend together add ~3.3% over the untuned v2 baseline (1.24 → 1.199).

### 4. Two-stage prediction (TR predictions → DC features)
Train TR models first, then feed their per-frame predicted positions into the DC model as additional features (`tr_x_pred`, `tr_y_pred`, `dist_to_tr_pred`, `dx_to_tr_pred`, `dy_to_tr_pred`). The idea: defenders react to where the receiver is *going*, not just where they were at throw time.

**Status:** done — integrated into `scripts/model_per_role_v2.py`

Results (val set):
- DC RMSE: **1.335** vs 1.332 without stacking — essentially flat.
- `tr_x_pred` ranks 4th in DC feature importance (108M avg gain), so the model does use it.

Key findings:
- The throw-time contextual features (`dist_to_tr`, `dx_to_tr`, relative velocity) already capture most of what the DC model needs. The predicted future TR position adds marginal signal on top.
- In-sample TR predictions during DC training introduce a slight optimism bias (TR was trained on the same data). Cross-validated TR predictions could close this gap but the marginal benefit doesn't justify the added complexity.
- This is a clean null result at this margin. The stacking infrastructure is in place if the TR model improves significantly in the future.

### 5. Sequence models (LSTM)
2-layer LSTM encoder over the last 10 input frames, with a per-output-frame MLP decoder. The encoder learns a hidden representation of the player's pre-throw trajectory; the decoder combines it with per-frame physics features (linear baseline, dist-to-ball, etc.) to predict (x, y). Trained separately per role.

**Status:** done — `scripts/model_lstm.py`

Results (val set, 50 epochs, no short-horizon blend):
- Overall RMSE: **1.670 yards** vs v2's 1.199 — not competitive.
- Targeted Receiver: 1.486 vs v2's 0.769
- Defensive Coverage: 1.738 vs v2's 1.335
- Beats linear starting at frame 6 (0.77 vs 0.80 yards), but is catastrophically worse at frames 1–5 (1.04 at frame 1 vs linear's 0.03).

Key findings:
- The LSTM does not learn to leverage the linear baseline features (`x_linear`, `y_linear`) effectively, despite them being in its input. LightGBM picks this up immediately and treats them as the dominant correction target.
- Both models were still improving at epoch 50 (DC never triggered early stopping), so more training would help at the margin — but the gap to v2 is too large to close with epochs alone.
- The short-horizon failure is severe: at very short horizons the player hasn't changed course yet, so linear extrapolation is nearly exact. The LSTM predicts some learned "average" trajectory instead.
- Verdict: on this dataset, well-engineered gradient boosting with physics-informed features dominates a vanilla LSTM. A sequence model could potentially compete with a skip-connection architecture (direct residual on the linear baseline) or attention, but that's beyond the scope here.

### Summary table (val set, weeks 15–18)

| Approach | Overall | TR | DC | Notes |
|---|---|---|---|---|
| Linear baseline | 2.41 | 1.99 | 2.56 | Physics extrapolation |
| Per-role LightGBM v1 | 1.45 | 0.90 | 1.62 | + physics features |
| Per-role LightGBM v2 | 1.199 | 0.769 | 1.335 | + trajectory, context, tuning, blend |
| v2 + two-stage stacking | 1.202 | 0.769 | 1.335 | TR preds → DC (null) |
| LSTM | 1.670 | 1.486 | 1.738 | Sequence model |
