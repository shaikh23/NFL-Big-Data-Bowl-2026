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

Results (LightGBM, weeks 1–14 train / 15–18 val):
- Overall RMSE: **1.24 yards** vs v1's 1.45 (15% improvement over v1, 49% over linear)
- Targeted Receiver: 0.79 vs v1's 0.90
- Defensive Coverage: 1.37 vs v1's 1.62 (defenders improved more — contextual features are most relevant here)

Key findings:
- Defenders benefited most from the new features. Knowing where the TR is and the geometry of the TR-to-ball line helps predict how defenders react.
- `y_vel_smooth` (trajectory) enters the top-15 feature importance — the pre-throw trajectory is contributing, particularly for y-direction prediction.
- The contextual features don't dominate individual importance rankings but are clearly driving the defender improvement.
- Short-horizon overshoot improved (0.28 at frame 1, down from 0.38 in v1) but still above linear's 0.03.

### 4. Physics-informed feature engineering
Engineer features that encode the relationship between each player and the ball landing location: distance, angle relative to current direction of motion, time-to-arrival given current speed, etc. These are especially high-signal for the targeted receiver. Feed these into whatever model is in use.

**Status:** pending

### 5. Sequence models (LSTM / Transformer)
Input is a time series, output is a time series. A small LSTM or Transformer could learn temporal movement patterns that flat features miss. Higher engineering cost, may not beat well-tuned gradient boosting on this dataset size. Stretch goal.

**Status:** pending
