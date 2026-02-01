"""
Central configuration for data paths.

If your data lives on an external drive, point DATA_DIR here.
Example: DATA_DIR = Path("/mnt/external/nfl_big_data/data")
"""

from pathlib import Path

# Root of this repo
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directory — change this if your data is on an external drive
DATA_DIR = PROJECT_ROOT / "data"

# Derived paths
TRAIN_DIR = DATA_DIR / "train"
TEST_INPUT = DATA_DIR / "test_input.csv"
TEST_CSV = DATA_DIR / "test.csv"
