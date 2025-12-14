from src.config.config import Config
from pathlib import Path
import pandas as pd

# -----------------------------
# Path helpers
# -----------------------------

def get_raw_data_path(filename="data.csv") -> Path:
    """
    Return the full path to a raw data file.
    """
    return Config.DATA_DIR / filename


def ensure_dirs():
    """
    Ensure that the raw data directory exists.
    """
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Data loading
# -----------------------------

def load_raw_data(filename="data.csv") -> pd.DataFrame:
    """
    Load a raw CSV file from the raw data directory.
    """
    path = get_raw_data_path(filename)
    return pd.read_csv(path)
