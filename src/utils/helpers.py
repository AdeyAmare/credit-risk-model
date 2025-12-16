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
    Load a raw CSV file from the raw data directory with basic error handling.
    
    Parameters:
        filename (str): Name of the CSV file to load.
    
    Returns:
        pd.DataFrame: Loaded data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = get_raw_data_path(filename)

    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found at path '{path}'.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading '{filename}': {e}")
        raise