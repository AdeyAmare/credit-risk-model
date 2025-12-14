import os
from pathlib import Path
import pandas as pd
import pytest
from src.utils.helpers import get_raw_data_path, ensure_dirs, load_raw_data
from src.eda.eda import EDAHelper
from src.config.config import Config


def test_get_raw_data_path():
    """Check that get_raw_data_path returns a Path object."""
    path = get_raw_data_path("data.csv")
    assert isinstance(path, Path)


def test_ensure_dirs():
    """Check that ensure_dirs creates the folder if it doesn't exist."""
    ensure_dirs()
    assert Config.DATA_DIR.exists()


def test_load_raw_data(tmp_path):
    """Test that load_raw_data raises FileNotFoundError for missing file."""
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_file)


def test_eda_helper():
    """Test EDAHelper methods on an example DataFrame."""
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 1000],
        "cat": ["a", "b", "a", "b", "a"]
    })
    eda = EDAHelper(df)
    results = eda.run_all()

    # Simple checks
    assert results["overview"]["n_rows"] == 5
    assert results["overview"]["n_cols"] == 2
    assert "num" in results["numeric_summary"].index
    assert "cat" in results["categorical_summary"]
    assert "Outliers Count" in results["outlier_summary"].columns
