# Notebooks

This folder contains Jupyter notebooks for **exploration and analysis** of the project data. Notebooks use the reusable code in `src/` for consistent workflows.

## Contents

- **eda.ipynb**: Runs the full exploratory data analysis (EDA) on the raw dataset using the `EDAHelper` class and utility functions from `src/utils/helpers.py`.

## Usage

1. Ensure your virtual environment is activated and dependencies are installed.
2. Make sure the raw data is available in the directory defined by `Config.DATA_DIR`.
3. Open `ena.ipynb` in Jupyter or VSCode.
4. The notebook will:
   - Load the raw data using `load_raw_data()` from helpers.
   - Ensure data directories exist with `ensure_dirs()` from helpers.
   - Run structured EDA using `EDAHelper`
   - Generate summaries for numeric and categorical columns, missing values, correlations, and outliers.

```python
from src.utils.helpers import load_raw_data, ensure_dirs
from src.eda.eda import EDAHelper

ensure_dirs()
df = load_raw_data("data.csv")
eda = EDAHelper(df)
results = eda.run_all(top_n=5)
