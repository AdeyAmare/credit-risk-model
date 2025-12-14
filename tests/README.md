# Helpers & EDA Unit Tests

This test suite verifies the functionality of the **data helpers** and **EDA pipeline** in this project. It ensures that:

1. File paths are correctly generated.
2. Data directories are created if missing.
3. CSV files can be loaded properly.
4. The `EDAHelper` class produces correct summary statistics.

---

## Tested Components

- **`src.config.config.Config`** — project configuration and data directory.
- **`src.utils.helpers`** — helper functions:
  - `get_raw_data_path(filename)` — returns the full path to a raw data file.
  - `ensure_dirs()` — ensures that the data folder exists.
  - `load_raw_data(filename)` — loads a CSV into a pandas DataFrame.
- **`src.eda.eda.EDAHelper`** — performs structured exploratory data analysis on a DataFrame, including:
  - Overview (rows, columns, dtypes)
  - Numeric summary
  - Categorical summary
  - Missing values
  - Correlation matrix
  - Outlier detection (IQR method)

---

## Running the Tests

1. **Activate your virtual environment**:

   ```cmd
   .venv\Scripts\activate
   ```
2. Run the tests using unittest:

From the project root:
```cmd
python -m unittest tests/test_helpers_eda.py
```
Or directly:
```cmd
python tests/test_helpers_eda.py
```
3. Expected output:
...
----------------------------------------------------------------------
Ran 3 tests in 0.XXXs

OK
