
# Helpers & EDA Unit Tests

This test suite verifies the functionality of **data helpers** and the **EDA pipeline** in this project. It ensures that:

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

   ```bash
   .venv\Scripts\activate
    ```

2. **Install `pytest`:**

   ```bash
   pip install pytest
    ```

3. **Run the tests** from the project root:

   ```bash
   pytest tests/test_helpers_eda.py
   ```

4. **Expected output**:

   ```
   ============================= test session starts ==============================
   collected 4 items

   tests/test_helpers_eda.py ....                                          [100%]

   ============================== 4 passed in 0.XXXs ===============================
   ```

---

### Notes

* `test_load_raw_data` uses a temporary path to check that missing files raise a `FileNotFoundError`.
* `test_eda_helper` runs `EDAHelper` on a small example DataFrame to validate numeric, categorical, and outlier summaries.
* All tests are written with **pytest**

