
#  Unit Tests

This test suite verifies the functionality of **data helpers**, the **EDA pipeline**, **Feature Engieering** and **Target Engineering** in this project. It ensures that:

1. File paths are correctly generated.
2. Data directories are created if missing.
3. CSV files can be loaded properly.
4. The `EDAHelper` class produces correct summary statistics.
5. **Target engineering and feature engineering pipelines** work as expected, including:
   - RFM calculation
   - High-risk customer generation
   - Time-based feature extraction
   - Full feature engineering pipeline with numeric and categorical processing

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
- **`src.target_engineering.ProxyTargetGenerator`** — generates customer-level targets, including RFM and high-risk indicators.
- **`src.data_processing.TimeFeaturesExtractor`** — extracts datetime-based features from transaction data.
- **`src.data_processing.FeatureEngineeringPipeline`** — performs full feature engineering, including numeric scaling, categorical encoding, and top-k selection.

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
   pytest tests/test_data_processing.py

   ```

4. **Expected output**:

  ============================= test session starts ==============================
collected 8 items

tests/test_helpers_eda.py ....                                          [100%]
tests/test_target_feature_engineering.py ....                            [100%]

============================== 8 passed in 0.XXXs ==============================


---

### Notes

* `test_load_raw_data` uses a temporary path to check that missing files raise a `FileNotFoundError`.

* `test_eda_helper` runs `EDAHelper` on a small example DataFrame to validate numeric, categorical, and outlier summaries.

* `test_calculate_rfm validates` that RFM features are correctly calculated per customer.

* `test_generate` verifies high-risk customer generation with clustering.

* `test_time_features_extractor` ensures datetime features are correctly extracted.

* `test_feature_engineering_pipeline` validates numeric and categorical feature processing in the full pipeline.

* All tests are written with **pytest**

