# EDA Module

`EDAHelper` is a utility class for performing **structured exploratory data analysis (EDA)** on a pandas DataFrame **without plotting or printing**. All methods return Python objects like DataFrames or dictionaries, making it easy to integrate into scripts or pipelines.

## Features

- **overview**: Basic dataset info (rows, columns, dtypes)
- **missing_values**: Counts and percentages of missing values
- **numeric_summary**: Descriptive statistics for numeric columns
- **categorical_summary**: Top N value counts for categorical columns
- **correlation_matrix**: Correlation matrix of numeric columns
- **outlier_summary**: Outlier statistics using the IQR method
- **run_all**: Run all EDA steps and return a dictionary of results

## Usage

```python
from src.eda.eda import EDAHelper
from src.utils.helpers import load_raw_data, ensure_dirs
import pandas as pd

# Ensure data directory exists
ensure_dirs()

# Load data using helpers
df = load_raw_data("data.csv")

# Initialize EDA helper
eda = EDAHelper(df)

# Run individual components
overview = eda.overview()
missing = eda.missing_values()
numeric_stats = eda.numeric_summary()
categorical_stats = eda.categorical_summary(top_n=5)
correlation = eda.correlation_matrix()
outliers = eda.outlier_summary()

# Or run everything at once
results = eda.run_all(top_n=5)
