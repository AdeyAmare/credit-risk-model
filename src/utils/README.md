# Helpers Module

`helpers.py` provides utility functions for **file paths and data loading**, centralizing common operations for working with raw project data.

## Features

- **get_raw_data_path(filename)**: Returns the full path to a raw data file in the project’s `DATA_DIR`.  
- **ensure_dirs()**: Creates the raw data directory if it doesn’t exist.  
- **load_raw_data(filename)**: Loads a CSV file from the raw data directory into a pandas DataFrame.

## Usage

```python
from src.utils.helpers import get_raw_data_path, ensure_dirs, load_raw_data

# Ensure data directory exists
ensure_dirs()

# Get full path to a file
path = get_raw_data_path("data.csv")

# Load CSV data
df = load_raw_data("data.csv")
