# src

The `src` folder contains all the **core Python code** for this project, including utilities, configuration, and modules for exploratory data analysis.

## Structure

```plaintext
src/
├── config/ # Project configuration (paths, constants, environment settings)
├── eda/ # Exploratory Data Analysis helper class
└── utils/ # Utility functions for data loading, path management, etc.
```

## Purpose

- **Centralized code**: Keeps all reusable Python code separate from notebooks and data.  
- **Reusability**: Functions and classes can be imported across scripts and notebooks.  
- **Consistency**: Standardizes paths, configuration, and data handling.

## Usage

```python
# Import configuration
from config.config import Config

# Load data using helpers
from utils.helpers import load_raw_data, ensure_dirs

# Run EDA
from eda.eda import EDAHelper
```

Keep notebooks in the notebooks/ folder and raw data in data/raw/. Use utils and config to manage paths and files consistently.