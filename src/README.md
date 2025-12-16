# src

The `src` folder contains all **core Python code** for this project.
It implements the full machine learning pipeline — from data preparation and target engineering to model training and batch prediction — in a reusable, testable manner.

## Structure

```plaintext
src/
├── config/
│   └── config.py            # Centralized configuration (paths, constants)
│
├── utils/
│   └── helpers.py           # Data loading, directory creation, utilities
│
├── eda/
│   └── eda.py               # EDAHelper class for structured exploratory analysis
│
├── target_engineering.py    # RFM computation, clustering, proxy target generation
│
├── train.py                 # Model training, evaluation, and persistence
│
├── predict.py               # Batch inference and prediction pipeline.
```

## Purpose

- **Centralized code**: Keeps all reusable Python code separate from notebooks and data.  
- **Reusability**: Functions and classes can be imported across scripts and notebooks.  
- **Consistency**: Standardizes paths, configuration, and data handling.

## Module Overview
1. `config/`

- Defines the Config class
- Central source of truth for:
    - Data directories
    - Model directories
    - File paths and constants

Used throughout the project to avoid hard-coded paths.

2. `utils/`

General-purpose helper functions, including:

- Loading CSV data safely
- Ensuring required directory structure exists
- Shared utilities used across EDA, training, and prediction

3. `eda/`

Contains the EDAHelper class used to perform structured exploratory data analysis, including:

- Numeric and categorical summaries
- Missing value analysis
- Correlations and outlier detection
- Primarily used by notebooks, but fully reusable as a Python module.

3. `target_engineering.py`

Implements proxy target generation using RFM analysis:

- Computes Recency, Frequency, Monetary (RFM) metrics
- Scales RFM features
- Performs customer clustering
- Identifies high-risk customer segments
- Produces a binary is_high_risk target

This module bridges raw transactional data and supervised modeling.

4. `train.py`

Handles model training and evaluation, including:

- Loading processed feature data
- Train / test splitting
- Training multiple models
- Evaluating models with standard classification metrics
- Persisting the best-performing model to disk

Designed to be run both programmatically and as part of automated workflows.

5. `predict.py`

Provides a production-safe batch prediction pipeline, including:

- Loading a trained model (singleton pattern)
- Validating input feature data
- Generating class predictions and probabilities
- Saving prediction outputs to CSV
- Command-line interface for inference jobs

This module is intended for deployment and downstream consumption.

## Usage

```python
from src.config.config import Config
from src.utils.helpers import load_raw_data
from src.eda.eda import EDAHelper
from src.target_engineering import ProxyTargetGenerator
from src.train import ModelTrainer
from src.predict import ModelPredictor

```

## Project Conventions

- Notebooks → notebooks/ (exploration, visualization, experimentation)

- Core logic → src/

- Raw data → data/raw/

- Processed data → data/processed/

- Trained models → models/

Notebooks should only orchestrate workflows using code from src, never reimplement logic.