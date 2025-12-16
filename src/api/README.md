
# API (`src/api`)

The `src/api` module exposes the trained credit risk model as a **FastAPI service** for real-time predictions.

## Purpose

* Serve the trained model via HTTP
* Validate inputs using Pydantic schemas
* Return credit risk predictions and probabilities
* Support low-latency, production inference

## Structure

```plaintext
src/api/
├── main.py               # FastAPI app and /predict endpoint
└── pydantic_models.py    # Request and response schemas
```

## Overview

* `pydantic_models.py` defines strict input and output schemas for prediction requests.
* `main.py` loads the trained model once and exposes a `/predict` endpoint that returns risk classification results.

This API complements batch prediction and training pipelines in `src/` and is intended for deployment and system integration.
