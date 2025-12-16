"""
Prediction Script for Credit Risk Model

This module provides a command-line interface and supporting classes
to load a trained machine learning model, apply it to new transactional
feature data, and generate credit risk predictions.

Key responsibilities:
- Load and validate input feature data from CSV
- Load a serialized ML model (singleton pattern)
- Generate class predictions and probabilities
- Persist prediction results to disk

The script is designed for batch inference and production-safe execution,
with extensive logging, error handling, and static typing.
"""

import pandas as pd
import joblib
import argparse
import logging
from typing import Optional, Any
from pathlib import Path

# -------------------- LOGGING CONFIG --------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- CONSTANTS --------------------
FEATURE_COLUMNS = [
    "Amount_sum",
    "Amount_mean",
    "Amount_count",
    "Amount_std",
    "transactionid_transactionid_76871_ratio",
    "transactionid_transactionid_73770_ratio",
    "transactionid_transactionid_26203_ratio",
    "batchid_batchid_67019_ratio",
    "batchid_batchid_51870_ratio",
    "batchid_batchid_113893_ratio",
    "accountid_accountid_4841_ratio",
    "accountid_accountid_4249_ratio",
    "accountid_accountid_4840_ratio",
    "subscriptionid_subscriptionid_3829_ratio",
    "subscriptionid_subscriptionid_4429_ratio",
    "subscriptionid_subscriptionid_1372_ratio",
    "customerid_customerid_7343_ratio",
    "customerid_customerid_3634_ratio",
    "customerid_customerid_647_ratio",
    "currencycode_ugx_ratio",
    "providerid_providerid_4_ratio",
    "providerid_providerid_6_ratio",
    "providerid_providerid_5_ratio",
    "productid_productid_6_ratio",
    "productid_productid_3_ratio",
    "productid_productid_10_ratio",
    "productcategory_financial_services_ratio",
    "productcategory_airtime_ratio",
    "productcategory_utility_bill_ratio",
    "channelid_channelid_3_ratio",
    "channelid_channelid_2_ratio",
    "channelid_channelid_5_ratio",
    "recency",
    "frequency",
    "monetary",
]

# -------------------- MODEL SINGLETON --------------------
_model: Optional[Any] = None


def load_model_once(model_path: str) -> Any:
    """
    Load a trained machine learning model from disk once and reuse it.

    This function implements a singleton pattern to prevent repeated
    disk I/O when predictions are made multiple times in the same process.

    Args:
        model_path (str): Path to the serialized model file (.pkl).

    Returns:
        Any: Loaded machine learning model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        Exception: If deserialization fails.
    """
    global _model

    try:
        if _model is None:
            logger.info("Loading model from disk...")
            model_file = Path(model_path)

            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            _model = joblib.load(model_file)
            logger.info(f"Model loaded successfully from {model_path}")

        return _model

    except Exception as exc:
        logger.error("Failed to load model", exc_info=True)
        raise exc


# -------------------- PREDICTOR CLASS --------------------
class ModelPredictor:
    """
    Encapsulates the full prediction workflow for a trained ML model.

    Responsibilities:
    - Load and validate feature data
    - Load a trained model
    - Generate predictions and probabilities
    - Save results to disk
    """

    def __init__(self, model_path: str, data_path: str) -> None:
        """
        Initialize the predictor with file paths.

        Args:
            model_path (str): Path to the trained model file.
            data_path (str): Path to the CSV file containing feature data.
        """
        self.model_path: str = model_path
        self.data_path: str = data_path
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.model: Optional[Any] = None
        self.customer_ids: Optional[pd.Series] = None

    def load_data(self) -> None:
        """
        Load input data from CSV and prepare the feature matrix.

        Ensures that all required feature columns exist and are ordered
        correctly for model inference.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If required feature columns are missing.
            Exception: If CSV loading fails.
        """
        try:
            logger.info("Loading input data...")
            data_file = Path(self.data_path)

            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found at {self.data_path}")

            self.df = pd.read_csv(data_file)
            logger.info("Input data loaded successfully")

        except Exception as exc:
            logger.error("Failed to load input data", exc_info=True)
            raise exc

        self.customer_ids = self.df.get("CustomerId")

        missing_features = [col for col in FEATURE_COLUMNS if col not in self.df.columns]
        if missing_features:
            logger.error(f"Missing required feature columns: {missing_features}")
            raise ValueError(f"Missing required feature columns: {missing_features}")

        self.X = self.df[FEATURE_COLUMNS]
        logger.info("Feature matrix prepared successfully")

    def load_model(self) -> None:
        """
        Load the trained model using the singleton loader.

        Raises:
            Exception: If model loading fails.
        """
        try:
            logger.info("Loading trained model...")
            self.model = load_model_once(self.model_path)
        except Exception as exc:
            logger.error("Model loading failed", exc_info=True)
            raise exc

    def predict(self) -> None:
        """
        Generate predictions and prediction probabilities.

        Adds the following columns to the DataFrame:
        - predicted_risk
        - predicted_risk_prob (if supported by the model)

        Raises:
            RuntimeError: If model or data has not been loaded.
            Exception: If prediction fails.
        """
        if self.model is None or self.X is None or self.df is None:
            raise RuntimeError("Model and data must be loaded before prediction")

        try:
            logger.info("Generating predictions...")
            self.df["predicted_risk"] = self.model.predict(self.X)

            if hasattr(self.model, "predict_proba"):
                self.df["predicted_risk_prob"] = self.model.predict_proba(self.X)[:, 1]
            else:
                self.df["predicted_risk_prob"] = None

            logger.info("Predictions generated successfully")

        except Exception as exc:
            logger.error("Prediction failed", exc_info=True)
            raise exc

    def save_predictions(self, output_path: str = "predictions.csv") -> None:
        """
        Save predictions to a CSV file.

        Args:
            output_path (str): Path where the predictions CSV will be saved.

        Raises:
            RuntimeError: If predictions have not been generated.
            Exception: If file writing fails.
        """
        if self.df is None:
            raise RuntimeError("No predictions to save")

        try:
            logger.info(f"Saving predictions to {output_path}...")
            self.df.to_csv(output_path, index=False)
            logger.info("Predictions saved successfully")

        except Exception as exc:
            logger.error("Failed to save predictions", exc_info=True)
            raise exc


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    """
    Command-line entry point for batch prediction execution.
    """
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model (.pkl)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input CSV data"
    )
    parser.add_argument(
        "--output_path", type=str, default="predictions.csv", help="Output CSV path"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting prediction pipeline...")
        predictor = ModelPredictor(model_path=args.model_path, data_path=args.data_path)
        predictor.load_data()
        predictor.load_model()
        predictor.predict()
        predictor.save_predictions(output_path=args.output_path)
        logger.info("Prediction pipeline completed successfully")

    except Exception as exc:
        logger.critical("Prediction pipeline failed", exc_info=True)
        raise exc