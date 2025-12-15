# predict.py
import pandas as pd
import joblib
import argparse
import logging
from typing import Optional

# -------------------- LOGGING CONFIG --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Class for loading a trained model and making predictions on new data.

    Attributes:
        model_path (str): Path to the trained model file (.pkl).
        data_path (str): Path to the input CSV data.
        df (pd.DataFrame): Loaded input data with predictions appended.
    """

    def __init__(self, model_path: str, data_path: str):
        self.model_path: str = model_path
        self.data_path: str = data_path
        self.df: Optional[pd.DataFrame] = None
        self.model = None

    def load_data(self) -> None:
        """
        Load CSV data for prediction.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            KeyError: If required columns are missing.
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            raise

        # Keep CustomerId for reference
        if 'CustomerId' not in self.df.columns:
            logger.warning("CustomerId column not found in data.")
            self.customer_ids = None
        else:
            self.customer_ids = self.df['CustomerId']

        # Features for prediction (drop CustomerId and target if exists)
        self.X = self.df.drop(columns=['CustomerId', 'is_high_risk'], errors='ignore')

    def load_model(self) -> None:
        """
        Load a trained model from disk.

        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: For other errors during loading.
        """
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self) -> None:
        """
        Make predictions using the loaded model.

        Adds:
            - 'predicted_risk' column with model predictions.
            - 'predicted_risk_prob' column with probability if model supports it.
        """
        if self.model is None or self.X is None:
            raise RuntimeError("Model and data must be loaded before predicting.")

        try:
            self.df['predicted_risk'] = self.model.predict(self.X)
            logger.info("Predictions generated successfully.")

            # Add probability if model supports it
            if hasattr(self.model, 'predict_proba'):
                self.df['predicted_risk_prob'] = self.model.predict_proba(self.X)[:, 1]
                logger.info("Prediction probabilities added.")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def save_predictions(self, output_path: str = 'predictions.csv') -> None:
        """
        Save predictions to CSV.

        Args:
            output_path (str): Path to save the CSV file.
        """
        if self.df is None:
            raise RuntimeError("No predictions to save. Run predict() first.")

        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            raise


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pkl)")
    parser.add_argument('--data_path', type=str, required=True, help="Path to input CSV data")
    parser.add_argument('--output_path', type=str, default='predictions.csv', help="Path to save predictions CSV")
    args = parser.parse_args()

    try:
        predictor = ModelPredictor(model_path=args.model_path, data_path=args.data_path)
        predictor.load_data()
        predictor.load_model()
        predictor.predict()
        predictor.save_predictions(output_path=args.output_path)
        logger.info("Prediction pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Prediction pipeline failed: {e}")
