"""
End-to-end training pipeline for high-risk customer classification.

This script:
- Loads processed modeling data from disk
- Trains multiple classification models using GridSearchCV
- Evaluates models using standard classification metrics
- Logs experiments, parameters, and metrics to MLflow
- Selects and persists the best-performing model based on ROC-AUC

Models trained:
- RandomForestClassifier (no scaling)
- LogisticRegression (with StandardScaler pipeline)

Artifacts:
- MLflow experiment runs
- Serialized best model saved as `best_model.pkl`
"""

import os
import joblib
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.config.config import Config


# -------------------- LOGGING CONFIG --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training, evaluation, and persistence of classification models
    for high-risk customer prediction.

    Responsibilities:
    - Load and split modeling data
    - Train models using hyperparameter tuning
    - Evaluate models on a hold-out test set
    - Log experiments and artifacts using MLflow
    - Select and save the best-performing model

    Attributes:
        data_path (str): Path to the processed modeling dataset
        random_state (int): Random seed for reproducibility
        save_dir (str): Directory where models and MLflow artifacts are stored
        models (Dict[str, object]): Trained models keyed by model name
        best_params (Dict[str, Dict]): Best hyperparameters per model
    """

    def __init__(
        self,
        data_path: str,
        random_state: int = 42,
        save_dir: str = Config.models_dir
    ):
        """
        Initialize the ModelTrainer.

        Args:
            data_path (str): Path to the processed CSV dataset
            random_state (int): Random seed for reproducibility
            save_dir (str): Directory to store trained models and MLflow data
        """
        self.data_path = data_path
        self.random_state = random_state
        self.save_dir = save_dir
        self.models: Dict[str, object] = {}
        self.best_params: Dict[str, Dict] = {}

        try:
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Model save directory ready: {self.save_dir}")
        except OSError as e:
            logger.critical(f"Failed to create model directory: {e}")
            raise

    # -------------------- DATA LOADING --------------------
    def load_data(self) -> None:
        """
        Load the modeling dataset from disk and perform a stratified train-test split.

        Expects the dataset to contain:
        - 'CustomerId' column (dropped)
        - 'is_high_risk' target column

        Raises:
            FileNotFoundError: If the data file does not exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(self.data_path):
            logger.critical(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Missing data file: {self.data_path}")

        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded from {self.data_path}")
        except Exception as e:
            logger.critical(f"Failed to read CSV file: {e}")
            raise

        required_columns = {"CustomerId", "is_high_risk"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logger.critical(f"Missing required columns: {missing}")
            raise ValueError(f"Dataset missing columns: {missing}")

        self.X = df.drop(columns=["CustomerId", "is_high_risk"])
        self.y = df["is_high_risk"]

        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                stratify=self.y,
                random_state=self.random_state
            )
            logger.info("Train-test split completed")
        except Exception as e:
            logger.critical(f"Train-test split failed: {e}")
            raise

    # -------------------- MODEL TRAINING --------------------
    def train_models(self) -> None:
        """
        Train classification models using GridSearchCV.

        Models:
        - RandomForestClassifier (no feature scaling)
        - LogisticRegression (with StandardScaler pipeline)

        Hyperparameter tuning is performed using ROC-AUC as the scoring metric.
        """
        logger.info("Starting model training")

        # -------- Random Forest --------
        logger.info("Training RandomForestClassifier")

        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced"
        )

        rf_params = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        }

        rf_grid = GridSearchCV(
            rf,
            param_grid=rf_params,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )

        rf_grid.fit(self.X_train, self.y_train)

        self.models["RandomForest"] = rf_grid.best_estimator_
        self.best_params["RandomForest"] = rf_grid.best_params_

        logger.info(f"RF best params: {rf_grid.best_params_}")

        # -------- Logistic Regression --------
        logger.info("Training LogisticRegression with scaling")

        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            ))
        ])

        lr_params = {
            "classifier__C": [0.01, 0.1, 1, 10]
        }

        lr_grid = GridSearchCV(
            lr_pipeline,
            param_grid=lr_params,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )

        lr_grid.fit(self.X_train, self.y_train)

        self.models["LogisticRegression"] = lr_grid.best_estimator_
        self.best_params["LogisticRegression"] = lr_grid.best_params_

        logger.info(f"LR best params: {lr_grid.best_params_}")

    # -------------------- EVALUATION + MLFLOW --------------------
    def evaluate_and_log(self) -> None:
        """
        Evaluate trained models on the test set and log results to MLflow.

        Metrics logged:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC-AUC

        Also:
        - Selects the best model based on ROC-AUC
        - Registers the best model in MLflow
        - Saves the best model locally as `best_model.pkl`
        """
        logger.info("Starting model evaluation and MLflow logging")

        mlruns_path = Path(self.save_dir) / "mlruns"

        try:
            mlflow.set_tracking_uri(
                f"file:///{mlruns_path.resolve().as_posix()}"
            )
            mlflow.set_experiment("HighRisk_Model_Comparison")
        except Exception as e:
            logger.critical(f"MLflow setup failed: {e}")
            raise

        results: Dict[str, Dict[str, float]] = {}

        for name, model in self.models.items():
            logger.info(f"Evaluating model: {name}")

            with mlflow.start_run(run_name=name):
                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)[:, 1]

                metrics = {
                    "accuracy": accuracy_score(self.y_test, y_pred),
                    "precision": precision_score(self.y_test, y_pred, zero_division=0),
                    "recall": recall_score(self.y_test, y_pred, zero_division=0),
                    "f1": f1_score(self.y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(self.y_test, y_prob),
                }

                results[name] = metrics

                mlflow.log_param("model_name", name)
                mlflow.log_params(self.best_params[name])
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, name="model")

                logger.info(f"{name} metrics: {metrics}")

        # -------- BEST MODEL SELECTION --------
        best_model_name = max(results, key=lambda m: results[m]["roc_auc"])
        best_model = self.models[best_model_name]

        logger.info(f"✅ Best model selected: {best_model_name}")

        with mlflow.start_run(run_name=f"Best_{best_model_name}"):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metrics(results[best_model_name])
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                registered_model_name="Best_HighRisk_Model_CreditRisk"
            )

        try:
            joblib.dump(
                best_model,
                Path(self.save_dir) / "best_model.pkl"
            )
            logger.info("Best model saved as best_model.pkl")
        except Exception as e:
            logger.critical(f"Failed to save best model: {e}")
            raise


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    try:
        trainer = ModelTrainer(
            data_path="data/processed/modeling_dataset.csv"
        )
        trainer.load_data()
        trainer.train_models()
        trainer.evaluate_and_log()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")
