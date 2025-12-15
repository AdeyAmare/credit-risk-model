# train.py
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
    Class for training and evaluating multiple classification models
    on high-risk customer prediction.

    Attributes:
        data_path (str): Path to the CSV dataset.
        random_state (int): Random seed for reproducibility.
        save_dir (str): Directory to save trained models and MLflow runs.
        models (Dict[str, object]): Dictionary storing trained model objects.
    """

    def __init__(
        self,
        data_path: str,
        random_state: int = 42,
        save_dir: str = Config.models_dir
    ):
        self.data_path: str = data_path
        self.random_state: int = random_state
        self.save_dir: str = save_dir
        self.models: Dict[str, object] = {}

        try:
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"Model save directory created: {self.save_dir}")
        except Exception as e:
            logger.error(f"Error creating save directory: {e}")
            raise

    # -------------------- DATA LOADING --------------------
    def load_data(self) -> None:
        """
        Load dataset from CSV and split into train and test sets.
        
        Raises:
            FileNotFoundError: If CSV file does not exist.
            KeyError: If required columns are missing.
        """
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            raise

        try:
            self.X = df.drop(columns=['CustomerId', 'is_high_risk'])
            self.y = df['is_high_risk']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                stratify=self.y,
                random_state=self.random_state
            )
            logger.info("Train-test split completed successfully")
        except KeyError as e:
            logger.error(f"Required column missing: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during train-test split: {e}")
            raise

    # -------------------- MODEL TRAINING --------------------
    def train_models(self) -> None:
        """
        Train RandomForestClassifier and LogisticRegression models
        with GridSearchCV and store the best estimators.
        
        Random Forest is trained without scaling.
        Logistic Regression is trained with StandardScaler.
        """
        try:
            # -------- Random Forest --------
            rf = RandomForestClassifier(random_state=self.random_state)
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            rf_grid = GridSearchCV(rf, param_grid=rf_params, cv=3, scoring='f1', n_jobs=-1)
            rf_grid.fit(self.X_train, self.y_train)
            self.models['RandomForest'] = rf_grid.best_estimator_
            logger.info(f"Random Forest best params: {rf_grid.best_params_}")

            # -------- Logistic Regression --------
            lr_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(max_iter=1000, solver='lbfgs', random_state=self.random_state))
            ])
            lr_params = {'lr__C': [0.01, 0.1, 1, 10]}
            lr_grid = GridSearchCV(lr_pipeline, param_grid=lr_params, cv=3, scoring='f1', n_jobs=-1)
            lr_grid.fit(self.X_train, self.y_train)
            self.models['LogisticRegression'] = lr_grid.best_estimator_
            logger.info(f"Logistic Regression best params: {lr_grid.best_params_}")

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    # -------------------- EVALUATION + MLFLOW --------------------
    def evaluate_and_log(self) -> None:
        """
        Evaluate trained models, log metrics to MLflow, and
        register the best model based on ROC-AUC score.

        Saves the best model as a pickle file in `save_dir`.
        """
        try:
            mlruns_path = Path(self.save_dir) / "mlruns"
            mlflow.set_tracking_uri(f"file:///{mlruns_path.resolve().as_posix()}")
            mlflow.set_experiment("HighRisk_Model_Comparison")
            logger.info(f"MLflow tracking set at: {mlruns_path.resolve()}")

            results: Dict[str, Dict[str, float]] = {}

            for name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)[:, 1]

                metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1_score': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_prob)
                }

                results[name] = metrics
                logger.info(f"\n{name} metrics: {metrics}")

                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model_name", name)
                    mlflow.log_params(model.get_params())
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(sk_model=model, name="model")

            # -------- BEST MODEL SELECTION --------
            best_model_name = max(results, key=lambda m: results[m]['roc_auc'])
            best_model = self.models[best_model_name]
            best_metrics = results[best_model_name]

            logger.info(f"\n✅ Best Model: {best_model_name}")
            logger.info(f"ROC-AUC: {best_metrics['roc_auc']}")

            # -------- REGISTER BEST MODEL --------
            with mlflow.start_run(run_name=f"Best_{best_model_name}"):
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metrics(best_metrics)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                    registered_model_name="Best_HighRisk_Model"
                )

                joblib.dump(best_model, os.path.join(self.save_dir, "best_model.pkl"))
                logger.info(f"Best model saved to {self.save_dir}/best_model.pkl")

        except Exception as e:
            logger.error(f"Error during evaluation or MLflow logging: {e}")
            raise


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    try:
        trainer = ModelTrainer(data_path="data/processed/modeling_dataset.csv")
        trainer.load_data()
        trainer.train_models()
        trainer.evaluate_and_log()
        logger.info("Training pipeline completed successfully.")
    except Exception as e:
        logger.critical(f"Training pipeline failed: {e}")
