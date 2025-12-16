from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import joblib
import pandas as pd
from pathlib import Path
from src.config.config import Config


# -------------------- APP INIT --------------------
app = FastAPI(title="Credit Risk Prediction API")


# -------------------- MODEL SINGLETON --------------------
_model = None
MODEL_PATH = Path(
    Config.models_dir / "best_model.pkl"
)


def load_model_once():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


# -------------------- PREDICTION ENDPOINT --------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        model = load_model_once()
        # Convert Pydantic request to DataFrame
        df = pd.DataFrame([request.dict()])
        predicted_risk = model.predict(df)[0]
        predicted_risk_prob = (
            model.predict_proba(df)[:, 1][0] if hasattr(model, "predict_proba") else None
        )

        return PredictionResponse(
            predicted_risk=int(predicted_risk),
            predicted_risk_prob=float(predicted_risk_prob)
            if predicted_risk_prob is not None
            else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
