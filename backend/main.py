from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import TransactionInput, PredictionResponse
from backend.inference import get_fraud_model

app = FastAPI(
    title="FraudGuard-AI Real-Time Fraud Scoring API",
    description=(
        "Real-time credit card fraud detection scoring service using a cost-optimized "
        "decision threshold and SHAP feature explainability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check")
def health_check():
    """Returns operational status of the service."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score Transaction for Fraud Risk",
    description="Engineers features, applies cost-optimal thresholding, and returns SHAP explanations."
)
def predict_transaction(transaction: TransactionInput):
    try:
        model = get_fraud_model()
        return model.predict(transaction.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model artifacts missing. Please copy model.joblib, scaler.joblib, "
                f"and metadata.json from model_training/artifacts/ into '{settings.artifacts_dir}'. "
                f"Original error: {str(e)}"
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
