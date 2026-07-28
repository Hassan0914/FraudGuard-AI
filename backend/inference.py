import os
import json
from functools import lru_cache
from typing import Dict, Any, List

import joblib
import numpy as np
import shap

from backend.config import settings
from backend.schemas import FeatureContribution, PredictionResponse


class FraudModel:
    def __init__(self, artifacts_dir: str = None):
        if artifacts_dir is None:
            artifacts_dir = settings.artifacts_dir

        model_path = os.path.join(artifacts_dir, "model.joblib")
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
        metadata_path = os.path.join(artifacts_dir, "metadata.json")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path)):
            raise FileNotFoundError(
                f"Missing model artifacts in directory '{artifacts_dir}'. "
                f"Please ensure model.joblib, scaler.joblib, and metadata.json are copied from "
                f"model_training/artifacts/ into '{artifacts_dir}'."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_cols: List[str] = self.metadata["feature_cols"]
        self.cost_optimal_threshold: float = float(self.metadata["cost_optimal_threshold"])

        # Initialize SHAP explainer once at construction time
        try:
            self.explainer = shap.Explainer(self.model)
        except Exception:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception:
                self.explainer = shap.Explainer(self.model.predict_proba)

    def engineer_features(self, raw_dict: Dict[str, Any]) -> np.ndarray:
        """
        Engineers server-side features ('hour_of_day' and 'log_amount') matching training pipeline,
        and orders features according to metadata.json feature_cols.
        """
        time_val = float(raw_dict["Time"])
        amount_val = float(raw_dict["Amount"])

        data = dict(raw_dict)
        data["hour_of_day"] = float((time_val % 86400) // 3600)
        data["log_amount"] = float(np.log1p(amount_val))

        ordered_values = [data[col] for col in self.feature_cols]
        return np.array([ordered_values], dtype=np.float64)

    def predict(self, raw_dict: Dict[str, Any]) -> PredictionResponse:
        """
        Performs feature engineering, scaling, probability estimation, cost-sensitive threshold scoring,
        risk tiering, and SHAP feature explainability.
        """
        raw_features = self.engineer_features(raw_dict)
        scaled_features = self.scaler.transform(raw_features)

        probabilities = self.model.predict_proba(scaled_features)[0]
        prob_score = float(probabilities[1])

        # Apply trained cost-optimal threshold
        threshold = self.cost_optimal_threshold
        fraud_pred = 1 if prob_score >= threshold else 0

        # Define risk levels
        medium_thresh = threshold * settings.medium_risk_ratio
        if prob_score >= threshold:
            risk_level = "High"
        elif prob_score >= medium_thresh:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Calculate SHAP explainability values
        shap_output = self.explainer(scaled_features)
        
        if hasattr(shap_output, "values"):
            shap_vals = shap_output.values
        else:
            shap_vals = shap_output

        # Handle 3D, 2D, or list SHAP shapes across different model types
        if isinstance(shap_vals, list):
            vals = np.array(shap_vals[1][0])
        elif shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1]
        elif shap_vals.ndim == 2:
            vals = shap_vals[0, :]
        else:
            vals = np.array(shap_vals).flatten()

        # Extract top 5 features by absolute SHAP impact
        top_indices = np.argsort(np.abs(vals))[::-1][:5]
        top_features = [
            FeatureContribution(
                feature=self.feature_cols[i],
                shap_value=float(vals[i])
            )
            for i in top_indices
        ]

        return PredictionResponse(
            fraud_prediction=fraud_pred,
            fraud_probability_score=round(prob_score, 4),
            risk_level=risk_level,
            threshold_used=round(threshold, 2),
            top_contributing_features=top_features,
        )


@lru_cache(maxsize=1)
def get_fraud_model() -> FraudModel:
    """Singleton provider for FraudModel instance."""
    return FraudModel()
