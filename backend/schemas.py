from typing import List
from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    Time: float = Field(..., description="Transaction timestamp in seconds")
    V1: float = Field(...)
    V2: float = Field(...)
    V3: float = Field(...)
    V4: float = Field(...)
    V5: float = Field(...)
    V6: float = Field(...)
    V7: float = Field(...)
    V8: float = Field(...)
    V9: float = Field(...)
    V10: float = Field(...)
    V11: float = Field(...)
    V12: float = Field(...)
    V13: float = Field(...)
    V14: float = Field(...)
    V15: float = Field(...)
    V16: float = Field(...)
    V17: float = Field(...)
    V18: float = Field(...)
    V19: float = Field(...)
    V20: float = Field(...)
    V21: float = Field(...)
    V22: float = Field(...)
    V23: float = Field(...)
    V24: float = Field(...)
    V25: float = Field(...)
    V26: float = Field(...)
    V27: float = Field(...)
    V28: float = Field(...)
    Amount: float = Field(..., description="Transaction monetary amount")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Time": 406.0,
                "V1": -2.31222654232745,
                "V2": 1.95199201064158,
                "V3": -1.60985073229769,
                "V4": 3.9979055875468,
                "V5": -0.522187864667764,
                "V6": -1.42654531920595,
                "V7": -2.53738730594232,
                "V8": 1.39165724829804,
                "V9": -2.77008927719433,
                "V10": -2.77227214465915,
                "V11": 3.20203320709656,
                "V12": -2.89990738849473,
                "V13": -0.595221881324605,
                "V14": -4.28925378244217,
                "V15": 0.389724120274487,
                "V16": -1.14074717980657,
                "V17": -2.83005567450437,
                "V18": -0.0168224681864079,
                "V19": 0.416955705037957,
                "V20": 0.126910559061474,
                "V21": 0.517232370861764,
                "V22": -0.0350493686052974,
                "V23": -0.465211076182388,
                "V24": 0.320198198514526,
                "V25": 0.0445191674731724,
                "V26": 0.177839798284401,
                "V27": 0.261145002567677,
                "V28": -0.143275874698919,
                "Amount": 149.62,
            }
        }
    )


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    fraud_prediction: int = Field(..., description="Binary classification (1=fraud, 0=legit)")
    fraud_probability_score: float = Field(..., description="Estimated probability of fraud [0-1]")
    risk_level: str = Field(..., description="Risk tier: Low, Medium, or High")
    threshold_used: float = Field(..., description="Cost-optimal decision threshold used")
    top_contributing_features: List[FeatureContribution] = Field(
        ..., description="Top 5 feature contributions by absolute SHAP value"
    )
