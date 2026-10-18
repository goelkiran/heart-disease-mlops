"""
Inference Pipeline and Prediction Engine for Heart Disease UCI Model
AIML ZG523 - MLOps Assignment 1
"""

from pathlib import Path
from typing import Dict, Any, List, Union
import joblib
import numpy as np
import pandas as pd

MODEL_ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "best_model.joblib"

# Required input features
FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


class HeartDiseasePredictor:
    """Wrapper class encapsulating trained model loading and production inference."""

    def __init__(self, model_path: Path = MODEL_ARTIFACT_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Train model using train.py first."
            )
        return joblib.load(self.model_path)

    def validate_and_format_input(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Validate presence of required feature columns and return numeric DataFrame."""
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise TypeError("Input must be a JSON dictionary or a list of dictionaries.")

        df = pd.DataFrame.from_records(records)
        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        df = df[FEATURE_COLUMNS].copy()
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def predict(
        self, data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions and confidence probabilities for single or batch inputs.
        Returns list of result dictionaries.
        """
        df = self.validate_and_format_input(data)
        probabilities = self.model.predict_proba(df)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results = []
        is_single = isinstance(data, dict)
        records = [data] if is_single else data

        for idx, (rec, pred, proba) in enumerate(zip(records, predictions, probabilities)):
            results.append(
                {
                    "prediction": int(pred),
                    "diagnosis": "Heart Disease Present" if pred == 1 else "Heart Disease Absent",
                    "probability": round(float(proba), 4),
                    "risk_level": "High" if proba >= 0.70 else ("Moderate" if proba >= 0.40 else "Low"),
                    "confidence": round(float(proba if pred == 1 else (1.0 - proba)), 4),
                }
            )

        return results
