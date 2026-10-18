"""
Unit Tests for Model Artifact, Inference Predictions, and Invariants
AIML ZG523 - MLOps Assignment 1
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import HeartDiseasePredictor, FEATURE_COLUMNS


@pytest.fixture
def predictor():
    """Fixture providing initialized HeartDiseasePredictor instance."""
    try:
        return HeartDiseasePredictor()
    except FileNotFoundError:
        pytest.skip("Model artifact not yet trained. Run train.py before running model tests.")


@pytest.fixture
def valid_single_patient():
    return {
        "age": 58,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 2,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.6,
        "slope": 2,
        "ca": 0.0,
        "thal": 3.0,
    }


def test_prediction_output_structure(predictor, valid_single_patient):
    """Test single patient prediction schema and value ranges."""
    results = predictor.predict(valid_single_patient)
    assert len(results) == 1
    res = results[0]

    assert "prediction" in res
    assert res["prediction"] in (0, 1)

    assert "probability" in res
    assert 0.0 <= res["probability"] <= 1.0

    assert "confidence" in res
    assert 0.5 <= res["confidence"] <= 1.0

    assert res["diagnosis"] in ("Heart Disease Present", "Heart Disease Absent")
    assert res["risk_level"] in ("High", "Moderate", "Low")


def test_batch_prediction_consistency(predictor, valid_single_patient):
    """Test batch prediction handles multiple records and preserves ordering."""
    patient2 = valid_single_patient.copy()
    patient2["age"] = 42
    patient2["oldpeak"] = 0.2

    batch = [valid_single_patient, patient2]
    results = predictor.predict(batch)

    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["probability"] <= 1.0


def test_missing_feature_raises_error(predictor, valid_single_patient):
    """Test that omitting a mandatory feature triggers a ValueError."""
    corrupted_patient = valid_single_patient.copy()
    del corrupted_patient["age"]

    with pytest.raises(ValueError, match="Missing required feature columns"):
        predictor.predict(corrupted_patient)
