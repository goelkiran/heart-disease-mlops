"""
Integration Tests for FastAPI Endpoints (/health, /predict, /metrics)
AIML ZG523 - MLOps Assignment 1
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import app

client = TestClient(app)


def test_index_endpoint():
    """Verify root index endpoint metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Heart Disease UCI Prediction Service"
    assert "documentation" in data


def test_health_endpoint():
    """Verify /health endpoint returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_endpoint_success():
    """Verify /predict endpoint produces valid prediction payload."""
    payload = {
        "age": 60,
        "sex": 1,
        "cp": 4,
        "trestbps": 145,
        "chol": 282,
        "fbs": 0,
        "restecg": 2,
        "thalach": 142,
        "exang": 1,
        "oldpeak": 2.8,
        "slope": 2,
        "ca": 2.0,
        "thal": 7.0,
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Model artifact not yet trained. Run train.py first.")

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "confidence" in data
    assert "risk_level" in data


def test_predict_validation_error():
    """Verify Pydantic validation rejects out-of-range inputs."""
    invalid_payload = {
        "age": 250,  # invalid age > 120
        "sex": 5,    # invalid sex
        "cp": 9,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


def test_metrics_endpoint():
    """Verify Prometheus metrics endpoint is available and exposes HTTP metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content
