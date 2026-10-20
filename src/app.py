"""
Production FastAPI Serving Application for Heart Disease Prediction API
Includes /health, /predict, /metrics (Prometheus), and structured request logging.
AIML ZG523 - MLOps Assignment 1
"""

import logging
import time
import uuid
import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from pipeline import HeartDiseasePredictor, FEATURE_COLUMNS
except ImportError:
    from src.pipeline import HeartDiseasePredictor, FEATURE_COLUMNS

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ReqID: %(request_id)s] %(message)s",
)
logger = logging.getLogger("heart_disease_api")

# Prometheus Metrics Definitions
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Latency in Seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total Predictions by Class",
    ["predicted_class", "risk_level"],
)

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-ready MLOps API for clinical risk classification with monitoring and logging.",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor singleton
try:
    predictor = HeartDiseasePredictor()
    model_ready = True
except Exception as e:
    predictor = None
    model_ready = False
    logger.warning(f"Predictor initialized in standby mode: {e}")


# Pydantic Schemas for Input Validation
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)")
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure in mm Hg")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1=true; 0=false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)")
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1=yes; 0=no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)")
    ca: float = Field(..., ge=0.0, le=3.0, description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: float = Field(..., ge=3.0, le=7.0, description="Thallium stress test result (3=normal, 6=fixed defect, 7=reversable defect)")


class PredictionResponse(BaseModel):
    prediction: int
    diagnosis: str
    probability: float
    risk_level: str
    confidence: float


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]


# Middleware: Latency, Request ID, and Logging
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start_time = time.time()

    # Pass request_id to logs via extra
    extra = {"request_id": req_id}
    logger.info(f"Incoming {request.method} {request.url.path} from {request.client.host}", extra=extra)

    try:
        response = await call_next(request)
        duration = time.time() - start_time
        status = response.status_code

        # Prometheus metrics
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=status).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

        logger.info(f"Completed {request.method} {request.url.path} [{status}] in {duration*1000:.2f}ms", extra=extra)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration*1000:.2f}"
        return response
    except Exception as exc:
        duration = time.time() - start_time
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=500).inc()
        logger.error(f"Failed {request.method} {request.url.path} in {duration*1000:.2f}ms: {exc}", extra=extra)
        raise exc


@app.get("/", tags=["General"])
def index():
    return {
        "service": "Heart Disease UCI Prediction Service",
        "status": "operational",
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "timestamp": time.time(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(patient: PatientData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded on server.")
    try:
        result = predictor.predict(patient.model_dump())[0]
        PREDICTION_COUNT.labels(
            predicted_class=str(result["prediction"]),
            risk_level=result["risk_level"]
        ).inc()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Inference"])
def predict_batch(batch: BatchPredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded on server.")
    try:
        inputs = [p.model_dump() for p in batch.patients]
        results = predictor.predict(inputs)
        for r in results:
            PREDICTION_COUNT.labels(
                predicted_class=str(r["prediction"]),
                risk_level=r["risk_level"]
            ).inc()
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch inference error: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
