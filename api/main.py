"""
main.py — FastAPI production application.

Endpoints:
  GET  /              — Redirect to docs
  GET  /health        — Liveness check
  GET  /model-info    — Loaded model metadata
  POST /predict       — Single engine failure prediction
  POST /predict/batch — Multi-engine batch prediction
  POST /drift-check   — Data drift detection

Auto-generated docs at:
  http://localhost:8000/docs     (Swagger UI)
  http://localhost:8000/redoc    (ReDoc)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftCheckRequest,
    DriftCheckResponse,
    DriftFeatureResult,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictionResponse,
)
from src.features.engineer import FeatureEngineer
from src.models.predictor import Predictor
from src.monitoring.drift_detector import DriftDetector
from src.utils.helpers import get_logger, load_config

logger  = get_logger(__name__)
cfg     = load_config()
VERSION = cfg["project"]["version"]

# ─── App state ────────────────────────────────────────────────────────────────

class AppState:
    predictor: Optional[Predictor]       = None
    engineer:  Optional[FeatureEngineer] = None
    detector:  Optional[DriftDetector]   = None
    baseline_df: Optional[pd.DataFrame]  = None

state = AppState()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup; release on shutdown."""
    logger.info("Starting up PredictiveMaintenance API v%s…", VERSION)

    try:
        state.predictor = Predictor(cfg).load_models()
        state.engineer  = FeatureEngineer(cfg)
        state.detector  = DriftDetector(cfg)

        # Try loading baseline for drift detection
        baseline_path = Path(cfg["paths"]["data_processed"]) / "train_featured.parquet"
        if baseline_path.exists():
            baseline_df = pd.read_parquet(baseline_path)
            feat_cols = [c for c in baseline_df.columns
                         if c not in ["unit_id", "cycle", "rul", "failure_label"]]
            state.detector.fit_baseline(baseline_df, feat_cols[:20])
            state.baseline_df = baseline_df
            logger.info("Drift baseline loaded from %s", baseline_path)
        else:
            logger.warning("No baseline found at %s — drift detection unavailable", baseline_path)

    except Exception as e:
        logger.error("Startup failed: %s", e)

    logger.info("API ready ✓")
    yield
    logger.info("Shutting down API…")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PredictiveMaintenance Pro API",
    description=(
        "Production-grade turbofan engine failure prediction service.\n\n"
        "Uses ensemble classical ML (RandomForest + XGBoost + LightGBM) "
        "to predict:\n"
        "- Binary failure probability within the next 30 cycles\n"
        "- Remaining Useful Life (RUL) in cycles\n"
        "- Risk categorisation: HEALTHY → WATCH → WARNING → CRITICAL\n\n"
        "Includes data drift detection via PSI + KS tests."
    ),
    version=VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Error handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _readings_to_df(readings: list, unit_id: Any = None) -> pd.DataFrame:
    """Convert SensorReading list → pandas DataFrame with unit_id column."""
    records = [r.model_dump() for r in readings]
    df = pd.DataFrame(records)
    if unit_id is not None:
        df["unit_id"] = unit_id
    return df


def _engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess + engineer features for raw sensor DataFrame."""
    from src.data.preprocessor import DataPreprocessor

    proc = DataPreprocessor(cfg)
    proc_path = Path(cfg["paths"]["data_processed"]) / "scaler.pkl"
    if proc_path.exists():
        proc.load_scaler()
        processed = proc.transform(raw_df)
    else:
        # Scaler not trained yet — use raw data with basic normalisation
        processed = raw_df.copy()
        processed["rul"] = 0
        processed["failure_label"] = 0

    return state.engineer.transform(processed)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe — returns model load status."""
    return HealthResponse(
        status="ok",
        classifier_loaded=state.predictor is not None and state.predictor._clf is not None,
        regressor_loaded=state.predictor is not None and state.predictor._reg is not None,
        version=VERSION,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return metadata about the currently loaded models."""
    if state.predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    info = state.predictor.model_info
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict failure probability and RUL for a single engine.

    Send 1–100 consecutive sensor readings. Using 10+ readings
    produces more stable predictions through averaging.
    """
    if state.predictor is None or state.predictor._clf is None:
        raise HTTPException(status_code=503, detail="Model not ready. Train first.")

    try:
        raw_df = _readings_to_df(request.readings, unit_id=request.unit_id)
        featured_df = _engineer_features(raw_df)
        result = state.predictor.predict_single(
            featured_df, unit_id=request.unit_id, use_last_n=request.use_last_n
        )
        return PredictionResponse(**result.to_dict())

    except Exception as e:
        logger.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Predict failure probability for multiple engines in one call.
    Returns summary counts + per-engine results.
    """
    if state.predictor is None or state.predictor._clf is None:
        raise HTTPException(status_code=503, detail="Model not ready. Train first.")

    try:
        all_results = []
        for engine_req in request.engines:
            raw_df = _readings_to_df(engine_req.readings, unit_id=engine_req.unit_id)
            featured_df = _engineer_features(raw_df)
            result = state.predictor.predict_single(
                featured_df, unit_id=engine_req.unit_id, use_last_n=engine_req.use_last_n
            )
            all_results.append(PredictionResponse(**result.to_dict()))

        return BatchPredictResponse(
            total_engines=len(all_results),
            critical_count=sum(1 for r in all_results if r.risk_level == "CRITICAL"),
            warning_count= sum(1 for r in all_results if r.risk_level == "WARNING"),
            watch_count=   sum(1 for r in all_results if r.risk_level == "WATCH"),
            healthy_count= sum(1 for r in all_results if r.risk_level == "HEALTHY"),
            results=all_results,
        )

    except Exception as e:
        logger.error("Batch prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift-check", response_model=DriftCheckResponse, tags=["Monitoring"])
async def drift_check(request: DriftCheckRequest):
    """
    Check if incoming sensor data has drifted from the training distribution.
    Requires baseline to be fitted (happens automatically during training).
    """
    if state.detector is None or not state.detector._baseline_stats:
        raise HTTPException(
            status_code=503,
            detail="Drift baseline not available. Run full training pipeline first."
        )

    try:
        raw_df = _readings_to_df(request.readings)
        featured_df = _engineer_features(raw_df)

        feat_cols = [c for c in featured_df.columns
                     if c not in ["unit_id", "cycle", "rul", "failure_label"]]

        drift_report = state.detector.detect_feature_drift(featured_df, feat_cols)
        summary      = state.detector.summarize_drift_report(drift_report)

        details = [
            DriftFeatureResult(
                feature=k,
                psi=v["psi"],
                ks_pvalue=v["ks_pvalue"],
                drifted=v["drifted"],
            )
            for k, v in drift_report.items()
        ]

        return DriftCheckResponse(
            alert_level=summary["alert_level"],
            total_features=summary["total_features"],
            drifted_features=summary["drifted_features"],
            avg_psi=summary["avg_psi"],
            details=details,
        )

    except Exception as e:
        logger.error("Drift check error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
        log_level="info",
    )
