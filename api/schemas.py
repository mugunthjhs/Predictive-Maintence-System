"""
schemas.py — Pydantic request/response schemas for FastAPI.

All input validation, serialisation, and documentation
of the predictive maintenance API contracts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ─── Prediction request ────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """One sensor reading (one cycle for one engine)."""
    T2_fan_inlet_temp:         float = Field(..., description="Fan inlet temperature (°R)")
    T24_LPC_outlet_temp:       float = Field(..., description="LPC outlet temperature (°R)")
    T30_HPC_outlet_temp:       float = Field(..., description="HPC outlet temperature (°R)")
    T50_LPT_outlet_temp:       float = Field(..., description="LPT outlet temperature (°R)")
    P2_fan_inlet_pressure:     float = Field(..., description="Fan inlet pressure (psia)")
    P15_bypass_duct_pressure:  float = Field(..., description="Bypass duct pressure (psia)")
    P30_HPC_outlet_pressure:   float = Field(..., description="HPC outlet pressure (psia)")
    Nf_fan_speed:              float = Field(..., description="Fan speed (rpm)")
    Nc_core_speed:             float = Field(..., description="Core speed (rpm)")
    epr_engine_pressure_ratio: float = Field(..., description="Engine pressure ratio (-)")
    Ps30_static_pressure:      float = Field(..., description="Static pressure at HPC (psia)")
    phi_fuel_flow:             float = Field(..., description="Fuel flow ratio (pps/psia)")
    NRf_corrected_fan_speed:   float = Field(..., description="Corrected fan speed (rpm)")
    NRc_corrected_core_speed:  float = Field(..., description="Corrected core speed (rpm)")
    op_setting_1:              float = Field(default=-0.0006)
    op_setting_2:              float = Field(default=0.0004)
    op_setting_3:              float = Field(default=100.0)
    cycle:                     int   = Field(default=1, ge=1)


class PredictRequest(BaseModel):
    """Request payload for failure prediction."""
    unit_id:   Any          = Field(default="unknown", description="Engine identifier")
    readings:  List[SensorReading] = Field(
        ..., min_length=1,
        description="Sensor readings (1+ cycles). More cycles → stabler estimate."
    )
    use_last_n: int = Field(default=10, ge=1, le=100,
                            description="Use last N cycles for prediction averaging.")


class BatchPredictRequest(BaseModel):
    """Request payload for batch failure prediction (multiple engines)."""
    engines: List[PredictRequest] = Field(..., min_length=1, max_length=1000)


# ─── Prediction response ───────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    unit_id:             Any
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    failure_predicted:   bool
    risk_level:          str   = Field(..., description="HEALTHY | WATCH | WARNING | CRITICAL")
    risk_emoji:          str
    rul_estimate:        Optional[float] = Field(None, description="Predicted remaining useful life (cycles)")
    confidence_margin:   Optional[float] = None
    message:             str  = ""

    @model_validator(mode="after")
    def populate_message(self):
        if self.risk_level == "CRITICAL":
            self.message = "⚠️  Immediate maintenance required! High failure probability."
        elif self.risk_level == "WARNING":
            self.message = "Schedule maintenance soon."
        elif self.risk_level == "WATCH":
            self.message = "Monitor closely — some degradation detected."
        else:
            self.message = "Engine is operating normally."
        return self


class BatchPredictResponse(BaseModel):
    total_engines: int
    critical_count: int
    warning_count: int
    watch_count: int
    healthy_count: int
    results: List[PredictionResponse]


# ─── Health & model info ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    regressor_loaded:  bool
    version: str


class ModelInfoResponse(BaseModel):
    classifier:    Optional[str]
    regressor:     Optional[str]
    threshold:     float
    feature_count: int


# ─── Drift detection ───────────────────────────────────────────────────────────

class DriftCheckRequest(BaseModel):
    """Provide a sample of recent sensor readings; returns drift report."""
    readings: List[SensorReading] = Field(..., min_length=30,
                                          description="At least 30 readings for statistical significance.")


class DriftFeatureResult(BaseModel):
    feature:   str
    psi:       float
    ks_pvalue: float
    drifted:   bool


class DriftCheckResponse(BaseModel):
    alert_level:      str
    total_features:   int
    drifted_features: int
    avg_psi:          float
    details:          List[DriftFeatureResult]
