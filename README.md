# Predictive Maintenance System

> Production-grade turbofan engine failure prediction using classical ML and time-series feature engineering

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-red)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Why This Project Exists

Industrial machines — turbofan engines, pumps, compressors — degrade silently over time. Conventional maintenance strategies both have significant drawbacks:

| Strategy | What it does | Problem |
|---|---|---|
| **Reactive Maintenance** | Fix after failure | Catastrophic downtime, safety risks, high repair costs |
| **Scheduled Maintenance** | Service every N days regardless | Wastes money replacing healthy parts |

**Predictive maintenance is the third way:** use the machine's own sensor data to predict *exactly when* it is going to fail and act just in time.

This project demonstrates that this is **solvable with classical ML** — no deep learning, no GPU required. Just rigorous engineering on time-series sensor data.

---

## What It Predicts

| Question | Method | Example Output |
|---|---|---|
| Will this engine fail in the next 30 cycles? | Binary classification | 87% failure probability |
| How many cycles does it have left? | Regression (RUL) | 12 cycles remaining |
| How urgent is the risk? | 4-level stratification | CRITICAL — immediate inspection required |

**Risk levels:** HEALTHY → WATCH → WARNING → CRITICAL

---

## Why Classical ML, Not Deep Learning?

This was a deliberate design decision:

| Reason | Explanation |
|---|---|
| **Interpretability** | Engineers need to understand *why* an alert fired. SHAP values and decision trees give that; LSTMs do not. |
| **Speed** | Millisecond inference per engine, no GPU required |
| **Reliability** | Ensemble methods have well-understood failure modes; neural networks can fail silently |
| **Industry compatibility** | Most industrial IoT platforms (SCADA, DCS) are tabular-data first |
| **Deployability** | Runs on any Linux VM or container — no CUDA, no special drivers |

The key insight: **temporal patterns that deep learning would learn automatically can be engineered explicitly as features** — rolling statistics, lag values, trend slopes, entropy. This project does exactly that.

---

## The Core Engineering Insight

Raw sensor data per cycle: `[T2=518.67, T30=1595.0, P30=545.0, ...]`

A classical ML model treats each row as independent — it cannot learn that "temperature has been rising for 10 cycles", which is the actual degradation signal.

**The solution is feature engineering.** We transform raw sensor readings into a rich feature matrix that captures temporal context:

| Feature Family | What it captures | Examples |
|---|---|---|
| Rolling statistics | Short-term behaviour | `roll_mean_5`, `roll_std_10`, `roll_max_30` |
| Lag features | Recent history vs. now | `T30_lag_1`, `T30_lag_3`, `T30_lag_10` |
| Rate of change | Velocity of degradation | `T30_diff_1`, `T30_diff_5` |
| EWMA | Exponentially weighted trend | `T30_ewm_5`, `T30_ewm_20` |
| Trend slope | Long-term rate via linear regression | `T30_slope_10` |
| Rolling entropy | Erraticism (sign of wear) | `P30_entropy_10` |
| Cross-sensor ratios | Physics-informed combinations | `pressure_ratio_30_2`, `temp_diff_50_30` |

This transforms **14 raw sensors into 100+ temporal features** per row, giving classical models the temporal context they need.

---

## System Architecture

```
Raw Sensor Data (14 sensors × N cycles per engine)
        │
        ▼
  DataPreprocessor
  ├── Compute RUL  (max_cycle - current_cycle)
  ├── Cap RUL at 125  (focus on failure zone)
  ├── Create binary label  (RUL ≤ 30 → failure = 1)
  └── StandardScaler  (fit on train only, no leakage)
        │
        ▼
  FeatureEngineer  (src/features/engineer.py)
  └── 100+ temporal features per sensor
        │
        ▼
  ModelTrainer
  ├── RandomForest, XGBoost, LightGBM, GradientBoosting  (classifiers)
  ├── XGBoost + RandomForest  (regressors for RUL)
  ├── TimeSeriesSplit CV — 5 folds  (no data leakage)
  └── MLflow experiment tracking
        │
        ▼
  ModelEvaluator
  ├── 15+ metrics  (AUC, F1, MCC, NASA Score, Business Cost)
  └── 8 evaluation plots per model
        │
        ▼
  Predictor
  ├── failure_probability
  ├── rul_estimate
  └── risk_level  (HEALTHY / WATCH / WARNING / CRITICAL)
        │
        ▼
  DriftDetector  (src/monitoring/drift_detector.py)
  └── PSI + KS test  (detects silent model degradation in production)
        │
        ▼
  FastAPI REST Service  (api/)
  └── /predict  /predict/batch  /drift-check  /health
```

---

## Project Structure

```
Predictive-Maintence-System/
│
├── config.yaml               ← Single source of truth for all parameters
├── train.py                  ← Training CLI entry point
├── demo.py                   ← Full end-to-end interactive demo
│
├── api/
│   ├── main.py               ← FastAPI app and all endpoints
│   └── schemas.py            ← Pydantic request/response models
│
├── src/
│   ├── features/
│   │   └── engineer.py       ← 100+ temporal features per sensor
│   ├── monitoring/
│   │   └── drift_detector.py ← PSI + KS drift detection
│   └── utils/
│       └── helpers.py        ← Logger, config loader, timer
│
├── tests/
│   ├── test_features.py      ← Feature engineering unit tests
│   └── test_models.py        ← Model and drift detection tests
│
├── CMAPSS_Data/              ← NASA C-MAPSS turbofan dataset (FD001–FD004)
│
├── data/                     ← Auto-created at runtime
│   ├── raw/                  ← Parquet files from data loader
│   └── processed/            ← Engineered features and scaler.pkl
│
├── models/                   ← Trained .pkl model artifacts
├── reports/figures/          ← Evaluation plots (8 per model)
├── mlruns/                   ← MLflow experiment artifacts
└── logs/                     ← Structured log files
```

---

## Quickstart

**Prerequisites:** Python 3.11+

```bash
# Clone and set up a virtual environment
git clone <repo-url>
cd Predictive-Maintence-System
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install pandas numpy scipy scikit-learn xgboost lightgbm mlflow \
            fastapi uvicorn pydantic pyyaml rich pytest pytest-cov
```

```bash
# Run the full end-to-end demo (recommended first step)
python demo.py --quick        # ~30 seconds
python demo.py                # Full version, ~3–5 minutes

# Train models only
python train.py --quick
python train.py               # Full training run

# Start the REST API (models must be trained first)
python -m uvicorn api.main:app --reload
# API docs available at: http://localhost:8000/docs

# View experiment tracking
mlflow ui
# MLflow UI available at: http://localhost:5000

# Run unit tests
pytest tests/ -v --cov=src
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — are models loaded? |
| `GET` | `/model-info` | Loaded model name and training metadata |
| `POST` | `/predict` | Single engine: failure probability + RUL + risk level |
| `POST` | `/predict/batch` | Fleet-wide batch prediction (up to 1,000 engines) |
| `POST` | `/drift-check` | PSI + KS drift detection against training baseline |

**Example — single engine prediction:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "unit_id": "engine_001",
    "readings": [{
      "T2_fan_inlet_temp": 518.67,
      "T30_HPC_outlet_temp": 1595.0,
      "P30_HPC_outlet_pressure": 545.0,
      "Nf_fan_speed": 2384.0,
      "cycle": 150
    }]
  }'
```

```json
{
  "unit_id": "engine_001",
  "failure_probability": 0.87,
  "will_fail_soon": true,
  "rul_estimate": 12,
  "risk_level": "CRITICAL",
  "recommendation": "Immediate inspection required"
}
```

---

## Dataset

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation** dataset, included in the `CMAPSS_Data/` directory.

| Dataset | Operating Conditions | Fault Modes | Train Engines | Test Engines |
|---|---|---|---|---|
| FD001 | 1 (sea level) | HPC degradation | 100 | 100 |
| FD002 | 6 | HPC degradation | 260 | 259 |
| FD003 | 1 (sea level) | HPC + fan degradation | 100 | 100 |
| FD004 | 6 | HPC + fan degradation | 248 | 249 |

**Format:** 26 columns (space-separated text)
- Column 1: Engine unit number
- Column 2: Operational cycle
- Columns 3–5: Operational settings (altitude, throttle, Mach number)
- Columns 6–26: 21 sensor measurements (pressures, temperatures, speeds, ratios)

The demo and training scripts also support **synthetic data generation** — a programmatic NASA-style simulator that produces engines with random lifetimes (150–350 cycles), sigmoid health degradation curves, and per-engine baseline variation to model fleet heterogeneity.

---

## Evaluation Metrics

### Classification (Failure Prediction)

| Metric | Why it is included |
|---|---|
| AUC-ROC | Discrimination ability across all thresholds |
| AUC-PR | More informative than AUC-ROC for imbalanced classes |
| F1 Score | Balance of precision and recall |
| **MCC** | **Best single metric for imbalanced binary classification** |
| Cohen's Kappa | Agreement corrected for random chance |
| Youden's J Threshold | Finds the optimal decision boundary (not always 0.5) |
| Calibration Curve | Are predicted probabilities honest? |
| Business Cost | Dollar impact of model errors (FN = $100, FP = $10) |

### Regression (RUL Prediction)

| Metric | Why it is included |
|---|---|
| RMSE | Penalises large errors heavily |
| MAE | Robust average error in cycles |
| MAPE | Percentage error, scale-independent |
| R² | Variance explained by the model |
| **NASA Score** | **Asymmetric: penalises late predictions more than early ones** |

---

## Key Design Decisions

### Why cap RUL at 125?

Without capping, the model treats RUL=300 and RUL=200 as very different problems — but from a maintenance perspective, both mean "healthy, do nothing." Capping at 125 focuses learning on the **degradation-sensitive zone**, the last 125 cycles before failure. This is standard practice in the C-MAPSS literature.

### Why `TimeSeriesSplit` instead of random K-fold?

Random K-fold would place future readings into training folds, letting the model "see the future" during training and producing optimistically biased scores. `TimeSeriesSplit` ensures every fold trains on the past and validates on the future, matching real deployment conditions.

```
Fold 1: Train [0–20%]  | Validate [20–40%]
Fold 2: Train [0–40%]  | Validate [40–60%]
Fold 3: Train [0–60%]  | Validate [60–80%]
Fold 4: Train [0–80%]  | Validate [80–100%]
```

### Why MCC as the primary metric?

Accuracy is misleading for imbalanced data — a model that always predicts "healthy" on an 85%-healthy dataset gets 85% accuracy but is useless. Matthews Correlation Coefficient (MCC) accounts for all four cells of the confusion matrix and is robust to class imbalance. It is the standard recommendation for imbalanced binary classification.

### Why the NASA asymmetric scoring function?

Missing a failure (false negative) causes catastrophic downtime or safety incidents. Predicting early just means slightly premature maintenance — inconvenient but safe. The NASA scoring function penalises late predictions more harshly than early predictions:

```
Early prediction (estimated RUL too high):  score = exp(−d / 13) − 1
Late  prediction (estimated RUL too low):   score = exp( d / 10) − 1
```

### Why drift detection?

Model performance silently degrades in production. Sensor calibration drifts, environmental conditions change, operating procedures shift. Without monitoring, you will not know the model is wrong until machines start failing.

- **PSI (Population Stability Index):** Measures the magnitude of distribution shift per feature. PSI < 0.1 → stable; PSI > 0.2 → significant drift, retrain recommended.
- **KS Test (Kolmogorov-Smirnov):** Non-parametric test for distribution shape differences (p-value < 0.05 flags an issue).

Together they detect both gradual drift and sudden distributional shock.

### Why a business cost model?

ML metrics alone do not tell the full story. This project quantifies the financial impact directly:
- **False Negative (missed failure): $100** — downtime, emergency repair, safety risk
- **False Positive (unnecessary maintenance): $10** — inspection cost, brief lost productivity

This lets engineers compare models on actual dollar impact, not just AUC or F1.

---

## Configuration

All parameters are centralised in `config.yaml`. Key sections:

| Section | Key Parameters |
|---|---|
| `data` | Dataset ID (FD001–FD004), RUL cap (125), failure horizon (30 cycles), train/test split |
| `features` | Rolling windows, lag steps, EWMA spans, informative sensor indices |
| `models` | Classifier list, regressor list, CV folds, champion selection metric |
| `evaluation` | Business cost matrix, classification threshold, optimal threshold flag |
| `monitoring` | PSI threshold (0.2), KS alpha (0.05) |
| `api` | Host, port, worker count |
| `mlflow` | Experiment name, tracking URI |
