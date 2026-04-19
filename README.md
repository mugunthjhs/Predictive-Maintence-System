# 🔧 PredictiveMaintenance Pro

> **Production-grade turbofan engine failure prediction using classical ML + time-series feature engineering**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-orange)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-green)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-red)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🎯 Why This Project Was Built

### The Real-World Problem

Industrial machines — turbofan engines, pumps, compressors — **degrade silently over time**. There are two conventional maintenance strategies, and both are flawed:

| Strategy | What it does | Problem |
|---|---|---|
| **Reactive Maintenance** | Fix it after it breaks | Catastrophic downtime, safety risks, huge repair costs |
| **Scheduled Maintenance** | Service every N days regardless | Wastes money — replaces healthy parts unnecessarily |

**Predictive maintenance is the third way:** use the machine's own sensor data to predict *exactly when* it is going to fail — and act just in time.

This project was built to demonstrate that this is **solvable with classical ML** — no LLMs, no deep learning, no GPU required. Just rigorous engineering on time-series sensor data.

---

### The Core Questions This Project Answers

1. **Will this engine fail in the next 30 cycles?** → Binary classification (Yes/No)
2. **How many cycles does it have left?** → Regression (Remaining Useful Life, i.e. RUL)
3. **How urgent is the risk?** → Risk stratification into 4 levels

| Output | Example |
|---|---|
| Failure probability | 0.87 (87% chance of failure within 30 cycles) |
| Remaining Useful Life | 12 cycles left |
| Risk level | 🔴 CRITICAL — immediate inspection required |

---

### Why Classical ML, Not Deep Learning?

This was a deliberate design choice, not a limitation:

| Reason | Explanation |
|---|---|
| **Interpretability** | Engineers need to understand *why* an alert fired. SHAP values and decision trees give that; LSTMs don't. |
| **Speed** | Millisecond inference per engine, with no GPU required. |
| **Reliability** | Ensemble methods have well-understood failure modes; neural networks can fail silently. |
| **Industry standard** | Most industrial IoT platforms (SCADA, DCS) are still tabular-data first. |
| **Deployability** | Runs on any Linux VM or container — no CUDA, no special drivers. |

The key insight is: **the temporal patterns that deep learning would learn automatically can be engineered explicitly as features** — rolling statistics, lag values, trend slopes, entropy. This project does exactly that.

---

## 🧠 The Core Engineering Insight

Raw sensor data looks like this per cycle: `[T2=518.67, T30=1595.0, P30=545.0, ...]`

A classical ML model (Random Forest, XGBoost) treats each row as **independent**. It doesn't know that cycle 50 came after cycle 49. So it cannot learn that "temperature has been rising for 10 cycles" — which is the actual signal for degradation.

**The solution: Feature Engineering.**

We transform raw sensor readings into a rich feature matrix per cycle that captures:
- **Rolling statistics** — mean, std, min, max over 5/10/20/30 cycle windows → *"How has this sensor behaved recently?"*
- **Lag features** — sensor value 1, 3, 5, 10 cycles ago → *"Where was it compared to now?"*
- **Rate of change (diff)** — current minus past reading → *"Is it rising or falling?"*
- **Exponentially Weighted Moving Average (EWMA)** — recent cycles weighted more → *"What's the trend?"*
- **Trend slope** — linear regression slope over last 10 cycles → *"How fast is it changing?"*
- **Rolling entropy** — disorder in the distribution → *"Is it getting erratic — a sign of wear?"*
- **Cross-sensor ratios** — Physics-inspired combinations like `P30/P2` (Overall Pressure Ratio) and `T50 - T30` (turbine efficiency) → *"What do domain experts look at?"*

This transforms **14 raw sensors into 100+ temporal features** per row, giving classical ML models the temporal context they need.

---

## 🏗️ System Architecture

```
Raw Sensor Data (14 sensors × N cycles per engine)
        │
        ▼
  DataPreprocessor
  ├── Compute RUL  (max_cycle - current_cycle)
  ├── Cap RUL at 125  (focus on failure zone)
  ├── Create binary label  (RUL ≤ 30 → failure=1)
  └── StandardScaler  (fit on train only, no leakage)
        │
        ▼
  FeatureEngineer
  └── 100+ temporal features per sensor (rolling, lag, diff, EWMA, slope, entropy, cross-sensor)
        │
        ▼
  ModelTrainer
  ├── RandomForest, XGBoost, LightGBM, GradientBoosting (classifiers)
  ├── XGBoost + RandomForest (regressors for RUL)
  ├── TimeSeriesSplit CV  (no data leakage)
  └── MLflow experiment tracking
        │
        ▼
  ModelEvaluator
  ├── 15+ metrics  (AUC, F1, MCC, NASA Score, Business Cost...)
  └── 8 evaluation plots per model
        │
        ▼
  Predictor
  ├── failure_probability
  ├── rul_estimate
  └── risk_level  (HEALTHY / WATCH / WARNING / CRITICAL)
        │
        ▼
  DriftDetector
  └── PSI + KS test  (catches silent model degradation in production)
        │
        ▼
  FastAPI REST Service
  └── /predict  /predict/batch  /drift-check  /health
```

---

## 📁 Project Structure

```
e:\time_series\
│
├── config.yaml               ← Single source of truth for ALL parameters
├── train.py                  ← Training CLI (run this to train models)
├── demo.py                   ← Full end-to-end interactive demo
├── requirements.txt
├── Dockerfile                ← Multi-stage slim Docker build
├── docker-compose.yml        ← API + MLflow services
│
├── src/
│   ├── data/
│   │   ├── loader.py         ← Synthetic NASA C-MAPSS-like data generator
│   │   └── preprocessor.py   ← RUL computation, labeling, scaling
│   ├── features/
│   │   └── engineer.py       ← 100+ temporal features per sensor
│   ├── models/
│   │   ├── trainer.py        ← Multi-model training + MLflow tracking
│   │   ├── evaluator.py      ← 15+ metrics + 8 evaluation plots
│   │   └── predictor.py      ← Real-time inference engine
│   ├── monitoring/
│   │   └── drift_detector.py ← PSI + KS drift detection
│   └── utils/
│       └── helpers.py        ← Logger, config loader, Timer
│
├── api/
│   ├── main.py               ← FastAPI app + all endpoints
│   └── schemas.py            ← Pydantic request/response models (type-safe)
│
├── tests/
│   ├── test_features.py      ← Data + feature unit tests
│   └── test_models.py        ← Model + drift unit tests
│
├── data/
│   ├── raw/                  ← Parquet files from generator (auto-created)
│   └── processed/            ← Engineered features + scaler.pkl
│
├── models/                   ← Trained .pkl files (champion model)
├── reports/figures/          ← 8 evaluation plots per model
├── mlruns/                   ← MLflow experiment artifacts
├── logs/                     ← Structured log files
└── .github/workflows/        ← CI/CD pipeline (GitHub Actions)
```

---

## ⚙️ Key Design Decisions Explained

### 1. Why cap RUL at 125?
Without capping, the model sees RUL=300 and RUL=200 as very different problems. But from a maintenance perspective, "healthy with 300 cycles left" and "healthy with 200 cycles left" are the same decision: do nothing. Capping at 125 focuses the model's learning on the **degradation-sensitive zone** — the last 125 cycles before failure — which is what actually matters. This is the standard practice in the academic C-MAPSS literature.

### 2. Why `TimeSeriesSplit` instead of random K-fold?
Time-series data has temporal ordering. Random K-fold would place future readings into training folds — the model would "know the future" during training, leading to optimistically biased scores. `TimeSeriesSplit` ensures every fold trains on the past and validates on the future, matching real deployment.

```
Fold 1: Train [0–20%]  | Validate [20–40%]
Fold 2: Train [0–40%]  | Validate [40–60%]
Fold 3: Train [0–60%]  | Validate [60–80%]
Fold 4: Train [0–80%]  | Validate [80–100%]
```

### 3. Why MCC as the primary metric?
Accuracy is misleading for imbalanced data (if 85% of cycles are healthy, a model that always says "healthy" gets 85% accuracy but is useless). **Matthews Correlation Coefficient (MCC)** accounts for all four cells of the confusion matrix and is robust to class imbalance. It's the recommondation by the machine learning community for imbalanced binary classification.

### 4. Why the NASA asymmetric scoring function?
In maintenance, **predicting failure too late is far worse than predicting too early**. Missing a failure (false negative) can cause catastrophic downtime or safety incidents. Predicting early just means slightly premature maintenance — inconvenient but safe. The NASA scoring function penalises late predictions 30% more harshly than early predictions:

```
Early prediction (RUL too high): score = exp(−d/13) − 1
Late  prediction (RUL too low):  score = exp( d/10) − 1
```

### 5. Why drift detection?
Model performance silently degrades in production. Sensor calibration drifts, environmental conditions change, or operating procedures shift. **Without monitoring, you won't know the model is wrong until machines start failing.**

- **PSI (Population Stability Index):** Measures *magnitude* of distribution shift per feature.
- **KS Test (Kolmogorov-Smirnov):** Non-parametric test for distribution shape differences.
- Together they detect both gradual drift and sudden distributional shock.

### 6. Why a business cost model?
ML metrics alone don't tell the full story. This project quantifies the financial impact:
- **False Negative (missed failure): $100** — downtime, emergency repair, safety risk
- **False Positive (unnecessary maintenance): $10** — inspection cost, lost productivity

This lets engineers compare models not just on F1 or AUC, but on actual dollar impact.

---

## 📊 Evaluation Metrics Summary

### Classification (Failure Prediction)

| Metric | Why it's included |
|---|---|
| AUC-ROC | Discrimination ability across all thresholds |
| AUC-PR | Better than AUC-ROC for imbalanced classes |
| F1 Score | Balance of precision and recall |
| **MCC** | **Best single metric for imbalanced data** |
| Cohen's Kappa | Agreement corrected for random chance |
| Youden's J Threshold | Finds optimal decision boundary (not always 0.5) |
| Calibration Curve | Are predicted probabilities honest? |
| Confusion Matrix | Visualise FP/FN tradeoffs |
| Business Cost | Dollar impact of model errors |

### Regression (RUL Prediction)

| Metric | Why it's included |
|---|---|
| RMSE | Large errors penalised heavily |
| MAE | Robust average error in cycles |
| MAPE | Percentage error (scale-independent) |
| R² / Adjusted R² | Variance explained by the model |
| **NASA Score** | **Asymmetric: penalises late predictions more** |

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full end-to-end demo (recommended first step)
python demo.py --quick      # ~30 seconds
python demo.py              # Full version ~3–5 min

# 3. Train only
python train.py --quick

# 4. Start REST API (must train first)
python -m uvicorn api.main:app --reload
# → Interactive API docs: http://localhost:8000/docs

# 5. View experiment tracking
mlflow ui
# → MLflow UI: http://localhost:5000

# 6. Run unit tests
pytest tests/ -v --cov=src
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — are models loaded? |
| `GET` | `/model-info` | Loaded model name + training metadata |
| `POST` | `/predict` | Single engine: failure probability + RUL + risk level |
| `POST` | `/predict/batch` | Fleet-wide batch prediction |
| `POST` | `/drift-check` | PSI + KS drift detection against training baseline |

**Example request:**
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

**Example response:**
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

## 🐳 Docker

```bash
# Full stack: API + MLflow
docker compose up

# API → http://localhost:8000
# MLflow → http://localhost:5000
```

The Dockerfile uses a multi-stage build: a builder stage installs all dependencies (including compiled C extensions for LightGBM/scipy), then a slim runtime stage copies only what's needed — keeping the final image as small as possible.

---

## 🔬 Dataset

This project uses **synthetic NASA C-MAPSS-like turbofan engine data** generated programmatically. Each simulated engine has:
- A random lifetime (150–350 cycles)
- A sigmoid health degradation curve with a random "knee" point where degradation accelerates
- 14 sensors with physically motivated degradation effects (e.g., temperatures rise, pressures fall)
- Per-engine baseline variation to model fleet heterogeneity

---

#   P r e d i c t i v e - M a i n t e n c e - S y s t e m  
 