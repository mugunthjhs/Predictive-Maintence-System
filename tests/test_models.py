"""
tests/test_models.py — Unit tests for model training, evaluation, and prediction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.models.evaluator import ModelEvaluator
from src.models.predictor import Predictor, PredictionResult
from src.monitoring.drift_detector import DriftDetector
from src.utils.helpers import load_config


@pytest.fixture
def cfg():
    c = load_config()
    c["evaluation"]["optimal_threshold"] = False
    return c


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10, random_state=42
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), pd.Series(y)


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=500, n_features=20, noise=5, random_state=42)
    y = np.clip(y, 0, 125)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(20)]), pd.Series(y)


@pytest.fixture
def trained_clf(binary_data):
    X, y = binary_data
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X[:400], y[:400])
    return clf, X[400:], y[400:]


@pytest.fixture
def trained_reg(regression_data):
    X, y = regression_data
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X[:400], y[:400])
    return reg, X[400:], y[400:]


class TestModelEvaluator:
    def test_evaluate_classifier_returns_dict(self, cfg, trained_clf):
        clf, X_test, y_test = trained_clf
        evaluator = ModelEvaluator(cfg)
        metrics = evaluator.evaluate_classifier(clf, X_test, y_test, model_name="test_rf")
        assert isinstance(metrics, dict)
        assert "roc_auc" in metrics
        assert "f1_score" in metrics
        assert "mcc" in metrics
        assert "cohen_kappa" in metrics
        assert "confusion_matrix" in metrics
        assert "business_cost" in metrics

    def test_auc_in_valid_range(self, cfg, trained_clf):
        clf, X_test, y_test = trained_clf
        metrics = ModelEvaluator(cfg).evaluate_classifier(clf, X_test, y_test)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_confusion_matrix_shape(self, cfg, trained_clf):
        clf, X_test, y_test = trained_clf
        metrics = ModelEvaluator(cfg).evaluate_classifier(clf, X_test, y_test)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_evaluate_regressor_returns_dict(self, cfg, trained_reg):
        reg, X_test, y_test = trained_reg
        metrics = ModelEvaluator(cfg).evaluate_regressor(reg, X_test, y_test)
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "nasa_score" in metrics
        assert metrics["rmse"] >= 0

    def test_nasa_score_symmetric(self):
        """Late prediction should cost more than early prediction."""
        y_true = np.array([50.0, 50.0])
        y_pred_early = np.array([40.0, 40.0])   # early (d < 0)
        y_pred_late  = np.array([60.0, 60.0])   # late  (d > 0)
        score_early = ModelEvaluator.nasa_score(y_true, y_pred_early)
        score_late  = ModelEvaluator.nasa_score(y_true, y_pred_late)
        assert score_late > score_early, "Late prediction should be penalised more"


class TestPredictor:
    def test_predict_returns_result(self, cfg, trained_clf, trained_reg):
        clf, X_test, _ = trained_clf
        reg, _, _ = trained_reg

        predictor = Predictor(cfg)
        predictor._clf = clf
        predictor._reg = reg

        # Add metadata columns
        X_with_meta = X_test.copy()
        X_with_meta["unit_id"] = 1
        X_with_meta["cycle"] = list(range(1, len(X_with_meta) + 1))

        result = predictor.predict_single(X_with_meta, unit_id=1)
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.failure_probability <= 1.0
        assert result.risk_level in {"HEALTHY", "WATCH", "WARNING", "CRITICAL"}

    def test_batch_predict(self, cfg, trained_clf, trained_reg):
        clf, X_test, _ = trained_clf
        reg, _, _ = trained_reg

        predictor = Predictor(cfg)
        predictor._clf = clf
        predictor._reg = reg

        X = X_test.copy()
        X["unit_id"] = np.repeat([1, 2], [len(X)//2, len(X) - len(X)//2])
        X["cycle"] = 1

        results = predictor.predict_batch(X)
        assert len(results) == 2


class TestDriftDetector:
    def test_psi_zero_for_same_distribution(self):
        arr = np.random.default_rng(0).normal(0, 1, 1000)
        psi = DriftDetector._compute_psi(arr, arr)
        assert psi < 0.01

    def test_psi_high_for_shifted_distribution(self):
        rng = np.random.default_rng(0)
        arr1 = rng.normal(0, 1, 1000)
        arr2 = rng.normal(5, 1, 1000)   # large shift
        psi = DriftDetector._compute_psi(arr1, arr2)
        assert psi > 0.2

    def test_drift_detection_no_drift(self, cfg):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"f1": rng.normal(0, 1, 200), "f2": rng.normal(5, 2, 200)})

        detector = DriftDetector(cfg)
        detector.fit_baseline(df, ["f1", "f2"])
        report = detector.detect_feature_drift(df, ["f1", "f2"])
        assert "f1" in report

    def test_drift_simulation_creates_shift(self, cfg):
        df = pd.DataFrame({"f1": np.random.normal(0, 1, 100), "f2": np.random.normal(0, 1, 100)})
        drifted = DriftDetector.simulate_drift(df, ["f1"], drift_strength=3.0)
        assert drifted["f1"].mean() > df["f1"].mean() + 1
