"""
drift_detector.py — Production Model & Data Drift Detection.

Why drift matters in predictive maintenance:
  → Sensor calibration changes over time
  → New equipment batches behave differently
  → Seasonal effects (temperature, humidity) shift readings
  → If drift is undetected, model accuracy silently degrades

Detection methods:
  ┌──────────────────────────────────────────────────────────────────┐
  │ 1. Population Stability Index (PSI)                              │
  │    Industry standard for detecting distribution shift            │
  │    PSI < 0.1  → No significant shift                            │
  │    PSI 0.1–0.2 → Moderate shift (investigate)                   │
  │    PSI > 0.2  → Significant shift (retrain required)            │
  │                                                                  │
  │ 2. Kolmogorov-Smirnov Test                                       │
  │    Non-parametric two-sample test                                │
  │    p < alpha → distributions are significantly different         │
  │                                                                  │
  │ 3. Performance Degradation Monitor                               │
  │    Track rolling model accuracy; alert if drops below threshold  │
  └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.helpers import get_logger, load_config

logger = get_logger(__name__)


class DriftDetector:
    """Statistical drift detection for feature distributions and model performance."""

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or load_config()
        monitor_cfg = self.cfg["monitoring"]
        self.psi_threshold: float = monitor_cfg["psi_threshold"]
        self.ks_alpha:      float = monitor_cfg["ks_alpha"]
        self._baseline_stats: Dict = {}
        logger.info(
            "DriftDetector | PSI threshold=%.2f | KS alpha=%.2f",
            self.psi_threshold, self.ks_alpha,
        )

    # ── PSI ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
        eps: float = 1e-6,
    ) -> float:
        """Compute Population Stability Index between two distributions."""
        combined = np.concatenate([expected, actual])
        bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
        bins[0] -= eps
        bins[-1] += eps

        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual,   bins=bins)

        exp_pct = exp_counts / max(len(expected), 1)
        act_pct = act_counts / max(len(actual),   1)

        # Replace zeros to avoid log(0)
        exp_pct = np.where(exp_pct == 0, eps, exp_pct)
        act_pct = np.where(act_pct == 0, eps, act_pct)

        psi_values = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        return float(np.sum(psi_values))

    # ── Fit baseline ──────────────────────────────────────────────────────

    def fit_baseline(self, df: pd.DataFrame, feature_cols: List[str]) -> "DriftDetector":
        """Store reference distribution statistics from training data."""
        self._baseline_stats = {}
        for col in feature_cols:
            if col not in df.columns:
                continue
            arr = df[col].dropna().values
            self._baseline_stats[col] = {
                "values": arr,
                "mean":   float(arr.mean()),
                "std":    float(arr.std()),
                "q25":    float(np.percentile(arr, 25)),
                "q75":    float(np.percentile(arr, 75)),
            }
        logger.info("Baseline fitted on %d features", len(self._baseline_stats))
        return self

    # ── Detect drift ──────────────────────────────────────────────────────

    def detect_feature_drift(
        self, new_df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Run PSI + KS test for each feature; return per-feature drift report.

        Returns::

            {
              "sensor_name": {
                "psi": 0.12,
                "psi_drift": False,
                "ks_stat": 0.08,
                "ks_pvalue": 0.23,
                "ks_drift": False,
                "drifted": False,
              },
              ...
            }
        """
        if not self._baseline_stats:
            raise RuntimeError("Baseline not fitted. Call fit_baseline() first.")

        cols = feature_cols or list(self._baseline_stats.keys())
        report: Dict[str, Dict] = {}

        drifted_features = []

        for col in cols:
            if col not in self._baseline_stats or col not in new_df.columns:
                continue

            baseline_arr = self._baseline_stats[col]["values"]
            new_arr      = new_df[col].dropna().values

            if len(new_arr) < 10:
                continue

            psi       = self._compute_psi(baseline_arr, new_arr)
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_arr, new_arr)

            psi_drift = psi > self.psi_threshold
            ks_drift  = ks_pvalue < self.ks_alpha
            drifted   = psi_drift or ks_drift

            report[col] = {
                "psi":       round(psi, 5),
                "psi_drift": psi_drift,
                "ks_stat":   round(ks_stat, 5),
                "ks_pvalue": round(ks_pvalue, 5),
                "ks_drift":  ks_drift,
                "drifted":   drifted,
            }

            if drifted:
                drifted_features.append(col)

        n_total   = len(report)
        n_drifted = len(drifted_features)

        logger.info(
            "Drift check | %d/%d features drifted | PSI threshold=%.2f",
            n_drifted, n_total, self.psi_threshold,
        )
        if drifted_features:
            logger.warning("Drifted features: %s", drifted_features[:10])

        return report

    # ── Summary ───────────────────────────────────────────────────────────

    @staticmethod
    def summarize_drift_report(report: Dict[str, Dict]) -> Dict:
        """Aggregate drift report into an executive summary."""
        total     = len(report)
        drifted   = sum(1 for v in report.values() if v.get("drifted"))
        psi_drift = sum(1 for v in report.values() if v.get("psi_drift"))
        ks_drift  = sum(1 for v in report.values() if v.get("ks_drift"))
        avg_psi   = np.mean([v["psi"] for v in report.values()]) if report else 0.0

        alert_level = "OK"
        if drifted / max(total, 1) > 0.5:
            alert_level = "CRITICAL — Retrain required"
        elif drifted / max(total, 1) > 0.2:
            alert_level = "WARNING — Investigate drift"
        elif drifted > 0:
            alert_level = "WATCH — Minor drift detected"

        return {
            "total_features":   total,
            "drifted_features": drifted,
            "psi_drifted":      psi_drift,
            "ks_drifted":       ks_drift,
            "avg_psi":          round(float(avg_psi), 5),
            "alert_level":      alert_level,
        }

    # ── Simulate degraded data (for testing) ──────────────────────────────

    @staticmethod
    def simulate_drift(df: pd.DataFrame, feature_cols: List[str], drift_strength: float = 2.0) -> pd.DataFrame:
        """
        Add artificial drift to test features.
        Used in demo.py to showcase drift detection.
        """
        drifted = df.copy()
        rng = np.random.default_rng(99)
        for col in feature_cols[:5]:   # drift a subset of features
            if col in drifted.columns:
                shift = drifted[col].std() * drift_strength
                drifted[col] = drifted[col] + shift + rng.normal(0, shift * 0.2, len(drifted))
        return drifted
