"""
engineer.py — Time-Series Feature Engineering.

Transforms preprocessed sensor data into an ML-ready feature matrix by
extracting temporal patterns that classical models cannot learn on their own.

Feature families:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Rolling statistics   — mean, std, min, max, skew     │
  │ 2. Lag features         — value k cycles ago            │
  │ 3. Rate-of-change       — first-order difference        │
  │ 4. EWMA                 — exponentially weighted mean   │
  │ 5. Trend slope          — linear regression over window │
  │ 6. Rolling entropy      — approximate entropy proxy     │
  │ 7. Cross-sensor ratio   — ratio between related sensors │
  └─────────────────────────────────────────────────────────┘

Processing is done per engine (group_by unit_id) so temporal ordering is
respected. Missing values from windowing are forward-filled then backward-
filled within each engine, then dropped if still NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.helpers import get_logger, load_config, Timer

logger = get_logger(__name__)


class FeatureEngineer:
    """Builds a rich feature matrix from preprocessed sensor DataFrame."""

    def __init__(self, cfg: Optional[Dict] = None):
        self.cfg = cfg or load_config()
        feat_cfg = self.cfg["features"]

        self.windows:      List[int] = feat_cfg["rolling_windows"]
        self.lags:         List[int] = feat_cfg["lag_steps"]
        self.ewma_spans:   List[int] = feat_cfg["ewma_spans"]
        self.trend_window: int       = feat_cfg["trend_window"]

        self._feature_names: List[str] = []
        logger.info(
            "FeatureEngineer | windows=%s | lags=%s | ewma_spans=%s",
            self.windows, self.lags, self.ewma_spans,
        )

    # ── Core helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _slope(values: np.ndarray) -> float:
        """Least-squares slope of a short time window."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        slope, *_ = np.polyfit(x, values, 1)
        return float(slope)

    @staticmethod
    def _rolling_entropy(series: pd.Series, window: int) -> pd.Series:
        """Rolling approximate entropy (variance-based proxy)."""
        def _ent(w: np.ndarray) -> float:
            if w.std() < 1e-8:
                return 0.0
            w_norm = w / (w.std() + 1e-8)
            # Histogram entropy over 4 bins
            counts, _ = np.histogram(w_norm, bins=4)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log(probs + 1e-10)))

        return series.rolling(window, min_periods=window // 2).apply(_ent, raw=True)

    # ── Feature families ─────────────────────────────────────────────────

    def _rolling_stats(self, grp: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        out = grp.copy()
        for col in sensor_cols:
            for w in self.windows:
                roll = grp[col].rolling(w, min_periods=max(w // 2, 1))
                out[f"{col}_roll_mean_{w}"] = roll.mean()
                out[f"{col}_roll_std_{w}"]  = roll.std().fillna(0)
                out[f"{col}_roll_min_{w}"]  = roll.min()
                out[f"{col}_roll_max_{w}"]  = roll.max()
        return out

    def _lag_features(self, grp: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        out = grp.copy()
        for col in sensor_cols:
            for lag in self.lags:
                out[f"{col}_lag_{lag}"] = grp[col].shift(lag)
        return out

    def _rate_of_change(self, grp: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        out = grp.copy()
        for col in sensor_cols:
            out[f"{col}_diff_1"] = grp[col].diff(1).fillna(0)
            out[f"{col}_diff_5"] = grp[col].diff(5).fillna(0)
        return out

    def _ewma_features(self, grp: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        out = grp.copy()
        for col in sensor_cols:
            for span in self.ewma_spans:
                out[f"{col}_ewm_{span}"] = grp[col].ewm(span=span, min_periods=1).mean()
        return out

    def _trend_slope(self, grp: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        out = grp.copy()
        for col in sensor_cols:
            out[f"{col}_slope_{self.trend_window}"] = (
                grp[col]
                .rolling(self.trend_window, min_periods=3)
                .apply(self._slope, raw=True)
                .fillna(0)
            )
        return out

    def _entropy_features(self, grp: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
        """Rolling entropy for top-4 sensors (computationally expensive)."""
        out = grp.copy()
        # Limit to most informative sensors to keep runtime reasonable
        top_sensors = sensor_cols[:4] if len(sensor_cols) > 4 else sensor_cols
        for col in top_sensors:
            out[f"{col}_entropy_10"] = self._rolling_entropy(grp[col], window=10)
        return out

    def _cross_sensor_features(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Physics-informed ratio features."""
        out = grp.copy()
        try:
            # Pressure ratio (compressor health indicator)
            if "P30_HPC_outlet_pressure" in grp and "P2_fan_inlet_pressure" in grp:
                denom = grp["P2_fan_inlet_pressure"].replace(0, np.nan)
                out["pressure_ratio_30_2"] = grp["P30_HPC_outlet_pressure"] / denom

            # Temperature differential (turbine efficiency)
            if "T50_LPT_outlet_temp" in grp and "T30_HPC_outlet_temp" in grp:
                out["temp_diff_50_30"] = grp["T50_LPT_outlet_temp"] - grp["T30_HPC_outlet_temp"]

            # Speed index
            if "Nf_fan_speed" in grp and "Nc_core_speed" in grp:
                denom = grp["Nc_core_speed"].replace(0, np.nan)
                out["fan_core_speed_ratio"] = grp["Nf_fan_speed"] / denom
        except Exception as e:
            logger.warning("Cross-sensor feature error: %s", e)
        return out

    # ── Orchestration ─────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature families on the full DataFrame.

        Processing is done per engine to preserve temporal integrity.
        """
        sensor_cols = [c for c in df.columns if c not in
                       ["unit_id", "cycle", "rul", "failure_label",
                        "op_setting_1", "op_setting_2", "op_setting_3"]]

        logger.info(
            "Engineering features for %d sensors on %d rows…",
            len(sensor_cols), len(df)
        )

        groups = []
        with Timer("Feature engineering") as t:
            for uid, grp in df.groupby("unit_id"):
                grp = grp.sort_values("cycle").copy()
                grp = self._rolling_stats(grp, sensor_cols)
                grp = self._lag_features(grp, sensor_cols)
                grp = self._rate_of_change(grp, sensor_cols)
                grp = self._ewma_features(grp, sensor_cols)
                grp = self._trend_slope(grp, sensor_cols)
                grp = self._entropy_features(grp, sensor_cols)
                grp = self._cross_sensor_features(grp)
                groups.append(grp)

        result = pd.concat(groups, ignore_index=True)
        result = result.groupby("unit_id").apply(
            lambda g: g.ffill().bfill()
        ).reset_index(drop=True)

        before = len(result)
        result.dropna(inplace=True)
        logger.info(
            "%s | %d → %d rows | %d features",
            t.elapsed_str, before, len(result), result.shape[1]
        )

        self._feature_names = [
            c for c in result.columns
            if c not in ["unit_id", "cycle", "rul", "failure_label"]
        ]
        return result

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def get_X_y_classifier(self, df: pd.DataFrame):
        """Return (X, y) for binary failure classification."""
        feature_cols = [c for c in df.columns if c not in
                        ["unit_id", "cycle", "rul", "failure_label"]]
        return df[feature_cols], df["failure_label"]

    def get_X_y_regressor(self, df: pd.DataFrame):
        """Return (X, y) for RUL regression."""
        feature_cols = [c for c in df.columns if c not in
                        ["unit_id", "cycle", "rul", "failure_label"]]
        return df[feature_cols], df["rul"]
