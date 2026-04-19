"""
tests/test_features.py — Unit tests for feature engineering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.utils.helpers import load_config


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["data"]["n_train_units"] = 5
    c["data"]["n_test_units"]  = 2
    c["features"]["rolling_windows"] = [5, 10]
    c["features"]["lag_steps"]       = [1, 3]
    c["features"]["ewma_spans"]      = [5]
    return c


@pytest.fixture(scope="module")
def raw_df(cfg):
    loader = DataLoader(cfg)
    return loader.load_sample(n_units=5)


@pytest.fixture(scope="module")
def processed_df(cfg, raw_df):
    proc = DataPreprocessor(cfg)
    return proc.fit_transform(raw_df)


@pytest.fixture(scope="module")
def featured_df(cfg, processed_df):
    eng = FeatureEngineer(cfg)
    return eng.transform(processed_df)


class TestDataLoader:
    def test_output_shape(self, raw_df):
        assert len(raw_df) > 0
        assert "unit_id" in raw_df.columns
        assert "cycle" in raw_df.columns

    def test_cycle_starts_at_1(self, raw_df):
        min_cycle = raw_df.groupby("unit_id")["cycle"].min()
        assert (min_cycle == 1).all()

    def test_no_missing_values_in_raw(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_sensor_columns_present(self, raw_df):
        assert "T30_HPC_outlet_temp" in raw_df.columns
        assert "Nf_fan_speed" in raw_df.columns


class TestDataPreprocessor:
    def test_rul_column_created(self, processed_df):
        assert "rul" in processed_df.columns

    def test_rul_min_is_zero(self, processed_df):
        # Each engine's last cycle has RUL=0 (or near 0)
        last_rul = processed_df.groupby("unit_id")["rul"].min()
        assert (last_rul >= 0).all()

    def test_rul_cap_applied(self, processed_df, cfg):
        rul_cap = cfg["data"]["rul_cap"]
        assert processed_df["rul"].max() <= rul_cap

    def test_binary_labels_created(self, processed_df):
        assert "failure_label" in processed_df.columns
        assert set(processed_df["failure_label"].unique()).issubset({0, 1})

    def test_label_balance(self, processed_df):
        # There should be both classes present
        counts = processed_df["failure_label"].value_counts()
        assert 0 in counts.index
        assert 1 in counts.index


class TestFeatureEngineer:
    def test_feature_count_increases(self, processed_df, featured_df):
        n_orig = processed_df.shape[1]
        n_feat = featured_df.shape[1]
        assert n_feat > n_orig, "Feature engineering should add columns"

    def test_no_unit_id_nan(self, featured_df):
        assert featured_df["unit_id"].isnull().sum() == 0

    def test_rolling_mean_exists(self, featured_df):
        roll_cols = [c for c in featured_df.columns if "roll_mean" in c]
        assert len(roll_cols) > 0

    def test_lag_features_exist(self, featured_df):
        lag_cols = [c for c in featured_df.columns if "_lag_" in c]
        assert len(lag_cols) > 0

    def test_ewma_features_exist(self, featured_df):
        ewm_cols = [c for c in featured_df.columns if "_ewm_" in c]
        assert len(ewm_cols) > 0

    def test_no_inf_values(self, featured_df):
        numeric = featured_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_get_X_y_classifier(self, cfg, featured_df):
        eng = FeatureEngineer(cfg)
        X, y = eng.get_X_y_classifier(featured_df)
        assert len(X) == len(y)
        assert "failure_label" not in X.columns
        assert y.name == "failure_label"

    def test_get_X_y_regressor(self, cfg, featured_df):
        eng = FeatureEngineer(cfg)
        X, y = eng.get_X_y_regressor(featured_df)
        assert len(X) == len(y)
        assert y.min() >= 0
