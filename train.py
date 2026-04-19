"""
train.py — Main training pipeline CLI.

Usage:
    python train.py                          # Full pipeline
    python train.py --quick                  # Fast demo mode (fewer engines)
    python train.py --skip-classifiers       # Only train regressors
    python train.py --force-regenerate       # Regenerate synthetic data

Pipeline stages:
    [1] Load / Generate synthetic C-MAPSS data
    [2] Preprocess: RUL calculation, labeling, scaling
    [3] Feature engineering: rolling stats, lags, trends, EWMA
    [4] Train classifiers + regressors with MLflow tracking
    [5] Evaluate champion models on holdout set
    [6] Save models, scaler, feature list, evaluation report
    [7] Fit drift detector baseline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.monitoring.drift_detector import DriftDetector
from src.utils.helpers import get_logger, load_config, ensure_dirs, Timer

logger = get_logger("train")


def parse_args():
    p = argparse.ArgumentParser(description="PredictiveMaintenance training pipeline")
    p.add_argument("--quick",            action="store_true", help="Quick demo mode (smaller dataset)")
    p.add_argument("--skip-classifiers", action="store_true", help="Skip classifier training")
    p.add_argument("--skip-regressors",  action="store_true", help="Skip regressor training")
    p.add_argument("--force-regenerate", action="store_true", help="Regenerate synthetic data")
    p.add_argument("--config",           default="config.yaml", help="Path to config file")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    ensure_dirs(cfg)

    # Quick-mode overrides
    if args.quick:
        cfg["data"]["n_train_units"] = 20
        cfg["data"]["n_test_units"]  = 5
        cfg["models"]["cv_folds"]    = 3
        cfg["features"]["rolling_windows"] = [5, 10]
        cfg["features"]["lag_steps"]       = [1, 3]
        cfg["features"]["ewma_spans"]      = [5]
        logger.info("Quick mode enabled — smaller dataset & fewer features")

    # ─── Stage 1: Data loading ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 — Data loading")
    logger.info("=" * 60)

    with Timer("Data loading") as t:
        loader = DataLoader(cfg)
        train_raw, test_raw = loader.load(force_regenerate=args.force_regenerate)

    logger.info(
        "%s | train=%d rows | test=%d rows",
        t.elapsed_str, len(train_raw), len(test_raw),
    )

    # ─── Stage 2: Preprocessing ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 — Preprocessing")
    logger.info("=" * 60)

    with Timer("Preprocessing") as t:
        proc = DataPreprocessor(cfg)
        train_proc = proc.fit_transform(train_raw)
        test_proc  = proc.transform(test_raw)

    logger.info(
        "%s | train failure rate=%.2f%% | test failure rate=%.2f%%",
        t.elapsed_str,
        train_proc["failure_label"].mean() * 100,
        test_proc["failure_label"].mean()  * 100,
    )

    # ─── Stage 3: Feature engineering ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 3 — Feature engineering")
    logger.info("=" * 60)

    with Timer("Feature engineering") as t:
        eng = FeatureEngineer(cfg)
        train_feat = eng.transform(train_proc)
        test_feat  = eng.transform(test_proc)

    feature_names = [c for c in train_feat.columns
                     if c not in ["unit_id", "cycle", "rul", "failure_label"]]
    logger.info(
        "%s | %d features | train=%d rows | test=%d rows",
        t.elapsed_str, len(feature_names), len(train_feat), len(test_feat),
    )

    # Save featured data for API drift baseline
    proc_dir = Path(cfg["paths"]["data_processed"])
    train_feat.to_parquet(proc_dir / "train_featured.parquet", index=False)
    test_feat.to_parquet( proc_dir / "test_featured.parquet",  index=False)
    logger.info("Featured data saved to %s", proc_dir)

    # ─── Stage 4: Training ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 4 — Model training")
    logger.info("=" * 60)

    X_train_clf, y_train_clf = eng.get_X_y_classifier(train_feat)
    X_train_reg, y_train_reg = eng.get_X_y_regressor(train_feat)
    X_test_clf,  y_test_clf  = eng.get_X_y_classifier(test_feat)
    X_test_reg,  y_test_reg  = eng.get_X_y_regressor(test_feat)

    trainer = ModelTrainer(cfg)

    if not args.skip_classifiers:
        logger.info("Training binary failure classifiers…")
        with Timer("Classifier training") as t:
            clf_results = trainer.train_classifiers(X_train_clf, y_train_clf)
        logger.info("%s", t.elapsed_str)

    if not args.skip_regressors:
        logger.info("Training RUL regressors…")
        with Timer("Regressor training") as t:
            reg_results = trainer.train_regressors(X_train_reg, y_train_reg)
        logger.info("%s", t.elapsed_str)

    trainer.save_training_metadata(feature_names)

    # ─── Stage 5: Evaluation ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 5 — Model evaluation")
    logger.info("=" * 60)

    evaluator = ModelEvaluator(cfg)

    if not args.skip_classifiers and trainer.best_clf_ is not None:
        clf_metrics = evaluator.evaluate_classifier(
            trainer.best_clf_, X_test_clf, y_test_clf,
            model_name=trainer._best_clf_name,
        )
        evaluator.plot_feature_importance(
            trainer.best_clf_, feature_names,
            model_name=trainer._best_clf_name,
        )

    if not args.skip_regressors and trainer.best_reg_ is not None:
        reg_metrics = evaluator.evaluate_regressor(
            trainer.best_reg_, X_test_reg, y_test_reg,
            model_name=trainer._best_reg_name,
        )

    evaluator.save_report()

    # ─── Stage 6: Drift baseline ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 6 — Drift detection baseline")
    logger.info("=" * 60)

    detector = DriftDetector(cfg)
    detector.fit_baseline(train_feat, feature_names[:30])
    logger.info("Drift baseline fitted on %d features", min(30, len(feature_names)))

    # ─── Done ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("✅  Training pipeline complete!")
    logger.info("    Models  → %s", cfg["paths"]["models"])
    logger.info("    Figures → %s", cfg["paths"]["figures"])
    logger.info("    Report  → reports/evaluation_report.json")
    logger.info("    MLflow  → mlflow ui --backend-store-uri %s", cfg["paths"]["mlruns"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
