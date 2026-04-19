"""
PREDICTIVE MAINTENANCE PRO — FULL DEMO
Turbofan Engine Failure Prediction System

Run this script to see the ENTIRE ML pipeline end-to-end:
  [OK] Data generation & exploration
  [OK] Preprocessing & feature engineering
  [OK] Multi-model training (RF, XGBoost, LightGBM)
  [OK] Comprehensive evaluation (15+ metrics, 8 plots)
  [OK] Individual engine failure prediction
  [OK] Batch fleet monitoring
  [OK] Data drift detection
  [OK] Business cost analysis

Usage:
    python demo.py                    # Full demo (~3-5 min)
    python demo.py --quick            # Fast demo (~30 sec)
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table

console = Console(force_terminal=True, highlight=False)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def section(title: str, tag: str = ""):
    console.print()
    console.print(Rule(f"  {tag}  {title}  ", style="bold cyan"))
    console.print()


def success(msg: str):
    console.print(f"  [bold green][OK][/bold green]  {msg}")


def info(msg: str):
    console.print(f"  [bold blue][i][/bold blue]  {msg}")


def warn(msg: str):
    console.print(f"  [bold yellow][!][/bold yellow]  {msg}")


def metric_panel(title: str, metrics: dict, color: str = "blue") -> Panel:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="dim white", no_wrap=True)
    table.add_column("Value",  style=f"bold {color}", justify="right")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    return Panel(table, title=f"[bold]{title}[/bold]", border_style=color, padding=(0, 1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Fast mode -- smaller dataset")
    return p.parse_args()


# ─── Main demo ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ─── Banner ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        "[bold cyan]PREDICTIVE MAINTENANCE PRO[/bold cyan]\n"
        "[dim]Turbofan Engine Failure Prediction | Classical ML | Production Grade[/dim]\n\n"
        "[yellow]NASA C-MAPSS Synthetic Dataset[/yellow]  |  "
        "[yellow]RandomForest + XGBoost + LightGBM[/yellow]  |  "
        "[yellow]MLflow Tracking[/yellow]",
        border_style="cyan",
        padding=(1, 4),
    ))

    from src.utils.helpers import load_config, ensure_dirs
    cfg = load_config()

    if args.quick:
        cfg["data"]["n_train_units"]        = 15
        cfg["data"]["n_test_units"]         = 4
        cfg["models"]["cv_folds"]           = 3
        cfg["features"]["rolling_windows"]  = [5, 10]
        cfg["features"]["lag_steps"]        = [1, 3]
        cfg["features"]["ewma_spans"]       = [5]
        console.print(
            Panel("[bold yellow]>> QUICK MODE <<[/bold yellow] -- smaller dataset for fast demonstration",
                  border_style="yellow")
        )

    ensure_dirs(cfg)

    # =========================================================================
    # STAGE 1 -- Data Generation
    # =========================================================================
    section("STAGE 1 -- Synthetic Data Generation", "[DATA]")

    from src.data.loader import DataLoader
    from src.data.preprocessor import SENSOR_COLS

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), console=console, transient=True,
    ) as prog:
        task = prog.add_task("Generating turbofan engine data...", total=None)
        loader = DataLoader(cfg)
        train_raw, test_raw = loader.load(force_regenerate=True)

    n_train_units = train_raw["unit_id"].nunique()
    n_test_units  = test_raw["unit_id"].nunique()
    avg_lifetime  = train_raw.groupby("unit_id")["cycle"].max().mean()

    success(f"Generated {n_train_units} training engines + {n_test_units} test engines")
    info(f"Average engine lifetime: {avg_lifetime:.0f} cycles")
    info(f"Total sensor readings:  {len(train_raw):,} (train) + {len(test_raw):,} (test)")

    stats_table = Table(title="Training Data -- Sensor Statistics (sample)", box=box.ROUNDED,
                        border_style="dim blue")
    stats_table.add_column("Sensor", style="cyan")
    stats_table.add_column("Mean",   justify="right")
    stats_table.add_column("Std",    justify="right")
    stats_table.add_column("Min",    justify="right")
    stats_table.add_column("Max",    justify="right")

    sample_sensors = [c for c in SENSOR_COLS if c in train_raw.columns][:6]
    for col in sample_sensors:
        s = train_raw[col]
        stats_table.add_row(col[:28], f"{s.mean():.2f}", f"{s.std():.3f}",
                            f"{s.min():.2f}", f"{s.max():.2f}")
    console.print(stats_table)

    lifetimes = train_raw.groupby("unit_id")["cycle"].max()
    info(f"Engine lifetime: min={lifetimes.min():.0f} | median={lifetimes.median():.0f} | max={lifetimes.max():.0f} cycles")

    # =========================================================================
    # STAGE 2 -- Preprocessing
    # =========================================================================
    section("STAGE 2 -- Preprocessing & Label Engineering", "[PROC]")

    from src.data.preprocessor import DataPreprocessor

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        task = prog.add_task("Running preprocessing pipeline...", total=None)
        proc = DataPreprocessor(cfg)
        train_proc = proc.fit_transform(train_raw)
        test_proc  = proc.transform(test_raw)

    success("RUL computed and capped at 125 cycles")
    success(f"Binary labels created (failure within {cfg['data']['failure_horizon']} cycles)")
    success("StandardScaler fitted and applied")

    label_table = Table(box=box.SIMPLE, show_header=True)
    label_table.add_column("Split",        style="cyan")
    label_table.add_column("Total Rows",   justify="right")
    label_table.add_column("Failure (1)",  justify="right", style="red")
    label_table.add_column("Healthy (0)",  justify="right", style="green")
    label_table.add_column("Failure Rate", justify="right", style="yellow")

    for split, df in [("Train", train_proc), ("Test", test_proc)]:
        n1 = (df["failure_label"] == 1).sum()
        n0 = (df["failure_label"] == 0).sum()
        label_table.add_row(
            split, f"{len(df):,}", f"{n1:,}", f"{n0:,}",
            f"{n1/len(df)*100:.1f}%"
        )
    console.print(label_table)

    # =========================================================================
    # STAGE 3 -- Feature Engineering
    # =========================================================================
    section("STAGE 3 -- Time-Series Feature Engineering", "[FEAT]")

    from src.features.engineer import FeatureEngineer

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        prog.add_task("Engineering temporal features...", total=None)
        eng = FeatureEngineer(cfg)
        train_feat = eng.transform(train_proc)
        test_feat  = eng.transform(test_proc)

    feature_names = [c for c in train_feat.columns
                     if c not in ["unit_id", "cycle", "rul", "failure_label"]]

    success(f"Feature matrix: {len(feature_names)} features engineered")

    families = {
        "Original sensors":   [f for f in feature_names if not any(x in f for x in ["roll", "lag", "diff", "ewm", "slope", "entropy", "ratio"])],
        "Rolling statistics": [f for f in feature_names if "roll" in f],
        "Lag features":       [f for f in feature_names if "lag_" in f],
        "Rate of change":     [f for f in feature_names if "diff_" in f],
        "EWMA":               [f for f in feature_names if "ewm_" in f],
        "Trend slope":        [f for f in feature_names if "slope" in f],
        "Entropy":            [f for f in feature_names if "entropy" in f],
        "Cross-sensor":       [f for f in feature_names if "ratio" in f or "temp_diff" in f],
    }

    feat_table = Table(title="Feature Engineering Summary", box=box.ROUNDED, border_style="dim cyan")
    feat_table.add_column("Family",  style="cyan")
    feat_table.add_column("Count",   justify="right", style="bold white")
    feat_table.add_column("Example", style="dim white")

    for family, feats in families.items():
        if feats:
            feat_table.add_row(family, str(len(feats)), feats[0][:40])
    console.print(feat_table)

    proc_dir = Path(cfg["paths"]["data_processed"])
    train_feat.to_parquet(proc_dir / "train_featured.parquet", index=False)
    test_feat.to_parquet( proc_dir / "test_featured.parquet",  index=False)

    # =========================================================================
    # STAGE 4 -- Model Training
    # =========================================================================
    section("STAGE 4 -- Multi-Model Training (MLflow Tracked)", "[TRAIN]")

    from src.models.trainer import ModelTrainer

    X_train_clf, y_train_clf = eng.get_X_y_classifier(train_feat)
    X_train_reg, y_train_reg = eng.get_X_y_regressor(train_feat)
    X_test_clf,  y_test_clf  = eng.get_X_y_classifier(test_feat)
    X_test_reg,  y_test_reg  = eng.get_X_y_regressor(test_feat)

    info(f"Training shape: X={X_train_clf.shape} | Positive class: {y_train_clf.mean()*100:.1f}%")
    console.print()

    trainer = ModelTrainer(cfg)

    console.print("[bold cyan]  Binary Failure Classifiers (TimeSeriesSplit CV):[/bold cyan]")
    clf_results = trainer.train_classifiers(X_train_clf, y_train_clf)

    clf_table = Table(title="Classifier Leaderboard", box=box.ROUNDED, border_style="cyan")
    clf_table.add_column("Rank",      justify="center")
    clf_table.add_column("Model",     style="cyan")
    clf_table.add_column("AUC-ROC",   justify="right", style="bold green")
    clf_table.add_column("F1",        justify="right")
    clf_table.add_column("Recall",    justify="right")
    clf_table.add_column("Precision", justify="right")
    clf_table.add_column("Time (s)",  justify="right", style="dim")

    medals = ["[1]", "[2]", "[3]", "[4]"]
    for i, r in enumerate(clf_results):
        s = r["scores"]
        clf_table.add_row(
            medals[i] if i < len(medals) else str(i+1),
            r["name"],
            f"[bold green]{s.get('roc_auc', 0):.4f}[/bold green]",
            f"{s.get('f1', 0):.4f}",
            f"{s.get('recall', 0):.4f}",
            f"{s.get('precision', 0):.4f}",
            str(r["time_s"]),
        )
    console.print(clf_table)

    console.print("\n[bold cyan]  RUL Regressors (TimeSeriesSplit CV):[/bold cyan]")
    reg_results = trainer.train_regressors(X_train_reg, y_train_reg)

    reg_table = Table(title="Regressor Leaderboard", box=box.ROUNDED, border_style="magenta")
    reg_table.add_column("Rank",     justify="center")
    reg_table.add_column("Model",    style="magenta")
    reg_table.add_column("R2",       justify="right", style="bold green")
    reg_table.add_column("RMSE",     justify="right")
    reg_table.add_column("MAE",      justify="right")
    reg_table.add_column("Time (s)", justify="right", style="dim")

    for i, r in enumerate(reg_results):
        s = r["scores"]
        reg_table.add_row(
            medals[i] if i < len(medals) else str(i+1),
            r["name"],
            f"[bold green]{s.get('r2', 0):.4f}[/bold green]",
            f"{abs(s.get('neg_rmse', 0)):.4f}",
            f"{abs(s.get('neg_mae', 0)):.4f}",
            str(r["time_s"]),
        )
    console.print(reg_table)

    trainer.save_training_metadata(feature_names)
    success(f"All models logged to MLflow: '{cfg['mlflow']['experiment_name']}'")
    info(f"View UI: mlflow ui --backend-store-uri {cfg['paths']['mlruns']}")

    # =========================================================================
    # STAGE 5 -- Evaluation
    # =========================================================================
    section("STAGE 5 -- Comprehensive Model Evaluation", "[EVAL]")

    from src.models.evaluator import ModelEvaluator

    evaluator = ModelEvaluator(cfg)

    console.print(f"[bold]Champion Classifier:[/bold] [cyan]{trainer._best_clf_name}[/cyan]")
    clf_metrics = evaluator.evaluate_classifier(
        trainer.best_clf_, X_test_clf, y_test_clf,
        model_name=trainer._best_clf_name,
    )
    evaluator.plot_feature_importance(
        trainer.best_clf_, feature_names, model_name=trainer._best_clf_name
    )

    cm  = clf_metrics["confusion_matrix"]
    biz = clf_metrics["business_cost"]

    col1 = metric_panel("Classification Metrics", {
        "Accuracy":      clf_metrics["accuracy"],
        "Precision":     clf_metrics["precision"],
        "Recall":        clf_metrics["recall"],
        "F1 Score":      clf_metrics["f1_score"],
        "AUC-ROC":       clf_metrics["roc_auc"],
        "Avg Precision": clf_metrics["avg_precision"],
        "MCC":           clf_metrics["mcc"],
        "Cohen Kappa":   clf_metrics["cohen_kappa"],
        "Threshold":     clf_metrics["threshold"],
    }, "green")

    col2 = metric_panel("Confusion Matrix", {
        "True Positives":  cm[1][1],
        "True Negatives":  cm[0][0],
        "False Positives": cm[0][1],
        "False Negatives": cm[1][0],
        "Total Samples":   sum(sum(r) for r in cm),
    }, "blue")

    col3 = metric_panel("Business Cost ($)", {
        "Total Cost":    f"{biz['total_business_cost']:,}",
        "Baseline Cost": f"{biz['baseline_alarm_cost']:,}",
        "Savings":       f"{biz['estimated_savings']:,}",
        "Savings %":     f"{biz['savings_pct']}%",
        "FN cost/unit":  biz["fn_cost_per_unit"],
        "FP cost/unit":  biz["fp_cost_per_unit"],
    }, "yellow")

    console.print(Columns([col1, col2, col3]))

    console.print()
    console.print(f"[bold]Champion Regressor:[/bold] [magenta]{trainer._best_reg_name}[/magenta]")
    reg_metrics = evaluator.evaluate_regressor(
        trainer.best_reg_, X_test_reg, y_test_reg,
        model_name=trainer._best_reg_name,
    )

    reg_col1 = metric_panel("Regression Metrics (RUL Prediction)", {
        "RMSE (cycles)": reg_metrics["rmse"],
        "MAE  (cycles)": reg_metrics["mae"],
        "MAPE (%)":      reg_metrics["mape_pct"],
        "Median AE":     reg_metrics["median_ae"],
        "R2 Score":      reg_metrics["r2"],
        "Adjusted R2":   reg_metrics["adjusted_r2"],
        "NASA Score":    f"{reg_metrics['nasa_score']:.0f} (lower=better)",
    }, "magenta")

    console.print(Columns([reg_col1]))

    evaluator.save_report()
    plots = list(Path(cfg["paths"]["figures"]).glob("*.png"))
    success(f"Evaluation report saved: reports/evaluation_report.json")
    success(f"Plots saved: {cfg['paths']['figures']}/ ({len(plots)} charts)")

    # =========================================================================
    # STAGE 6 -- Live Predictions
    # =========================================================================
    section("STAGE 6 -- Live Engine Failure Prediction", "[PRED]")

    from src.models.predictor import Predictor

    predictor = Predictor(cfg)
    predictor._clf = trainer.best_clf_
    predictor._reg = trainer.best_reg_

    console.print("[bold]Single Engine Prediction (last 10 cycles):[/bold]\n")

    demo_engines = []
    for uid, grp in test_feat.groupby("unit_id"):
        last_rul = grp["rul"].iloc[-1]
        if last_rul < 10:   demo_engines.append(("CRITICAL", uid, grp))
        elif last_rul < 30: demo_engines.append(("WARNING",  uid, grp))
        elif last_rul < 80: demo_engines.append(("WATCH",    uid, grp))
        else:               demo_engines.append(("HEALTHY",  uid, grp))
        if len(demo_engines) >= 4:
            break

    pred_table = Table(box=box.ROUNDED, border_style="cyan")
    pred_table.add_column("Engine ID",  style="cyan",   justify="center")
    pred_table.add_column("True RUL",   justify="right")
    pred_table.add_column("Pred RUL",   justify="right", style="yellow")
    pred_table.add_column("P(Failure)", justify="right", style="bold")
    pred_table.add_column("Alert",      justify="center")
    pred_table.add_column("Risk Level", justify="center")
    pred_table.add_column("True State", justify="center")

    color_map = {"HEALTHY": "green", "WATCH": "yellow", "WARNING": "dark_orange", "CRITICAL": "red"}
    badge_map  = {"HEALTHY": "[OK]", "WATCH": "[--]", "WARNING": "[!!]", "CRITICAL": "[!!]"}

    for true_state, uid, grp in demo_engines:
        result = predictor.predict_single(grp, unit_id=uid, use_last_n=10)
        true_rul   = grp["rul"].iloc[-1]
        risk_color = color_map.get(result.risk_level, "white")
        pred_table.add_row(
            f"Engine-{uid}",
            f"{true_rul:.0f}",
            f"{result.rul_estimate:.0f}" if result.rul_estimate else "N/A",
            f"[bold {risk_color}]{result.failure_probability:.2%}[/bold {risk_color}]",
            "[bold red]FAIL[/bold red]" if result.failure_predicted else "[green] OK [/green]",
            f"[bold {risk_color}]{result.risk_level}[/bold {risk_color}]",
            true_state,
        )
    console.print(pred_table)

    console.print("\n[bold]Fleet-Wide Batch Prediction:[/bold]")
    batch_results = predictor.predict_batch(test_feat, use_last_n=10)

    fleet_summary = {"HEALTHY": 0, "WATCH": 0, "WARNING": 0, "CRITICAL": 0}
    for r in batch_results:
        fleet_summary[r.risk_level] = fleet_summary.get(r.risk_level, 0) + 1

    fleet_table = Table(box=box.SIMPLE)
    fleet_table.add_column("Risk Level", justify="center")
    fleet_table.add_column("Count",      justify="right")
    fleet_table.add_column("Pct",        justify="right")
    fleet_table.add_column("Recommended Action", style="dim")

    actions = {
        "HEALTHY":  "Continue normal operations",
        "WATCH":    "Increase monitoring frequency",
        "WARNING":  "Schedule maintenance within 2 weeks",
        "CRITICAL": "*** Immediate inspection required! ***",
    }

    for level in ["CRITICAL", "WARNING", "WATCH", "HEALTHY"]:
        count = fleet_summary.get(level, 0)
        pct   = count / max(len(batch_results), 1) * 100
        fleet_table.add_row(
            f"[bold {color_map[level]}]{level}[/bold {color_map[level]}]",
            f"[bold]{count}[/bold]",
            f"{pct:.1f}%",
            actions[level],
        )
    console.print(fleet_table)

    # =========================================================================
    # STAGE 7 -- Drift Detection
    # =========================================================================
    section("STAGE 7 -- Data Drift Detection", "[DRIFT]")

    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector(cfg)
    sensor_cols = [c for c in feature_names if "_roll_" not in c and "_lag_" not in c][:15]
    detector.fit_baseline(train_feat, sensor_cols)

    drifted_test = DriftDetector.simulate_drift(test_feat, sensor_cols[:5], drift_strength=2.5)

    console.print("[bold]A) No-drift scenario (normal test data):[/bold]")
    normal_report  = detector.detect_feature_drift(test_feat, sensor_cols)
    normal_summary = detector.summarize_drift_report(normal_report)

    console.print("[bold]B) Drift scenario (artificially shifted sensors):[/bold]")
    drift_report  = detector.detect_feature_drift(drifted_test, sensor_cols)
    drift_summary = detector.summarize_drift_report(drift_report)

    def drift_color(level: str) -> str:
        if "CRITICAL" in level: return "red"
        if "WARNING"  in level: return "yellow"
        if "WATCH"    in level: return "dark_orange"
        return "green"

    drift_table = Table(box=box.ROUNDED, border_style="yellow")
    drift_table.add_column("Scenario",         style="cyan")
    drift_table.add_column("Drifted Features", justify="right", style="bold")
    drift_table.add_column("Avg PSI",          justify="right")
    drift_table.add_column("Alert Level",      style="bold")

    for label, summary in [("Normal data", normal_summary), ("Drifted data", drift_summary)]:
        dc = drift_color(summary["alert_level"])
        drift_table.add_row(
            label,
            f"{summary['drifted_features']}/{summary['total_features']}",
            f"{summary['avg_psi']:.5f}",
            f"[bold {dc}]{summary['alert_level']}[/bold {dc}]",
        )
    console.print(drift_table)

    sorted_drift = sorted(drift_report.items(), key=lambda x: x[1]["psi"], reverse=True)[:5]
    if sorted_drift:
        console.print("\n[bold]Top Drifting Features:[/bold]")
        top_drift_table = Table(box=box.SIMPLE)
        top_drift_table.add_column("Feature",  style="cyan")
        top_drift_table.add_column("PSI",      justify="right", style="yellow")
        top_drift_table.add_column("KS p-val", justify="right")
        top_drift_table.add_column("Drifted?", justify="center")

        for feat, stats in sorted_drift:
            top_drift_table.add_row(
                feat[:35],
                f"{stats['psi']:.5f}",
                f"{stats['ks_pvalue']:.5f}",
                "[bold red]YES[/bold red]" if stats["drifted"] else "[green]NO[/green]",
            )
        console.print(top_drift_table)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    section("DEMO COMPLETE -- Project Summary", "[DONE]")

    console.print(Panel(
        f"[bold green][OK] Model Training:[/bold green]    {len(clf_results)} classifiers + {len(reg_results)} regressors\n"
        f"[bold green][OK] Champion Classifier:[/bold green] {trainer._best_clf_name} "
        f"(AUC={clf_metrics['roc_auc']:.4f} | F1={clf_metrics['f1_score']:.4f} | Recall={clf_metrics['recall']:.4f})\n"
        f"[bold green][OK] Champion Regressor:[/bold green]  {trainer._best_reg_name} "
        f"(R2={reg_metrics['r2']:.4f} | RMSE={reg_metrics['rmse']:.2f} | NASA={reg_metrics['nasa_score']:.0f})\n"
        f"[bold green][OK] Feature Engineering:[/bold green] {len(feature_names)} features from {len(SENSOR_COLS)} sensors\n"
        f"[bold green][OK] Fleet Monitoring:[/bold green]   CRITICAL={fleet_summary.get('CRITICAL',0)} | "
        f"WARNING={fleet_summary.get('WARNING',0)} | "
        f"WATCH={fleet_summary.get('WATCH',0)} | "
        f"HEALTHY={fleet_summary.get('HEALTHY',0)}\n"
        f"[bold green][OK] Business Cost:[/bold green]      ${biz['estimated_savings']:,} saved ({biz['savings_pct']}%)\n"
        f"[bold green][OK] Evaluation Plots:[/bold green]   {len(plots)} charts -> {cfg['paths']['figures']}/\n"
        f"[bold green][OK] MLflow Tracking:[/bold green]    All runs logged\n\n"
        f"[bold yellow]Next steps:[/bold yellow]\n"
        f"  python -m uvicorn api.main:app --reload   -> Start REST API (http://localhost:8000/docs)\n"
        f"  mlflow ui                                  -> View experiment tracker\n"
        f"  python train.py                            -> Full production training",
        title="[bold]PredictiveMaintenance Pro -- Complete[/bold]",
        border_style="cyan",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    main()
