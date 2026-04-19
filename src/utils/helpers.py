"""
helpers.py — Shared utility functions.

Covers:
  - Structured logging setup
  - Config loading from YAML
  - Directory creation
  - Context-manager timer
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml


# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler — logs/ directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "predictive_maintenance.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


# ─── Config ──────────────────────────────────────────────────────────────────

_config_cache: Dict[str, Any] = {}


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML config with in-process caching."""
    global _config_cache
    if _config_cache:
        return _config_cache

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


# ─── Directory helpers ───────────────────────────────────────────────────────

def ensure_dirs(cfg: Dict[str, Any]) -> None:
    """Create all project directories defined in config['paths']."""
    for key, rel_path in cfg.get("paths", {}).items():
        Path(rel_path).mkdir(parents=True, exist_ok=True)


# ─── Timer ────────────────────────────────────────────────────────────────────

class Timer:
    """Context-manager wall-clock timer.

    Usage::

        with Timer("Training XGBoost") as t:
            model.fit(X, y)
        print(t.elapsed_str)          # "Training XGBoost completed in 3.21 s"
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.start: float = 0.0
        self.end: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.end - self.start

    @property
    def elapsed_str(self) -> str:
        secs = self.elapsed
        if secs >= 60:
            return f"{self.label} completed in {secs/60:.1f} min"
        return f"{self.label} completed in {secs:.2f} s"

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.end = time.perf_counter()
