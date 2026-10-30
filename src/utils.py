"""
src/utils.py — shared utility helpers.

Responsibilities:
- Directory creation helpers.
- Structured logging setup.
- Timer context manager.
- JSON / CSV read-write wrappers.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a human-readable formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Filesystem helpers ────────────────────────────────────────────────────────

def ensure_dirs(*paths: Path) -> None:
    """Create all supplied directories (and parents) if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def save_json(data: dict[str, Any], path: Path) -> None:
    """Write *data* to *path* as pretty-printed JSON (creates parent dirs)."""
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return as dict.  Raises FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write DataFrame to CSV (creates parent dirs)."""
    ensure_dirs(path.parent)
    df.to_csv(path, index=False, **kwargs)


def load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Load CSV; raises FileNotFoundError if absent."""
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return pd.read_csv(path, **kwargs)


# ── Timer ─────────────────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer context manager."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __str__(self) -> str:
        return f"{self.elapsed:.2f}s"
