from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def record_time(nt: int, dt: float) -> float:
    if nt <= 0:
        raise ValueError("nt must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return float(nt) * float(dt)


def array_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, mmap_mode="r")
    finite = bool(np.isfinite(arr).all())
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": finite,
        "min": float(np.min(arr)) if finite else None,
        "max": float(np.max(arr)) if finite else None,
        "mean": float(np.mean(arr)) if finite else None,
    }


def strict_disjoint(left: list[int], right: list[int]) -> bool:
    return set(int(v) for v in left).isdisjoint(int(v) for v in right)


def wavelet_summary(f0: float, source_delay: float | None = None) -> dict[str, float | None]:
    if f0 <= 0.0:
        raise ValueError("f0 must be positive")
    peak_time = float(1.0 / f0 if source_delay is None else source_delay)
    return {
        "f0": float(f0),
        "source_delay": None if source_delay is None else float(source_delay),
        "peak_time": peak_time,
        "dominant_period": float(1.0 / f0),
    }
