from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip() or "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and not fieldnames:
        raise ValueError(f"no rows or fieldnames to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def markdown_table(rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> str:
    cols = list(columns or (rows[0].keys() if rows else []))
    if not cols:
        return "_No rows._\n"
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(out) + "\n"


def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-12) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    if aa.size != bb.size or aa.size == 0:
        return float("nan")
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= eps:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def finite_array(path: Path) -> np.ndarray:
    arr = np.load(path)
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return arr


def image_stats(arr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(arr, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return {"abs_mean": float("nan"), "abs_p95": float("nan"), "energy": float("nan")}
    abs_values = np.abs(values)
    return {
        "abs_mean": float(np.mean(abs_values)),
        "abs_p95": float(np.percentile(abs_values, 95.0)),
        "energy": float(np.mean(values * values)),
    }


def edge_mae(model: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None) -> float:
    gy_m, gx_m = np.gradient(np.asarray(model, dtype=float))
    gy_r, gx_r = np.gradient(np.asarray(reference, dtype=float))
    diff = np.abs(np.hypot(gx_m, gy_m) - np.hypot(gx_r, gy_r))
    if mask is not None:
        diff = diff[np.asarray(mask, dtype=bool)]
    return float(np.mean(diff)) if diff.size else float("nan")
