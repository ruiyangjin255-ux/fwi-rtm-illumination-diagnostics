from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from admit_fwi.diagnostics.admit_common import edge_mae


def _norm01(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def model_metrics(model: np.ndarray, true: np.ndarray) -> dict[str, float]:
    diff = np.asarray(model, dtype=float) - np.asarray(true, dtype=float)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "edge_mae": edge_mae(model, true),
        "gradient_mae": edge_mae(model, true),
    }


def run_lightweight_proxy(model_dir: Path) -> dict[str, Any]:
    true = np.load(model_dir / "true_velocity.npy").astype(np.float32)
    initial = np.load(model_dir / "initial_velocity.npy").astype(np.float32)
    if true.shape != initial.shape:
        raise ValueError(f"shape mismatch: {true.shape} vs {initial.shape}")
    if np.allclose(true, initial):
        raise ValueError("initial model must differ from true model")
    delta = true - initial
    illum = _norm01(1.0 / (1.0 + np.abs(np.gradient(initial)[0])))
    consensus = _norm01(1.0 / (1.0 + np.abs(np.gradient(delta)[0])))
    ecg = _norm01(illum * consensus)
    rng = np.random.default_rng(0)
    random_gate = (rng.random(true.shape) >= 0.75).astype(np.float32)
    gates = {
        "full": np.ones_like(true),
        "global": np.full_like(true, 0.25),
        "illumination": (illum >= np.percentile(illum, 65.0)).astype(np.float32) * 0.25,
        "ecg": (ecg >= np.percentile(ecg, 65.0)).astype(np.float32) * 0.25,
        "inverse": (illum <= np.percentile(illum, 35.0)).astype(np.float32) * 0.25,
        "random_seed_0": random_gate * 0.25,
    }
    rows = []
    for method, gate in gates.items():
        model = initial + gate * delta if method != "full" else initial + 0.25 * delta
        metrics = model_metrics(model, true)
        rows.append(
            {
                "method": method,
                "update_l2": float(np.linalg.norm((model - initial))),
                "active_fraction": float(np.mean(gate > 0.0)),
                "model_mae": metrics["mae"],
                "model_rmse": metrics["rmse"],
                "edge_mae": metrics["edge_mae"],
                "gradient_mae": metrics["gradient_mae"],
                "heldout_nrms": float(metrics["rmse"] / (np.mean(true) + 1.0e-8)),
                "trace_corr": float(np.corrcoef(model.ravel(), true.ravel())[0, 1]),
                "image_proxy": "SIMPLIFIED_IMAGE_PROXY",
            }
        )
    initial_metrics = model_metrics(initial, true)
    return {
        "status": "READY",
        "proxy_type": "SIMPLIFIED_DIAGNOSTIC_PROXY_NOT_FWI",
        "shape": list(true.shape),
        "initial_metrics": initial_metrics,
        "rows": rows,
    }
