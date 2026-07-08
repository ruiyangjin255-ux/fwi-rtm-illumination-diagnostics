"""Corrected PASD Phase-3R metrics computed from fresh prediction archives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import global_ssim_np, gradient_magnitude_np, gradient_metrics, masked_mae_identity, sha256_file


OFFICIAL_CORRECTED_METRICS = (
    "MAE",
    "RMSE",
    "Relative_Error",
    "SSIM",
    "PSNR",
    "source_threshold_edge_MAE",
    "nonedge_MAE",
    "gradient_l1_all",
    "gradient_l1_edge",
    "gradient_magnitude_MAE_all",
    "gradient_magnitude_MAE_edge",
    "edge_precision",
    "edge_recall",
    "edge_F1",
)

DEPRECATED_METRIC_FIELDS = (
    "edge_MAE",
    "gradient_error",
    "edge_mae",
    "gradient_error",
    "target_adaptive_edge_MAE",
)


def stable_id_hash(sample_ids: np.ndarray) -> str:
    values = np.asarray(sample_ids, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def source_threshold_from_config(config: dict[str, Any]) -> float:
    raw = str(config.get("edge_threshold_source", ""))
    if "tau_source_90=" in raw:
        return float(raw.split("tau_source_90=", 1)[1].split()[0].rstrip(",;"))
    if "source_train_edge_threshold_tau90" in config:
        return float(config["source_train_edge_threshold_tau90"])
    return 9.999999974752427e-07


def archive_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        required = {"sample_id", "prediction", "target"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path} missing archive keys: {sorted(missing)}")
        return {key: np.asarray(payload[key]) for key in required}


def binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dz in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dz : dz + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def edge_prf_np(pred_mask: np.ndarray, true_mask: np.ndarray, tolerance_pixels: int = 1) -> dict[str, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    true_d = binary_dilate(true, tolerance_pixels)
    pred_d = binary_dilate(pred, tolerance_pixels)
    tp_p = int((pred & true_d).sum())
    tp_t = int((true & pred_d).sum())
    pred_count = int(pred.sum())
    true_count = int(true.sum())
    precision = tp_p / pred_count if pred_count else 0.0
    recall = tp_t / true_count if true_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"edge_precision": float(precision), "edge_recall": float(recall), "edge_F1": float(f1)}


def corrected_sample_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    tau_source: float,
    tau_pred: float,
    dx: float = 1.0,
    dz: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    error = prediction - target
    abs_error = np.abs(error)
    true_grad = gradient_magnitude_np(target, dx=dx, dz=dz)
    pred_grad = gradient_magnitude_np(prediction, dx=dx, dz=dz)
    true_mask = true_grad > float(tau_source)
    pred_mask = pred_grad > float(tau_pred)
    identity = masked_mae_identity(prediction, target, true_mask)
    grad = gradient_metrics(prediction, target, mask=true_mask, dx=dx, dz=dz)
    mag_abs = np.abs(pred_grad - true_grad)
    data_range = max(float(target.max() - target.min()), 1e-6)
    mse = float((error * error).mean())
    metrics = {
        "MAE": identity["full_MAE"],
        "RMSE": float(np.sqrt(mse)),
        "Relative_Error": float(np.linalg.norm(error.reshape(-1)) / max(np.linalg.norm(target.reshape(-1)), 1e-12)),
        "SSIM": global_ssim_np(prediction, target),
        "PSNR": float(20.0 * np.log10(data_range / max(np.sqrt(mse), 1e-12))),
        "source_threshold_edge_MAE": identity["edge_MAE"],
        "nonedge_MAE": identity["nonedge_MAE"],
        "gradient_l1_all": grad["gradient_l1_all"],
        "gradient_l1_edge": grad["gradient_l1_edge"],
        "gradient_magnitude_MAE_all": float(mag_abs.mean()),
        "gradient_magnitude_MAE_edge": float(mag_abs[true_mask].mean()) if true_mask.any() else 0.0,
        **edge_prf_np(pred_mask, true_mask, tolerance_pixels=1),
    }
    coverage = {
        "edge_pixels": int(true_mask.sum()),
        "nonedge_pixels": int(true_mask.size - true_mask.sum()),
        "edge_coverage": float(true_mask.mean()),
        "tau_source": float(tau_source),
        "tau_pred": float(tau_pred),
        "mask_condition": "strict_gt",
    }
    return metrics, coverage


def archive_sha(path: str | Path) -> str:
    return sha256_file(path)
