"""Phase-1b metric diagnostics for edge masks, gradients, and branch behavior."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gradient_xy(array: np.ndarray, dx: float = 1.0, dz: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(array, dtype=np.float32)
    gx = np.zeros_like(a, dtype=np.float32)
    gz = np.zeros_like(a, dtype=np.float32)
    gx[..., :-1] = (a[..., 1:] - a[..., :-1]) / float(dx)
    gx[..., -1] = gx[..., -2]
    gz[..., :-1, :] = (a[..., 1:, :] - a[..., :-1, :]) / float(dz)
    gz[..., -1, :] = gz[..., -2, :]
    return gx, gz


def gradient_magnitude_np(array: np.ndarray, dx: float = 1.0, dz: float = 1.0) -> np.ndarray:
    gx, gz = gradient_xy(array, dx=dx, dz=dz)
    return np.sqrt(gx * gx + gz * gz + 1e-12)


def edge_mask_from_threshold(true_velocity: np.ndarray, threshold: float, dx: float = 1.0, dz: float = 1.0) -> np.ndarray:
    return gradient_magnitude_np(true_velocity, dx=dx, dz=dz) > float(threshold)


def edge_mask_percentile(true_velocity: np.ndarray, percentile: float, dx: float = 1.0, dz: float = 1.0) -> tuple[np.ndarray, float]:
    g = gradient_magnitude_np(true_velocity, dx=dx, dz=dz)
    threshold = float(np.percentile(g.reshape(-1), float(percentile)))
    return g > threshold, threshold


def masked_mae_identity(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    err = np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    mask = np.asarray(mask, dtype=bool)
    edge_pixels = int(mask.sum())
    nonedge_pixels = int(mask.size - edge_pixels)
    edge_sum = float(err[mask].sum()) if edge_pixels else 0.0
    nonedge_sum = float(err[~mask].sum()) if nonedge_pixels else 0.0
    full_mae = float((edge_sum + nonedge_sum) / max(1, mask.size))
    edge_mae = float(edge_sum / edge_pixels) if edge_pixels else 0.0
    nonedge_mae = float(nonedge_sum / nonedge_pixels) if nonedge_pixels else 0.0
    coverage = float(edge_pixels / max(1, mask.size))
    weighted = coverage * edge_mae + (1.0 - coverage) * nonedge_mae
    return {
        "edge_pixels": edge_pixels,
        "nonedge_pixels": nonedge_pixels,
        "edge_coverage": coverage,
        "full_MAE": full_mae,
        "edge_MAE": edge_mae,
        "nonedge_MAE": nonedge_mae,
        "weighted_identity_error": float(abs(full_mae - weighted)),
    }


def gradient_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None, dx: float = 1.0, dz: float = 1.0) -> dict[str, float]:
    pgx, pgz = gradient_xy(prediction, dx=dx, dz=dz)
    tgx, tgz = gradient_xy(target, dx=dx, dz=dz)
    diff = np.abs(pgx - tgx) + np.abs(pgz - tgz)
    pmag = np.sqrt(pgx * pgx + pgz * pgz + 1e-12)
    tmag = np.sqrt(tgx * tgx + tgz * tgz + 1e-12)
    if mask is None:
        mask = np.ones_like(tmag, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    denom = np.maximum(pmag * tmag, 1e-12)
    cosine = (pgx * tgx + pgz * tgz) / denom
    return {
        "gradient_l1_all": float(diff.mean()),
        "gradient_l1_edge": float(diff[mask].mean()) if mask.any() else float("nan"),
        "gradient_magnitude_MAE": float(np.abs(pmag - tmag).mean()),
        "gradient_direction_cosine_error": float((1.0 - np.clip(cosine[mask], -1.0, 1.0)).mean()) if mask.any() else float("nan"),
    }


def dilate_mask(mask: np.ndarray, tolerance_pixels: int) -> np.ndarray:
    if tolerance_pixels <= 0:
        return np.asarray(mask, dtype=bool)
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    kernel = 2 * int(tolerance_pixels) + 1
    out = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=int(tolerance_pixels))
    return out[0, 0].numpy() > 0.5


def edge_prf(pred_mask: np.ndarray, true_mask: np.ndarray, tolerance_pixels: int = 1) -> dict[str, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    true = np.asarray(true_mask, dtype=bool)
    true_d = dilate_mask(true, tolerance_pixels)
    pred_d = dilate_mask(pred, tolerance_pixels)
    tp_p = int((pred & true_d).sum())
    tp_t = int((true & pred_d).sum())
    pred_count = int(pred.sum())
    true_count = int(true.sum())
    precision = tp_p / pred_count if pred_count else 0.0
    recall = tp_t / true_count if true_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"edge_precision": float(precision), "edge_recall": float(recall), "edge_F1": float(f1)}


def gaussian_smooth_np(array: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    x = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    radius = max(1, int(round(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1 = torch.exp(-(coords.square()) / (2.0 * float(sigma) * float(sigma)))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1)[None, None]
    y = F.conv2d(x, k2, padding=radius)
    return y[0, 0].numpy()


def global_ssim_np(prediction: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    mux, muy = x.mean(), y.mean()
    varx, vary = ((x - mux) ** 2).mean(), ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()
    dr = max(float(max(x.max(), y.max()) - min(x.min(), y.min())), 1e-6)
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (varx + vary + c2)))


def simple_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    err = np.asarray(prediction, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    out = {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err * err).mean())), "SSIM": global_ssim_np(prediction, target)}
    if mask is not None:
        mask_bool = np.asarray(mask, dtype=bool)
        out["edge_MAE"] = float(np.abs(err)[mask_bool].mean()) if mask_bool.any() else 0.0
    return out
