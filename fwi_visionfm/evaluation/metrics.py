from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_batch(value: Any) -> np.ndarray:
    array = _to_numpy(value).astype(np.float64, copy=False)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim == 3:
        return array
    raise ValueError(f"expected [B,H,W], [B,1,H,W], or [H,W], got {array.shape}")


def _paired_batches(prediction: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = _to_batch(prediction)
    truth = _to_batch(target)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction and target shapes must match, got {pred.shape} vs {truth.shape}")
    return pred, truth


def _resolve_data_range(prediction: Any, target: Any, data_range: float | None = None, eps: float = 1.0e-8) -> float:
    if data_range is not None:
        return max(float(data_range), eps)
    pred = _to_batch(prediction)
    truth = _to_batch(target)
    derived = max(float(pred.max() - pred.min()), float(truth.max() - truth.min()))
    return max(derived, eps)


def _sobel_edges(batch: np.ndarray) -> np.ndarray:
    gx_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64) / 4.0
    gy_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64) / 4.0
    padded = np.pad(batch, ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = np.zeros_like(batch, dtype=np.float64)
    gy = np.zeros_like(batch, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            patch = padded[:, i:i + batch.shape[1], j:j + batch.shape[2]]
            gx += gx_kernel[i, j] * patch
            gy += gy_kernel[i, j] * patch
    return np.sqrt(gx * gx + gy * gy)


def _laplacian(batch: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(batch, ((0, 0), (1, 1), (1, 1)), mode="edge")
    output = np.zeros_like(batch, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            output += kernel[i, j] * padded[:, i:i + batch.shape[1], j:j + batch.shape[2]]
    return output


def mae(prediction: Any, target: Any) -> float:
    pred, truth = _paired_batches(prediction, target)
    return float(np.mean(np.abs(pred - truth)))


def rmse(prediction: Any, target: Any) -> float:
    pred, truth = _paired_batches(prediction, target)
    diff = pred - truth
    return float(np.sqrt(np.mean(diff * diff)))


def relative_l2(prediction: Any, target: Any, eps: float = 1.0e-8) -> float:
    pred, truth = _paired_batches(prediction, target)
    numerator = np.sqrt(np.mean((pred - truth) ** 2))
    denominator = max(float(np.sqrt(np.mean(truth**2))), eps)
    return float(numerator / denominator)


def relative_mae(prediction: Any, target: Any, eps: float = 1.0e-8) -> float:
    pred, truth = _paired_batches(prediction, target)
    denominator = np.maximum(np.abs(truth), eps)
    return float(np.mean(np.abs(pred - truth) / denominator))


def relative_rmse(prediction: Any, target: Any, eps: float = 1.0e-8) -> float:
    pred, truth = _paired_batches(prediction, target)
    normalized = (pred - truth) / np.maximum(np.abs(truth), eps)
    return float(np.sqrt(np.mean(normalized * normalized)))


def psnr(prediction: Any, target: Any, data_range: float | None = None, eps: float = 1.0e-8) -> float:
    error = rmse(prediction, target)
    if error <= eps:
        return float("inf")
    resolved_range = _resolve_data_range(prediction, target, data_range=data_range, eps=eps)
    return float(20.0 * np.log10(resolved_range / error))


def ssim(prediction: Any, target: Any, data_range: float | None = None, eps: float = 1.0e-8) -> tuple[float | None, bool]:
    _paired_batches(prediction, target)
    _ = (data_range, eps)
    return None, False


def _fallback_ssim(pred: np.ndarray, truth: np.ndarray, data_range: float, eps: float) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    values = []
    for pred_image, truth_image in zip(pred, truth):
        pred_mean = float(pred_image.mean())
        truth_mean = float(truth_image.mean())
        pred_var = float(pred_image.var())
        truth_var = float(truth_image.var())
        covariance = float(((pred_image - pred_mean) * (truth_image - truth_mean)).mean())
        numerator = (2.0 * pred_mean * truth_mean + c1) * (2.0 * covariance + c2)
        denominator = (pred_mean**2 + truth_mean**2 + c1) * (pred_var + truth_var + c2)
        values.append(float(numerator / max(denominator, eps)))
    return float(np.mean(values))


def horizon_gradient_mae(prediction: Any, target: Any) -> float:
    pred, truth = _paired_batches(prediction, target)
    return float(np.mean(np.abs(np.diff(pred, axis=2) - np.diff(truth, axis=2))))


def vertical_gradient_mae(prediction: Any, target: Any) -> float:
    pred, truth = _paired_batches(prediction, target)
    return float(np.mean(np.abs(np.diff(pred, axis=1) - np.diff(truth, axis=1))))


def velocity_gradient_error(prediction: Any, target: Any) -> float:
    hz = horizon_gradient_mae(prediction, target)
    vz = vertical_gradient_mae(prediction, target)
    return float(0.5 * (hz + vz))


def edge_mae(prediction: Any, target: Any) -> float:
    pred, truth = _paired_batches(prediction, target)
    return float(np.mean(np.abs(_sobel_edges(pred) - _sobel_edges(truth))))


def edge_overlap(prediction: Any, target: Any, eps: float = 1.0e-8) -> float:
    pred, truth = _paired_batches(prediction, target)
    pred_edges = _sobel_edges(pred)
    truth_edges = _sobel_edges(truth)
    pred_mask = pred_edges > np.quantile(pred_edges, 0.75)
    truth_mask = truth_edges > np.quantile(truth_edges, 0.75)
    tp = float(np.logical_and(pred_mask, truth_mask).sum())
    fp = float(np.logical_and(pred_mask, np.logical_not(truth_mask)).sum())
    fn = float(np.logical_and(np.logical_not(pred_mask), truth_mask).sum())
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    return float(2.0 * precision * recall / max(precision + recall, eps))


def laplacian_mae(prediction: Any, target: Any) -> float:
    pred, truth = _paired_batches(prediction, target)
    return float(np.mean(np.abs(_laplacian(pred) - _laplacian(truth))))


def compute_velocity_metrics(
    prediction: Any,
    target: Any,
    *,
    data_range: float | None = None,
    eps: float = 1.0e-8,
) -> dict[str, Any]:
    ssim_value, ssim_available = ssim(prediction, target, data_range=data_range, eps=eps)
    pred, truth = _paired_batches(prediction, target)
    if ssim_value is None:
        ssim_value = _fallback_ssim(pred, truth, _resolve_data_range(pred, truth, data_range=data_range, eps=eps), eps)
        ssim_available = True
    mse = float(np.mean((pred - truth) ** 2))
    return {
        "loss": mse,
        "mae": mae(pred, truth),
        "rmse": rmse(pred, truth),
        "relative_l2": relative_l2(pred, truth, eps=eps),
        "relative_mae": relative_mae(pred, truth, eps=eps),
        "relative_rmse": relative_rmse(pred, truth, eps=eps),
        "psnr": psnr(pred, truth, data_range=data_range, eps=eps),
        "ssim": ssim_value,
        "ssim_available": ssim_available,
        "edge_mae": edge_mae(pred, truth),
        "edge_overlap": edge_overlap(pred, truth),
        "boundary_f1": edge_overlap(pred, truth),
        "laplacian_mae": laplacian_mae(pred, truth),
        "horizon_gradient_mae": horizon_gradient_mae(pred, truth),
        "vertical_gradient_mae": vertical_gradient_mae(pred, truth),
        "gradient_mae": velocity_gradient_error(pred, truth),
        "gradient_error": velocity_gradient_error(pred, truth),
    }
