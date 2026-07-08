from __future__ import annotations

import numpy as np

Array = np.ndarray


def fit_linear_calibration(predictions: list[Array], targets: list[Array]) -> dict[str, float]:
    if not predictions or not targets or len(predictions) != len(targets):
        raise ValueError("predictions and targets must be non-empty lists with the same length")
    x = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in predictions])
    y = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in targets])
    if x.shape != y.shape:
        raise ValueError("flattened predictions and targets must have the same shape")
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    centered_x = x - mean_x
    variance_x = float(np.mean(centered_x * centered_x))
    if variance_x <= 1.0e-12:
        scale = 0.0
        bias = mean_y
    else:
        covariance_xy = float(np.mean(centered_x * (y - mean_y)))
        scale = covariance_xy / variance_x
        bias = mean_y - scale * mean_x
    return {"scale": float(scale), "bias": float(bias)}


def apply_linear_calibration(prediction: Array, params: dict[str, float]) -> Array:
    scale = float(params["scale"])
    bias = float(params["bias"])
    return (scale * np.asarray(prediction, dtype=np.float32) + bias).astype(np.float32)


def train_linear_calibration(
    predictions: list[Array],
    targets: list[Array],
    *,
    epochs: int = 20,
    learning_rate: float = 1.0e-8,
    initial: dict[str, float] | None = None,
) -> dict[str, object]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not predictions or not targets or len(predictions) != len(targets):
        raise ValueError("predictions and targets must be non-empty lists with the same length")
    x = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in predictions])
    y = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in targets])
    if x.shape != y.shape:
        raise ValueError("flattened predictions and targets must have the same shape")
    scale = float(initial["scale"]) if initial is not None and "scale" in initial else 1.0
    bias = float(initial["bias"]) if initial is not None and "bias" in initial else 0.0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        residual = scale * x + bias - y
        mse = float(np.mean(residual * residual))
        mae = float(np.mean(np.abs(residual)))
        grad_scale = float(2.0 * np.mean(residual * x))
        grad_bias = float(2.0 * np.mean(residual))
        history.append(
            {
                "epoch": float(epoch),
                "mse": mse,
                "mae": mae,
                "scale": scale,
                "bias": bias,
                "grad_scale": grad_scale,
                "grad_bias": grad_bias,
            }
        )
        scale -= learning_rate * grad_scale
        bias -= learning_rate * grad_bias
    return {"params": {"scale": float(scale), "bias": float(bias)}, "history": history}
