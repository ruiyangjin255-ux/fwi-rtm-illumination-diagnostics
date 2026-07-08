from __future__ import annotations

import numpy as np

from fwi_visionfm.evaluation.metrics import (
    compute_velocity_metrics as velocity_metrics,
    edge_mae,
    horizon_gradient_mae,
    laplacian_mae,
    mae,
    psnr,
    relative_l2,
    relative_mae,
    relative_rmse,
    rmse,
    ssim as _ssim_with_flag,
    velocity_gradient_error,
    vertical_gradient_mae,
)


def ssim_2d(prediction, target, data_range: float | None = None, eps: float = 1.0e-8):
    value, _ = _ssim_with_flag(prediction, target, data_range=data_range, eps=eps)
    if value is not None:
        return value
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if data_range is None:
        data_range = float(max(pred.max(), truth.max()) - min(pred.min(), truth.min()))
        if data_range <= eps:
            data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pred_mean = float(pred.mean())
    truth_mean = float(truth.mean())
    pred_var = float(pred.var())
    truth_var = float(truth.var())
    covariance = float(((pred - pred_mean) * (truth - truth_mean)).mean())
    numerator = (2.0 * pred_mean * truth_mean + c1) * (2.0 * covariance + c2)
    denominator = (pred_mean**2 + truth_mean**2 + c1) * (pred_var + truth_var + c2)
    return float(numerator / max(denominator, eps))


ssim_like = ssim_2d
gradient_error = velocity_gradient_error
