from __future__ import annotations

import numpy as np

from fwi_visionfm.config import LossConfig
from fwi_visionfm.metrics import gradient_error, mae, ssim_like

Array = np.ndarray


def combined_velocity_loss(prediction: Array, target: Array, cfg: LossConfig | None = None) -> dict[str, float]:
    cfg = cfg or LossConfig()
    mae_value = mae(prediction, target)
    gradient_value = gradient_error(prediction, target)
    ssim_value = ssim_like(prediction, target)
    ssim_loss = 0.0 if ssim_value is None else max(0.0, 1.0 - ssim_value)
    total = cfg.mae_weight * mae_value + cfg.gradient_weight * gradient_value + cfg.ssim_weight * ssim_loss
    return {
        "mae": float(mae_value),
        "gradient": float(gradient_value),
        "ssim": float(ssim_loss),
        "total": float(np.asarray(total, dtype=np.float64)),
    }
